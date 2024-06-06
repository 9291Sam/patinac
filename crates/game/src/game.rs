use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use dashmap::DashMap;
use gfx::{glm, nal};
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::geometry::ColliderHandle;
use util::AtomicF32;

use crate::entity::CollideableSmallVec;
use crate::renderpasses::RenderPassManager;
use crate::{Entity, SelfManagedEntity};

pub struct TickTag(());

pub struct Game
{
    this_weak: Weak<Game>,

    delta_time: AtomicF32,

    entities:              DashMap<util::Uuid, Weak<dyn Entity>>,
    // collideables:
    //     DashMap<util::Uuid, Option<(RigidBodyHandle, CollideableSmallVec<ColliderHandle>)>>,
    self_managed_entities: DashMap<util::Uuid, Arc<dyn SelfManagedEntity>>,

    // TODO: move to a seperate class
    renderer:            Arc<gfx::Renderer>,
    render_pass_manager: Arc<RenderPassManager>,
    render_pass_updater: util::WindowUpdater<gfx::RenderPassSendFunction>
}

impl UnwindSafe for Game {}
impl RefUnwindSafe for Game {}

impl Debug for Game
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Game")
    }
}

impl Drop for Game
{
    fn drop(&mut self)
    {
        self.entities
            .iter()
            .filter_map(|r| r.value().upgrade())
            .for_each(|strong| log::warn!("Retained Entity! {:?}", &*strong));
    }
}

// TODO: does Game need to know about renderer at all? it shouldn't...

impl Game
{
    pub fn new(
        renderer: Arc<gfx::Renderer>,
        render_pass_updater: util::WindowUpdater<gfx::RenderPassSendFunction>
    ) -> Arc<Self>
    {
        Arc::new_cyclic(|this_weak| {
            let this = Game {
                renderer: renderer.clone(),
                entities: DashMap::new(),
                self_managed_entities: DashMap::new(),
                delta_time: AtomicF32::new(0.0),
                this_weak: this_weak.clone(),
                render_pass_manager: Arc::new(RenderPassManager::new(renderer)),
                render_pass_updater
            };

            this.render_pass_updater
                .update(this.render_pass_manager.clone().generate_renderpass_vec());

            this
        })
    }

    pub fn get_renderer(&self) -> &Arc<gfx::Renderer>
    {
        &self.renderer
    }

    pub fn get_renderpass_manager(&self) -> &RenderPassManager
    {
        &self.render_pass_manager
    }

    pub fn get_delta_time(&self) -> f32
    {
        self.delta_time.load(Ordering::Acquire)
    }

    pub fn register(&self, entity: Arc<dyn Entity>)
    {
        if let Some(self_managed) = entity.clone().as_self_managed()
        {
            self.self_managed_entities
                .insert(self_managed.get_uuid(), self_managed);
        }

        if let Some(collideable) = entity.as_collideable()
        {}

        self.entities
            .insert(entity.get_uuid(), Arc::downgrade(&entity));
    }

    pub fn enter_tick_loop(&self, poll_continue_func: &dyn Fn() -> bool)
    {
        let mut prev = std::time::Instant::now();
        let mut delta_time: f64;

        let tick_pool = util::ThreadPool::new(4, "Game Tick");

        // use rapier3d::prelude::*;

        // let mut rigid_body_set = RigidBodySet::new();
        // let mut collider_set = ColliderSet::new();

        // let mut player_controller = KinematicCharacterController::default();

        // // Create the ground.
        // let collider = ColliderBuilder::cuboid(128.0, 1.0, 128.0).build();
        // collider_set.insert(collider);

        // // Create the bounding ball.
        // let rigid_body = RigidBodyBuilder::dynamic()
        //     .translation(vector![0.0, 100.0, 0.0])
        //     .build();
        // let collider = ColliderBuilder::ball(5.0)
        //     .contact_force_event_threshold(f32::MIN_POSITIVE)
        //     .restitution(1.0)
        //     .build();
        // let ball_body_handle = rigid_body_set.insert(rigid_body);
        // collider_set.insert_with_parent(collider, ball_body_handle, &mut
        // rigid_body_set);

        // // Create other structures necessary for the simulation.
        // let gravity = vector![0.0, -0.823241, 0.0];
        // let mut physics_pipeline = PhysicsPipeline::new();
        // let mut island_manager = IslandManager::new();
        // let mut broad_phase = BroadPhaseMultiSap::new();
        // let mut narrow_phase = NarrowPhase::new();
        // let mut impulse_joint_set = ImpulseJointSet::new();
        // let mut multibody_joint_set = MultibodyJointSet::new();
        // let mut ccd_solver = CCDSolver::new();
        // let mut query_pipeline = QueryPipeline::new();
        // let physics_hooks = ();
        // let event_handler = ();

        // let minimum_tick_time = Duration::from_micros(100);

        while poll_continue_func()
        {
            let now = std::time::Instant::now();

            let delta_duration = now - prev;
            delta_time = delta_duration.as_secs_f64();
            prev = now;

            // if let Some(d) = minimum_tick_time.checked_sub(delta_duration)
            // {
            //     spin_sleep::sleep(d);
            // }

            self.delta_time.store(delta_time as f32, Ordering::Release);

            let strong_game = self.this_weak.upgrade().unwrap();

            self.self_managed_entities
                .retain(|_, strong_entity| strong_entity.is_alive());

            let mut futures = Vec::new();

            self.entities.retain(|_, weak_entity| {
                if let Some(strong_entity) = weak_entity.upgrade()
                {
                    let local_game: Arc<Game> = strong_game.clone();
                    futures.push(tick_pool.run_async(move || {
                        strong_entity.tick(&local_game, TickTag(()));
                    }));

                    true
                }
                else
                {
                    false
                }
            });

            futures.into_iter().for_each(|f| f.get());
        }
    }
}

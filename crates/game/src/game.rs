use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::Ordering::{self, *};
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use gfx::{glm, nal as nalgebra, InputManager};
use nalgebra::{Isometry3, Quaternion, UnitQuaternion};
use rapier3d::control::KinematicCharacterController;
use util::AtomicF32;

use crate::renderpasses::RenderPassManager;
use crate::{Entity, SelfManagedEntity};

pub struct TickTag(());

#[no_mangle]
static DEMO_FLOAT_HEIGHT: AtomicF32 = AtomicF32::new(0.0);

pub trait World: Send + Sync
{
    fn get_height(&self, pos: glm::Vec3) -> f32;
}

pub struct Game
{
    this_weak:             Weak<Game>,
    renderer:              Arc<gfx::Renderer>,
    entities:              util::Registrar<util::Uuid, Weak<dyn Entity>>,
    self_managed_entities: util::Registrar<util::Uuid, Arc<dyn SelfManagedEntity>>,
    float_delta_time:      AtomicU32,
    float_time_alive:      AtomicU64,
    world:                 Mutex<Option<Weak<dyn World>>>,
    render_pass_manager:   Arc<RenderPassManager>,
    render_pass_updater:   util::WindowUpdater<gfx::RenderPassSendFunction>
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
            .access()
            .into_iter()
            .filter_map(|(_, weak)| weak.upgrade())
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
                entities: util::Registrar::new(),
                self_managed_entities: util::Registrar::new(),
                float_delta_time: AtomicU32::new(0.0f32.to_bits()),
                float_time_alive: AtomicU64::new(0.0f64.to_bits()),
                this_weak: this_weak.clone(),

                world: Mutex::new(None),
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
        f32::from_bits(self.float_delta_time.load(Acquire))
    }

    pub fn register(&self, entity: Arc<dyn Entity>)
    {
        if let Some(self_managed) = entity.clone().as_self_managed()
        {
            self.self_managed_entities
                .insert(self_managed.get_uuid(), self_managed);
        }

        self.entities
            .insert(entity.get_uuid(), Arc::downgrade(&entity));
    }

    pub fn register_chunk(&self, chunk: Weak<dyn World>)
    {
        *self.world.lock().unwrap() = Some(chunk);
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

        let minimum_tick_time = Duration::from_micros(100);

        let renderer = self.renderer.clone();

        while poll_continue_func()
        {
            let now = std::time::Instant::now();

            let delta_duration = now - prev;
            delta_time = delta_duration.as_secs_f64();
            prev = now;

            if let Some(d) = minimum_tick_time.checked_sub(delta_duration)
            {
                spin_sleep::sleep(d);
            }

            self.float_delta_time
                .store((delta_time as f32).to_bits(), Release);
            self.float_time_alive.store(
                (f64::from_bits(self.float_time_alive.load(Acquire)) + delta_time).to_bits(),
                Release
            );

            let thread_entities = &self.entities;

            let strong_game = self.this_weak.upgrade().unwrap();

            self.self_managed_entities
                .access()
                .into_iter()
                .for_each(|(uuid, strong_entity)| {
                    if !strong_entity.is_alive()
                    {
                        self.self_managed_entities.delete(uuid);
                    }
                });

            self.entities
                .access()
                .into_iter()
                .filter_map(|(uuid, weak_renderable)| {
                    match weak_renderable.upgrade()
                    {
                        Some(s) => Some(s),
                        None =>
                        {
                            thread_entities.delete(uuid);
                            None
                        }
                    }
                })
                .map(|strong_entity| {
                    let local = strong_game.clone();
                    tick_pool.run_async(move || {
                        strong_entity.tick(&local, TickTag(()));
                    })
                })
                .for_each(|future| future.get());
        }
    }
}

use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::geometry::ColliderHandle;
use rapier3d::prelude::*;
use util::AtomicF32;

use crate::renderpasses::RenderPassManager;
use crate::{Entity, SelfManagedEntity};

pub struct TickTag(());

pub struct Game
{
    delta_time: AtomicF32,

    entities:              DashMap<util::Uuid, Weak<dyn Entity>>,
    collideables:          DashMap<util::Uuid, (RigidBodyHandle, Vec<ColliderHandle>)>,
    self_managed_entities: DashMap<util::Uuid, Arc<dyn SelfManagedEntity>>,

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
        self.display_holdover_entities_info();
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
        let this = Arc::new(Game {
            renderer: renderer.clone(),
            entities: DashMap::new(),
            collideables: DashMap::new(),
            self_managed_entities: DashMap::new(),
            delta_time: AtomicF32::new(0.0),
            render_pass_manager: Arc::new(RenderPassManager::new(renderer)),
            render_pass_updater
        });

        this.render_pass_updater
            .update(this.render_pass_manager.clone().generate_renderpass_vec());

        this
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

        self.entities
            .insert(entity.get_uuid(), Arc::downgrade(&entity));
    }

    pub fn enter_tick_loop(&self, poll_continue_func: &dyn Fn() -> bool)
    {
        let mut prev = std::time::Instant::now();
        let mut delta_time: f32;

        let tick_pool = util::ThreadPool::new(4, "Game Tick");

        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        // Create other structures necessary for the simulation.
        let gravity = vector![0.0, -128.64542732, 0.0];
        let mut physics_pipeline = PhysicsPipeline::new();
        let mut island_manager = IslandManager::new();
        let mut broad_phase = BroadPhaseMultiSap::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut impulse_joint_set = ImpulseJointSet::new();
        let mut multibody_joint_set = MultibodyJointSet::new();
        let mut ccd_solver = CCDSolver::new();
        let mut query_pipeline = QueryPipeline::new();
        let physics_hooks = ();
        let event_handler = ();

        // Create the ground.
        let collider = ColliderBuilder::cuboid(10000.0, 1.0, 10000.0).build();
        collider_set.insert(collider);

        // // Create the bounding ball.
        // let rigid_body = RigidBodyBuilder::dynamic()
        //     .translation(vector![0.0, 10.0, 0.0])
        //     .build();
        // let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
        // let ball_body_handle = rigid_body_set.insert(rigid_body);
        // collider_set.insert_with_parent(collider, ball_body_handle, &mut
        // rigid_body_set);

        while poll_continue_func()
        {
            // delta time initialization
            {
                let now = std::time::Instant::now();

                delta_time = (now - prev).as_secs_f32();
                prev = now;

                self.delta_time.store(delta_time, Ordering::SeqCst);
            }

            // Cull self-managed entities
            {
                self.self_managed_entities
                    .retain(|_, strong_entity| strong_entity.is_alive());
            }

            // Collect entities that will influence this tick
            let strong_entities: Vec<Arc<dyn Entity>> = {
                let mut entities_to_tick = Vec::new();

                self.entities.retain(|uuid, weak_entity| {
                    if let Some(strong_entity) = weak_entity.upgrade()
                    {
                        entities_to_tick.push(strong_entity);
                        true
                    }
                    else
                    {
                        if let Some((_, (rigid_body_handle, _))) = self.collideables.remove(&uuid)
                        {
                            // This Entity had collideables, we need to free those handles now that
                            // its owner is gone
                            rigid_body_set
                                .remove(
                                    rigid_body_handle,
                                    &mut island_manager,
                                    &mut collider_set,
                                    &mut impulse_joint_set,
                                    &mut multibody_joint_set,
                                    true
                                )
                                .unwrap();
                        }

                        false
                    }
                });

                entities_to_tick
            };

            let entities: Vec<&dyn Entity> = strong_entities
                .iter()
                .map(|e| unsafe { util::modify_lifetime(&**e) })
                .collect();

            // Physics Tick
            {
                // Handle first frame physics init things
                for collideable in entities.iter().cloned().filter_map(|e| e.as_collideable())
                {
                    if let Entry::Vacant(e) = self.collideables.entry(collideable.get_uuid())
                    {
                        let (rigid_body, colliders) = collideable.init_collideable();

                        let rigid_body_handle = rigid_body_set.insert(rigid_body);
                        let collider_handles = colliders
                            .into_iter()
                            .map(|c| {
                                collider_set.insert_with_parent(
                                    c,
                                    rigid_body_handle,
                                    &mut rigid_body_set
                                )
                            })
                            .collect();

                        e.insert((rigid_body_handle, collider_handles));
                    }
                }

                // propagate this info
                for collideable in entities.into_iter().filter_map(|e| e.as_collideable())
                {
                    collideable.physics_tick(
                        self,
                        gravity,
                        rigid_body_set
                            .get_mut(
                                self.collideables
                                    .get(&collideable.get_uuid())
                                    .expect("CollideableHandle Not Contained!")
                                    .value()
                                    .0
                            )
                            .unwrap(),
                        TickTag(())
                    );
                }

                // and then do the actual tick
                physics_pipeline.step(
                    &gravity,
                    &IntegrationParameters {
                        dt: delta_time as f32,
                        ..Default::default()
                    },
                    &mut island_manager,
                    &mut broad_phase,
                    &mut narrow_phase,
                    &mut rigid_body_set,
                    &mut collider_set,
                    &mut impulse_joint_set,
                    &mut multibody_joint_set,
                    &mut ccd_solver,
                    Some(&mut query_pipeline),
                    &physics_hooks,
                    &event_handler
                );
            }

            let mut tick_futures = Vec::new();

            if strong_entities.len() > 128
            {
                for e in strong_entities
                {
                    let me = unsafe { util::modify_lifetime(self) };
                    tick_futures.push(tick_pool.run_async(move || e.tick(me, TickTag(()))))
                }
            }
            else
            {
                for e in strong_entities
                {
                    e.tick(self, TickTag(()))
                }
            }

            tick_futures.into_iter().for_each(|f| f.get());
        }
    }

    pub fn display_holdover_entities_info(&self)
    {
        self.entities
            .iter()
            .filter_map(|r| r.value().upgrade())
            .for_each(|strong| log::warn!("Retained Entity! {:?}", &*strong));
    }
}

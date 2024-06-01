use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use gfx::glm;

use crate::renderpasses::RenderPassManager;
use crate::{Entity, SelfManagedEntity};

pub struct TickTag(());

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
    camera:                Mutex<gfx::Camera>,
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
                camera: Mutex::new(gfx::Camera::new(
                    glm::Vec3::new(-186.0, 154.0, -168.0),
                    0.218903,
                    0.748343
                )),
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
        self.entities
            .insert(entity.get_uuid(), Arc::downgrade(&entity));
    }

    pub fn register_self_managed(&self, entity: Arc<dyn SelfManagedEntity>)
    {
        self.register(entity.clone());

        self.self_managed_entities.insert(entity.get_uuid(), entity);
    }

    pub fn register_chunk(&self, chunk: Weak<dyn World>)
    {
        *self.world.lock().unwrap() = Some(chunk);
    }

    pub fn poll_input_updates(
        &self,
        input_manager: &gfx::InputManager,
        camera_delta_time: f32
    ) -> gfx::Camera
    {
        let mut camera = self.camera.lock().unwrap();

        let move_scale = 10.0
            * if input_manager.is_key_pressed(gfx::KeyCode::ShiftLeft)
            {
                20.0
            }
            else
            {
                4.0
            };
        let rotate_scale = 10.0;

        if input_manager.is_key_pressed(gfx::KeyCode::KeyW)
        {
            let v = camera.get_forward_vector() * move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::KeyS)
        {
            let v = camera.get_forward_vector() * -move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::KeyD)
        {
            let v = camera.get_right_vector() * move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::KeyA)
        {
            let v = camera.get_right_vector() * -move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::Space)
        {
            let v = *gfx::Transform::global_up_vector() * move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::ControlLeft)
        {
            let v = *gfx::Transform::global_up_vector() * -move_scale;

            camera.add_position(v * camera_delta_time);
        };

        if input_manager.is_key_pressed(gfx::KeyCode::Backslash)
        {
            input_manager.detach_cursor();
        };

        if input_manager.is_key_pressed(gfx::KeyCode::KeyK)
        {
            log::info!("Camera: {}", camera)
        };

        let mouse_diff_px: glm::Vec2 = {
            let mouse_cords_diff_px_f32: (f32, f32) = input_manager.get_mouse_delta();

            glm::Vec2::new(mouse_cords_diff_px_f32.0, mouse_cords_diff_px_f32.1)
        };

        let screen_size_px: glm::Vec2 = {
            let screen_size_u32 = self.get_renderer().get_framebuffer_size();

            glm::Vec2::new(screen_size_u32.x as f32, screen_size_u32.y as f32)
        };

        // delta over the whole screen -1 -> 1
        let normalized_delta = mouse_diff_px.component_div(&screen_size_px);

        let delta_rads = normalized_delta
            .component_div(&glm::Vec2::repeat(2.0))
            .component_mul(&self.get_renderer().get_fov());

        if self.renderer.get_delta_time() != 0.0
        {
            camera.add_yaw(delta_rads.x * rotate_scale);
            camera.add_pitch(delta_rads.y * rotate_scale);
        }

        // // process world interaction
        // camera.add_position(glm::Vec3::new(0.0, -100.0, 0.0) * camera_delta_time);

        // if let Some(w) = &*self.world.lock().unwrap()
        // {
        //     if let Some(world) = w.upgrade()
        //     {
        //         let p = camera.get_position().y;
        //         let target: f32 = world.get_height(camera.get_position()) + 32.5;

        //         let diff = target - p;
        //         if diff > 0.0
        //         {
        //             camera.add_position(glm::Vec3::new(0.0, diff, 0.0));
        //         }
        //     }
        // };

        camera.clone()
    }

    pub fn enter_tick_loop(&self, poll_continue_func: &dyn Fn() -> bool)
    {
        let mut prev = std::time::Instant::now();
        let mut delta_time: f64;

        let tick_pool = util::ThreadPool::new(4, "Game Tick");

        let minimum_tick_time = Duration::from_micros(10);

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

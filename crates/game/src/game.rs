use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use gfx::glm;

use crate::Entity;

pub struct TickTag(());

pub struct Game
{
    this_weak:        Weak<Game>,
    renderer:         Arc<gfx::Renderer>,
    entities:         util::Registrar<util::Uuid, Weak<dyn Entity>>,
    float_delta_time: AtomicU32,
    float_time_alive: AtomicU64
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
    pub fn new(renderer: Arc<gfx::Renderer>) -> Arc<Self>
    {
        Arc::new_cyclic(|this_weak| {
            Game {
                renderer,
                entities: util::Registrar::new(),
                float_delta_time: AtomicU32::new(0.0f32.to_bits()),
                float_time_alive: AtomicU64::new(0.0f64.to_bits()),
                this_weak: this_weak.clone()
            }
        })
    }

    pub fn get_renderer(&self) -> &Arc<gfx::Renderer>
    {
        &self.renderer
    }

    #[deprecated]
    pub fn get_time_alive(&self) -> f64
    {
        f64::from_bits(self.float_time_alive.load(Acquire))
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

    pub fn enter_camera_loop(
        &self,
        input_manager_receiver: &(Mutex<Option<Arc<gfx::InputManager>>>, Condvar),
        poll_continue_func: &dyn Fn() -> bool
    )
    {
        let input_manager = {
            let mut guard = input_manager_receiver.0.lock().unwrap();

            while (*guard).is_none()
            {
                guard = input_manager_receiver
                    .1
                    .wait(input_manager_receiver.0.lock().unwrap())
                    .unwrap();
            }

            guard.take().unwrap()
        };

        let mut prev = std::time::Instant::now();
        let mut camera_delta_time: f32;

        let mut camera = gfx::Camera::new(
            glm::Vec3::new(-658.22, 1062.2232, 623.242),
            0.318903,
            -3.978343
        );

        while poll_continue_func()
        {
            let now = std::time::Instant::now();

            camera_delta_time = (now - prev).as_secs_f32();
            prev = now;

            let move_scale = 10.0
                * if input_manager.is_key_pressed(gfx::KeyCode::ShiftLeft)
                {
                    25.0
                }
                else
                {
                    20.0
                };
            let rotate_scale = 10.0;

            if input_manager.is_key_pressed(gfx::KeyCode::KeyK)
            {
                log::info!(
                    "Camera: {} | Frame Time (ms): {:.03} | FPS: {:.03} | Memory Used: {}",
                    camera,
                    self.get_delta_time() * 1000.0,
                    1.0 / self.get_delta_time(),
                    util::bytes_as_string(
                        util::get_bytes_of_active_allocations() as f64,
                        util::SuffixType::Full
                    )
                );
            }

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

            if input_manager.is_key_pressed(gfx::KeyCode::KeyP)
            {
                input_manager.attach_cursor();
            };

            if input_manager.is_key_pressed(gfx::KeyCode::KeyO)
            {
                input_manager.detach_cursor();
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

            if (self.renderer.get_delta_time() != 0.0)
            {
                camera.add_yaw(
                    delta_rads.x * rotate_scale * camera_delta_time /* / self.renderer.
                                                                     * get_delta_time() */
                );
                camera.add_pitch(
                    delta_rads.y * rotate_scale * camera_delta_time /* / self.renderer.
                                                                     * get_delta_time() */
                );
            }

            self.get_renderer().camera_updater.update(camera.clone());
        }
    }

    pub fn enter_tick_loop(&self, poll_continue_func: &dyn Fn() -> bool)
    {
        let mut prev = std::time::Instant::now();
        let mut delta_time: f64;

        while poll_continue_func()
        {
            let now = std::time::Instant::now();

            delta_time = (now - prev).as_secs_f64();
            prev = now;

            self.float_delta_time
                .store((delta_time as f32).to_bits(), Release);
            self.float_time_alive.store(
                (f64::from_bits(self.float_time_alive.load(Acquire)) + delta_time).to_bits(),
                Release
            );

            let thread_entities = &self.entities;

            let strong_game = self.this_weak.upgrade().unwrap();

            // TODO: deadlock and too long detection
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
                    util::run_async(move || {
                        strong_entity.tick(&local, TickTag(()));
                    })
                })
                .for_each(|future| future.get());
        }
    }
}

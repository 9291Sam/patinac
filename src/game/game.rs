use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use nalgebra::UnitQuaternion;
use nalgebra_glm as glm;

use crate::gfx::{self};
pub struct Game<'r>
{
    renderer: &'r gfx::Renderer
}

impl<'r> Game<'r>
{
    pub fn new(renderer: &'r gfx::Renderer) -> Self
    {
        Game {
            renderer
        }
    }

    pub fn enter_tick_loop(&self, should_stop: &AtomicBool)
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();

        for x in -5..=5
        {
            for z in -5..=5
            {
                objs.push(gfx::flat_textured::FlatTextured::new(
                    self.renderer,
                    glm::Vec3::new(x as f32, 0.0, z as f32),
                    gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
                    gfx::flat_textured::FlatTextured::PENTAGON_INDICES
                ));
            }
        }

        for i in 0..25
        {
            objs.push(gfx::lit_textured::LitTextured::new_cube(
                self.renderer,
                gfx::Transform {
                    translation: glm::Vec3::new(0.0, 5.0, (i * 2) as f32),
                    rotation:    *UnitQuaternion::from_axis_angle(
                        &gfx::Transform::global_up_vector(),
                        i as f32 / 4.0
                    ),
                    scale:       glm::Vec3::repeat(1.0)
                }
            ));
        }

        let obj = gfx::lit_textured::LitTextured::new_cube(
            self.renderer,
            gfx::Transform {
                translation: glm::Vec3::new(5.0, 5.0, 5.0),
                rotation:    *UnitQuaternion::from_axis_angle(
                    &gfx::Transform::global_up_vector(),
                    0.0
                ),
                scale:       glm::Vec3::repeat(1.0)
            }
        );

        let mut prev = std::time::Instant::now();
        let mut delta_time = 0.0f32;

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {
            {
                let mut guard = obj.transform.lock().unwrap();
                let quat = guard.rotation
                    * *UnitQuaternion::from_axis_angle(
                        &gfx::Transform::global_up_vector(),
                        1.0 * delta_time
                    );

                guard.rotation = quat.normalize();
            }

            let now = std::time::Instant::now();
            delta_time = (now - prev).as_secs_f32();
            prev = now;

            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

use std::sync::Arc;

use world_gen::BrickMapBuffers;

use super::Entity;

#[derive(Debug)]
pub struct TestScene
{
    _objs:  Vec<Arc<dyn gfx::Recordable>>,
    world:  world_gen::BrickMap,
    // cube:   Arc<gfx::lit_textured::LitTextured>,
    voxels: Arc<gfx::parallax_raymarched::ParallaxRaymarched>,
    id:     util::Uuid
}

impl TestScene
{
    pub fn new(game: &super::Game) -> Arc<Self>
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();
        let mut cube: Option<Arc<gfx::lit_textured::LitTextured>> = None;

        let (
            mut world,
            BrickMapBuffers {
                tracking_buffer,
                brick_buffer
            }
        ) = world_gen::BrickMap::new(game.get_renderer());

        for x in -64..64
        {
            for y in -64..64
            {
                for z in -64..64
                {
                    if x == 0
                    {
                        world.set_voxel(world_gen::Voxel::Green, gfx::I64Vec3::new(x, y, z));
                    }
                }
            }
        }

        tracking_buffer.unmap();
        brick_buffer.unmap();

        let bind_group = Arc::new(
            game.get_renderer()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label:   Some("Brick Map Bind Group"),
                    layout:  game
                        .get_renderer()
                        .render_cache
                        .lookup_bind_group_layout(gfx::BindGroupType::BrickMap),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding:  0,
                            resource: tracking_buffer.as_entire_binding()
                        },
                        wgpu::BindGroupEntry {
                            binding:  1,
                            resource: brick_buffer.as_entire_binding()
                        }
                    ]
                })
        );

        let voxels = gfx::parallax_raymarched::ParallaxRaymarched::new_cube(
            game.get_renderer(),
            gfx::Transform {
                translation: gfx::Vec3::new(0.0, 0.0, 0.0),
                scale: gfx::Vec3::repeat(1.25),
                ..Default::default()
            },
            bind_group.clone()
        );

        // objs.push(gfx::parallax_raymarched::ParallaxRaymarched::new_cube(
        //     game.get_renderer(),
        //     gfx::Transform {
        //         translation: gfx::Vec3::repeat(0.0),
        //         scale: gfx::Vec3::repeat(1.25),
        //         ..Default::default()
        //     }
        // ));

        objs.push(gfx::parallax_raymarched::ParallaxRaymarched::new_cube(
            game.get_renderer(),
            gfx::Transform {
                translation: gfx::Vec3::new(10.1, 2.0, 0.0),
                scale: gfx::Vec3::repeat(4.0),
                ..Default::default()
            },
            bind_group.clone()
        ));

        objs.push(gfx::parallax_raymarched::ParallaxRaymarched::new_cube(
            game.get_renderer(),
            gfx::Transform {
                translation: gfx::Vec3::new(-8.0, 2.0, 12.0),
                scale: gfx::Vec3::repeat(5.99),
                ..Default::default()
            },
            bind_group.clone()
        ));

        // objs.push(
        //     gfx::parallax_raymarched::ParallaxRaymarched::new_camera_tracked(game.
        // get_renderer()) );

        for x in -5..=5
        {
            for z in -5..=5
            {
                objs.push(gfx::flat_textured::FlatTextured::new(
                    game.get_renderer(),
                    gfx::Vec3::new(x as f32, 0.0, z as f32),
                    gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
                    gfx::flat_textured::FlatTextured::PENTAGON_INDICES
                ));

                let a = gfx::lit_textured::LitTextured::new_cube(
                    game.get_renderer(),
                    gfx::Transform {
                        translation: gfx::Vec3::new(x as f32, 4.0, z as f32),
                        rotation:    *gfx::UnitQuaternion::from_axis_angle(
                            &gfx::Transform::global_up_vector(),
                            (x + z) as f32 / 4.0
                        ),
                        scale:       gfx::Vec3::repeat(0.4)
                    }
                );

                if x == 0 && z == 0
                {
                    cube = Some(a.clone());
                }

                objs.push(a);
            }
        }

        let this = Arc::new(TestScene {
            _objs: objs,
            world,
            voxels,
            id: util::Uuid::new()
        });

        game.register(this.clone());

        this
    }
}

impl Entity for TestScene
{
    fn get_name(&self) -> &str
    {
        "Test Scene"
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn get_position(&self) -> Option<gfx::Vec3>
    {
        None
    }

    fn tick(&self, game: &super::Game, _: super::TickTag)
    {
        // {
        //     let mut guard = self.cube.transform.lock().unwrap();

        //     let quat = guard.rotation
        //         * *gfx::UnitQuaternion::from_axis_angle(
        //           &gfx::Transform::global_up_vector(), 1.0 *
        //           game.get_delta_time()
        //         );

        //     guard.rotation = quat.normalize();
        // }

        // {
        //     let mut guard = self.voxels.transform.lock().unwrap();

        //     let quat = guard.rotation
        //         * *gfx::UnitQuaternion::from_axis_angle(
        //           &gfx::Transform::global_up_vector(), -1.0 *
        //           game.get_delta_time()
        //         );

        //     guard.rotation = quat.normalize();
        // }
    }
}

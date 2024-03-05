use std::borrow::Cow;
use std::sync::Arc;

use gfx::glm;
use noise::NoiseFn;

use crate::recordables::flat_textured::FlatTextured;
use crate::recordables::lit_textured::LitTextured;

#[derive(Debug)]
pub struct TestScene
{
    _objs:           Vec<Arc<dyn gfx::Recordable>>,
    rotate_objs:     Vec<Arc<LitTextured>>,
    voxel_chunk:     Arc<dyn game::Entity>,
    brick_map_chunk: Arc<voxel::BrickMapChunk>,
    id:              util::Uuid
}

impl TestScene
{
    pub fn new(game: &game::Game) -> Arc<Self>
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();
        let mut rotate_objs: Vec<Arc<LitTextured>> = Vec::new();

        let mut voxel_transforms: Vec<gfx::Transform> = Vec::new();

        for x in -5..=5
        {
            for z in -5..=5
            {
                objs.push(FlatTextured::new(
                    game.get_renderer(),
                    glm::Vec3::new(x as f32, 0.0, z as f32),
                    FlatTextured::PENTAGON_VERTICES,
                    FlatTextured::PENTAGON_INDICES
                ));

                let a = LitTextured::new_cube(
                    game.get_renderer(),
                    gfx::Transform {
                        translation: glm::Vec3::new(x as f32, 4.0, z as f32),
                        rotation:    *glm::UnitQuaternion::from_axis_angle(
                            &gfx::Transform::global_up_vector(),
                            (x + z) as f32 / 4.0
                        ),
                        scale:       glm::Vec3::repeat(0.4)
                    }
                );

                if x == 0 && z == 0
                {
                    rotate_objs.push(a.clone());
                }

                // objs.push(voxel::Chunk::new(
                //     game,
                //     gfx::Transform {
                //         translation: glm::Vec3::new(x as f32, 8.0, z as f32),
                //         rotation:    *glm::UnitQuaternion::from_axis_angle(
                //             &gfx::Transform::global_up_vector(),
                //             (x + z) as f32 / 4.0
                //         ),
                //         scale:       glm::Vec3::repeat(1.1)
                //     },
                //     false
                // ));

                objs.push(a);
            }
        }

        // for (x, y, z) in itertools::iproduct!(16..24, 16..24, 16..24)
        // {
        //     voxel_transforms.push(gfx::Transform {
        //         translation: glm::Vec3::new(x as f32, y as f32, z as f32),
        //         rotation:    *glm::UnitQuaternion::from_axis_angle(
        //             &gfx::Transform::global_up_vector(),
        //             (x + z) as f32 / 4.0
        //         ),
        //         scale:       glm::Vec3::repeat(3.0)
        //     });
        // }

        // let b = 3i32;

        // for (x, y, z) in itertools::iproduct!(-b..=b, -b..=b, -b..=b)
        // {
        // if x.abs() == b || y.abs() == b || z.abs() == b
        // {
        // voxel_transforms.push(gfx::Transform {
        //     translation: glm::Vec3::new(0.0, 0.0, 0.0),
        //     scale: glm::Vec3::repeat(64.0),
        //     ..Default::default()
        // });
        // }
        // }

        let voxel_chunk = voxel::Chunk::new(game, voxel_transforms);

        let this = Arc::new(TestScene {
            _objs: objs,
            rotate_objs,
            brick_map_chunk: voxel::BrickMapChunk::new(game, glm::Vec3::new(484.0, 494.0, 484.0)),
            id: util::Uuid::new(),
            voxel_chunk
        });

        {
            let data_manager: &mut voxel::VoxelChunkDataManager =
                &mut this.brick_map_chunk.access_data_manager().lock().unwrap();

            let perlin = noise::Perlin::new(1347234789);

            let noise_func = |x: u16, z: u16, layer: u16| -> u16 {
                let value = perlin.get([x as f64 / 64.0, layer as f64 / 32.0, z as f64 / 64.0])
                    * 48.0
                    + 32.9;
                (value as u16).clamp(0, 1024)
            };
            let b: u16 = 1024;
            let layers = 24;

            for l in 0..layers
            {
                for (x, z) in itertools::iproduct!(0..b, 0..b)
                {
                    data_manager.write_voxel(
                        voxel::Voxel::Green,
                        voxel::ChunkPosition::new(
                            x,
                            noise_func(x, z, l) + (l * (0.8 * (b / layers) as f32) as u16),
                            z
                        )
                    );
                }
            }

            data_manager.stop_writes();
        }

        game.register(this.clone());

        this
    }
}

impl game::EntityCastDepot for TestScene
{
    fn as_entity(&self) -> Option<&dyn game::Entity>
    {
        Some(self)
    }

    fn as_positionable(&self) -> Option<&dyn game::Positionable>
    {
        None
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
    }
}

impl game::Entity for TestScene
{
    fn get_name(&self) -> Cow<'_, str>
    {
        "Test Scene".into()
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn tick(&self, game: &game::Game, _: game::TickTag)
    {
        for o in &self.rotate_objs
        {
            let mut guard = o.transform.lock().unwrap();

            let quat = guard.rotation
                * *glm::UnitQuaternion::from_axis_angle(
                    &gfx::Transform::global_up_vector(),
                    1.0 * game.get_delta_time()
                );

            guard.rotation = quat.normalize();
        }

        // self.brick_map_chunk.get_position_mut(&|t| {
        //     *t = glm::Vec3::new(
        //         (507.0 + 2.0 * game.get_time_alive().add(std::f64::consts::PI
        // / 4.0).cos() as f32),         (507.0 + 2.0 *
        // game.get_time_alive().add(std::f64::consts::FRAC_PI_2).sin()) as f32,
        //         (507.0 + -2.0 * game.get_time_alive().mul(2.0).cos()) as f32
        //     );
        // });
    }
}

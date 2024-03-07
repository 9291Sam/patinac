use std::borrow::Cow;
use std::sync::Arc;

use gfx::glm;
use noise::NoiseFn;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::recordables::flat_textured::FlatTextured;
use crate::recordables::lit_textured::LitTextured;

#[derive(Debug)]
pub struct TestScene
{
    _objs:           Vec<Arc<dyn gfx::Recordable>>,
    rotate_objs:     Vec<Arc<LitTextured>>,
    brick_map_chunk: Arc<voxel::BrickMapChunk>,
    id:              util::Uuid
}

impl TestScene
{
    pub fn new(game: &game::Game) -> Arc<Self>
    {
        let mut objs: Vec<Arc<dyn gfx::Recordable>> = Vec::new();
        let mut rotate_objs: Vec<Arc<LitTextured>> = Vec::new();

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

                objs.push(a);
            }
        }

        let this = Arc::new(TestScene {
            _objs: objs,
            rotate_objs,
            brick_map_chunk: voxel::BrickMapChunk::new(game, glm::Vec3::new(0.0, 0.0, 0.0)),
            id: util::Uuid::new()
        });

        {
            let data_manager: &mut voxel::VoxelChunkDataManager =
                &mut this.brick_map_chunk.access_data_manager().lock().unwrap();

            let perlin = noise::Perlin::new(1347234789);

            let noise_func = |x: i16, z: i16, layer: i16| -> i16 {
                let value =
                    perlin.get([x as f64 / 64.0, layer as f64 / 32.0, z as f64 / 64.0]) * 48.0;
                (value as i16).clamp(-512, 511)
            };
            let b: i16 = 1024;
            // let layers = 24;

            let mut rand = rand::thread_rng();

            for (x, z) in itertools::iproduct!(-b / 2..b / 2, -b / 2..b / 2)
            {
                data_manager.write_voxel(
                    match rand.gen_range(0..3)
                    {
                        0 => voxel::Voxel::Red,
                        1 => voxel::Voxel::Green,
                        2 => voxel::Voxel::Blue,
                        _ => unreachable!()
                    },
                    voxel::ChunkPosition::new(
                        x,
                        noise_func(x, z, 1), // + (l as f32 * (0.8 * (b / layers) as f32)),
                        z
                    )
                );
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

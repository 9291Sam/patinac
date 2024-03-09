use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::Arc;

use gfx::glm;
use itertools::iproduct;
use noise::NoiseFn;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use voxel::Voxel;

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

            let noise_generator = noise::SuperSimplex::new(
                (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
            );

            let mut random_generator = RefCell::new(rand::thread_rng());
            let mut pos_voxel_cache = std::collections::HashMap::<glm::I16Vec3, Voxel>::new();

            let noise_sampler = |x: i16, z: i16| {
                (noise_generator.get([x as f64 / 256.0, 0.0, z as f64 / 256.0]) * 24.0) as i16
            };

            let get_rand_grass_voxel = || -> Voxel {
                random_generator
                    .borrow_mut()
                    .gen_range(7..=12)
                    .try_into()
                    .unwrap()
            };

            let get_rand_stone_voxel = || -> Voxel {
                random_generator
                    .borrow_mut()
                    .gen_range(1..=6)
                    .try_into()
                    .unwrap()
            };

            // let generate_voxel_at_pos = |pos: glm::I16Vec3| -> Voxel {};

            // let mut get_voxel_at_pos = |pos: glm::I16Vec3| {
            //     match pos_voxel_cache.get(&pos).cloned()
            //     {
            //         Some(v) => v,
            //         None =>
            //         {
            //             let generated_voxel = generate_voxel_at_pos(pos);
            //             pos_voxel_cache.insert(pos, generated_voxel);
            //             generated_voxel
            //         }
            //     }
            // };

            let c = 64i16;

            for (x, y, z) in iproduct!(-c..c, -c..c, -c..c)
            {
                let pos = glm::I16Vec3::new(x, y, z);

                data_manager.write_brick(get_rand_stone_voxel(), glm::I16Vec3::new(x, y, z));

                // if x == z
                // {
                //     log::info!("genpos: {}", pos);
                // }
            }

            let b = 512i16;

            for (x, z) in iproduct!(-b..b, -b..b)
            {
                let pos = glm::I16Vec3::new(x, noise_sampler(x, z), z);

                let mut above_brick_pos = glm::I16Vec3::new(
                    pos.x.div_euclid(8),
                    (pos.y).div_euclid(8),
                    pos.z.div_euclid(8)
                );

                while above_brick_pos.y < 64
                {
                    data_manager.write_brick(Voxel::Air, above_brick_pos);

                    above_brick_pos.y += 1;
                }
            }

            for (x, z) in iproduct!(-b..b, -b..b)
            {
                let pos = glm::I16Vec3::new(x, noise_sampler(x, z), z);

                let noise_height = noise_sampler(pos.x, pos.z);

                let diff = pos.y - noise_height;

                let v = match diff
                {
                    1.. => Voxel::Air,
                    -5..=0 => get_rand_grass_voxel(),
                    ..=-6 => get_rand_stone_voxel()
                };

                data_manager.write_voxel(v, pos);

                let mut this_brick_y = pos.y.div_euclid(8);

                if pos.y.rem_euclid(8) == 0
                {
                    this_brick_y -= 1
                }

                let mut stone_fill_pos = pos;

                stone_fill_pos.y -= 1;

                while stone_fill_pos.y.div_euclid(8) == this_brick_y
                {
                    data_manager.write_voxel(get_rand_stone_voxel(), stone_fill_pos);

                    stone_fill_pos.y -= 1;
                }
            }

            data_manager.stop_writes();
        }

        // {
        //     let data_manager: &mut voxel::VoxelChunkDataManager =
        //         &mut this.brick_map_chunk.access_data_manager().lock().unwrap();

        //     for i in -64..64
        //     {
        //         data_manager.write_brick(voxel::Voxel::Green,
        // glm::I16Vec3::repeat(i))     }

        //     let perlin = noise::Perlin::new(1347234789);

        //     let noise_func = |x: i16, z: i16, layer: i16| -> i16 {
        //         let value =
        //
        //     };
        //     let b: i16 = 1024;
        //     let layers = 24;

        //     let mut rand = rand::thread_rng();

        //     for l in -layers / 2..layers
        //     {
        //         for (x, z) in itertools::iproduct!(-b / 2..b / 2, -b / 2..b / 2)
        //         {
        //             let noise = noise_func(x, z, l);

        //             if noise.abs() > 4
        //             {
        //                 continue;
        //             }

        //             let height = noise + (b / layers) * l;
        //             let height = height.clamp(-512, 511);

        //             if height == -512 || height == 511
        //             {
        //                 continue;
        //             }

        //             data_manager.write_voxel(
        //                 match height.abs() % 3
        //                 {
        //                     0 => voxel::Voxel::Red,
        //                     1 => voxel::Voxel::Green,
        //                     2 => voxel::Voxel::Blue,
        //                     _ => unreachable!()
        //                 },
        //                 voxel::ChunkPosition::new(x, height, z)
        //             );
        //         }
        //     }

        //     data_manager.stop_writes();
        // }

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

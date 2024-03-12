use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use gfx::glm::{self, any};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;
use voxel::Voxel;

#[derive(Debug)]
pub struct TestScene
{
    brick_map_chunk: Mutex<Vec<util::Promise<Arc<super::BrickMapChunk>>>>,
    id:              util::Uuid
}

impl TestScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let brick_game = game.clone();

        let this = Arc::new(TestScene {
            brick_map_chunk: Mutex::new(
                // iproduct!(-2..=0, -2..=0)
                iproduct!(0..=0, 0..=0)
                    .map(|(x, z)| -> util::Promise<_> {
                        let local_game = brick_game.clone();

                        util::run_async(move || {
                            create_and_fill(
                                &local_game,
                                glm::Vec3::new(
                                    (x as f32) * 1024.0 - 512.0,
                                    -512.0,
                                    (z as f32) * 1024.0 - 512.0
                                )
                            )
                        })
                        .into()
                    })
                    .collect()
            ),
            id:              util::Uuid::new()
        });

        game.register(this.clone());

        this
    }
}

fn create_and_fill(brick_game: &game::Game, pos: glm::Vec3) -> Arc<super::BrickMapChunk>
{
    let chunk = super::BrickMapChunk::new(brick_game, pos);

    {
        let data_manager: &voxel::VoxelChunkDataManager = chunk.access_data_manager();

        let noise_generator = noise::SuperSimplex::new(
            (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
        );

        let random_generator = RefCell::new(rand::thread_rng());

        let noise_sampler = |x: u16, z: u16| {
            ((noise_generator.get([
                ((pos.x as f64) + (x as f64)) / 256.0,
                0.0,
                ((pos.z as f64) + (z as f64)) / 256.0
            ]) * 484.0)
                + 64.0) as u16
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

        let c = 128u16;

        for (x, y, z) in iproduct!(0..c, 0..c, 0..c)
        {
            let pos = glm::U16Vec3::new(x, y, z);

            data_manager.write_brick(get_rand_stone_voxel(), pos);
        }

        data_manager.flush_entire();

        let b = 1024u16;

        // for (x, z) in iproduct!(0..b, 0..b)
        // {
        //     let pos = glm::U16Vec3::new(x, noise_sampler(x, z), z);

        //     let mut above_brick_pos =

        //     while above_brick_pos.y < 128
        //     {
        //         data_manager.write_brick(Voxel::Air, above_brick_pos);

        //         above_brick_pos.y += 1;
        //     }
        // }

        // data_manager.flush_entire();

        for (x, z) in iproduct!(0..b, 0..b)
        {
            let pos = glm::U16Vec3::new(x, noise_sampler(x, z), z);
            // data_manager.write_brick(
            //     Voxel::Air,
            //     glm::U16Vec3::new(
            //         pos.x.div_euclid(8),
            //         pos.y.div_euclid(8).max(1) - 1,
            //         pos.z.div_euclid(8)
            //     )
            // );
            data_manager.write_voxel(get_rand_grass_voxel(), pos);

            // for h in (pos.y..((pos.y / 8) * 8))
            // {
            //     data_manager.write_voxel(Voxel::Air, glm::U16Vec3::new(x, h,
            // z)); }
        }

        data_manager.flush_entire();
    }

    chunk
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

    fn tick(&self, _: &game::Game, _: game::TickTag)
    {
        // let mut guard = self.brick_map_chunk.lock().unwrap();

        // guard.iter_mut().for_each(|p| {
        //     p.poll_ref();

        //     if let util::Promise::Resolved(chunk) = &*p
        //     {
        //         let manager = chunk.access_data_manager();

        //         for _ in 0..64
        //         {
        //             // let diff = 8u16;
        //             let base: glm::U16Vec3 = glm::U16Vec3::new(
        //                 rand::thread_rng().gen_range(512u16..=698),
        //                 rand::thread_rng().gen_range(512u16..=698),
        //                 rand::thread_rng().gen_range(512u16..=698)
        //             );

        //             // let top = base.add_scalar(diff);

        //             // for (x, y, z) in iproduct!(base.x..top.x,
        // base.y..top.y, base.z..top.z)             // {
        //             manager.write_voxel(
        //
        // rand::thread_rng().gen_range(1..=12).try_into().unwrap(),
        //                 base
        //             );
        //             // }
        //         }

        //         manager.flush_to_gpu();
        //     }
        // });

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

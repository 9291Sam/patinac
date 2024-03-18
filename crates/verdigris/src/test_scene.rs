use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;
use voxel::Voxel;

#[derive(Debug)]
pub struct TestScene
{
    brick_map_chunk: Mutex<util::Promise<Vec<util::Promise<Arc<super::BrickMapChunk>>>>>,
    id:              util::Uuid
}

impl TestScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let brick_game = game.clone();
        let chunk_r = 0;
        let this = Arc::new(TestScene {
            brick_map_chunk: Mutex::new(
                util::run_async(move || {
                    iproduct!(-chunk_r..=chunk_r, -chunk_r..=chunk_r)
                // iproduct!(-1..=1, -1..=1)
                // iproduct!(-0..=0, -0..=0)
                    .map(|(x, z)| -> util::Promise<_> {
                        let local_game = brick_game.clone();

                        std::thread::sleep(Duration::from_millis(25));

                        util::run_async(move || {

                            // std::thread::sleep(Duration::from_millis(rand::thread_rng().gen_range(225..=875)));
                            create_and_fill(
                                &local_game,
                                glm::Vec3::new(
                                    (x as f32) * voxel::CHUNK_VOXEL_SIZE as f32 - (voxel::CHUNK_VOXEL_SIZE / 2) as f32,
                                    -((voxel::CHUNK_VOXEL_SIZE / 2) as f32),
                                    (z as f32) * voxel::CHUNK_VOXEL_SIZE as f32 - (voxel::CHUNK_VOXEL_SIZE / 2) as f32
                                )
                            )
                        })
                        .into()
                    })
                    .collect()
                })
                .into()
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
            ]) * 218.0)
                + 384.0) as u16
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

        let c = voxel::BRICK_MAP_EDGE_SIZE as u16;;

        for (x, y, z) in iproduct!(0..c, 0..c, 0..c)
        {
            let pos = glm::U16Vec3::new(x, y, z);

            data_manager.write_brick(get_rand_stone_voxel(), pos);
        }

        for (b_x, b_z) in iproduct!(0..c, 0..c)
        {
            let mut top_free_brick_height: u16 = 0;
            // iter over columns
            for (l_x, l_z) in iproduct!(0..8, 0..8)
            {
                let chunk_x = b_x * voxel::VOXEL_BRICK_EDGE_LENGTH as u16 + l_x;
                let chunk_z = b_z * voxel::VOXEL_BRICK_EDGE_LENGTH as u16 + l_z;

                let height = noise_sampler(chunk_x, chunk_z);

                let pos = glm::U16Vec3::new(chunk_x, height, chunk_z);

                top_free_brick_height = top_free_brick_height.max(height.div_euclid(voxel::VOXEL_BRICK_EDGE_LENGTH as u16) + 1);

                data_manager.write_voxel(get_rand_grass_voxel(), pos);
            }

            for h in top_free_brick_height..(voxel::BRICK_MAP_EDGE_SIZE as u16)
            {
                data_manager.write_brick(Voxel::Air, glm::U16Vec3::new(b_x, h, b_z));
            }

            for (l_x, l_z) in iproduct!(0..voxel::VOXEL_BRICK_EDGE_LENGTH as u16, 0..voxel::VOXEL_BRICK_EDGE_LENGTH as u16)
            {
                let chunk_x = b_x * voxel::VOXEL_BRICK_EDGE_LENGTH as u16 + l_x;
                let chunk_z = b_z * voxel::VOXEL_BRICK_EDGE_LENGTH as u16 + l_z;

                let grass_height = noise_sampler(chunk_x, chunk_z);

                let pos = glm::U16Vec3::new(chunk_x, grass_height + 1, chunk_z);

                for y_p in pos.y..(top_free_brick_height * voxel::VOXEL_BRICK_EDGE_LENGTH as u16 + voxel::VOXEL_BRICK_EDGE_LENGTH as u16)
                {
                    data_manager.write_voxel(Voxel::Air, glm::U16Vec3::new(chunk_x, y_p, chunk_z));
                }
            }
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
        let mut guard = self.brick_map_chunk.lock().unwrap();

        guard.poll_ref();

        if let util::Promise::Resolved(chunk_vec) = &mut *guard
        {
            chunk_vec.iter_mut().for_each(|p| {
                p.poll_ref();

                if let util::Promise::Resolved(chunk) = &*p
                {
                    let manager = chunk.access_data_manager();

                    for _ in 0..256
                    {
                        let center = 512u16;
                        let edge = 64;
                        let range = (center - edge)..(center + edge);

                        let base: glm::U16Vec3 = glm::U16Vec3::new(
                            rand::thread_rng().gen_range(range.clone()),
                            rand::thread_rng().gen_range(range.clone()),
                            rand::thread_rng().gen_range(range.clone())
                        );

                        manager.write_voxel(
                            rand::thread_rng().gen_range(1..=12).try_into().unwrap(),
                            base
                        );
                    }

                    manager.flush_to_gpu();
                }
            })
        };
    }
}

use std::borrow::Cow;
use std::sync::Arc;

use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;

use crate::{FaceVoxelChunk, FaceVoxelChunkVoxelInstance, VoxelFace};

#[derive(Debug)]
pub struct DemoScene
{
    chunks: Vec<Arc<FaceVoxelChunk>>,
    id:     util::Uuid
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let start = std::time::Instant::now();

        let mut this = DemoScene {
            id:     util::Uuid::new(),
            chunks: Vec::new()
        };

        let noise_generator = noise::SuperSimplex::new(
            (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
        );

        for (o_x, o_z) in iproduct!(0..1, 0..1)
        // iproduct!(-8..8, -8..8)
        {
            let w_x = 32.0 * o_x as f32;
            let w_z = 32.0 * o_z as f32;

            let mut chunk = FaceVoxelChunk::new(
                &game,
                gfx::Transform {
                    translation: glm::Vec3::new(w_x, 0.0, w_z),
                    ..Default::default()
                }
            );

            let noise_sampler = |x: i16, z: i16| {
                let h = 8.0f64;

                (noise_generator.get([
                    (x as f64 + w_x as f64) / 256.0,
                    0.0,
                    (z as f64 + w_z as f64) / 256.0
                ]) * h)
                    + 16.0
            };

            let mut v = Vec::new();

            for (x, y, z) in iproduct!(0..32, 0..32, 0..32)
            {
                let val = noise_sampler(x, z) as u32;

                if val < y
                {
                    continue;
                }

                v.push(FaceVoxelChunkVoxelInstance::new(
                    x.try_into().unwrap(),
                    y,
                    z.try_into().unwrap(),
                    VoxelFace::Top,
                    rand::thread_rng().gen_range(1..=12)
                ))
            }

            unsafe { Arc::get_mut_unchecked(&mut chunk) }.update_voxels(v);

            this.chunks.push(chunk);
        }

        let this = Arc::new(this);

        game.register(this.clone());

        let end = std::time::Instant::now();

        log::info!("Generated chunks in {}ms", (end - start).as_millis());

        this
    }
}

impl game::EntityCastDepot for DemoScene
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

impl game::Entity for DemoScene
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

        // guard.poll_ref();

        // if let util::Promise::Resolved(chunk_vec) = &mut *guard
        // {
        //     chunk_vec.iter_mut().for_each(|p| {
        //         p.poll_ref();

        //         // if let util::Promise::Resolved(chunk) = &*p
        //         // {
        //         //     let manager = chunk.access_data_manager();

        //         //     for _ in 0..256
        //         //     {
        //         //         let center = 448u16;
        //         //         let edge = 64;
        //         //         let range = (center - edge)..(center + edge);

        //         //         let base: glm::U16Vec3 = glm::U16Vec3::new(
        //         //             rand::thread_rng().gen_range(range.clone()),
        //         //             rand::thread_rng().gen_range(range.clone()),
        //         //             rand::thread_rng().gen_range(range.clone())
        //         //         );

        //         //         manager.write_voxel(
        //         //
        //         // rand::thread_rng().gen_range(1..=12).try_into().unwrap(),
        //         //             base
        //         //         );
        //         //     }

        //         //     log::trace!(" hmmm");

        //         //     manager.flush_to_gpu();
        //         // }
        //     })
        // };
    }
}

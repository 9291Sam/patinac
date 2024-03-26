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

        let mut number_of_faces = 0;

        let e = 0;
        for (o_x, o_z) in iproduct!(-e..=e, -e..=e)
        {
            let w_x = 512.0 * o_x as f32;
            let w_z = 512.0 * o_z as f32;

            let scale = 1.0;

            let mut chunk = FaceVoxelChunk::new(
                &game,
                gfx::Transform {
                    translation: glm::Vec3::new(scale * w_x - 256.0, 0.0, scale * w_z - 256.0),
                    scale: glm::Vec3::new(scale, scale, scale),
                    ..Default::default()
                }
            );

            let noise_sampler = |x: i32, z: i32| {
                let h = 128.0f64;

                (noise_generator.get([
                    (x as f64 + w_x as f64) / 256.0,
                    0.0,
                    (z as f64 + w_z as f64) / 256.0
                ]) * h)
                    + h
            };

            let occupied = |x: i32, y: i32, z: i32| noise_sampler(x, z) as i32 > y;

            let mut v = Vec::new();

            for (x, z) in iproduct!(0..512u32, 0..512u32)
            {
                let h = noise_sampler(x as i32, z as i32) as u32;

                for y in h.saturating_sub(8)..h.saturating_add(8)
                {
                    if !occupied(x as i32, y as i32, z as i32)
                    {
                        continue;
                    }

                    let c = rand::thread_rng().gen_range(1..=12);

                    VoxelFace::iter().for_each(|f| {
                        let dir = f.get_axis();

                        if !occupied(x as i32 + dir.x, y as i32 + dir.y, z as i32 + dir.z)
                        {
                            v.push(FaceVoxelChunkVoxelInstance::new(x, y, z, 0, 0, f, c));
                            number_of_faces += 1;
                        }
                    });
                }
            }

            unsafe { Arc::get_mut_unchecked(&mut chunk) }.update_voxels(v);

            this.chunks.push(chunk);
        }

        let this = Arc::new(this);

        game.register(this.clone());

        let end = std::time::Instant::now();

        log::info!(
            "Generated chunks in {}ms with {} faces",
            (end - start).as_millis(),
            number_of_faces
        );

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

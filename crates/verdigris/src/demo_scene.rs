use std::borrow::Cow;
use std::sync::Arc;

use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;

use crate::{RasterChunk, RasterizedVoxelVertexOffsetPosition};

#[derive(Debug)]
pub struct DemoScene
{
    raster: Arc<RasterChunk>,
    id:     util::Uuid
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let mut this = Arc::new(DemoScene {
            id:     util::Uuid::new(),
            raster: super::RasterChunk::new(
                &game,
                gfx::Transform {
                    translation: glm::Vec3::new(35.0, 975.0, -35.0),
                    ..Default::default()
                }
            )
        });

        let noise_generator = noise::SuperSimplex::new(
            (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
        );

        let noise_sampler = |x: i16, z: i16| {
            let h = 84.0f64;

            (noise_generator.get([(x as f64) / 256.0, 0.0, (z as f64) / 256.0]) * h) as i16
        };

        let d = 1024;

        let mut v = iproduct!(-d..d, -d..d)
            .map(|(x, z)| {
                RasterizedVoxelVertexOffsetPosition {
                    offset: glm::I16Vec4::new(
                        x,
                        noise_sampler(x, z),
                        z,
                        rand::thread_rng().gen_range(1..=12)
                    )
                }
            })
            .collect::<Vec<_>>();

        v.push(RasterizedVoxelVertexOffsetPosition {
            offset: glm::I16Vec4::new(0, 512, 0, 0)
        });

        let t: &mut DemoScene = Arc::get_mut(&mut this).unwrap();
        let c: &mut RasterChunk = unsafe { Arc::get_mut_unchecked(&mut t.raster) };

        c.update_voxels(v);

        game.register(this.clone());

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

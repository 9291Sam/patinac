use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;

use crate::{RasterChunk, VoxelFace, VoxelFaceDirection};

#[derive(Debug)]
pub struct DemoScene
{
    raster: Mutex<Vec<util::Promise<Arc<RasterChunk>>>>,
    id:     util::Uuid
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let r = 1i16;

        let chunks: Vec<util::Promise<Arc<RasterChunk>>> = iproduct!(-r..=r, -r..=r)
            .map(|(chunk_x, chunk_z)| {
                let game = game.clone();
                util::run_async(move || {
                    let noise_generator = noise::SuperSimplex::new(
                        (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
                    );

                    let chunk_offset_x = 511.0 * chunk_x as f64 - 256.0;
                    let chunk_offset_z = 511.0 * chunk_z as f64 - 256.0;

                    let noise_sampler = |x: i16, z: i16| -> i16 {
                        let h = 256.0f64;

                        (noise_generator.get([
                            (x as f64 + chunk_offset_x) / 256.0,
                            0.0,
                            (z as f64 + chunk_offset_z) / 256.0
                        ]) * h
                            + h) as i16
                    };

                    let occupied = |x: i16, y: i16, z: i16| (y <= noise_sampler(x, z));

                    RasterChunk::new(
                        &game,
                        gfx::Transform {
                            translation: glm::Vec3::new(
                                chunk_offset_x as f32,
                                0.0,
                                chunk_offset_z as f32
                            ),
                            scale: glm::Vec3::new(1.0, 1.0, 1.0),
                            ..Default::default()
                        },
                        iproduct!(0..511i16, 0..511i16).flat_map(|(x, z)| {
                            let voxel = rand::thread_rng().gen_range(0..=3);
                            let h = noise_sampler(x, z).max(0) as u16;

                            VoxelFaceDirection::iterate().filter_map(move |d| {
                                let axis = d.get_axis();

                                if occupied(x + axis.x, h as i16 + axis.y, z + axis.z)
                                {
                                    None
                                }
                                else
                                {
                                    Some(VoxelFace {
                                        direction: d,
                                        voxel,
                                        lw_size: glm::U16Vec2::new(1, 1),
                                        position: glm::U16Vec3::new(
                                            x.max(0) as u16,
                                            h.max(0),
                                            z.max(0) as u16
                                        )
                                    })
                                }
                            })
                        })
                    )
                })
                .into()
            })
            .collect();

        let this = Arc::new(DemoScene {
            id:     util::Uuid::new(),
            raster: Mutex::new(chunks)
        });

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
        self.raster
            .lock()
            .unwrap()
            .iter_mut()
            .for_each(|c| c.poll_ref());
    }
}

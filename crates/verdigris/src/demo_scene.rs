use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;
use voxel::{RasterChunk, VoxelFace, VoxelFaceDirection};

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

        let noise_generator = noise::SuperSimplex::new(
            (234782378948923489238948972347234789342u128 % u32::MAX as u128) as u32
        );

        let chunks: Vec<util::Promise<Arc<RasterChunk>>> = iproduct!(-r..=r, -r..=r)
            .map(|(chunk_x, chunk_z)| {
                let game = game.clone();
                util::run_async(move || {
                    create_chunk(
                        &game,
                        &noise_generator,
                        glm::DVec3::new(
                            511.0 * chunk_x as f64 - 256.0,
                            0.0,
                            511.0 * chunk_z as f64 - 256.0
                        ),
                        1.0
                    )
                })
                .into()
            })
            .chain(
                iproduct!(0..=3, 0..=3)
                    .filter(|(x, z)| *x == 0 || *z == 0 || *x == 3 || *z == 3)
                    .map(|(x, z)| {
                        let game = game.clone();
                        util::run_async(move || {
                            create_chunk(
                                &game,
                                &noise_generator,
                                glm::DVec3::new(
                                    2.5 + (511.0 * 1.5 * x as f64 - 256.0 * 2.0 * 3.0),
                                    0.0,
                                    2.5 + (511.0 * 1.5 * z as f64 - 256.0 * 2.0 * 3.0)
                                ),
                                1.5
                            )
                        })
                        .into()
                    })
            )
            .chain(
                iproduct!(0..=3, 0..=3)
                    .filter(|(x, z)| *x == 0 || *z == 0 || *x == 3 || *z == 3)
                    .map(|(x, z)| {
                        let game = game.clone();
                        util::run_async(move || {
                            create_chunk(
                                &game,
                                &noise_generator,
                                glm::DVec3::new(
                                    5.5 + (511.0 * 6.0 * x as f64 - 256.0 * 12.0),
                                    0.0,
                                    5.5 + (511.0 * 6.0 * z as f64 - 256.0 * 12.0)
                                ),
                                3.0
                            )
                        })
                        .into()
                    })
            )
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

fn create_chunk(
    game: &game::Game,
    noise: &impl NoiseFn<f64, 2>,
    offset: glm::DVec3,
    scale: f64
) -> Arc<RasterChunk>
{
    let noise_sampler = |x: i32, z: i32| -> f64 {
        let h = 84.0f64;

        noise.get([(x as f64) / 256.0, (z as f64) / 256.0]) * h + h
    };

    let occupied = |x: i32, y: i32, z: i32| (y <= noise_sampler(x, z) as i32);

    assert!(offset.y == 0.0);

    RasterChunk::new(
        game,
        gfx::Transform {
            translation: glm::Vec3::new(offset.x as f32, 0.0, offset.z as f32),
            scale: glm::Vec3::new(scale as f32, scale as f32, scale as f32),
            ..Default::default()
        },
        iproduct!(0..511i32, 0..511i32).flat_map(|(local_x, local_z)| {
            let world_x = (scale * local_x as f64 + offset.x) as i32;
            let world_z = (scale * local_z as f64 + offset.z) as i32;

            let noise_h_world = noise_sampler(world_x, world_z);
            let local_h = (noise_h_world / scale) as i32;

            ((-4 + local_h)..(local_h + 4)).flat_map(move |sample_h_local| {
                let sample_h_world = (sample_h_local as f64) * scale;
                let voxel = if world_x.abs() < 1 || world_z.abs() < 1
                {
                    0
                }
                else
                {
                    rand::thread_rng().gen_range(1..=12)
                };

                VoxelFaceDirection::iterate().filter_map(move |d| {
                    if !occupied(world_x, sample_h_world as i32, world_z)
                    {
                        return None;
                    }

                    let axis = d.get_axis();

                    if occupied(
                        (world_x as f64 + scale * 1.001 * axis.x as f64).round() as i32,
                        (sample_h_world + scale * 1.001 * axis.y as f64).round() as i32,
                        (world_z as f64 + scale * 1.001 * axis.z as f64).round() as i32
                    )
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
                                local_x as u16,
                                (sample_h_world.max(0.0) / scale) as u16,
                                local_z as u16
                            )
                        })
                    }
                })
            })
        })
    )
}

// VoxelFaceDirection::iterate().flat_map(move |d| {

//     ((-5 + world_h)..(world_h + 5))
//         .step_by(scale as usize)
//         .filter_map(move |sample_world_h| {
//
//         })
// })

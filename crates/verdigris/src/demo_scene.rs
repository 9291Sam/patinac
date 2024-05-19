use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use gfx::glm::{self};
use gfx::wgpu;
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;
use voxel3::{VoxelFace, VoxelFaceDirection, VoxelManager};
#[derive(Debug)]
pub struct DemoScene
{
    dm: Arc<voxel3::VoxelManager>,
    id: util::Uuid // future: Mutex<util::Promise<()>>
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let noise_generator = noise::SuperSimplex::new(3478293422);

        let dm = VoxelManager::new(game.clone());

        let c_game = game.clone();
        let c_dm = dm.clone();

        util::run_async(move || {
            create_chunk(
                c_dm,
                c_game,
                &noise_generator,
                glm::DVec3::new(-256.0, 0.0, -256.0),
                1.0
            );
        })
        .detach();

        let this = Arc::new(DemoScene {
            dm: dm.clone(),
            id: util::Uuid::new() // future: Mutex::new(future.into())
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
        // self.future.lock().unwrap().poll_ref();
    }
}

fn create_chunk(
    dm: Arc<VoxelManager>,
    game: Arc<game::Game>,
    noise: &(impl NoiseFn<f64, 2> + Sync),
    offset: glm::DVec3,
    scale: f64
)
{
    let raw_noise_sampler = |x: i32, z: i32| -> f64 {
        let h = 84.0f64;

        noise.get([(x as f64) / 256.0, (z as f64) / 256.0]) * h + h
    };

    let mut noise_cache: Box<[[f64; 512]; 512]> = unsafe { Box::new_zeroed().assume_init() };

    for x in 0..512
    {
        for z in 0..512
        {
            let world_x = x as f64 + offset.x;
            let world_z = z as f64 + offset.z;

            noise_cache[x][z] = raw_noise_sampler(world_x as i32, world_z as i32);
        }
    }

    let noise_sampler = |world_x: i32, world_z: i32| -> f64 {
        noise_cache[(world_x as f64 - offset.x) as usize][(world_z as f64 - offset.z) as usize]
    };

    let occupied = |x: i32, y: i32, z: i32| (y <= noise_sampler(x, z) as i32);

    let face_iterator =
        spiral::ChebyshevIterator::new(256, 256, 255).flat_map(|(local_x, local_z)| {
            let world_x = (scale * local_x as f64 + offset.x) as i32;
            let world_z = (scale * local_z as f64 + offset.z) as i32;

            let noise_h_world = noise_sampler(world_x, world_z);
            let local_h = (noise_h_world / scale) as i32;

            ((-4 + local_h)..(local_h + 4)).flat_map(move |sample_h_local| {
                let sample_h_world = (sample_h_local as f64) * scale;
                let voxel = if world_x.abs() == 0 || world_z.abs() == 0
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
                        (world_x as f64 + scale * 1.01 * axis.x as f64).round() as i32,
                        (sample_h_world + scale * 1.01 * axis.y as f64).round() as i32,
                        (world_z as f64 + scale * 1.01 * axis.z as f64).round() as i32
                    )
                    {
                        None
                    }
                    else
                    {
                        Some(VoxelFace {
                            direction: d,
                            voxel,
                            position: glm::I32Vec3::new(
                                local_x as i32,
                                (sample_h_world.max(0.0) / scale) as i32,
                                local_z as i32
                            ),
                            material: rand::thread_rng().gen_range(0..=12)
                        })
                    }
                })
            })
        });

    for face in face_iterator
    {
        dm.insert_face(face)
    }
}

// fn create_chunk_rt(
//     game: &game::Game,
//     noise: &impl NoiseFn<f64, 2>,
//     offset: glm::DVec3,
//     scale: f64,
//     rx: oneshot::Receiver<Arc<VoxelChunkDataManager>>
// ) -> Arc<BrickMapChunk>
// {
//     // let noise_sampler = |x: i32, z: i32| -> f64 {
//     //     let h = 278.0f64;

//     //     (noise.get([(x as f64) / 256.0, (z as f64) / 256.0]) * h +
// h).clamp(0.0,     // 510.0) };

//     // let occupied = |x: i32, y: i32, z: i32| (y <= noise_sampler(x, z) as
// i32);

//     let c = BrickMapChunk::new(
//         game,
//         glm::Vec3::new(offset.x as f32, offset.y as f32, offset.z as f32),
//         Some(rx.recv().unwrap())
//     );

//     c.access_data_manager().flush_entire();

//     // let dm = c.access_data_manager();

//     // iproduct!(0..511i32, 0..511i32).for_each(|(local_x, local_z)| {
//     //     let world_x = (scale * local_x as f64 + offset.x) as i32;
//     //     let world_z = (scale * local_z as f64 + offset.z) as i32;

//     //     let noise_h_world = noise_sampler(world_x, world_z);
//     //     let local_h = (noise_h_world / scale) as i32;

//     //     ((-4 + local_h)..(local_h + 4)).for_each(move |sample_h_local| {
//     //         let sample_h_world = (sample_h_local as f64) * scale;
//     //         let voxel = if world_x.abs() == 0 || world_z.abs() == 0
//     //         {
//     //             0
//     //         }
//     //         else
//     //         {
//     //             rand::thread_rng().gen_range(1..=12)
//     //         };

//     //         VoxelFaceDirection::iterate().for_each(move |d| {
//     //             let axis = d.get_axis();

//     //             if !occupied(
//     //                 (world_x as f64 + scale * 1.01 * axis.x as
// f64).round() as     // i32,                 (sample_h_world + scale * 1.01 *
// axis.y as     // f64).round() as i32,                 (world_z as f64 + scale
// * 1.01 *     // axis.z as f64).round() as i32             ) //             {
//   //                 dm.write_voxel( // Voxel::try_from(voxel).unwrap(), //
//   glm::U16Vec3::new( //                         local_x as u16, //
//   (sample_h_world.max(0.0) / scale) as u16, // local_z as u16 // ) // ); // }
//   //         }) //     }) // });

//     // dm.flush_entire();

//     c
// }

// VoxelFaceDirection::iterate().flat_map(move |d| {

//     ((-5 + world_h)..(world_h + 5))
//         .step_by(scale as usize)
//         .filter_map(move |sample_world_h| {
//
//         })
// })

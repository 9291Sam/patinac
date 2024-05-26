use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use dot_vox::DotVoxData;
use gfx::glm::{self};
use gfx::wgpu;
use itertools::iproduct;
use noise::NoiseFn;
use rand::{Rng, SeedableRng};
use voxel3::{VoxelWorld, WorldPosition};
#[derive(Debug)]
pub struct DemoScene
{
    dm: Arc<voxel3::VoxelWorld>,
    id: util::Uuid // future: Mutex<util::Promise<()>>
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let noise_generator = noise::SuperSimplex::new(3478293422);

        let dm = VoxelWorld::new(game.clone());

        let c_dm = dm.clone();

        util::run_async(move || {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(238902348902348);

            let it = iproduct!(-127..0, 0..127, 0..127).map(|(x, y, z)| {
                (
                    WorldPosition(glm::I32Vec3::new(x, y, z)),
                    rng.gen_range(0..=18).try_into().unwrap()
                )
            });

            c_dm.insert_many_voxel(it);

            // for (x, y, z) in
            // {
            //     c_dm.insert_voxel(
            //         ,

            //     );
            // }

            // let data =
            // dot_vox::load_bytes(include_bytes!("../../../room.vox")).
            // unwrap();

            // let dot_vox::Size {
            //     x,
            //     y,
            //     z
            // } = data.models[0].size;

            // for (xx, yy, zz) in iproduct!(0..3, 0..3, 0..3)
            // {
            //     load_model_from_file_into(
            //         glm::I32Vec3::new(xx * x as i32 + 1, yy * z as i32 + 1,
            // zz * y as i32
            // + 1),         &c_dm, &data );
            // }

            // arbitrary_landscape_demo(&c_dm)
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

fn load_model_from_file_into(world_offset: glm::I32Vec3, dm: &VoxelWorld, data: &DotVoxData)
{
    let voxels: HashMap<glm::U8Vec3, u8> = data.models[0]
        .voxels
        .iter()
        .map(|mv| (glm::U8Vec3::new(mv.x, mv.y, mv.z), mv.i))
        .collect::<HashMap<_, _>>();

    for (pos, _) in voxels.iter()
    {
        let pos = pos.xzy();

        dm.insert_voxel(
            WorldPosition(pos.cast() + world_offset),
            ((pos.x % 3 + pos.y % 4 + pos.z % 7) as u16)
                .try_into()
                .unwrap()
        );
    }
}

fn arbitrary_landscape_demo(dm: &VoxelWorld)
{
    let noise = noise::OpenSimplex::new(2384247834);

    spiral::ChebyshevIterator::new(0, 0, 512).for_each(|(x, z)| {
        dm.insert_voxel(
            WorldPosition(glm::I32Vec3::new(
                x,
                (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 256.0) as i32,
                z
            )),
            rand::thread_rng().gen_range(1..=8).try_into().unwrap()
        )
    })
}

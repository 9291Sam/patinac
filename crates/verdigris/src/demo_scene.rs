use std::borrow::Cow;
use std::sync::Arc;

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::{Rng, SeedableRng};
use voxel::{VoxelWorld, WorldPosition};
#[derive(Debug)]
pub struct DemoScene
{
    dm: Arc<voxel::VoxelWorld>,
    id: util::Uuid // future: Mutex<util::Promise<()>>
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
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

            std::thread::sleep_ms(10);

            load_model_from_file_into(
                glm::I32Vec3::new(0, 126, 0),
                &c_dm,
                &dot_vox::load_bytes(include_bytes!("../../../menger.vox")).unwrap()
            );

            std::thread::sleep_ms(10);

            arbitrary_landscape_demo(&c_dm);
        })
        .detach();

        let this = Arc::new(DemoScene {
            dm: dm.clone(),
            id: util::Uuid::new()
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
    let it = data.models[0].voxels.iter().map(|pos| {
        (
            WorldPosition(glm::U8Vec3::new(pos.x, pos.y, pos.z).cast() + world_offset),
            ((pos.x % 3 + pos.y % 4 + pos.z % 7) as u16)
                .try_into()
                .unwrap()
        )
    });

    dm.insert_many_voxel(it);
}

fn arbitrary_landscape_demo(dm: &VoxelWorld)
{
    let noise = noise::OpenSimplex::new(2384247834);

    let it = spiral::ChebyshevIterator::new(0, 0, 1024).map(|(x, z)| {
        (
            WorldPosition(glm::I32Vec3::new(
                x,
                (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 256.0) as i32,
                z
            )),
            rand::thread_rng().gen_range(1..=8).try_into().unwrap()
        )
    });

    dm.insert_many_voxel(it);
}

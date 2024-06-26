use std::borrow::Cow;
use std::sync::Arc;

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use voxel::{ChunkCoordinate, ChunkLocalPosition, ChunkPool};
use voxel2::{ChunkCollider, ChunkManager, WorldPosition};

use crate::Player;

pub struct DemoScene
{
    chunk_pool: Arc<voxel::ChunkPool>,
    id:         util::Uuid,

    player:         Arc<Player>,
    camera_updater: util::WindowUpdater<gfx::Camera> /* future_collider:
                                                      * util::Promise<Arc<ChunkCollider>> */
}
unsafe impl Sync for DemoScene {}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>, camera_updater: util::WindowUpdater<gfx::Camera>)
    -> Arc<Self>
    {
        let chunk_pool = ChunkPool::new(game.clone());
        let pool2 = chunk_pool.clone();

        let future_collider = util::run_async(move || {
            let chunk = pool2.allocate_chunk(ChunkCoordinate(glm::I32Vec3::new(0, 0, 0)));

            let it = iproduct!(0..64, 0..64, 0..64).map(|(x, y, z)| {
                (
                    ChunkLocalPosition(glm::U8Vec3::new(x, y, z)),
                    voxel::Voxel::Dirt2
                )
            });

            pool2.write_many_voxel(&chunk, it);

            // load_model_from_file_into(
            //     glm::I32Vec3::new(0, 126, 0),
            //     &c_dm2,
            //     &dot_vox::load_bytes(include_bytes!("../../../models/menger.
            // vox")). unwrap() );

            // arbitrary_landscape_demo(&pool2);

            // pool2.build_collision_info()
        })
        .detach();

        let player = Player::new(
            &game,
            gfx::Camera::new(glm::Vec3::new(-186.0, 354.0, -168.0), 0.218903, 0.748343)
        );

        camera_updater.update(player.get_camera());

        let this = Arc::new(DemoScene {
            chunk_pool: chunk_pool.clone(),
            id: util::Uuid::new(),
            player,
            camera_updater
        });

        game.register(this.clone());

        this
    }
}

impl game::EntityCastDepot for DemoScene
{
    fn as_self_managed(self: Arc<Self>) -> Option<Arc<dyn game::SelfManagedEntity>>
    {
        None
    }

    fn as_positionalable(&self) -> Option<&dyn game::Positionalable>
    {
        None
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
    }

    fn as_collideable(&self) -> Option<&dyn game::Collideable>
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
        self.camera_updater.update(self.player.get_camera());
    }
}

fn load_model_from_file_into(world_offset: glm::I32Vec3, dm: &ChunkManager, data: &DotVoxData)
{
    let it = data.models[0]
        .voxels
        .iter()
        .map(|pos| WorldPosition(glm::U8Vec3::new(pos.x, pos.y, pos.z).cast() + world_offset));

    dm.insert_many_voxel(it);
}

fn arbitrary_landscape_demo(dm: &ChunkManager)
{
    let noise = noise::OpenSimplex::new(2384247834);

    let it = spiral::ChebyshevIterator::new(0, 0, 512).map(|(x, z)| {
        WorldPosition(glm::I32Vec3::new(
            x,
            (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 128.0) as i32,
            z
        ))
    });

    // in a spiral formation stating at the center broding out, sample a height map
    // and collect a vector of the samplied points and a random color value

    dm.insert_many_voxel(it);
}

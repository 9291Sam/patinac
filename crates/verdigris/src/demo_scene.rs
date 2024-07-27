use std::borrow::Cow;
use std::sync::Arc;

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;

use crate::recordables::skybox::Skybox;
use crate::voxel_world::{VoxelWorld, WorldPosition};
use crate::Player;

pub struct DemoScene
{
    world: Arc<crate::voxel_world::VoxelWorld>,
    id:    util::Uuid,

    player:         Arc<Player>,
    camera_updater: util::WindowUpdater<gfx::Camera>,
    _skybox:        Arc<Skybox>
}
unsafe impl Sync for DemoScene {}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>, camera_updater: util::WindowUpdater<gfx::Camera>)
    -> Arc<Self>
    {
        let world = VoxelWorld::new(game.clone());
        let world2 = world.clone();

        util::run_async(move || {
            let w = world2;

            // let it = iproduct!(0..64, 0..64, 0..64).map(|(x, y, z)| {
            //     (
            //         WorldPosition(glm::I32Vec3::new(x + 1, y + 1, z + 1)),
            //         rand::thread_rng().gen_range(12..=14).try_into().unwrap()
            //     )
            // });

            let it = iproduct!(-32..32, -32..32).map(|(x, z)| {
                (
                    WorldPosition(glm::I32Vec3::new(x, 32, z)),
                    rand::thread_rng().gen_range(15..=18).try_into().unwrap()
                )
            });

            w.write_many_voxel(it);

            w.flush_all_voxel_updates();

            // load_model_from_file_into(
            //     glm::I32Vec3::new(0, 126, 0),
            //     &w,
            //     &dot_vox::load_bytes(include_bytes!("../../../models/menger.vox")).
            // unwrap() );

            w.flush_all_voxel_updates();

            // arbitrary_landscape_demo(&w);

            flat_demo(&w);

            // pool2.build_collision_info()

            w.flush_all_voxel_updates();
        })
        .detach();

        let player = Player::new(
            &game,
            gfx::Camera::new(glm::Vec3::new(-173.0, 184.0, -58.0), 0.218903, 0.748343)
        );

        camera_updater.update(player.get_camera());

        let this = Arc::new(DemoScene {
            world,
            id: util::Uuid::new(),
            player,
            camera_updater,
            _skybox: Skybox::new_skybox(game.clone(), gfx::Transform::default())
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
        let camera = self.player.get_camera();

        self.world
            .update_with_camera_position(camera.get_position());

        self.camera_updater.update(camera);
    }
}

fn load_model_from_file_into(world_offset: glm::I32Vec3, world: &VoxelWorld, data: &DotVoxData)
{
    let it = data.models[0].voxels.iter().map(|pos| {
        (
            WorldPosition(glm::U8Vec3::new(pos.x, pos.y, pos.z).cast() + world_offset),
            rand::thread_rng().gen_range(15..=18).try_into().unwrap()
        )
    });

    world.write_many_voxel(it);
}

fn arbitrary_landscape_demo(world: &VoxelWorld)
{
    let noise = noise::OpenSimplex::new(2384247834);

    let it = spiral::ChebyshevIterator::new(0, 0, 512).map(|(x, z)| {
        (
            WorldPosition(glm::I32Vec3::new(
                x,
                (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 128.0
                    + -32.0 * f64::exp(-(x as f64 * x as f64 + z as f64 * z as f64) / 4096.0))
                    as i32,
                z
            )),
            rand::thread_rng().gen_range(1..=11).try_into().unwrap()
        )
    });

    world.write_many_voxel(it);
}

fn flat_demo(world: &VoxelWorld)
{
    let it = spiral::ChebyshevIterator::new(0, 0, 1024).map(|(x, z)| {
        (
            WorldPosition(glm::I32Vec3::new(x, 0, z)),
            voxel::Voxel::SilverMeta0
        )
    });

    world.write_many_voxel(it);
}

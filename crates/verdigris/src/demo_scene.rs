use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::SeedableRng;
use voxel2::{ChunkManager, WorldPosition};

use crate::player::Player;
use crate::recordables::lit_textured::LitTextured;

pub struct DemoScene
{
    _dm:               Arc<voxel2::ChunkManager>,
    lit_textured_cube: Arc<LitTextured>,
    id:                util::Uuid,

    player:         Arc<Player>,
    camera_updater: util::WindowUpdater<gfx::Camera>
}

impl DemoScene
{
    pub fn new(game: Arc<game::Game>, camera_updater: util::WindowUpdater<gfx::Camera>)
    -> Arc<Self>
    {
        let dm = ChunkManager::new(game.clone());
        let c_dm = dm.clone();
        let c_dm2 = dm.clone();

        let mut rng = rand::rngs::SmallRng::seed_from_u64(23879234789234);

        util::run_async(move || {
            let it = iproduct!(0..64, -64..0, 0..64)
                .map(|(x, y, z)| WorldPosition(glm::I32Vec3::new(x, y, z)));

            c_dm2.insert_many_voxel(it);

            load_model_from_file_into(
                glm::I32Vec3::new(0, 126, 0),
                &c_dm2,
                &dot_vox::load_bytes(include_bytes!("../../../models/menger.vox")).unwrap()
            );

            arbitrary_landscape_demo(&c_dm2);
        })
        .detach();

        let player = Player::new(
            &game,
            gfx::Camera::new(glm::Vec3::new(-186.0, 154.0, -168.0), 0.218903, 0.748343)
        );

        camera_updater.update(player.get_camera());

        let this = Arc::new(DemoScene {
            _dm: dm.clone(),
            id: util::Uuid::new(),
            lit_textured_cube: LitTextured::new_cube(
                game.clone(),
                gfx::Transform {
                    scale: glm::Vec3::new(5.0, 5.0, 5.0),
                    ..Default::default()
                }
            ),
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

    fn tick(&self, game: &game::Game, _: game::TickTag)
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

    let it = spiral::ChebyshevIterator::new(0, 0, 768).map(|(x, z)| {
        WorldPosition(glm::I32Vec3::new(
            x,
            (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 256.0) as i32,
            z
        ))
    });

    // in a spiral formation stating at the center broding out, sample a height map
    // and collect a vector of the samplied points and a random color value

    dm.insert_many_voxel(it);
}

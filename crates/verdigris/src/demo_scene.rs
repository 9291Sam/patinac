use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Mul;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::Rng;
use voxel::PointLight;

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

impl Debug for DemoScene
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "DemoScene")
    }
}

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
                    WorldPosition(glm::I32Vec3::new(x, 50, z)),
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

        camera_updater.update(player.get_player_vision_camera());

        let this = Arc::new(DemoScene {
            world,
            id: util::Uuid::new(),
            player,
            camera_updater,
            _skybox: Skybox::new_skybox(game.clone(), gfx::Transform::default())
        });

        game.register(this.clone());
        game.get_renderer().register(this.clone());

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
        self.world
            .update_with_camera_position(self.player.get_collider_position());

        self.camera_updater
            .update(self.player.get_player_vision_camera());
    }
}

impl gfx::Recordable for DemoScene
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Demo Scene")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn pre_record_update(
        &self,
        encoder: &mut gfx::wgpu::CommandEncoder,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        static TIME_ALIVE: util::AtomicF32 = util::AtomicF32::new(0.0);
        const MAX_LIGHTS: usize = 256;
        const RADIUS: usize = 256;

        static LIGHTS: Mutex<[PointLight; MAX_LIGHTS]> = Mutex::new(
            [PointLight {
                position:        glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                color_and_power: glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                falloffs:        glm::Vec4::new(0.0, 0.0, 1.0, 0.00)
            }; MAX_LIGHTS]
        );

        let lights: &mut [PointLight; MAX_LIGHTS] = &mut LIGHTS.lock().unwrap();

        let time_alive = TIME_ALIVE.load(Ordering::Acquire);

        for (idx, l) in lights.iter_mut().enumerate()
        {
            let percent_around = (idx as f32) / (MAX_LIGHTS as f32);
            let angle = percent_around * std::f32::consts::PI * 2.0;

            l.color_and_power = glm::Vec4::new(1.0, 1.0, 1.0, idx as f32 * idx as f32);

            l.position = glm::Vec4::new(
                angle.sin() * ((8.0 * angle + time_alive).cos().mul(120.0) + RADIUS as f32),
                13.0,
                angle.cos() * ((8.0 * angle + time_alive).cos().mul(120.0) + RADIUS as f32),
                0.0
            );
        }

        TIME_ALIVE.aba_add(renderer.get_delta_time(), Ordering::SeqCst);

        self.world.write_lights(lights);

        gfx::RecordInfo::NoRecord
    }

    fn record<'s>(&'s self, _: &mut gfx::GenericPass<'s>, _: Option<gfx::DrawId>)
    {
        unreachable!()
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
    let it = spiral::ChebyshevIterator::new(0, 0, 444).map(|(x, z)| {
        (
            WorldPosition(glm::I32Vec3::new(x, 0, z)),
            rand::thread_rng().gen_range(15..=18).try_into().unwrap()
        )
    });

    world.write_many_voxel(it);
}

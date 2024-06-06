use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use dot_vox::DotVoxData;
use gfx::glm::{self};
use itertools::iproduct;
use noise::NoiseFn;
use rand::SeedableRng;
use voxel2::{ChunkManager, WorldPosition};

use crate::recordables::lit_textured::LitTextured;

#[derive(Debug)]
pub struct DemoScene
{
    _dm:               Arc<voxel2::ChunkManager>,
    lit_textured_cube: Arc<LitTextured>,
    id:                util::Uuid,

    camera:         Mutex<gfx::Camera>,
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

        let inital_camera =
            gfx::Camera::new(glm::Vec3::new(-186.0, 154.0, -168.0), 0.218903, 0.748343);

        camera_updater.update(inital_camera.clone());

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
            camera: Mutex::new(inital_camera),
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
        let renderer = game.get_renderer().clone();
        let mut camera = self.camera.lock().unwrap();

        *camera = modify_camera_for_next_frame(
            camera.clone(),
            renderer.get_framebuffer_size(),
            renderer.get_fov(),
            renderer.get_delta_time(),
            game.get_delta_time(),
            renderer.get_input_manager()
        );

        self.camera_updater.update(camera.clone());
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

    let it = spiral::ChebyshevIterator::new(0, 0, 256).map(|(x, z)| {
        WorldPosition(glm::I32Vec3::new(
            x,
            (noise.get([x as f64 / 256.0, z as f64 / 256.0]) * 256.0 - 128.0) as i32,
            z
        ))
    });

    // in a spiral formation stating at the center broding out, sample a height map
    // and collect a vector of the samplied points and a random color value

    dm.insert_many_voxel(it);
}

pub fn modify_camera_for_next_frame(
    old_camera: gfx::Camera,
    renderer_framebuffer_size: glm::UVec2,
    renderer_fov: glm::Vec2,
    renderer_delta_time: f32,
    game_delta_time: f32,
    input_manager: &gfx::InputManager
) -> gfx::Camera
{
    let mut camera = old_camera;

    let move_scale = 10.0
        * if input_manager.is_key_pressed(gfx::KeyCode::ShiftLeft)
        {
            20.0
        }
        else
        {
            4.0
        };
    let rotate_scale = 10.0;

    if input_manager.is_key_pressed(gfx::KeyCode::KeyW)
    {
        let v = camera.get_forward_vector() * move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyS)
    {
        let v = camera.get_forward_vector() * -move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyD)
    {
        let v = camera.get_right_vector() * move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyA)
    {
        let v = camera.get_right_vector() * -move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::Space)
    {
        let v = *gfx::Transform::global_up_vector() * move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::ControlLeft)
    {
        let v = *gfx::Transform::global_up_vector() * -move_scale;

        camera.add_position(v * game_delta_time);
    };

    if input_manager.is_key_pressed(gfx::KeyCode::Backslash)
    {
        input_manager.detach_cursor();
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyK)
    {
        log::info!("Camera: {}", camera)
    };

    let mouse_diff_px: glm::Vec2 = {
        let mouse_cords_diff_px_f32: (f32, f32) = input_manager.get_mouse_delta();

        glm::Vec2::new(mouse_cords_diff_px_f32.0, mouse_cords_diff_px_f32.1)
    };

    let screen_size_px: glm::Vec2 = {
        let screen_size_u32 = renderer_framebuffer_size;

        glm::Vec2::new(screen_size_u32.x as f32, screen_size_u32.y as f32)
    };

    // delta over the whole screen -1 -> 1
    let normalized_delta = mouse_diff_px.component_div(&screen_size_px);

    let delta_rads = normalized_delta
        .component_div(&glm::Vec2::repeat(2.0))
        .component_mul(&renderer_fov);

    if renderer_delta_time != 0.0
    {
        camera.add_yaw(delta_rads.x / renderer_delta_time * rotate_scale * game_delta_time);
        camera.add_pitch(delta_rads.y / renderer_delta_time * rotate_scale * game_delta_time);
    }

    camera
}

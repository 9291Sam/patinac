use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use gfx::nal::Isometry3;
use gfx::{glm, nal};
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::geometry::{Collider, ColliderBuilder};
use rapier3d::prelude::RigidBody;

pub struct Player
{
    uuid: util::Uuid,

    camera: Mutex<gfx::Camera>
}

impl Player
{
    pub fn new(game: &game::Game, inital_camera: gfx::Camera) -> Arc<Self>
    {
        let this = Arc::new(Player {
            uuid:   util::Uuid::new(),
            camera: Mutex::new(inital_camera)
        });

        game.register(this.clone());

        this
    }

    pub fn get_camera(&self) -> gfx::Camera
    {
        self.camera.lock().unwrap().clone()
    }
}

impl game::Entity for Player
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Player")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn tick(&self, _: &game::Game, _: game::TickTag) {}
}

impl game::EntityCastDepot for Player
{
    fn as_self_managed(
        self: std::sync::Arc<Self>
    ) -> Option<std::sync::Arc<dyn game::SelfManagedEntity>>
    {
        None
    }

    fn as_positionalable(&self) -> Option<&dyn game::Positionalable>
    {
        Some(self)
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        Some(self)
    }

    fn as_collideable(&self) -> Option<&dyn game::Collideable>
    {
        Some(self)
    }
}

impl game::Positionalable for Player
{
    fn get_position(&self) -> gfx::glm::Vec3
    {
        self.camera.lock().unwrap().get_position()
    }
}

impl game::Transformable for Player
{
    fn get_transform(&self) -> gfx::Transform
    {
        self.camera.lock().unwrap().get_transform()
    }
}

impl game::Collideable for Player
{
    fn init_collideable(&self) -> (RigidBody, Vec<Collider>)
    {
        (
            RigidBodyBuilder::kinematic_position_based().build(),
            [ColliderBuilder::capsule_y(48.0, 16.0).build()].to_vec()
        )
    }

    fn physics_tick(&self, rigid_body: &mut RigidBody, game: &game::Game, _: game::TickTag)
    {
        let mut current_camera = self.camera.lock().unwrap();

        let next_frame_camera = modify_camera_based_on_inputs(current_camera.clone(), game);

        rigid_body.set_next_kinematic_position(Isometry3 {
            rotation:    glm::UnitQuaternion::new_unchecked(
                next_frame_camera.get_transform().rotation.normalize()
            ),
            translation: nal::Translation {
                vector: next_frame_camera.get_position()
            }
        });

        *current_camera = next_frame_camera;
    }

    // fn init_collideable(&self) -> RigidBody
    // {
    //     RigidBodyBuilder::kinematic_position_based()
    // }

    // fn physics_tick(&self, rigid_body: &mut RigidBody, game: &game::Game, _:
    // game::TickTag) {
    //     self.camera = rigid_body.position();

    //     modify_camera_based_on_inputs();

    //     rigid_body.set_next_kinematic_translation(translation);
    //     rigid_body.set_next_kinematic_rotation(rotation);
    // }
}

pub fn modify_camera_based_on_inputs(mut camera: gfx::Camera, game: &game::Game) -> gfx::Camera
{
    let renderer = game.get_renderer();

    let renderer_framebuffer_size = renderer.get_framebuffer_size();
    let renderer_fov: glm::Vec2 = renderer.get_fov();
    let renderer_delta_time: f32 = renderer.get_delta_time();
    let game_delta_time = game.get_delta_time();
    let input_manager = renderer.get_input_manager();

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

use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use gfx::glm;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::geometry::{Collider, ColliderBuilder};
use rapier3d::prelude::RigidBody;
use util::AtomicF32;

pub struct Player
{
    uuid: util::Uuid,

    camera:        Mutex<gfx::Camera>,
    time_floating: AtomicF32
}

impl Player
{
    pub fn new(game: &game::Game, inital_camera: gfx::Camera) -> Arc<Self>
    {
        let this = Arc::new(Player {
            uuid:          util::Uuid::new(),
            camera:        Mutex::new(inital_camera),
            time_floating: AtomicF32::new(0.0)
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
            RigidBodyBuilder::dynamic()
                .ccd_enabled(true)
                .translation(self.camera.lock().unwrap().get_position())
                .can_sleep(false)
                .lock_rotations()
                .additional_solver_iterations(16)
                .build(),
            vec![
                ColliderBuilder::capsule_y(24.0, 6.0)
                    .contact_force_event_threshold(0.001)
                    .friction(1.5)
                    .friction_combine_rule(rapier3d::dynamics::CoefficientCombineRule::Multiply)
                    .enabled(true)
                    .build(),
            ]
        )
    }

    fn physics_tick(
        &self,
        game: &game::Game,
        _: glm::Vec3,
        this_body: &mut RigidBody,
        _: game::TickTag
    )
    {
        // goals
        // movement
        // gravity
        // no clipping
        // proper jump
        // TODO: do something to disable infinite user provided accelleration
        let mut camera = self.camera.lock().unwrap();

        let DesiredMovementResult {
            mut desired_translation,
            delta_pitch,
            delta_yaw,
            wants_to_jump
        } = calculate_desired_movement(
            camera.get_forward_vector(),
            camera.get_right_vector(),
            game
        );

        desired_translation.y = 0.0;

        if self.time_floating.load(Ordering::Acquire) != 0.0
        {
            desired_translation = glm::Vec3::zeros();
        }

        this_body.apply_impulse(desired_translation / game.get_delta_time(), true);

        if wants_to_jump && self.time_floating.load(Ordering::Acquire) < 0.01
        {
            this_body.apply_impulse(glm::Vec3::new(0.0, 600.0, 0.0), true)
        }

        if this_body.linvel().y.abs() > 0.1
        // TODO: fix
        {
            self.time_floating
                .aba_add(game.get_delta_time(), Ordering::AcqRel);
        }
        else
        {
            self.time_floating.store(0.0, Ordering::Release)
        }

        // log::trace!(
        //     "BLOCK_ONPosition: {:?} | Integrated Velocity: {:?} {} | Time Floating:
        // {}",     this_body.translation(),
        //     this_body.linvel(),
        //     this_body.linvel().magnitude(),
        //     self.time_floating.load(Ordering::Acquire)
        // );

        camera.add_pitch(delta_pitch);
        camera.add_yaw(delta_yaw);
        camera.set_position(*this_body.translation());
    }

    // fn init_collideable(&self) -> (RigidBody, Vec<Collider>)
    // {
    //     // We actually don't want this to be managed by the main system, so just
    // put a     // fixed immovable without any colldiers at the origin.
    //     (
    //         RigidBodyBuilder::fixed()
    //             .enabled(false)
    //             .sleeping(true)
    //             .build(),
    //         Vec::new()
    //     )
    // }

    // fn physics_tick(
    //     &self,
    //     game: &game::Game,
    //     gravity: glm::Vec3,
    //     this_handle: RigidBodyHandle,
    //     rigid_body_set: &mut RigidBodySet,
    //     collider_set: &mut ColliderSet,
    //     query_pipeline: &QueryPipeline,
    //     _: game::TickTag
    // )
    // {
    //     let mut camera = self.camera.lock().unwrap();

    //     let desired_camera = modify_camera_based_on_inputs(camera.clone(), game);

    //     let move_result = self.player_controller.move_shape(
    //         game.get_delta_time(),
    //         rigid_body_set,
    //         collider_set,
    //         query_pipeline,
    //         &Capsule::new_y(24.0, 16.0),
    //         &get_isometry_of_camera(&camera),
    //         desired_camera.get_position()
    //             + get_gravity_influenced_velocity_given_time_floating(
    //               self.time_floating.load(Ordering::Acquire), gravity
    //             ) * game.get_delta_time(),
    //         QueryFilter::new().exclude_rigid_body(this_handle),
    //         |_| {}
    //     );

    //     rigid_body_set.get(this_handle).unwrap().pred

    //     if !move_result.grounded
    //     {
    //         self.time_floating.store(
    //             self.time_floating.load(Ordering::Acquire) +
    // game.get_delta_time(),             Ordering::Release
    //         )
    //     }
    //     else
    //     {
    //         self.time_floating.store(0.0, Ordering::Release)
    //     }

    //     camera.set_position(move_result.translation);
    //     camera.set_yaw(desired_camera.get_yaw());
    //     camera.set_pitch(desired_camera.get_pitch());
    // }
}

struct DesiredMovementResult
{
    desired_translation: glm::Vec3,
    delta_pitch:         f32,
    delta_yaw:           f32,
    wants_to_jump:       bool
}

fn calculate_desired_movement(
    forward_vector: glm::Vec3,
    right_vector: glm::Vec3,
    game: &game::Game
) -> DesiredMovementResult
{
    let mut net_translation = glm::Vec3::repeat(0.0);

    let renderer = game.get_renderer();

    let renderer_framebuffer_size = renderer.get_framebuffer_size();
    let renderer_fov: glm::Vec2 = renderer.get_fov();
    let renderer_delta_time: f32 = renderer.get_delta_time();
    let game_delta_time = game.get_delta_time();
    let input_manager = renderer.get_input_manager();

    let move_scale = 10.0
        * if input_manager.is_key_pressed(gfx::KeyCode::ShiftLeft)
        {
            7.0
        }
        else
        {
            1.0
        };
    let rotate_scale = 10.0;

    if input_manager.is_key_pressed(gfx::KeyCode::KeyW)
    {
        let v = forward_vector * move_scale;

        net_translation += v * game_delta_time;
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyS)
    {
        let v = forward_vector * -move_scale;

        net_translation += v * game_delta_time;
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyD)
    {
        let v = right_vector * move_scale;

        net_translation += v * game_delta_time;
    };

    if input_manager.is_key_pressed(gfx::KeyCode::KeyA)
    {
        let v = right_vector * -move_scale;

        net_translation += v * game_delta_time;
    };

    let mut wants_to_jump = false;

    if input_manager.is_key_pressed(gfx::KeyCode::Space)
    {
        // let v = *gfx::Transform::global_up_vector() * move_scale;

        wants_to_jump = true;
        // net_translation += v * game_delta_time;
    };

    if input_manager.is_key_pressed(gfx::KeyCode::ControlLeft)
    {
        let v = *gfx::Transform::global_up_vector() * -move_scale;

        net_translation += v * game_delta_time;
    };

    if input_manager.is_key_pressed(gfx::KeyCode::Backslash)
    {
        input_manager.detach_cursor();
    };

    // if input_manager.is_key_pressed(gfx::KeyCode::KeyK)
    // {
    //     log::info!("Camera: {}", camera)
    // };

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

    let mut pitch = 0.0;
    let mut yaw = 0.0;
    if renderer_delta_time != 0.0
    {
        pitch = delta_rads.y / renderer_delta_time * rotate_scale * game_delta_time;

        yaw = delta_rads.x / renderer_delta_time * rotate_scale * game_delta_time
    }

    net_translation.y = 0.0;

    DesiredMovementResult {
        desired_translation: net_translation,
        delta_pitch:         pitch,
        delta_yaw:           yaw,
        wants_to_jump:       wants_to_jump
    }
}

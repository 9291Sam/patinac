use std::f32::consts::{FRAC_PI_2, TAU};
use std::fmt::Display;

use nalgebra::UnitQuaternion;
use nalgebra_glm as glm;

#[derive(Debug, Clone)]
pub struct Camera
{
    pitch:     f32,
    yaw:       f32,
    transform: Transform
}

impl Display for Camera
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            f,
            "Position {{ {}, {}, {} }} | Pitch: {} | Yaw: {}",
            self.transform.translation.x,
            self.transform.translation.y,
            self.transform.translation.z,
            self.pitch,
            self.yaw
        )
    }
}

impl Camera
{
    pub fn new(translation: glm::Vec3, pitch: f32, yaw: f32) -> Camera
    {
        let mut this = Camera {
            pitch,
            yaw,
            transform: Transform {
                translation,
                ..Default::default()
            }
        };

        this.enforce_invariants();

        this
    }

    pub fn get_perspective(&self, renderer: &super::Renderer, transform: &Transform) -> glm::Mat4
    {
        let mut projection = glm::infinite_perspective_rh_zo(
            renderer.get_framebuffer_size().x as f32 / renderer.get_framebuffer_size().y as f32,
            renderer.get_fov().y,
            0.001
        );

        projection[(3, 2)] *= -1.0;
        projection[(2, 2)] *= -1.0;

        projection * self.get_view_matrix() * transform.as_model_matrix()
    }

    pub fn get_view_matrix(&self) -> glm::Mat4
    {
        (self.transform.as_translation_matrix() * self.transform.as_rotation_matrix())
            .try_inverse()
            .unwrap()
    }

    pub fn get_forward_vector(&self) -> glm::Vec3
    {
        self.transform.get_forward_vector()
    }

    pub fn get_right_vector(&self) -> glm::Vec3
    {
        self.transform.get_right_vector()
    }

    pub fn get_up_vector(&self) -> glm::Vec3
    {
        self.transform.get_up_vector()
    }

    pub fn get_position(&self) -> glm::Vec3
    {
        self.transform.translation
    }

    pub fn add_position(&mut self, position: glm::Vec3)
    {
        self.transform.translation += position;
    }

    pub fn add_pitch(&mut self, pitch_to_add: f32)
    {
        self.pitch += pitch_to_add;

        self.enforce_invariants();
    }

    pub fn add_yaw(&mut self, yaw_to_add: f32)
    {
        self.yaw += yaw_to_add;

        self.enforce_invariants();
    }

    fn enforce_invariants(&mut self)
    {
        self.pitch = self.pitch.clamp(-FRAC_PI_2, FRAC_PI_2);
        self.yaw %= TAU;

        self.transform.rotation =
            *(UnitQuaternion::from_axis_angle(&Transform::global_up_vector(), self.yaw)
                * UnitQuaternion::from_axis_angle(&Transform::global_right_vector(), self.pitch));
    }
}

#[derive(Debug, Clone)]
pub struct Transform
{
    pub translation: glm::Vec3,
    pub rotation:    nalgebra::Quaternion<f32>,
    pub scale:       glm::Vec3
}

impl Default for Transform
{
    fn default() -> Self
    {
        Self::new()
    }
}

impl Transform
{
    pub fn new() -> Transform
    {
        Transform {
            translation: glm::Vec3::repeat(0.0),
            rotation:    glm::Quat::new(1.0, 0.0, 0.0, 0.0),
            scale:       glm::Vec3::repeat(1.0)
        }
    }

    pub fn global_forward_vector() -> nalgebra::UnitVector3<f32>
    {
        nalgebra::UnitVector3::new_normalize(glm::Vec3::new(0.0, 0.0, 1.0))
    }

    pub fn global_right_vector() -> nalgebra::UnitVector3<f32>
    {
        nalgebra::UnitVector3::new_normalize(glm::Vec3::new(1.0, 0.0, 0.0))
    }

    pub fn global_up_vector() -> nalgebra::UnitVector3<f32>
    {
        nalgebra::UnitVector3::new_normalize(glm::Vec3::new(0.0, 1.0, 0.0))
    }

    pub fn as_model_matrix(&self) -> glm::Mat4
    {
        self.as_translation_matrix() * self.as_rotation_matrix() * self.as_scale_matrix()
    }

    pub fn as_translation_matrix(&self) -> glm::Mat4
    {
        glm::translate(&glm::Mat4::identity(), &self.translation)
    }

    pub fn as_rotation_matrix(&self) -> glm::Mat4
    {
        glm::quat_to_mat4(&self.rotation.normalize())
    }

    pub fn as_scale_matrix(&self) -> glm::Mat4
    {
        glm::scale(&glm::Mat4::identity(), &self.scale)
    }

    pub fn get_forward_vector(&self) -> glm::Vec3
    {
        *(UnitQuaternion::new_normalize(self.rotation).to_rotation_matrix()
            * Transform::global_forward_vector())
    }

    pub fn get_right_vector(&self) -> glm::Vec3
    {
        *(UnitQuaternion::new_normalize(self.rotation).to_rotation_matrix()
            * Transform::global_right_vector())
    }

    pub fn get_up_vector(&self) -> glm::Vec3
    {
        *(UnitQuaternion::new_normalize(self.rotation).to_rotation_matrix()
            * Transform::global_up_vector())
    }
}

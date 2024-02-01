use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::{Mutex, RwLock};

use nalgebra_glm as glm;

#[derive(Debug)]
pub struct Camera
{
    critical_section: RwLock<CameraCriticalSection>
}

impl Camera
{
    pub fn new(translation: glm::Vec3, pitch: f32, yaw: f32) -> Camera
    {
        let mut critical_section = CameraCriticalSection {
            translation,
            pitch,
            yaw,
            transform: Transform::default()
        };

        Camera::enforce_invariants(&mut critical_section);

        Camera {
            critical_section: RwLock::new(critical_section)
        }
    }

    pub fn get_perspective(&self, renderer: &super::Renderer, transform: &Transform) -> glm::Mat4
    {
        //! sync with shaders!
        let projection = glm::perspective::<f32>(
            renderer.get_fov().y,
            renderer.get_aspect_ratio(),
            0.1,
            100000.0
        ) * OPENGL_TO_WGPU_MATRIX;

        #[rustfmt::skip]
        const OPENGL_TO_WGPU_MATRIX: glm::Mat4 = glm::Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );

        projection * self.get_view_matrix() * transform.as_model_matrix()
    }

    pub fn get_view_matrix(&self) -> glm::Mat4
    {
        let guard = self.critical_section.read().unwrap();
        let CameraCriticalSection {
            translation,
            pitch,
            yaw,
            transform
        } = &*guard;

        (transform.as_translation_matrix() * transform.as_rotation_matrix())
            .try_inverse()
            .unwrap()
    }

    pub fn get_forward_vector(&self) -> glm::Vec3
    {
        self.critical_section
            .read()
            .unwrap()
            .transform
            .get_forward_vector()
    }

    pub fn get_right_vector(&self) -> glm::Vec3
    {
        self.critical_section
            .read()
            .unwrap()
            .transform
            .get_right_vector()
    }

    pub fn get_up_vector(&self) -> glm::Vec3
    {
        self.critical_section
            .read()
            .unwrap()
            .transform
            .get_up_vector()
    }

    pub fn get_position(&self) -> glm::Vec3
    {
        self.critical_section.read().unwrap().transform.translation
    }

    pub fn add_position(&self, position: glm::Vec3)
    {
        self.critical_section.write().unwrap().transform.translation += position;
    }

    pub fn add_pitch(&self, pitch_to_add: f32)
    {
        let mut guard = self.critical_section.write().unwrap();

        {
            let CameraCriticalSection {
                translation,
                pitch,
                yaw,
                transform
            } = &mut *guard;

            *pitch += pitch_to_add;
        }

        Camera::enforce_invariants(&mut *guard);
    }

    pub fn add_yaw(&self, yaw_to_add: f32)
    {
        let mut guard = self.critical_section.write().unwrap();

        {
            let CameraCriticalSection {
                translation,
                pitch,
                yaw,
                transform
            } = &mut *guard;

            *yaw += yaw_to_add;
        }

        Camera::enforce_invariants(&mut *guard);
    }

    fn enforce_invariants(
        CameraCriticalSection {
            translation,
            pitch,
            yaw,
            transform
        }: &mut CameraCriticalSection
    )
    {
        let mut q = glm::Quat::new(1.0, 0.0, 0.0, 0.0);

        *pitch = pitch.clamp(-FRAC_PI_2, FRAC_PI_2);
        *yaw = glm::modf(*yaw, TAU);

        q *= glm::quat_angle_axis(*pitch, &Transform::RIGHT_VECTOR);

        q = glm::quat_angle_axis(*yaw, &Transform::UP_VECTOR) * q;

        transform.rotation = q;
    }
}

impl Clone for Camera
{
    fn clone(&self) -> Self
    {
        let guard = self.critical_section.read().unwrap();
        let critical_section = &*guard;

        Self {
            critical_section: RwLock::new(critical_section.clone())
        }
    }
}

#[derive(Debug, Clone)]
struct CameraCriticalSection
{
    translation: glm::Vec3,
    pitch:       f32,
    yaw:         f32,
    transform:   Transform
}

#[derive(Debug, Clone, Default)]
pub struct Transform
{
    translation: glm::Vec3,
    rotation:    glm::Quat,
    scale:       glm::Vec3
}

impl Transform
{
    const FORWARD_VECTOR: glm::Vec3 = glm::Vec3::new(0.0, 0.0, 1.0);
    const RIGHT_VECTOR: glm::Vec3 = glm::Vec3::new(1.0, 0.0, 0.0);
    const UP_VECTOR: glm::Vec3 = glm::Vec3::new(0.0, 1.0, 0.0);

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
        todo!()
        // UnitQuaternion::new()
        // nalgebraself.rotation.to
    }

    pub fn as_scale_matrix(&self) -> glm::Mat4
    {
        todo!()
    }

    pub fn get_forward_vector(&self) -> glm::Vec3
    {
        todo!()
    }

    pub fn get_right_vector(&self) -> glm::Vec3
    {
        todo!()
    }

    pub fn get_up_vector(&self) -> glm::Vec3
    {
        todo!()
    }
}

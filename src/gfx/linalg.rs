use std::sync::Mutex;

use nalgebra_glm as glm;

#[derive(Debug)]
pub struct Camera
{
    critical_section: Mutex<CameraCriticalSection>
}

impl Camera
{
    pub fn new(translation: glm::Vec3, pitch: f32, yaw: f32) -> Camera
    {
        Camera {
            critical_section: Mutex::new(CameraCriticalSection {
                translation,
                pitch,
                yaw
            })
        }
    }

    pub fn get_perspective(&self, renderer: &super::Renderer, transform: &Transform) {}

    pub fn get_view_matrix(&self) -> glm::Vec3
    {
        todo!()
    }

    pub fn get_forward_vector() -> glm::Vec3
    {
        todo!()
    }

    pub fn get_right_vector() -> glm::Vec3
    {
        todo!()
    }

    pub fn get_up_vector() -> glm::Vec3
    {
        todo!()
    }

    pub fn get_position() -> glm::Vec3
    {
        todo!()
    }
}

impl Clone for Camera
{
    fn clone(&self) -> Self
    {
        let guard = self.critical_section.lock().unwrap();
        let critical_section = &*guard;

        Self {
            critical_section: Mutex::new(critical_section.clone())
        }
    }
}

#[derive(Debug, Clone)]
struct CameraCriticalSection
{
    translation: glm::Vec3,
    pitch:       f32,
    yaw:         f32
}

#[derive(Debug, Clone)]
pub struct Transform {}

impl Transform {}

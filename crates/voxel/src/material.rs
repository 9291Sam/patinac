use std::sync::Arc;

use bytemuck::{NoUninit, Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::{glm, wgpu};

use crate::Voxel;
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Zeroable)]
pub struct VoxelMaterial
{
    diffuse_color:             glm::Vec4,
    subsurface_color:          glm::Vec4,
    diffuse_subsurface_weight: f32,

    specular_color: glm::Vec4,
    specular:       f32,
    roughness:      f32,
    metallic:       f32,

    emissive_color_and_power: glm::Vec4,
    coat_color_and_power:     glm::Vec4,

    special: u32
}

unsafe impl NoUninit for VoxelMaterial {}

impl Default for VoxelMaterial
{
    fn default() -> Self
    {
        Self {
            diffuse_color:             glm::Vec4::new(1.0, 0.0, 1.0, 1.0),
            subsurface_color:          glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            diffuse_subsurface_weight: 0.0,
            specular_color:            glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            specular:                  0.0,
            roughness:                 1.0,
            metallic:                  0.0,
            emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
            coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
            special:                   0
        }
    }
}

pub struct MaterialManager
{
    material_buffer: Arc<wgpu::Buffer>
}

impl MaterialManager
{
    pub fn new(renderer: &gfx::Renderer) -> MaterialManager
    {
        MaterialManager {
            material_buffer: Arc::new(renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("Raster Vertex Buffer {}"),
                contents: bytemuck::cast_slice(
                    &Self::generate_material_array(get_material_from_voxel)[..]
                ),
                usage:    wgpu::BufferUsages::VERTEX
            }))
        }
    }

    pub fn get_material_buffer(&self) -> Arc<wgpu::Buffer>
    {
        self.material_buffer.clone()
    }

    fn generate_material_array(mat_func: impl Fn(Voxel) -> VoxelMaterial) -> Vec<VoxelMaterial>
    {
        (u16::MIN..=u16::MAX)
            .map(|i| {
                match Voxel::try_from(i)
                {
                    Ok(v) => mat_func(v),
                    Err(()) => VoxelMaterial::default()
                }
            })
            .collect()
    }
}

fn get_material_from_voxel(v: Voxel) -> VoxelMaterial
{
    match v
    {
        Voxel::Air => VoxelMaterial::default(),
        Voxel::Rock0 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            }
        }
        Voxel::Rock1 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.6, 0.6, 0.6, 1.0),
                ..Default::default()
            }
        }
        Voxel::Rock2 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.7, 0.7, 0.7, 1.0),
                ..Default::default()
            }
        }
        Voxel::Rock3 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.8, 0.8, 0.8, 1.0),
                ..Default::default()
            }
        }
        Voxel::Rock4 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.9, 0.9, 0.9, 1.0),
                ..Default::default()
            }
        }
        Voxel::Rock5 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass0 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.0, 0.5, 0.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass1 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.0, 0.6, 0.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass2 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.0, 0.7, 0.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass3 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.0, 0.8, 0.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass4 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.0, 0.9, 0.0, 1.0),
                ..Default::default()
            }
        }
        Voxel::Grass5 =>
        {
            VoxelMaterial {
                diffuse_color: glm::Vec4::new(0.2, 0.5, 0.2, 1.0),
                ..Default::default()
            }
        }
    }
}

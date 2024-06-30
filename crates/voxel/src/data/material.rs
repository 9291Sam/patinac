use std::sync::Arc;

use bytemuck::{NoUninit, Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::{glm, wgpu};
use num_enum::{IntoPrimitive, TryFromPrimitive};

#[repr(u16)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, IntoPrimitive, TryFromPrimitive,
)]
pub enum Voxel
{
    Air = 0,
    Dirt0,
    Dirt1,
    Dirt2,
    Dirt3,
    Dirt4,
    Dirt5,
    Dirt6,
    Dirt7,
    Stone0,
    Stone1,
    Stone14,
    Wood0,
    Wood1,
    Wood2,
    SilverMeta0,
    SilverMeta1,
    GoldMetal0,
    GoldMetal1
}

unsafe impl Zeroable for Voxel {}
unsafe impl Pod for Voxel {}

impl Voxel
{
    pub fn as_bytes(self) -> [u8; 2]
    {
        (self as u16).to_ne_bytes()
    }

    pub fn is_air(self) -> bool
    {
        if let Voxel::Air = self
        {
            true
        }
        else
        {
            false
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Zeroable)]
pub struct VoxelMaterial
{
    diffuse_color:             glm::Vec4,
    subsurface_color:          glm::Vec4,
    specular_color:            glm::Vec4,
    diffuse_subsurface_weight: f32,
    specular:                  f32,
    roughness:                 f32,
    metallic:                  f32,
    emissive_color_and_power:  glm::Vec4,
    coat_color_and_power:      glm::Vec4
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
            coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
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
                usage:    wgpu::BufferUsages::STORAGE
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
                    Err(_) => mat_func(Voxel::Air)
                }
            })
            .collect()
    }
}

#[inline(never)]
#[cold]
fn get_material_from_voxel(v: Voxel) -> VoxelMaterial
{
    match v
    {
        Voxel::Air =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.075, 0.5, 0.6, 0.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(1.0, 1.0, 1.0, 0.0),
                specular:                  0.0,
                roughness:                 1.0,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt0 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.1, 0.07, 0.03, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.1, 0.1, 0.1, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt1 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.15, 0.1, 0.05, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.15, 0.15, 0.15, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt2 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.2, 0.13, 0.07, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.2, 0.2, 0.2, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt3 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.25, 0.17, 0.09, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.25, 0.25, 0.25, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt4 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.3, 0.2, 0.1, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.3, 0.3, 0.3, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt5 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.35, 0.23, 0.12, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.35, 0.35, 0.35, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt6 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.4, 0.27, 0.14, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.4, 0.4, 0.4, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Dirt7 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.45, 0.3, 0.15, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.45, 0.45, 0.45, 0.0),
                specular:                  38.3,
                roughness:                 0.9,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Stone0 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.3, 0.3, 0.3, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
                specular:                  38.3,
                roughness:                 0.8,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Stone1 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.32, 0.32, 0.32, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
                specular:                  38.3,
                roughness:                 0.78,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Stone14 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.35, 0.35, 0.35, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
                specular:                  38.3,
                roughness:                 0.85,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Wood0 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.4, 0.25, 0.1, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.3, 0.2, 0.1, 1.0),
                specular:                  0.15,
                roughness:                 0.6,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Wood1 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.42, 0.27, 0.12, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.32, 0.22, 0.12, 1.0),
                specular:                  0.18,
                roughness:                 0.62,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::Wood2 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.45, 0.3, 0.15, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.35, 0.25, 0.15, 1.0),
                specular:                  0.2,
                roughness:                 0.7,
                metallic:                  0.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::SilverMeta0 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.75, 0.75, 0.75, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.95, 0.95, 0.95, 1.0),
                specular:                  0.9,
                roughness:                 0.1,
                metallic:                  1.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::SilverMeta1 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.8, 0.8, 0.8, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(0.98, 0.98, 0.98, 1.0),
                specular:                  0.85,
                roughness:                 0.15,
                metallic:                  1.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::GoldMetal0 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.8, 0.65, 0.2, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(1.0, 0.85, 0.3, 1.0),
                specular:                  0.9,
                roughness:                 0.1,
                metallic:                  1.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
        Voxel::GoldMetal1 =>
        {
            VoxelMaterial {
                diffuse_color:             glm::Vec4::new(0.82, 0.67, 0.22, 1.0),
                subsurface_color:          glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                diffuse_subsurface_weight: 0.0,
                specular_color:            glm::Vec4::new(1.0, 0.87, 0.32, 1.0),
                specular:                  0.85,
                roughness:                 0.12,
                metallic:                  1.0,
                emissive_color_and_power:  glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                coat_color_and_power:      glm::Vec4::new(0.0, 0.0, 0.0, 0.0)
            }
        }
    }
}

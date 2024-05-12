use gfx::glm;

use crate::Voxel;

#[repr(C)]
pub struct VoxelMaterial
{
    diffuse_color:             glm::Vec4,
    subsurface_color:          glm::Vec4,
    diffuse_subsurface_weight: f32,

    specular_color: glm::Vec4,
    specular:       f32,
    roughness:      f32,
    metallic:       f32,
    anisotropic:    f32,

    emissive_color_and_power: glm::Vec4,
    coat_color_and_power:     glm::Vec4,

    special: u32
}

pub struct MaterialManager
{
    material_buffer: wgpu::Buffer
}

impl MaterialManager
{
    pub fn new() {}
}

fn get_material_from_voxel(v: Voxel) -> VoxelMaterial
{
    match v
    {
        Voxel::Air => todo!(),
        Voxel::Rock0 => todo!(),
        Voxel::Rock1 => todo!(),
        Voxel::Rock2 => todo!(),
        Voxel::Rock3 => todo!(),
        Voxel::Rock4 => todo!(),
        Voxel::Rock5 => todo!(),
        Voxel::Grass0 => todo!(),
        Voxel::Grass1 => todo!(),
        Voxel::Grass2 => todo!(),
        Voxel::Grass3 => todo!(),
        Voxel::Grass4 => todo!(),
        Voxel::Grass5 => todo!()
    }
}

use std::default;

use phf_macros::phf_map;
use phf_shared::PhfBorrow;
use strum::{EnumIter, IntoEnumIterator};

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, EnumIter)]
pub enum Voxel
{
    #[default]
    Air = 0
}

impl phf::PhfHash for Voxel
{
    fn phf_hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        state.write_u16(*self as u16)
    }
}

impl Voxel
{
    #[inline(never)]
    pub fn get_material(&self) -> VoxelMaterial
    {
        match *self
        {
            Voxel::Air =>
            {
                VoxelMaterial {
                    alpha_or_emissive: 0,
                    srgb_r:            todo!(),
                    srgb_g:            todo!(),
                    srgb_b:            todo!(),
                    special:           todo!(),
                    specular:          todo!(),
                    roughness:         todo!(),
                    metallic:          todo!()
                }
            }
        }
    }
}

pub fn build_lookup_table() -> Vec<VoxelMaterial>
{
    Voxel::iter().map(|v| v.get_material()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VoxelMaterial
// TODO: literally just rip off blender's bsdf
{
    visibility:          Visibility,
    visibility_strength: u16,

    srgb_r: u8,
    srgb_g: u8,
    srgb_b: u8,

    special:   u8,
    specular:  u8,
    roughness: u8,
    metallic:  u8
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Visibility
{
    #[default]
    Invisible,
    Translucent,
    Opaque,
    Emissive
}

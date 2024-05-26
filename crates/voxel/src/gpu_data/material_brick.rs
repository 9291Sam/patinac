use bytemuck::{AnyBitPattern, NoUninit};

use crate::material::Voxel;
use crate::{BrickLocalPosition, BRICK_EDGE_LEN_VOXELS};

#[repr(C)]
#[derive(Clone, Copy, Debug, AnyBitPattern, NoUninit)]
pub(crate) struct MaterialBrick
{
    data: [[[Voxel; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]
}

impl MaterialBrick
{
    pub fn new_filled(voxel: Voxel) -> Self
    {
        MaterialBrick {
            data: [[[voxel; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]
        }
    }

    pub fn get_voxel(&self, BrickLocalPosition(local_pos): BrickLocalPosition) -> Voxel
    {
        let [x, y, z] = [
            local_pos.x as usize,
            local_pos.y as usize,
            local_pos.z as usize
        ];

        debug_assert!(x < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {x}");
        debug_assert!(y < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {y}");
        debug_assert!(z < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {z}");

        unsafe {
            *self
                .data
                .get_unchecked(x as usize)
                .get_unchecked(y as usize)
                .get_unchecked(z as usize)
        }
    }

    // Returns the old voxel
    pub fn set_voxel(
        &mut self,
        BrickLocalPosition(local_pos): BrickLocalPosition,
        new_voxel: Voxel
    ) -> Voxel
    {
        let [x, y, z] = [
            local_pos.x as usize,
            local_pos.y as usize,
            local_pos.z as usize
        ];

        debug_assert!(x < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {x}");
        debug_assert!(y < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {y}");
        debug_assert!(z < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {z}");

        let voxel_to_modify: &mut Voxel = unsafe {
            self.data
                .get_unchecked_mut(x as usize)
                .get_unchecked_mut(y as usize)
                .get_unchecked_mut(z as usize)
        };

        let previous_voxel = voxel_to_modify.clone();

        *voxel_to_modify = new_voxel;

        previous_voxel
    }
}
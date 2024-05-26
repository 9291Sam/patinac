use std::fmt::Debug;
use std::sync::Arc;

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::wgpu::{self};
use gfx::{glm, CpuTrackedDenseSet};

pub(crate) struct FaceManager
{
    face_id_allocator: util::FreelistAllocator,
    face_id_buffer:    gfx::CpuTrackedDenseSet<u32>,
    face_data_buffer:  gfx::CpuTrackedBuffer<GpuFaceData>
}

impl Debug for FaceManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Voxel Manager")
    }
}

impl FaceManager
{
    pub fn new(game: Arc<game::Game>) -> Self
    {
        const INITIAL_SIZE: usize = 1024;
        let renderer = game.get_renderer().clone();

        let id_buffer = CpuTrackedDenseSet::new(
            renderer.clone(),
            INITIAL_SIZE,
            String::from("Face Id Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let data_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            INITIAL_SIZE,
            String::from("Face Data Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        FaceManager {
            face_id_allocator: util::FreelistAllocator::new(INITIAL_SIZE),
            face_id_buffer:    id_buffer,
            face_data_buffer:  data_buffer
        }
    }

    // no chunks for now, just one global chunk

    #[must_use]
    pub fn insert_face(&mut self, face: GpuFaceData) -> FaceId
    {
        let new_face_id = if let Ok(id) = self.face_id_allocator.allocate()
        {
            id
        }
        else
        {
            let realloc_size = self.face_id_allocator.get_total_blocks() * 3 / 2;
            self.face_id_allocator.extend_size(realloc_size);
            self.face_data_buffer.realloc(realloc_size);

            self.face_id_allocator.allocate().unwrap()
        };

        self.face_id_buffer.insert(new_face_id as u32);

        self.face_data_buffer.write(new_face_id, face);

        FaceId(new_face_id as u32)
    }

    pub fn remove_face(&mut self, face_id: FaceId)
    {
        unsafe { self.face_id_allocator.free(face_id.0 as usize) };

        self.face_id_buffer.remove(face_id.0).unwrap();
    }

    pub fn access_buffers<K>(&self, func: impl FnOnce(FaceManagerBuffers) -> K) -> K
    {
        self.face_id_buffer.get_buffer(|face_id_buffer| {
            self.face_data_buffer.get_buffer(|face_data_buffer| {
                func(FaceManagerBuffers {
                    face_id_buffer,
                    face_data_buffer
                })
            })
        })
    }

    pub fn replicate_to_gpu(&mut self) -> bool
    {
        let mut needs_resize = false;

        needs_resize |= self.face_id_buffer.replicate_to_gpu();
        needs_resize |= self.face_data_buffer.replicate_to_gpu();

        needs_resize
    }

    pub fn get_number_of_faces(&self) -> u32
    {
        (self.face_id_buffer.get_number_of_elements()) as u32
    }
}

pub struct FaceManagerBuffers<'b>
{
    pub face_id_buffer:   &'b wgpu::Buffer,
    pub face_data_buffer: &'b wgpu::Buffer
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoxelFaceDirection
{
    Top    = 0,
    Bottom = 1,
    Left   = 2,
    Right  = 3,
    Front  = 4,
    Back   = 5
}

impl TryFrom<u8> for VoxelFaceDirection
{
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error>
    {
        use VoxelFaceDirection::*;

        match value
        {
            0 => Ok(Top),
            1 => Ok(Bottom),
            2 => Ok(Left),
            3 => Ok(Right),
            4 => Ok(Front),
            5 => Ok(Back),
            _ => Err(value)
        }
    }
}

impl VoxelFaceDirection
{
    pub fn to_bits(self) -> u8
    {
        self as u8
    }

    pub fn iterate() -> impl Iterator<Item = VoxelFaceDirection>
    {
        [
            VoxelFaceDirection::Top,
            VoxelFaceDirection::Bottom,
            VoxelFaceDirection::Left,
            VoxelFaceDirection::Right,
            VoxelFaceDirection::Front,
            VoxelFaceDirection::Back
        ]
        .into_iter()
    }

    pub fn get_axis(self) -> glm::I16Vec3
    {
        match self
        {
            VoxelFaceDirection::Top => glm::I16Vec3::new(0, 1, 0),
            VoxelFaceDirection::Bottom => glm::I16Vec3::new(0, -1, 0),
            VoxelFaceDirection::Left => glm::I16Vec3::new(-1, 0, 0),
            VoxelFaceDirection::Right => glm::I16Vec3::new(1, 0, 0),
            VoxelFaceDirection::Front => glm::I16Vec3::new(0, 0, -1),
            VoxelFaceDirection::Back => glm::I16Vec3::new(0, 0, 1)
        }
    }

    pub fn opposite(self) -> VoxelFaceDirection
    {
        match self
        {
            VoxelFaceDirection::Top => VoxelFaceDirection::Bottom,
            VoxelFaceDirection::Bottom => VoxelFaceDirection::Top,
            VoxelFaceDirection::Left => VoxelFaceDirection::Right,
            VoxelFaceDirection::Right => VoxelFaceDirection::Left,
            VoxelFaceDirection::Front => VoxelFaceDirection::Back,
            VoxelFaceDirection::Back => VoxelFaceDirection::Front
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug)]
pub(crate) struct GpuFaceData
// is allocated at a specific index
{
    material_and_chunk_id: u32,
    // 9 bits x
    // 9 bits y
    // 9 bits z
    // 3 bits normal
    // 1 bit visibility
    // 1 bit unused
    location_within_chunk: u32
}

impl GpuFaceData
{
    pub fn new(material: u16, chunk_id: u16, pos: glm::U16Vec3, dir: VoxelFaceDirection) -> Self
    {
        assert!(pos.x < 2u16.pow(9), "{:?}", pos.x);
        assert!(pos.y < 2u16.pow(9), "{:?}", pos.y);
        assert!(pos.z < 2u16.pow(9), "{:?}", pos.z);

        GpuFaceData {
            material_and_chunk_id: (material as u32) | ((chunk_id as u32) << 16),
            location_within_chunk: (pos.x as u32)
                | ((pos.y as u32) << 9)
                | ((pos.z as u32) << 18)
                | ((dir.to_bits() as u32) << 27)
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug, Hash)]
pub struct FaceId(u32);
// face manager: write_face(pos dir vox)
// voxel manager: write_voxel(world pos, vox)

use std::sync::{Arc, Mutex};

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::{glm, wgpu};
use num_enum::{IntoPrimitive, TryFromPrimitive};

pub(crate) struct FaceManager
{
    face_id_allocator: Mutex<util::FreelistAllocator>,

    face_id_buffer:   gfx::CpuTrackedDenseSet<u32>,
    face_data_buffer: gfx::CpuTrackedBuffer<GpuFaceData>
}

impl FaceManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        FaceManager {
            face_id_allocator: todo!(),
            face_id_buffer:    todo!(),
            face_data_buffer:  todo!()
        }
    }

    #[must_use]
    pub fn insert_face(&self, face_data: GpuFaceData) -> FaceId
    {
        let new_face_id = self.face_id_allocator.lock().unwrap().allocate().unwrap();

        self.face_id_buffer.insert(new_face_id as u32);

        self.face_data_buffer.write(new_face_id, face_data);

        FaceId(new_face_id as u32)
    }

    pub fn remove_face(&self, face_id: FaceId)
    {
        let mut face_id_allocator = self.face_id_allocator.lock().unwrap();

        self.face_id_buffer.remove(face_id.0).unwrap();

        unsafe { face_id_allocator.free(face_id.0 as usize) }
    }

    pub fn replicate_to_gpu(&self) -> bool
    {
        let mut should_resize = false;

        should_resize |= std::hint::black_box(self.face_id_buffer.replicate_to_gpu());
        should_resize |= std::hint::black_box(self.face_data_buffer.replicate_to_gpu());

        should_resize
    }

    pub fn get_number_of_faces_to_draw(&self) -> usize
    {
        self.face_id_buffer.get_number_of_elements()
    }

    pub fn access_buffers<K>(&self, access_func: impl FnOnce(FaceManagerBuffers<'_>) -> K) -> K
    {
        self.face_id_buffer.get_buffer(|face_id_buffer| {
            self.face_data_buffer.get_buffer(|face_data_buffer| {
                access_func(FaceManagerBuffers {
                    face_id_buffer,
                    face_data_buffer
                })
            })
        })
    }
}

pub struct FaceManagerBuffers<'b>
{
    pub face_id_buffer:   &'b wgpu::Buffer,
    pub face_data_buffer: &'b wgpu::Buffer
}

// insert face -> Faceid
// remove face(faceId) -> void
pub(crate) struct FaceId(u32);
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, TryFromPrimitive, IntoPrimitive)]
pub enum VoxelFaceDirection
{
    Top    = 0,
    Bottom = 1,
    Left   = 2,
    Right  = 3,
    Front  = 4,
    Back   = 5
}

impl VoxelFaceDirection
{
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
                | ((dir as u32) << 27)
        }
    }
}

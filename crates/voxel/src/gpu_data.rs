use core::slice;
use std::assert_matches::assert_matches;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Contiguous, Pod, Zeroable};
use gfx::glm;
use gfx::wgpu::{self};

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Voxel
{
    Air    = 0,
    Rock0  = 1,
    Rock1  = 2,
    Rock2  = 3,
    Rock3  = 4,
    Rock4  = 5,
    Rock5  = 6,
    Grass0 = 7,
    Grass1 = 8,
    Grass2 = 9,
    Grass3 = 10,
    Grass4 = 11,
    Grass5 = 12
}

unsafe impl Zeroable for Voxel {}
unsafe impl Pod for Voxel {}

impl TryFrom<u16> for Voxel
{
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error>
    {
        match value
        {
            0 => Ok(Voxel::Air),
            1 => Ok(Voxel::Rock0),
            2 => Ok(Voxel::Rock1),
            3 => Ok(Voxel::Rock2),
            4 => Ok(Voxel::Rock3),
            5 => Ok(Voxel::Rock4),
            6 => Ok(Voxel::Rock5),
            7 => Ok(Voxel::Grass0),
            8 => Ok(Voxel::Grass1),
            9 => Ok(Voxel::Grass2),
            10 => Ok(Voxel::Grass3),
            11 => Ok(Voxel::Grass4),
            12 => Ok(Voxel::Grass5),
            _ => Err(())
        }
    }
}

impl Voxel
{
    pub fn as_bytes(self) -> [u8; 2]
    {
        (self as u16).to_ne_bytes()
    }
}

// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
// #[repr(C)]
// pub struct VoxelBrick
// {
//     data: [[[Voxel; VOXEL_BRICK_EDGE_LENGTH]; VOXEL_BRICK_EDGE_LENGTH];
// VOXEL_BRICK_EDGE_LENGTH] }

// const VOXEL_STORAGE_BITS: usize =
//     VOXEL_BRICK_EDGE_LENGTH * VOXEL_BRICK_EDGE_LENGTH *
// VOXEL_BRICK_EDGE_LENGTH; const VOXEL_STORAGE_BYTES: usize =
// VOXEL_STORAGE_BITS.div_ceil(8);

// impl VoxelBrick
// {
//     pub fn new_empty() -> VoxelBrick
//     {
//         VoxelBrick {
//             data: [[[Voxel::Air; VOXEL_BRICK_EDGE_LENGTH];
// VOXEL_BRICK_EDGE_LENGTH];                 VOXEL_BRICK_EDGE_LENGTH]
//         }
//     }

//     pub fn write(&mut self, pos: glm::U16Vec3, voxel: Voxel)
//     {
//         self.data[pos.x as usize][pos.y as usize][pos.z as usize] = voxel
//     }

//     pub fn fill(&mut self, voxel: Voxel)
//     {
//         for slice in self.data.iter_mut()
//         {
//             for layer in slice.iter_mut()
//             {
//                 for v in layer.iter_mut()
//                 {
//                     *v = voxel;
//                 }
//             }
//         }
//     }
// }

/// Value Table
/// TODO: flip the range on its head so that 0 is a valid brick pointer and that
/// 0xFF... is the nullptr (and also air voxel!)
///  0 - null brick pointer
/// [1, 2^32 - 2^16) - valid brick pointer
/// [2^32 - 2^16 - 2^32] - voxel

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
struct VoxelBrickPointer
{
    data: u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum VoxelBrickPointerType
{
    Null,
    ValidBrickPointer(u32),
    Voxel(Voxel)
}

impl VoxelBrickPointer
{
    /// The first voxel
    const POINTER_TO_VOXEL_BOUND: u32 = u32::MAX - 2u32.pow(16);

    pub fn new_null() -> VoxelBrickPointer
    {
        VoxelBrickPointer {
            data: 0
        }
    }

    pub fn new_ptr(ptr: u32) -> VoxelBrickPointer
    {
        let maybe_new_ptr = VoxelBrickPointer {
            data: ptr
        };

        assert_matches!(
            maybe_new_ptr.classify(),
            VoxelBrickPointerType::ValidBrickPointer(_ptr),
            "Tried to create a pointer with an invalid value. Valid range is [1, 2^32 - 2^16)"
        );

        maybe_new_ptr
    }

    pub fn new_voxel(voxel: Voxel) -> VoxelBrickPointer
    {
        let maybe_new_ptr = VoxelBrickPointer {
            data: voxel as u16 as u32 + Self::POINTER_TO_VOXEL_BOUND
        };

        assert!(voxel != Voxel::Air);
        assert_matches!(
            maybe_new_ptr.classify(),
            VoxelBrickPointerType::Voxel(_voxel)
        );

        maybe_new_ptr
    }

    pub fn get_ptr(&self) -> u32
    {
        if let VoxelBrickPointerType::ValidBrickPointer(ptr) = self.classify()
        {
            ptr
        }
        else
        {
            panic!()
        }
    }

    pub fn classify(&self) -> VoxelBrickPointerType
    {
        match self.data
        {
            0 => VoxelBrickPointerType::Null,
            1..Self::POINTER_TO_VOXEL_BOUND => VoxelBrickPointerType::ValidBrickPointer(self.data),
            Self::POINTER_TO_VOXEL_BOUND..=u32::MAX =>
            {
                VoxelBrickPointerType::Voxel(
                    Voxel::try_from((self.data - Self::POINTER_TO_VOXEL_BOUND) as u16).unwrap()
                )
            }
        }
    }
}

// type BrickPointer = NonZeroU32;

type BrickMap =
    [[[VoxelBrickPointer; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE];

pub const VOXEL_BRICK_EDGE_LENGTH: usize = 8;
pub const BRICK_MAP_EDGE_SIZE: usize = 256;
pub const CHUNK_VOXEL_SIZE: usize = VOXEL_BRICK_EDGE_LENGTH * BRICK_MAP_EDGE_SIZE;

// TODO: this shit very much isnt MT safe and will have races all over the
// place!
#[derive(Debug)]
pub struct VoxelChunkDataManager
{
    renderer: Arc<gfx::Renderer>,

    window_voxel_bind_group: util::Window<Arc<wgpu::BindGroup>>,
    buffer_critical_section: Mutex<VoxelChunkDataManagerBufferCriticalSection>,

    bind_group_layout: Arc<wgpu::BindGroupLayout>
}

pub struct VoxelChunkDataManagerBufferCriticalSection
{
    cpu_brick_map:   Box<BrickMap>,
    gpu_brick_map:   wgpu::Buffer,
    delta_brick_map: BTreeSet<util::SendSyncMutPtr<VoxelBrickPointer>>,

    cpu_brick_buffer:   Vec<BitBrick>,
    gpu_brick_buffer:   wgpu::Buffer,
    delta_brick_buffer: BTreeSet<util::SendSyncMutPtr<BitBrick>>,
    needs_resize_flush: bool,

    voxel_bind_group:                Arc<wgpu::BindGroup>,
    window_updater_voxel_bind_group: util::WindowUpdater<Arc<wgpu::BindGroup>>,

    brick_allocator: util::FreelistAllocator
}

impl Debug for VoxelChunkDataManagerBufferCriticalSection
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("VoxelChunkDataManagerBufferCriticalSection")
            .field("needs_resize_flush", &self.needs_resize_flush)
            .field("brick_allocator", &self.brick_allocator)
            .finish()
    }
}

pub type ChunkPosition = glm::U16Vec3;

impl VoxelChunkDataManager
{
    pub fn peek_allocated_bricks(&self) -> (NonZeroUsize, NonZeroUsize)
    {
        self.buffer_critical_section
            .lock()
            .unwrap()
            .brick_allocator
            .peek()
    }

    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        static BINDINGS: &[wgpu::BindGroupLayoutEntry] = &[
            wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage {
                        read_only: true
                    },
                    has_dynamic_offset: false,
                    min_binding_size:   None // MIN_0_BINDING_SIZE
                },
                count:      None
            },
            wgpu::BindGroupLayoutEntry {
                binding:    1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage {
                        read_only: true
                    },
                    has_dynamic_offset: false,
                    min_binding_size:   None //  MIN_0_BINDING_SIZE
                },
                count:      None
            }
        ];

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Brick Map Chunk BinGroup Layouts"),
                    entries: BINDINGS
                });

        assert!(VOXEL_BRICK_EDGE_LENGTH == 8);

        let number_of_starting_bricks = BRICK_MAP_EDGE_SIZE * BRICK_MAP_EDGE_SIZE * 2;

        let r = renderer.clone();

        let gpu_brick_map = r.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Voxel Chunk Data Manager Brick Map Buffer"),
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size:               std::mem::size_of::<BrickMap>() as u64,
            mapped_at_creation: false
        });

        let gpu_brick_buffer = r.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Voxel Chunk Data Manager Brick Buffer"),
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size:               std::mem::size_of::<BitBrick>() as u64
                * number_of_starting_bricks as u64,
            mapped_at_creation: false
        });

        let voxel_bind_group = Arc::new(r.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Voxel Bind Group"),
            layout:  &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: gpu_brick_map.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: gpu_brick_buffer.as_entire_binding()
                }
            ]
        }));

        let (window_voxel_bind_group, window_updater_voxel_bind_group) =
            util::Window::new(voxel_bind_group.clone());

        Self {
            renderer,
            window_voxel_bind_group,
            buffer_critical_section: Mutex::new(VoxelChunkDataManagerBufferCriticalSection {
                cpu_brick_map: vec![
                    [[VoxelBrickPointer::new_null(); BRICK_MAP_EDGE_SIZE];
                        BRICK_MAP_EDGE_SIZE];
                    BRICK_MAP_EDGE_SIZE
                ]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
                gpu_brick_map,
                delta_brick_map: BTreeSet::new(),
                cpu_brick_buffer: vec![BitBrick::new(false); number_of_starting_bricks],
                gpu_brick_buffer,
                delta_brick_buffer: BTreeSet::new(),
                needs_resize_flush: false,
                window_updater_voxel_bind_group,
                voxel_bind_group,
                brick_allocator: util::FreelistAllocator::new(
                    (number_of_starting_bricks - 1).try_into().unwrap()
                )
            }),
            bind_group_layout
        }
    }

    pub fn write_brick(&self, v: Voxel, unsigned_pos: ChunkPosition)
    {
        let unsigned_bound = BRICK_MAP_EDGE_SIZE as u16;

        assert!(
            unsigned_pos.x < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.x
        );

        assert!(
            unsigned_pos.y < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.y
        );

        assert!(
            unsigned_pos.z < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.z
        );

        let VoxelChunkDataManagerBufferCriticalSection {
            ref mut cpu_brick_map,
            ref mut delta_brick_map,
            ref mut brick_allocator,
            ..
        } = &mut *self.buffer_critical_section.lock().unwrap();

        let current_brick_ptr = &mut cpu_brick_map[unsigned_pos.x as usize]
            [unsigned_pos.y as usize][unsigned_pos.z as usize];

        if let VoxelBrickPointerType::ValidBrickPointer(old_ptr) = current_brick_ptr.classify()
        {
            brick_allocator.free((old_ptr.into_integer() as usize).try_into().unwrap())
        }

        if v == Voxel::Air
        {
            *current_brick_ptr = VoxelBrickPointer::new_null();
        }
        else
        {
            *current_brick_ptr = VoxelBrickPointer::new_voxel(v);
        }

        delta_brick_map.insert((current_brick_ptr as *mut VoxelBrickPointer).into());
    }

    pub fn write_voxel(&self, v: Voxel, unsigned_pos: ChunkPosition)
    {
        let unsigned_bound = (BRICK_MAP_EDGE_SIZE * VOXEL_BRICK_EDGE_LENGTH) as u16;

        assert!(
            unsigned_pos.x < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.x
        );

        assert!(
            unsigned_pos.y < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.y
        );

        assert!(
            unsigned_pos.z < unsigned_bound,
            "Bound {} is out of bounds",
            unsigned_pos.z
        );

        let voxel_pos = glm::U16Vec3::new(
            unsigned_pos.x % VOXEL_BRICK_EDGE_LENGTH as u16,
            unsigned_pos.y % VOXEL_BRICK_EDGE_LENGTH as u16,
            unsigned_pos.z % VOXEL_BRICK_EDGE_LENGTH as u16
        );

        let brick_pos = glm::U16Vec3::new(
            unsigned_pos.x / VOXEL_BRICK_EDGE_LENGTH as u16,
            unsigned_pos.y / VOXEL_BRICK_EDGE_LENGTH as u16,
            unsigned_pos.z / VOXEL_BRICK_EDGE_LENGTH as u16
        );

        let VoxelChunkDataManagerBufferCriticalSection {
            ref mut cpu_brick_map,
            ref mut gpu_brick_map,
            ref mut delta_brick_map,
            ref mut cpu_brick_buffer,
            ref mut gpu_brick_buffer,
            ref mut delta_brick_buffer,
            ref mut needs_resize_flush,
            ref mut window_updater_voxel_bind_group,
            ref mut voxel_bind_group,
            ref mut brick_allocator
        } = &mut *self.buffer_critical_section.lock().unwrap();

        let this_brick_ptr =
            &mut cpu_brick_map[brick_pos.x as usize][brick_pos.y as usize][brick_pos.z as usize];

        let mut allocate_brick = || -> VoxelBrickPointer {
            match brick_allocator.allocate()
            {
                Ok(new_ptr) =>
                {
                    VoxelBrickPointer::new_ptr(new_ptr.into_integer().try_into().unwrap())
                }
                Err(_) =>
                {
                    *needs_resize_flush = true;

                    let new_len = ((cpu_brick_buffer.len() as f64) * 1.25) as usize;

                    cpu_brick_buffer.resize(new_len, BitBrick::new(false));

                    let _ = std::mem::replace(
                        gpu_brick_buffer,
                        self.renderer.create_buffer(&wgpu::BufferDescriptor {
                            label:              Some("Voxel Chunk Data Manager Brick Buffer"),
                            usage:              wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::STORAGE,
                            size:               std::mem::size_of::<BitBrick>() as u64
                                * cpu_brick_buffer.len() as u64,
                            mapped_at_creation: false
                        })
                    );

                    *voxel_bind_group =
                        Arc::new(self.renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                            label:   Some("Voxel Bind Group"),
                            layout:  &self.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding:  0,
                                    resource: gpu_brick_map.as_entire_binding()
                                },
                                wgpu::BindGroupEntry {
                                    binding:  1,
                                    resource: gpu_brick_buffer.as_entire_binding()
                                }
                            ]
                        }));

                    window_updater_voxel_bind_group.update(voxel_bind_group.clone());

                    brick_allocator.extend_size(cpu_brick_buffer.len() - 1);

                    if let Ok(new_ptr) = brick_allocator.allocate()
                    {
                        VoxelBrickPointer::new_ptr(new_ptr.into_integer().try_into().unwrap())
                    }
                    else
                    {
                        unreachable!()
                    }
                }
            }
        };

        match this_brick_ptr.classify()
        {
            VoxelBrickPointerType::ValidBrickPointer(_) =>
            {}
            VoxelBrickPointerType::Voxel(old_solid_voxel) =>
            {
                // allocate new brick
                {
                    *this_brick_ptr = allocate_brick();

                    delta_brick_map.insert((this_brick_ptr as *mut VoxelBrickPointer).into());
                }

                // fill brick
                let brick_to_fill = &mut cpu_brick_buffer[this_brick_ptr.get_ptr() as usize];

                brick_to_fill.fill(old_solid_voxel != Voxel::Air);

                delta_brick_buffer.insert((brick_to_fill as *mut BitBrick).into());
            }
            VoxelBrickPointerType::Null =>
            {
                // allocate new brick
                {
                    *this_brick_ptr = allocate_brick();

                    delta_brick_map.insert((this_brick_ptr as *mut VoxelBrickPointer).into());
                }
            }
        }

        // write voxel
        let this_brick: &mut BitBrick = &mut cpu_brick_buffer[this_brick_ptr.get_ptr() as usize];

        this_brick.write_voxel(voxel_pos, v != Voxel::Air);

        delta_brick_buffer.insert((this_brick as *mut BitBrick).into());
    }

    pub fn read_voxel(&self, pos: ChunkPosition) -> Voxel
    {
        let unsigned_bound = (BRICK_MAP_EDGE_SIZE * VOXEL_BRICK_EDGE_LENGTH) as u16;

        assert!(pos.x < unsigned_bound, "Bound {} is out of bounds", pos.x);

        assert!(pos.y < unsigned_bound, "Bound {} is out of bounds", pos.y);

        assert!(pos.z < unsigned_bound, "Bound {} is out of bounds", pos.z);

        let voxel_pos = glm::U16Vec3::new(
            pos.x % VOXEL_BRICK_EDGE_LENGTH as u16,
            pos.y % VOXEL_BRICK_EDGE_LENGTH as u16,
            pos.z % VOXEL_BRICK_EDGE_LENGTH as u16
        );

        let brick_pos = glm::U16Vec3::new(
            pos.x / VOXEL_BRICK_EDGE_LENGTH as u16,
            pos.y / VOXEL_BRICK_EDGE_LENGTH as u16,
            pos.z / VOXEL_BRICK_EDGE_LENGTH as u16
        );

        let guard = &*self.buffer_critical_section.lock().unwrap();

        let ptr =
            guard.cpu_brick_map[brick_pos.x as usize][brick_pos.y as usize][brick_pos.z as usize];

        match ptr.classify()
        {
            VoxelBrickPointerType::Null => Voxel::Air,
            VoxelBrickPointerType::ValidBrickPointer(ptr) =>
            {
                if guard
                    .cpu_brick_buffer
                    .get(ptr as usize)
                    .unwrap()
                    .read_voxel(voxel_pos)
                {
                    Voxel::Rock0
                }
                else
                {
                    Voxel::Air
                }
            }
            VoxelBrickPointerType::Voxel(v) => v
        }
    }

    pub fn get_bind_group(&self) -> Arc<wgpu::BindGroup>
    {
        self.window_voxel_bind_group.get()
    }

    pub fn get_bind_group_layout(&self) -> &Arc<wgpu::BindGroupLayout>
    {
        &self.bind_group_layout
    }

    pub fn flush_entire(&self)
    {
        let guard = &mut *self.buffer_critical_section.lock().unwrap();

        self.flush_entire_internal(guard);
    }

    fn flush_entire_internal(
        &self,
        critical_section: &mut VoxelChunkDataManagerBufferCriticalSection
    )
    {
        let VoxelChunkDataManagerBufferCriticalSection {
            ref mut cpu_brick_map,
            ref mut gpu_brick_map,
            ref mut delta_brick_map,
            ref mut cpu_brick_buffer,
            ref mut gpu_brick_buffer,
            ref mut delta_brick_buffer,
            ref mut needs_resize_flush,
            ..
        } = critical_section;

        delta_brick_buffer.clear();
        delta_brick_map.clear();

        self.renderer.queue.write_buffer(gpu_brick_map, 0, unsafe {
            slice::from_raw_parts(
                &cpu_brick_map[0][0][0] as *const VoxelBrickPointer as *const _,
                std::mem::size_of_val::<BrickMap>(&**cpu_brick_map)
            )
        });

        self.renderer
            .queue
            .write_buffer(gpu_brick_buffer, 0, unsafe {
                slice::from_raw_parts(
                    cpu_brick_buffer.as_ptr() as *const _,
                    cpu_brick_buffer.len() * std::mem::size_of::<BitBrick>()
                )
            });

        *needs_resize_flush = false;
    }

    pub fn flush_to_gpu(&self)
    {
        let guard = &mut *self.buffer_critical_section.lock().unwrap();

        if guard.needs_resize_flush
        {
            self.flush_entire_internal(guard);

            return;
        }

        let VoxelChunkDataManagerBufferCriticalSection {
            ref mut cpu_brick_map,
            ref mut gpu_brick_map,
            ref mut delta_brick_map,
            ref mut cpu_brick_buffer,
            ref mut gpu_brick_buffer,
            ref mut delta_brick_buffer,
            ..
        } = guard;

        let head_brick_map: *const VoxelBrickPointer = &cpu_brick_map[0][0][0] as *const _;

        delta_brick_map.extract_if(|_| true).for_each(|ptr| {
            self.renderer.queue.write_buffer(
                gpu_brick_map,
                unsafe { ptr.byte_offset_from(head_brick_map).try_into().unwrap() },
                unsafe { bytes_of::<VoxelBrickPointer>(&**ptr) }
            )
        });

        let head_brick_buffer: *const BitBrick = &cpu_brick_buffer[0] as *const _;

        delta_brick_buffer.extract_if(|_| true).for_each(|ptr| {
            self.renderer.queue.write_buffer(
                gpu_brick_buffer,
                unsafe { ptr.byte_offset_from(head_brick_buffer).try_into().unwrap() },
                unsafe { bytes_of::<BitBrick>(&**ptr) }
            )
        })
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BitBrick
{
    bit_data: [u32; 16]
}

impl BitBrick
{
    pub fn new(filled: bool) -> BitBrick
    {
        let fill_val = if filled
        {
            !0
        }
        else
        {
            0
        };

        BitBrick {
            bit_data: [fill_val; 16]
        }
    }

    fn read_voxel(&self, pos: glm::U16Vec3) -> bool
    {
        // Verify if each component of the input position is in the range [0, 7]
        for &p in pos.iter()
        {
            if p > 7
            {
                panic!("Position out of range");
            }
        }

        let index = (pos[0] as usize * 64 + pos[1] as usize * 8 + pos[2] as usize) / 32;
        let bit_offset = (pos[0] as usize * 64 + pos[1] as usize * 8 + pos[2] as usize) % 32;
        let bit_mask = 1 << bit_offset;

        self.bit_data[index] & bit_mask != 0
    }

    fn write_voxel(&mut self, pos: glm::U16Vec3, voxel: bool)
    {
        // Verify if each component of the input position is in the range [0, 7]
        for &p in pos.iter()
        {
            if p > 7
            {
                panic!("Position out of range");
            }
        }

        let index = (pos[0] as usize * 64 + pos[1] as usize * 8 + pos[2] as usize) / 32;
        let bit_offset = (pos[0] as usize * 64 + pos[1] as usize * 8 + pos[2] as usize) % 32;
        let bit_mask = 1 << bit_offset;

        if voxel
        {
            self.bit_data[index] |= bit_mask;
        }
        else
        {
            self.bit_data[index] &= !bit_mask;
        }
    }

    fn fill(&mut self, filled: bool)
    {
        *self = Self::new(filled);
    }
}

#[cfg(test)]
mod test
{
    pub use super::*;

    #[test]
    fn assert_values()
    {
        assert_matches!(
            VoxelBrickPointer::new_null().classify(),
            VoxelBrickPointerType::Null
        );

        for i in 1..4
        {
            let v = Voxel::try_from(i).unwrap();

            assert_matches!(
                VoxelBrickPointer::new_voxel(v).classify(),
                VoxelBrickPointerType::Voxel(_v)
            );
        }

        for i in 1..100
        {
            assert_matches!(
                VoxelBrickPointer::new_ptr(i).classify(),
                VoxelBrickPointerType::ValidBrickPointer(_i)
            );
        }

        for i in (VoxelBrickPointer::POINTER_TO_VOXEL_BOUND - 100)
            ..VoxelBrickPointer::POINTER_TO_VOXEL_BOUND
        {
            assert_matches!(
                VoxelBrickPointer::new_ptr(i).classify(),
                VoxelBrickPointerType::ValidBrickPointer(_i)
            );
        }
    }

    // chatgpt wrote this... this is good.
    #[test]
    fn test_read_write_voxel()
    {
        let mut brick = BitBrick::new(false);

        // Test filling in one bit at a time and then accessing it
        for x in 0..8
        {
            for y in 0..8
            {
                for z in 0..8
                {
                    let pos = glm::U16Vec3::new(x, y, z);
                    brick.write_voxel(pos, true);

                    assert_eq!(brick.read_voxel(pos), true);

                    // Verify other bits remain unchanged
                    for xx in 0..8
                    {
                        for yy in 0..8
                        {
                            for zz in 0..8
                            {
                                let other_pos = glm::U16Vec3::new(xx, yy, zz);

                                if pos == other_pos
                                {
                                    continue;
                                }

                                assert_eq!(brick.read_voxel(other_pos), false);
                            }
                        }
                    }

                    brick.write_voxel(pos, false);
                    assert_eq!(brick.read_voxel(pos), false);
                }
            }
        }
    }
}

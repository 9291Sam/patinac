use core::slice;
use std::assert_matches::assert_matches;
use std::collections::BTreeSet;
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

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VoxelBrick
{
    data: [[[Voxel; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
}

impl VoxelBrick
{
    pub fn new_empty() -> VoxelBrick
    {
        VoxelBrick {
            data: [[[Voxel::Air; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
        }
    }

    pub fn write(
        &mut self
    ) -> &mut [[[Voxel; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
    {
        &mut self.data
    }

    pub fn fill(&mut self, voxel: Voxel)
    {
        for slice in self.data.iter_mut()
        {
            for layer in slice.iter_mut()
            {
                for v in layer.iter_mut()
                {
                    *v = voxel;
                }
            }
        }
    }
}

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

const VOXEL_BRICK_SIZE: usize = 8;
const BRICK_MAP_EDGE_SIZE: usize = 128;
// const CHUNK_VOXEL_SIZE: usize = VOXEL_BRICK_SIZE * BRICK_MAP_EDGE_SIZE;

// TODO: this shit very much isnt MT safe and will have races all over the
// place!
#[derive(Debug)]
pub struct VoxelChunkDataManager
{
    renderer: Arc<gfx::Renderer>,

    cpu_brick_map:            Mutex<Box<BrickMap>>,
    pub(crate) gpu_brick_map: wgpu::Buffer,
    delta_brick_map:          Mutex<BTreeSet<*const VoxelBrickPointer>>,

    // TODO: integrate into larger single mutexes
    cpu_brick_buffer:            Mutex<Vec<VoxelBrick>>,
    pub(crate) gpu_brick_buffer: Mutex<wgpu::Buffer>,
    delta_brick_buffer:          Mutex<BTreeSet<*const VoxelBrick>>,
    needs_resize_flush:          Mutex<bool>,

    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    bind_group:        Mutex<Arc<wgpu::BindGroup>>,

    brick_allocator: Mutex<util::FreelistAllocator>
}

pub type ChunkPosition = glm::U16Vec3;

impl VoxelChunkDataManager
{
    pub fn new(renderer: Arc<gfx::Renderer>, bind_group_layout: Arc<wgpu::BindGroupLayout>)
    -> Self
    {
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
            size:               std::mem::size_of::<VoxelBrick>() as u64
                * number_of_starting_bricks as u64,
            mapped_at_creation: false
        });

        let voxel_bind_group = r.create_bind_group(&wgpu::BindGroupDescriptor {
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
        });

        Self {
            renderer,
            cpu_brick_map: Mutex::new(
                vec![
                    [[VoxelBrickPointer::new_null(); BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE];
                    BRICK_MAP_EDGE_SIZE
                ]
                .into_boxed_slice()
                .try_into()
                .unwrap()
            ),
            gpu_brick_map,
            delta_brick_map: Mutex::new(BTreeSet::new()),
            cpu_brick_buffer: Mutex::new(vec![VoxelBrick::new_empty(); number_of_starting_bricks]),
            gpu_brick_buffer: Mutex::new(gpu_brick_buffer),
            delta_brick_buffer: Mutex::new(BTreeSet::new()),
            needs_resize_flush: Mutex::new(false),
            bind_group_layout,
            bind_group: Mutex::new(Arc::new(voxel_bind_group)),
            brick_allocator: Mutex::new(util::FreelistAllocator::new(
                (number_of_starting_bricks - 1).try_into().unwrap()
            ))
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

        let mut locked_brick_map = self.cpu_brick_map.lock().unwrap();

        let current_brick_ptr = &mut locked_brick_map[unsigned_pos.x as usize]
            [unsigned_pos.y as usize][unsigned_pos.z as usize];

        if let VoxelBrickPointerType::ValidBrickPointer(old_ptr) = current_brick_ptr.classify()
        {
            self.brick_allocator
                .lock()
                .unwrap()
                .free((old_ptr.into_integer() as usize).try_into().unwrap())
        }

        if v == Voxel::Air
        {
            *current_brick_ptr = VoxelBrickPointer::new_null();
        }
        else
        {
            *current_brick_ptr = VoxelBrickPointer::new_voxel(v);
        }

        self.delta_brick_map
            .lock()
            .unwrap()
            .insert(current_brick_ptr as *const _);
    }

    pub fn write_voxel(&self, v: Voxel, unsigned_pos: ChunkPosition)
    {
        let unsigned_bound = (BRICK_MAP_EDGE_SIZE * VOXEL_BRICK_SIZE) as u16;

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
            unsigned_pos.x % VOXEL_BRICK_SIZE as u16,
            unsigned_pos.y % VOXEL_BRICK_SIZE as u16,
            unsigned_pos.z % VOXEL_BRICK_SIZE as u16
        );

        let brick_pos = glm::U16Vec3::new(
            unsigned_pos.x / VOXEL_BRICK_SIZE as u16,
            unsigned_pos.y / VOXEL_BRICK_SIZE as u16,
            unsigned_pos.z / VOXEL_BRICK_SIZE as u16
        );

        let mut locked_brick_map = self.cpu_brick_map.lock().unwrap();
        let mut locked_brick_buffer = self.cpu_brick_buffer.lock().unwrap();
        let mut locked_allocator = self.brick_allocator.lock().unwrap();

        let this_brick_ptr =
            &mut locked_brick_map[brick_pos.x as usize][brick_pos.y as usize][brick_pos.z as usize];

        let mut allocate_brick = || -> VoxelBrickPointer {
            match locked_allocator.allocate()
            {
                Ok(new_ptr) =>
                {
                    VoxelBrickPointer::new_ptr(new_ptr.into_integer().try_into().unwrap())
                }
                Err(_) =>
                {
                    let mut locked_gpu_brick_buffer = self.gpu_brick_buffer.lock().unwrap();

                    *self.needs_resize_flush.lock().unwrap() = true;

                    let new_len = ((locked_brick_buffer.len() as f64) * 1.25) as usize;

                    locked_brick_buffer.resize(new_len, VoxelBrick::new_empty());

                    *locked_gpu_brick_buffer =
                        self.renderer.create_buffer(&wgpu::BufferDescriptor {
                            label:              Some("Voxel Chunk Data Manager Brick Buffer"),
                            usage:              wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::STORAGE,
                            size:               std::mem::size_of::<VoxelBrick>() as u64
                                * locked_brick_buffer.len() as u64,
                            mapped_at_creation: false
                        });

                    *self.bind_group.lock().unwrap() =
                        Arc::new(self.renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                            label:   Some("Voxel Bind Group"),
                            layout:  &self.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding:  0,
                                    resource: self.gpu_brick_map.as_entire_binding()
                                },
                                wgpu::BindGroupEntry {
                                    binding:  1,
                                    resource: locked_gpu_brick_buffer.as_entire_binding()
                                }
                            ]
                        }));

                    locked_allocator.extend_size(locked_brick_buffer.len() - 1);

                    if let Ok(new_ptr) = locked_allocator.allocate()
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
            VoxelBrickPointerType::Voxel(_) =>
            {
                // allocate new brick
                {
                    *this_brick_ptr = allocate_brick();

                    self.delta_brick_map
                        .lock()
                        .unwrap()
                        .insert(this_brick_ptr as *const _);
                }

                // fill brick
                let brick_to_fill = &mut locked_brick_buffer[this_brick_ptr.get_ptr() as usize];

                self.delta_brick_buffer
                    .lock()
                    .unwrap()
                    .insert(brick_to_fill as *const _);
            }
            VoxelBrickPointerType::Null =>
            {
                // allocate new brick
                {
                    *this_brick_ptr = allocate_brick();

                    self.delta_brick_map
                        .lock()
                        .unwrap()
                        .insert(this_brick_ptr as *const _);
                }
            }
        }

        // write voxel
        let this_brick: &mut VoxelBrick =
            &mut locked_brick_buffer[this_brick_ptr.get_ptr() as usize];

        this_brick.write()[voxel_pos.x as usize][voxel_pos.y as usize][voxel_pos.z as usize] = v;

        self.delta_brick_buffer
            .lock()
            .unwrap()
            .insert(this_brick as *const _);
    }

    pub fn get_bind_group(&self) -> Arc<wgpu::BindGroup>
    {
        self.bind_group.lock().unwrap().clone()
    }

    pub fn flush_entire(&self)
    {
        let mut delta_brick_buffer = self.delta_brick_buffer.lock().unwrap();
        let mut delta_brick_map = self.delta_brick_map.lock().unwrap();

        let locked_brick_map = self.cpu_brick_map.lock().unwrap();
        let locked_brick_buffer = self.cpu_brick_buffer.lock().unwrap();

        delta_brick_buffer.clear();
        delta_brick_map.clear();

        self.renderer
            .queue
            .write_buffer(&self.gpu_brick_map, 0, unsafe {
                slice::from_raw_parts(
                    &locked_brick_map[0][0][0] as *const VoxelBrickPointer as *const _,
                    std::mem::size_of_val::<BrickMap>(&*locked_brick_map)
                )
            });

        self.renderer
            .queue
            .write_buffer(&self.gpu_brick_buffer.lock().unwrap(), 0, unsafe {
                slice::from_raw_parts(
                    locked_brick_buffer.as_ptr() as *const _,
                    locked_brick_buffer.len() * std::mem::size_of::<VoxelBrick>()
                )
            });

        *self.needs_resize_flush.lock().unwrap() = false;
    }

    pub fn flush_to_gpu(&self)
    {
        let mut delta_brick_buffer = self.delta_brick_buffer.lock().unwrap();
        let mut delta_brick_map = self.delta_brick_map.lock().unwrap();

        let locked_brick_map = self.cpu_brick_map.lock().unwrap();
        let locked_brick_buffer = self.cpu_brick_buffer.lock().unwrap();

        {
            let head_brick_map: *const VoxelBrickPointer = &locked_brick_map[0][0][0] as *const _;

            delta_brick_map.extract_if(|_| true).for_each(|ptr| {
                self.renderer.queue.write_buffer(
                    &self.gpu_brick_map,
                    unsafe { ptr.byte_offset_from(head_brick_map).try_into().unwrap() },
                    unsafe { bytes_of::<VoxelBrickPointer>(&*ptr) }
                )
            });
        }

        if *self.needs_resize_flush.lock().unwrap()
        {
            delta_brick_buffer.clear();

            self.renderer
                .queue
                .write_buffer(&self.gpu_brick_buffer.lock().unwrap(), 0, unsafe {
                    slice::from_raw_parts(
                        locked_brick_buffer.as_ptr() as *const _,
                        locked_brick_buffer.len() * std::mem::size_of::<VoxelBrick>()
                    )
                });

            *self.needs_resize_flush.lock().unwrap() = false;
        }
        else
        {
            let head_brick_buffer: *const VoxelBrick = &locked_brick_buffer[0] as *const _;

            delta_brick_buffer.extract_if(|_| true).for_each(|ptr| {
                self.renderer.queue.write_buffer(
                    &self.gpu_brick_buffer.lock().unwrap(),
                    unsafe { ptr.byte_offset_from(head_brick_buffer).try_into().unwrap() },
                    unsafe { bytes_of::<VoxelBrick>(&*ptr) }
                )
            })
        }
    }

    // TODO: functions for writing to each array that automatically handle flushing
    // + resizes
}

#[cfg(test)]
mod test
{
    pub use super::*;

    #[test]
    fn assert_sizes()
    {
        assert_eq!(1024, std::mem::size_of::<VoxelBrick>());
        assert_eq!(8 * 1024 * 1024, std::mem::size_of::<BrickMap>());
    }

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
}

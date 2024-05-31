use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::{cast_slice, Pod};
use fnv::FnvHashMap;
use gfx::glm::all;
use gfx::wgpu;
use nonmax::NonMaxU32;
use offset_allocator::NodeIndexNonMax;
pub use offset_allocator::{Allocation as OAllocation, Allocator as OAllocator};
use util::NoElementContained;

pub struct SubAllocatedCpuTrackedBuffer<T: Pod>
{
    renderer:  Arc<gfx::Renderer>,
    allocator: OAllocator<u32>,

    gpu_buffer: wgpu::Buffer,
    cpu_buffer: Box<[T]>,

    flush_ranges: Vec<RangeInclusive<u32>>,

    #[cfg(debug_assertions)]
    uuid: util::Uuid
}

pub struct BufferAllocation<T: Pod>
{
    internal_allocation: OAllocation,
    length:              u32,
    phantom:             PhantomData<T>,
    #[cfg(debug_assertions)]
    uuid:                util::Uuid
}

impl<T: Pod> Hash for BufferAllocation<T>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.internal_allocation.offset.hash(state);
        self.length.hash(state);
    }
}

impl<T: Pod> BufferAllocation<T>
{
    pub fn to_global_valid_range(&self) -> Range<u32>
    {
        self.internal_allocation.offset..(self.internal_allocation.offset + self.length)
    }

    pub fn get_length(&self) -> u32
    {
        self.length
    }
}

impl<T: Pod> SubAllocatedCpuTrackedBuffer<T>
{
    pub fn new(
        renderer: Arc<gfx::Renderer>,
        capacity: u32,
        buffer_label: &str,
        usages: wgpu::BufferUsages
    ) -> Self
    {
        SubAllocatedCpuTrackedBuffer {
            renderer:                      renderer.clone(),
            allocator:                     OAllocator::new(capacity),
            gpu_buffer:                    renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(buffer_label),
                size:               capacity as u64 * std::mem::size_of::<T>() as u64,
                usage:              usages | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }),
            cpu_buffer:                    unsafe {
                Box::new_zeroed_slice(capacity as usize).assume_init()
            },
            flush_ranges:                  Vec::new(),
            #[cfg(debug_assertions)]
            uuid:                          util::Uuid::new()
        }
    }

    #[must_use]
    pub fn allocate(&mut self, length: u32) -> BufferAllocation<T>
    {
        let internal_allocation = self.allocator.allocate(length).unwrap_or_else(|| {
            panic!(
                "Allocation of {} {}s failed!",
                length,
                std::any::type_name::<T>()
            )
        });

        BufferAllocation {
            internal_allocation,
            length,
            phantom: PhantomData,
            #[cfg(debug_assertions)]
            uuid: self.uuid
        }
    }

    pub fn realloc(
        &mut self,
        new_length: u32,
        old_allocation: BufferAllocation<T>
    ) -> BufferAllocation<T>
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == old_allocation.uuid);

        if new_length
            <= self
                .allocator
                .allocation_size(old_allocation.internal_allocation)
        {
            BufferAllocation {
                internal_allocation:           old_allocation.internal_allocation,
                length:                        new_length,
                phantom:                       PhantomData,
                #[cfg(debug_assertions)]
                uuid:                          self.uuid.clone()
            }
        }
        else
        {
            let old_data_range = old_allocation.to_global_valid_range();
            self.allocator.free(old_allocation.internal_allocation);

            let new_allocation_internal =
                self.allocator.allocate(new_length).unwrap_or_else(|| {
                    panic!(
                        "Allocation of {} {}s failed!",
                        new_length,
                        std::any::type_name::<T>()
                    )
                });

            let new_allocation = BufferAllocation {
                internal_allocation:           new_allocation_internal,
                length:                        new_length,
                phantom:                       PhantomData,
                #[cfg(debug_assertions)]
                uuid:                          self.uuid
            };

            self.cpu_buffer.copy_within(
                (old_data_range.start as usize)..(old_data_range.end as usize),
                new_allocation.to_global_valid_range().start as usize
            );

            let new_allocation_range = new_allocation.to_global_valid_range();

            self.flush_ranges
                .push(new_allocation_range.start..=new_allocation_range.end - 1);

            new_allocation
        }
    }

    pub fn deallocate(&mut self, allocation: BufferAllocation<T>)
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        self.allocator.free(allocation.internal_allocation)
    }

    pub fn read(&self, allocation: &BufferAllocation<T>, local_range: Range<u32>) -> &[T]
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        let global_range = self.get_allocation_range(allocation, local_range);

        &self.cpu_buffer[(global_range.start as usize)..(global_range.end as usize)]
    }

    pub fn write(&mut self, allocation: &BufferAllocation<T>, local_range: Range<u32>, data: &[T])
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        debug_assert!(local_range.len() == data.len());

        let global_range = self.get_allocation_range(allocation, local_range);

        self.cpu_buffer[(global_range.start as usize)..(global_range.end as usize)]
            .copy_from_slice(data);

        self.flush_ranges
            .push(global_range.start..=(global_range.end - 1))
    }

    pub fn access_ref<K>(
        &self,
        allocation: &BufferAllocation<T>,
        local_range: Range<u32>,
        func: impl FnOnce(&[T]) -> K
    ) -> K
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        func(self.read(allocation, local_range))
    }

    pub fn access_mut<K>(
        &mut self,
        allocation: &BufferAllocation<T>,
        local_range: Range<u32>,
        func: impl FnOnce(&mut [T]) -> K
    ) -> K
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        let global_range = self.get_allocation_range(allocation, local_range);

        let k =
            func(&mut self.cpu_buffer[(global_range.start as usize)..(global_range.end as usize)]);

        self.flush_ranges
            .push(global_range.start..=(global_range.end - 1));

        k
    }

    pub fn access_buffer(&self) -> &wgpu::Buffer
    {
        &self.gpu_buffer
    }

    pub fn get_buffer_size_bytes(&self) -> NonZeroU64
    {
        NonZeroU64::new(self.cpu_buffer.len() as u64 * std::mem::size_of::<T>() as u64).unwrap()
    }

    pub fn replicate_to_gpu(&mut self)
    {
        let flush_ranges = combine_into_ranges(
            std::mem::replace(&mut self.flush_ranges, Vec::new()),
            2u32.pow(14),
            32
        );

        for range in flush_ranges
        {
            let range_len_elements = range.end() + 1 - range.start();

            let usize_range = (*range.start() as usize)..=(*range.end() as usize);

            if range_len_elements > 0
            {
                self.renderer
                    .queue
                    .write_buffer_with(
                        &self.gpu_buffer,
                        *range.start() as u64 * std::mem::size_of::<T>() as u64,
                        NonZeroU64::new(
                            range_len_elements as u64 * std::mem::size_of::<T>() as u64
                        )
                        .unwrap()
                    )
                    .unwrap()
                    .copy_from_slice(cast_slice(&self.cpu_buffer[usize_range]))
            }
        }
    }

    fn get_allocation_range(
        &self,
        allocation: &BufferAllocation<T>,
        local_range: Range<u32>
    ) -> Range<u32>
    {
        #[cfg(debug_assertions)]
        assert!(self.uuid == allocation.uuid);

        let front = allocation.internal_allocation.offset + local_range.start;
        let back = allocation.internal_allocation.offset + local_range.end;
        let read_range = front..back;

        #[cfg(debug_assertions)]
        {
            let allocated_range = allocation.to_global_valid_range();

            debug_assert!(read_range.start >= allocated_range.start);
            debug_assert!(read_range.end <= allocated_range.end);
        }

        read_range
    }
}

pub struct SubAllocatedDenseSet<T: Pod + Eq + Hash>
{
    element_to_idx_map:        FnvHashMap<T, u32>,
    data:                      BufferAllocation<T>,
    number_of_stored_elements: u32
}

impl<T: Pod + Eq + Hash> SubAllocatedDenseSet<T>
{
    pub fn new(initial_elements: usize, allocator: &mut SubAllocatedCpuTrackedBuffer<T>) -> Self
    {
        assert!(std::mem::size_of::<T>() < 64);

        SubAllocatedDenseSet {
            element_to_idx_map:        FnvHashMap::with_capacity_and_hasher(
                initial_elements,
                Default::default()
            ),
            data:                      allocator.allocate(initial_elements as u32),
            number_of_stored_elements: 0
        }
    }

    /// Returns the previous element (if contained)
    pub fn insert(&mut self, t: T, allocator: &mut SubAllocatedCpuTrackedBuffer<T>) -> Option<T>
    {
        match self.element_to_idx_map.entry(t.clone())
        {
            Entry::Occupied(occupied_entry) =>
            {
                let idx: u32 = *occupied_entry.get();
                let old_t: T = occupied_entry.replace_key();

                allocator.write(&self.data, idx..(idx + 1), &[t.clone()]);

                Some(old_t)
            }
            Entry::Vacant(vacant_entry) =>
            {
                let new_element_idx = self.number_of_stored_elements;

                self.number_of_stored_elements += 1;

                allocator.write(
                    &self.data,
                    new_element_idx..(new_element_idx + 1),
                    &[t.clone()]
                );

                vacant_entry.insert(new_element_idx);

                None
            }
        }
    }

    pub fn remove(
        &mut self,
        t: T,
        allocator: &mut SubAllocatedCpuTrackedBuffer<T>
    ) -> Result<(), NoElementContained>
    {
        match self.element_to_idx_map.entry(t.clone())
        {
            Entry::Occupied(occupied_entry) =>
            {
                let (_, rem_idx) = occupied_entry.remove_entry();

                let last_idx = self.number_of_stored_elements - 1;

                if rem_idx != last_idx
                {
                    let last_val = allocator
                        .read(&self.data, last_idx..(last_idx + 1))
                        .get(0)
                        .unwrap()
                        .clone();

                    // swap last element to the front
                    allocator.write(&self.data, rem_idx..(rem_idx + 1), &[last_val]);

                    // fixup
                    *self.element_to_idx_map.get_mut(&last_val).unwrap() = rem_idx;
                }

                self.number_of_stored_elements -= 1;
                Ok(())
            }
            Entry::Vacant(_) => Err(NoElementContained)
        }
    }
}

use std::ops::RangeInclusive;

pub(crate) fn combine_into_ranges(
    mut ranges: Vec<RangeInclusive<u32>>,
    max_distance: u32,
    max_ranges: usize
) -> Vec<RangeInclusive<u32>>
{
    if ranges.is_empty()
    {
        return vec![];
    }

    // Sort ranges by their start value
    ranges.sort_unstable_by_key(|range| *range.start());

    let mut combined_ranges = Vec::with_capacity(ranges.len());
    let mut current_range = unsafe { ranges.get_unchecked(0).clone() };

    for i in 1..ranges.len()
    {
        let range = unsafe { ranges.get_unchecked(i) };
        if *range.start() <= *current_range.end() + max_distance
        {
            current_range = *current_range.start()..=*range.end().max(current_range.end());
        }
        else
        {
            combined_ranges.push(current_range);
            current_range = range.clone();
        }
    }
    combined_ranges.push(current_range);

    unsafe {
        if combined_ranges.len() > max_ranges
        {
            while combined_ranges.len() > max_ranges
            {
                let mut min_distance = u32::MAX;
                let mut min_index = 0;

                for i in 0..combined_ranges.len() - 1
                {
                    let distance = *combined_ranges.get_unchecked(i + 1).start()
                        - *combined_ranges.get_unchecked(i).end();
                    if distance < min_distance
                    {
                        min_distance = distance;
                        min_index = i;
                    }
                }

                let merged_range = *combined_ranges.get_unchecked(min_index).start()
                    ..=*combined_ranges.get_unchecked(min_index + 1).end();
                *combined_ranges.get_unchecked_mut(min_index) = merged_range;
                combined_ranges.remove(min_index + 1);
            }

            combined_ranges.set_len(max_ranges);
        }
    }

    combined_ranges
}

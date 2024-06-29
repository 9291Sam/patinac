use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};

pub struct CpuTrackedBuffer<T: AnyBitPattern + NoUninit + Debug>
{
    renderer: Arc<super::Renderer>,

    name:                String,
    usage:               wgpu::BufferUsages,
    buffer_len_elements: usize,
    gpu_data:            wgpu::Buffer,

    cpu_data: Vec<T>,

    flush_list:         Vec<u64>,
    needs_resize_flush: bool
}

impl<T: AnyBitPattern + NoUninit + Debug> CpuTrackedBuffer<T>
{
    #[inline(always)]
    pub fn new(
        renderer: Arc<super::Renderer>,
        elements: usize,
        name: String,
        usage: wgpu::BufferUsages
    ) -> CpuTrackedBuffer<T>
    {
        let gpu_data = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(&name),
            size:               elements as u64 * std::mem::size_of::<T>() as u64,
            usage:              usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        CpuTrackedBuffer {
            renderer,
            buffer_len_elements: elements,
            gpu_data,
            cpu_data: unsafe { Box::new_zeroed_slice(elements).assume_init().into_vec() },
            flush_list: Vec::new(),
            needs_resize_flush: false,
            name,
            usage
        }
    }

    #[inline(always)]
    pub fn get_cpu_len(&self) -> usize
    {
        self.cpu_data.len()
    }

    #[inline(always)]
    #[allow(clippy::clone_on_copy)]
    pub fn read_clone(&self, index: usize) -> T
    {
        self.cpu_data
            .get(index)
            .unwrap_or_else(|| {
                panic!(
                    "Out of Bounds access of CpuTrackedBuffer {} / {}",
                    index, self.buffer_len_elements
                )
            })
            .clone()
    }

    #[inline(always)]
    pub fn access_ref<K>(&self, index: usize, access_func: impl FnOnce(&T) -> K) -> K
    {
        access_func(self.cpu_data.get(index).unwrap_or_else(|| {
            panic!(
                "Out of Bounds access of CpuTrackedBuffer {} / {}",
                index, self.buffer_len_elements
            )
        }))
    }

    #[inline(always)]
    pub fn access_mut<K>(&mut self, index: usize, access_func: impl FnOnce(&mut T) -> K) -> K
    {
        let k = access_func(self.cpu_data.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Out of Bounds access of CpuTrackedBuffer {} / {}",
                index, self.buffer_len_elements
            )
        }));

        self.flush_list.push(index as u64);

        k
    }

    /// # Safety
    ///
    /// Calling this method with overlapping or out-of-bounds indices is
    /// undefined behavior even if the resulting references are not used.
    #[inline(always)]
    pub unsafe fn access_many_unchecked_mut<const N: usize, K>(
        &mut self,
        indices: [usize; N],
        access_func: impl FnOnce([&mut T; N]) -> K
    ) -> K
    {
        access_func(self.cpu_data.get_many_unchecked_mut(indices))
    }

    #[inline(always)]
    pub fn write(&mut self, index: usize, t: T)
    {
        self.flush_list.push(index as u64);

        let len = self.cpu_data.len();

        *self.cpu_data.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Cpu Tracked Buffer Index Out Of Bounds @ {} / {}",
                index, len
            )
        }) = t;
    }

    #[inline(always)]
    pub fn realloc(&mut self, elements: usize)
    {
        self.cpu_data.resize_with(elements, T::zeroed);

        self.needs_resize_flush = true;
    }

    // TODO: replace with a get ptr...
    #[inline(always)]
    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        buf_access_func(&self.gpu_data)
    }

    #[inline(always)]
    #[must_use]
    /// true if the buffer was recreated
    pub fn replicate_to_gpu(&mut self) -> bool
    {
        let mut did_resize_occur = false;

        if self.buffer_len_elements != self.cpu_data.len()
        {
            self.gpu_data = self.renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&self.name),
                size:               self.cpu_data.len() as u64 * std::mem::size_of::<T>() as u64,
                usage:              self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            });

            self.buffer_len_elements = self.cpu_data.len();

            self.needs_resize_flush = true;

            did_resize_occur = true;
        }

        let points_to_flush = std::mem::take(&mut self.flush_list);

        if std::mem::replace(&mut self.needs_resize_flush, false)
        {
            // log::trace!("full flush of {}", self.name);

            self.renderer
                .queue
                .write_buffer_with(
                    &self.gpu_data,
                    0,
                    NonZeroU64::new(self.cpu_data.len() as u64 * std::mem::size_of::<T>() as u64)
                        .unwrap()
                )
                .unwrap()
                .copy_from_slice(cast_slice(&self.cpu_data[..]));
        }
        else
        {
            let flush_ranges = crate::combine_into_ranges(points_to_flush, 65535, 64);

            for range in flush_ranges
            {
                let range_len_elements = range.end() + 1 - range.start();

                let usize_range = (*range.start() as usize)..=(*range.end() as usize);

                if range_len_elements > 0
                {
                    self.renderer
                        .queue
                        .write_buffer_with(
                            &self.gpu_data,
                            *range.start() * std::mem::size_of::<T>() as u64,
                            NonZeroU64::new(range_len_elements * std::mem::size_of::<T>() as u64)
                                .unwrap()
                        )
                        .unwrap()
                        .copy_from_slice(cast_slice(&self.cpu_data[usize_range]))
                }
            }
        }

        did_resize_occur
    }
}

impl<T: AnyBitPattern + NoUninit + Debug + Eq> CpuTrackedBuffer<T>
{
    pub fn write_eq_testing(&mut self, index: usize, t: T)
    {
        let len = self.cpu_data.len();

        let elem_to_test: &mut T = self.cpu_data.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Cpu Tracked Buffer Index Out Of Bounds @ {} / {}",
                index, len
            )
        });

        if *elem_to_test != t
        {
            self.flush_list.push(index as u64);

            *elem_to_test = t;
        }
    }
}

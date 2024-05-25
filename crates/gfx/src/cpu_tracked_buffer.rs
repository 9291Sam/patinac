use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};

pub struct CpuTrackedBuffer<T: AnyBitPattern + NoUninit + Debug>
{
    renderer:           Arc<super::Renderer>,
    name:               String,
    usage:              wgpu::BufferUsages,
    critical_section:   Mutex<CpuTrackedBufferCriticalSection<T>>,
    needs_resize_flush: AtomicBool
}

struct CpuTrackedBufferCriticalSection<T>
{
    buffer_len_elements: usize,
    gpu_data:            wgpu::Buffer,
    cpu_data:            Vec<T>,
    flush_list:          Vec<u64>
}

impl<T: AnyBitPattern + NoUninit + Debug> CpuTrackedBuffer<T>
{
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
            critical_section: Mutex::new(CpuTrackedBufferCriticalSection {
                buffer_len_elements: elements,
                gpu_data,
                cpu_data: vec![T::zeroed(); elements],
                flush_list: Vec::new()
            }),
            needs_resize_flush: AtomicBool::new(false),
            name,
            usage
        }
    }

    pub fn get_cpu_len(&self) -> usize
    {
        self.critical_section.lock().unwrap().cpu_data.len()
    }

    #[allow(clippy::clone_on_copy)]
    pub fn read_clone(&self, index: usize) -> T
    {
        self.critical_section
            .lock()
            .unwrap()
            .cpu_data
            .get(index)
            .expect("Out of Bounds access of CpuTrackedBuffer")
            .clone()
    }

    pub fn access_ref<K>(&self, index: usize, access_func: impl FnOnce(&T) -> K) -> K
    {
        access_func(
            self.critical_section
                .lock()
                .unwrap()
                .cpu_data
                .get_mut(index)
                .expect("Out of Bounds access of CpuTrackedBuffer")
        )
    }

    pub fn access_mut<K>(&self, index: usize, access_func: impl FnOnce(&mut T) -> K) -> K
    {
        let CpuTrackedBufferCriticalSection {
            cpu_data,
            flush_list,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        let k = access_func(
            cpu_data
                .get_mut(index)
                .expect("Out of Bounds access of CpuTrackedBuffer")
        );

        // if flush_list.len() < MAX_FLUSHES_BEFORE_ENTIRE
        // {
        flush_list.push(index as u64);
        // }

        k
    }

    pub fn write(&self, index: usize, t: T)
    {
        let CpuTrackedBufferCriticalSection {
            cpu_data,
            flush_list,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        // if flush_list.len() < MAX_FLUSHES_BEFORE_ENTIRE
        // {
        flush_list.push(index as u64);
        // }

        let len = cpu_data.len();

        *cpu_data.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Cpu Tracked Buffer Index Out Of Bounds @ {} / {}",
                index, len
            )
        }) = t;
    }

    pub fn realloc(&self, elements: usize)
    {
        let CpuTrackedBufferCriticalSection {
            cpu_data, ..
        } = &mut *self.critical_section.lock().unwrap();

        cpu_data.resize_with(elements, T::zeroed);

        self.needs_resize_flush.store(true, Ordering::SeqCst);
    }

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        buf_access_func(&self.critical_section.lock().unwrap().gpu_data)
    }

    #[must_use]
    /// true if the buffer was recreated
    pub fn replicate_to_gpu(&self) -> bool
    {
        let CpuTrackedBufferCriticalSection {
            cpu_data,
            flush_list,
            gpu_data,
            buffer_len_elements
        } = &mut *self.critical_section.lock().unwrap();

        let mut did_resize_occur = false;

        if *buffer_len_elements != cpu_data.len()
        {
            *gpu_data = self.renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&self.name),
                size:               cpu_data.len() as u64 * std::mem::size_of::<T>() as u64,
                usage:              self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            });

            *buffer_len_elements = cpu_data.len();

            self.needs_resize_flush.store(true, Ordering::SeqCst);

            did_resize_occur = true;
        }

        let points_to_flush = std::mem::take(flush_list);

        if self.needs_resize_flush.swap(false, Ordering::SeqCst)
        {
            log::trace!("full flush of {}", self.name);

            self.renderer
                .queue
                .write_buffer_with(
                    gpu_data,
                    0,
                    NonZeroU64::new(cpu_data.len() as u64 * std::mem::size_of::<T>() as u64)
                        .unwrap()
                )
                .unwrap()
                .copy_from_slice(cast_slice(&cpu_data[..]));
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
                            gpu_data,
                            *range.start() * std::mem::size_of::<T>() as u64,
                            NonZeroU64::new(range_len_elements * std::mem::size_of::<T>() as u64)
                                .unwrap()
                        )
                        .unwrap()
                        .copy_from_slice(cast_slice(&cpu_data[usize_range]))
                }
            }
        }

        did_resize_occur
    }
}

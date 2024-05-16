use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::Zeroable;

const MAX_FLUSHES_BEFORE_ENTIRE: usize = 128;

pub struct CpuTrackedBuffer<T: Zeroable + Clone>
{
    renderer:                Arc<super::Renderer>,
    gpu_data:                wgpu::Buffer,
    cpu_data_and_flush_list: Mutex<(Box<[T]>, Vec<usize>)>,
    needs_resize_flush:      AtomicBool
}

impl<T: Zeroable + Clone> CpuTrackedBuffer<T>
{
    pub fn new(
        renderer: Arc<super::Renderer>,
        elements: usize,
        name: &str,
        usage: wgpu::BufferUsages
    ) -> CpuTrackedBuffer<T>
    {
        let cpu_data = vec![T::zeroed(); elements].into_boxed_slice();
        let gpu_data = renderer.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: elements as u64 * std::mem::size_of::<T>() as u64,
            usage,
            mapped_at_creation: false
        });

        CpuTrackedBuffer {
            renderer,
            gpu_data,
            cpu_data_and_flush_list: Mutex::new((cpu_data, Vec::new())),
            needs_resize_flush: AtomicBool::new(false)
        }
    }

    pub fn read(&self, index: usize) -> T
    {
        self.cpu_data_and_flush_list
            .lock()
            .unwrap()
            .0
            .get(index)
            .expect("Out of Bounds access of CpuTrackedBuffer")
            .clone()
    }

    pub fn write(&self, index: usize, t: T)
    {
        let (cpu_data, flush_list) = &mut *self.cpu_data_and_flush_list.lock().unwrap();

        if flush_list.len() > MAX_FLUSHES_BEFORE_ENTIRE
        {
            flush_list.push(index);
        }

        *cpu_data
            .get_mut(index)
            .expect("Out of bounds CpuTrackedBuffer") = t;
    }

    pub fn realloc(&self, elements: usize)
    {
        let (cpu_data, flush_list) = &mut *self.cpu_data_and_flush_list.lock().unwrap();

        let mut new_data = vec![T::zeroed(); elements].into_boxed_slice();

        unsafe {
            std::ptr::copy_nonoverlapping(
                cpu_data,
                &mut new_data,
                cpu_data.len().min(new_data.len())
            )
        }

        *cpu_data = new_data;
        flush_list.clear();
        self.needs_resize_flush.store(true, Ordering::SeqCst);
    }

    pub fn replicate_to_gpu(&self)
    {
        let (cpu_data, flush_list) = &mut *self.cpu_data_and_flush_list.lock().unwrap();

        if flush_list.len() > MAX_FLUSHES_BEFORE_ENTIRE
            || self.needs_resize_flush.load(Ordering::SeqCst)
        {
            flush_list.clear();

            let buf: *mut T = self
                .renderer
                .queue
                .write_buffer_with(
                    &self.gpu_data,
                    0,
                    NonZeroU64::new(cpu_data.len() as u64 * std::mem::size_of::<T>() as u64)
                        .unwrap()
                )
                .unwrap()
                .as_mut_ptr() as *mut T;

            unsafe {
                std::ptr::copy_nonoverlapping::<T>((**cpu_data).as_ptr(), buf, cpu_data.len())
            };
        }
        else
        {
            flush_list.drain(..).for_each(|i| {
                let buf: *mut T = self
                    .renderer
                    .queue
                    .write_buffer_with(
                        &self.gpu_data,
                        i as u64 * std::mem::size_of::<T>() as u64,
                        NonZeroU64::new(std::mem::size_of::<T>() as u64).unwrap()
                    )
                    .unwrap()
                    .as_mut_ptr() as *mut T;

                unsafe { std::ptr::copy_nonoverlapping::<T>(&cpu_data[i], buf, 1) };
            });
        }
    }
}

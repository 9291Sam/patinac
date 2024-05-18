use std::hash::Hash;
use std::num::NonZero;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard};

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};
use gfx::wgpu;
use util::NoElementContained;

pub(crate) struct CpuTrackedDenseSet<T: AnyBitPattern + NoUninit + Hash + Eq>
{
    renderer:                Arc<gfx::Renderer>,
    name:                    String,
    usage:                   wgpu::BufferUsages,
    gpu_buffer_len_elements: AtomicUsize,
    gpu_buffer:              RwLock<wgpu::Buffer>,
    dense_set:               Mutex<util::DenseSet<T>>
}

impl<T: AnyBitPattern + NoUninit + Hash + Eq> CpuTrackedDenseSet<T>
{
    pub fn new(
        renderer: Arc<gfx::Renderer>,
        initial_gpu_buffer_len_elements: usize,
        name: String,
        buffer_usage: wgpu::BufferUsages
    ) -> Arc<Self>
    {
        Arc::new(CpuTrackedDenseSet {
            renderer:                renderer.clone(),
            name:                    name.clone(),
            usage:                   buffer_usage,
            gpu_buffer_len_elements: AtomicUsize::new(initial_gpu_buffer_len_elements),
            gpu_buffer:              RwLock::new(renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&name),
                size:               std::mem::size_of::<T>() as u64
                    * initial_gpu_buffer_len_elements as u64,
                usage:              buffer_usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            })),
            dense_set:               Mutex::new(util::DenseSet::new())
        })
    }

    pub fn insert(&self, t: T) -> Option<T>
    {
        self.dense_set.lock().unwrap().insert(t)
    }

    pub fn remove(&self, t: T) -> Result<(), NoElementContained>
    {
        self.dense_set.lock().unwrap().remove(t)
    }

    pub fn get_number_of_elements(&self) -> usize
    {
        self.dense_set.lock().unwrap().to_dense_elements().len()
    }

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        buf_access_func(&self.gpu_buffer.read().unwrap())
    }

    #[must_use]
    // returns whether or not a resize of the gpu-internal buffer ocurred
    pub fn flush_to_gpu(&self) -> bool
    {
        let mut resize_occurred = false;

        let mut gpu_buffer = self.gpu_buffer.write().unwrap();

        if self.get_number_of_elements() < self.gpu_buffer_len_elements.load(Ordering::SeqCst) / 2
            || self.get_number_of_elements() > self.gpu_buffer_len_elements.load(Ordering::SeqCst)
        {
            *gpu_buffer = self.renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&self.name),
                size:               self.get_number_of_elements() as u64 * 3 / 2
                    * std::mem::size_of::<T>() as u64,
                usage:              self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            });

            self.gpu_buffer_len_elements.store(
                self.get_number_of_elements() * 3 / 2 * std::mem::size_of::<T>(),
                Ordering::SeqCst
            );
            resize_occurred = true;
        }

        let data_mtx = &self.dense_set.lock().unwrap();
        let data_to_write: &[u8] = cast_slice(data_mtx.to_dense_elements());

        self.renderer
            .queue
            .write_buffer_with(
                &gpu_buffer,
                0,
                NonZero::new(data_to_write.len() as u64).unwrap()
            )
            .unwrap()
            .copy_from_slice(cast_slice(data_to_write));

        resize_occurred
    }
}
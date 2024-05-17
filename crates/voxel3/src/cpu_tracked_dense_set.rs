use std::hash::Hash;
use std::num::NonZero;
use std::sync::Arc;

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};
use gfx::wgpu;
use util::NoElementContained;

pub(crate) struct CpuTrackedDenseSet<T: AnyBitPattern + NoUninit + Hash + Eq>
{
    renderer:                Arc<gfx::Renderer>,
    name:                    String,
    usage:                   wgpu::BufferUsages,
    gpu_buffer_len_elements: usize,
    gpu_buffer:              wgpu::Buffer,
    dense_set:               util::DenseSet<T>
}

impl<T: AnyBitPattern + NoUninit + Hash + Eq> CpuTrackedDenseSet<T>
{
    pub fn insert(&mut self, t: T) -> Option<T>
    {
        self.dense_set.insert(t)
    }

    pub fn remove(&mut self, t: T) -> Result<(), NoElementContained>
    {
        self.dense_set.remove(t)
    }

    pub fn get_number_of_elements(&self) -> usize
    {
        self.dense_set.to_dense_elements().len()
    }

    pub fn flush_to_gpu(&mut self)
    {
        if self.get_number_of_elements() < self.gpu_buffer_len_elements / 2
            || self.get_number_of_elements() > self.gpu_buffer_len_elements
        {
            self.gpu_buffer = self.renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&self.name),
                size:               self.get_number_of_elements() as u64 * 3 / 2
                    * std::mem::size_of::<T>() as u64,
                usage:              self.usage,
                mapped_at_creation: false
            });
        }

        self.renderer
            .queue
            .write_buffer_with(
                &self.gpu_buffer,
                0,
                NonZero::new(self.gpu_buffer_len_elements as u64 * std::mem::size_of::<T>() as u64)
                    .unwrap()
            )
            .unwrap()
            .copy_from_slice(cast_slice(self.dense_set.to_dense_elements()));
    }
}

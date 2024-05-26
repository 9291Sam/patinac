use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use bytemuck::{AnyBitPattern, NoUninit};
use util::NoElementContained;

use crate::Renderer;

pub struct CpuTrackedDenseSet<T: AnyBitPattern + NoUninit + Hash + Eq + Debug>
{
    // TODO: inline this struct and remove the cpu side duplication
    gpu_buffer: crate::CpuTrackedBuffer<T>,
    dense_set:  util::DenseSet<T>
}

impl<T: AnyBitPattern + NoUninit + Hash + Eq + Debug> CpuTrackedDenseSet<T>
{
    pub fn new(
        renderer: Arc<Renderer>,
        initial_gpu_buffer_len_elements: usize,
        name: String,
        buffer_usage: wgpu::BufferUsages
    ) -> Self
    {
        CpuTrackedDenseSet {
            gpu_buffer: crate::CpuTrackedBuffer::new(
                renderer,
                initial_gpu_buffer_len_elements,
                name,
                buffer_usage
            ),
            dense_set:  util::DenseSet::new()
        }
    }

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

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        self.gpu_buffer.get_buffer(buf_access_func)
    }

    #[must_use]
    // returns whether or not a resize of the gpu-internal buffer ocurred
    pub fn replicate_to_gpu(&mut self) -> bool
    {
        let dense_set_data = self.dense_set.to_dense_elements();

        if dense_set_data.len() > self.gpu_buffer.get_cpu_len()
        {
            self.gpu_buffer.realloc(dense_set_data.len() * 2);
        }

        dense_set_data
            .iter()
            .enumerate()
            .for_each(|(idx, t)| self.gpu_buffer.write_eq_testing(idx, *t));

        self.gpu_buffer.replicate_to_gpu()
    }
}

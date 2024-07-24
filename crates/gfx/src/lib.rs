#![feature(hasher_prefixfree_extras)]
#![feature(map_try_insert)]
#![feature(get_many_mut)]
#![feature(new_uninit)]
#![allow(clippy::type_complexity)]

mod cpu_tracked_buffer;
mod cpu_tracked_dense_set;
mod input_manager;
mod linalg;
mod recordables;
mod render_cache;
mod renderer;
mod screen_sized_texture;

pub use cpu_tracked_buffer::CpuTrackedBuffer;
pub use input_manager::*;
pub use linalg::*;
pub use recordables::{DrawId, PassStage, RecordInfo, Recordable, RenderPassId};
pub use render_cache::{
    CacheableComputePipelineDescriptor,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor,
    GenericPass,
    GenericPipeline
};
pub use renderer::{EncoderToPassFn, RenderPassSendFunction, Renderer};
pub use screen_sized_texture::*;
pub use wgpu;
pub use winit::keyboard::KeyCode;

pub mod glm
{
    pub use nalgebra::UnitQuaternion;
    pub use nalgebra_glm::*;
}
use std::ops::RangeInclusive;

pub use cpu_tracked_dense_set::CpuTrackedDenseSet;
pub use nalgebra as nal;

#[macro_export]
macro_rules! include_many_wgsl {
    ($first:expr, $($rest:expr),*) => {
        {
            wgpu::ShaderModuleDescriptor {
                label: Some(concat!($first)),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(include_str!($first), "\n", $(include_str!($rest), "\n",)*).into()
                ),
            }
        }
    };
}

pub(crate) fn combine_into_ranges(
    mut points: Vec<u64>,
    max_distance: u64,
    max_ranges: usize
) -> Vec<RangeInclusive<u64>>
{
    if points.is_empty()
    {
        return vec![];
    }

    points.sort_unstable();

    let mut ranges = Vec::with_capacity(points.len());
    let mut start = unsafe { *points.get_unchecked(0) };
    let mut end = start;

    let mut points_ptr = points.as_ptr();
    let points_end_ptr = unsafe { points_ptr.add(points.len()) };

    unsafe {
        points_ptr = points_ptr.add(1);

        while points_ptr != points_end_ptr
        {
            let point = *points_ptr;
            if point - end <= max_distance
            {
                end = point;
            }
            else
            {
                ranges.push(start..=end);
                start = point;
                end = point;
            }
            points_ptr = points_ptr.add(1);
        }

        ranges.push(start..=end);

        if ranges.len() > max_ranges
        {
            while ranges.len() > max_ranges
            {
                let mut min_distance = u64::MAX;
                let mut min_index = 0;

                for i in 0..ranges.len() - 1
                {
                    let distance =
                        *ranges.get_unchecked(i + 1).start() - *ranges.get_unchecked(i).end();
                    if distance < min_distance
                    {
                        min_distance = distance;
                        min_index = i;
                    }
                }

                let merged_range = *ranges.get_unchecked(min_index).start()
                    ..=*ranges.get_unchecked(min_index + 1).end();
                *ranges.get_unchecked_mut(min_index) = merged_range;
                ranges.remove(min_index + 1);
            }

            ranges.set_len(max_ranges);
        }
    }

    ranges
}

#[test]
fn ranges()
{
    let points = vec![0, 1, 2, 5, 6, 7, 89, 87, 82, 92];
    let max_distance = 5;
    let max_ranges = 128;
    let ranges = combine_into_ranges(points, max_distance, max_ranges);

    assert_eq!(ranges[0], 0..=7);
    assert_eq!(ranges[1], 82..=92);
    assert_eq!(ranges.len(), 2);
}

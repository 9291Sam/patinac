#![feature(lazy_cell)]
#![feature(hasher_prefixfree_extras)]
#![feature(map_try_insert)]
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

pub(crate) fn combine_into_ranges(
    points: Vec<u64>,
    max_distance: u64,
    max_ranges: usize
) -> Vec<RangeInclusive<u64>>
{
    if points.is_empty()
    {
        return vec![];
    }

    let mut points = points;
    points.sort();

    let mut ranges = Vec::new();
    let mut start = points[0];
    let mut end = points[0];

    for &point in &points[1..]
    {
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
    }

    ranges.push(start..=end);

    // Post-processing to merge ranges if there are more than `max_ranges` ranges
    while ranges.len() > max_ranges
    {
        let mut min_distance = u64::MAX;
        let mut min_index = 0;

        for i in 0..ranges.len() - 1
        {
            let distance = *ranges[i + 1].start() - *ranges[i].end();
            if distance < min_distance
            {
                min_distance = distance;
                min_index = i;
            }
        }

        let merged_range = *ranges[min_index].start()..=*ranges[min_index + 1].end();
        ranges[min_index] = merged_range;
        ranges.remove(min_index + 1);
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

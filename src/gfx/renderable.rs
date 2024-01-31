use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use strum::{EnumIter, IntoEnumIterator};

pub trait Renderable: Send + Sync + Sealed
{
    // fn get_pass_type(&self) -> PassStage;
    // fn get_pipeline_type(&self) -> PipelineType;
    // fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4];
    // fn get_bind_group_ids(&self) -> [Option<NonZeroU64>; 4]
    // {
    //     // TODO: replace with wgpu::Id<wgpu::BindGroup>'s Ord impl once that gets
    //     // stabilized
    //     std::array::from_fn(|i| {
    //         self.get_bind_groups()[i]
    //             .map(|group| NonZeroU64::new(group.global_id().inner()).unwrap())
    //     })
    // }

    // fn should_draw(&self) -> bool;

    // fn ord(&self, other: &impl Renderable) -> Ordering
    // {
    //     Equal
    //         .then(self.get_pass_type().cmp(&other.get_pass_type()))
    //         .then(self.get_pipeline_type().cmp(&other.get_pipeline_type()))
    //         .then(self.get_bind_group_ids().cmp(&other.get_bind_group_ids()))
    // }

    // fn bind_and_draw(&self, render_pass: &mut wgpu::RenderPass);
}

trait Sealed {}
impl<T> Sealed for Arc<T> {}

struct Bar {}

impl Bar
{
    fn new() -> Arc<Bar>
    {
        Arc::new(Bar {})
    }
}

impl Renderable for Arc<Bar> {}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PassStage
{
    GraphicsSimpleColor
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PipelineType
{
    GraphicsFlat
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum BindGroupType
{
    GlobalCamera
}

pub struct RenderCache
{
    bind_group_layout_cache: HashMap<BindGroupType, wgpu::BindGroupLayout>,
    pipeline_layout_cache:   HashMap<PipelineType, wgpu::PipelineLayout>,

    render_pipeline_cache:  HashMap<PipelineType, wgpu::RenderPipeline>,
    compute_pipeline_cache: HashMap<PipelineType, wgpu::ComputePipeline>
}

impl RenderCache
{
    pub fn new(device: &wgpu::Device) -> Self
    {
        let bind_group_layout_cache = BindGroupType::iter()
            .map(|bind_group_type| {
                let new_bind_group_layout = match bind_group_type
                {
                    BindGroupType::GlobalCamera =>
                    {
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label:   Some("GlobalCamera"),
                            entries: &[]
                        })
                    }
                };

                (bind_group_type, new_bind_group_layout)
            })
            .collect();

        let pipeline_layout_cache = PipelineType::iter()
            .map(|pipeline_type| {
                let new_pipeline_layout = match pipeline_type
                {
                    PipelineType::GraphicsFlat =>
                    {
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:                Some("GraphicsFlat"),
                            bind_group_layouts:   &[],
                            push_constant_ranges: &[]
                        })
                    }
                };

                (pipeline_type, new_pipeline_layout)
            })
            .collect();

        let render_pipeline_cache = PipelineType::iter()
            .filter_map(|pipeline_type| {
                let maybe_new_pipeline = match pipeline_type
                {
                    PipelineType::GraphicsFlat =>
                    {
                        Some(
                            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label:         todo!(),
                                layout:        todo!(),
                                vertex:        todo!(),
                                primitive:     todo!(),
                                depth_stencil: todo!(),
                                multisample:   todo!(),
                                fragment:      todo!(),
                                multiview:     todo!()
                            })
                        )
                    }
                };

                maybe_new_pipeline.map(|p| (pipeline_type, p))
            })
            .collect();
        let compute_pipeline_cache = HashMap::new();

        RenderCache {
            bind_group_layout_cache,
            pipeline_layout_cache,
            render_pipeline_cache,
            compute_pipeline_cache
        }
    }

    pub fn lookup_render_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline
    {
        self.render_pipeline_cache.get(&pipeline_type).unwrap()
    }
}

impl PipelineType
{
    pub fn classify(&self) -> PipelinePass
    {
        match *self
        {
            PipelineType::GraphicsFlat => PipelinePass::Render
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum PipelinePass
{
    Compute,
    Render
}

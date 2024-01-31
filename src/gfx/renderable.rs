use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use strum::EnumIter;

pub trait Renderable
{
    fn get_pass_type(&self) -> PassStage;
    fn get_pipeline_type(&self) -> PipelineType;
    fn get_bind_groups(&self) -> [&'_ wgpu::BindGroup; 4];
    fn get_bind_group_ids(&self) -> [NonZeroU64; 4]
    {
        std::array::from_fn(|i| {
            NonZeroU64::new(self.get_bind_groups()[i].global_id().inner()).unwrap()
        })
    }

    fn should_draw(&self) -> bool;

    fn ord(&self, other: &impl Renderable) -> Ordering
    {
        Equal
            .then(self.get_pass_type().cmp(&other.get_pass_type()))
            .then(self.get_pipeline_type().cmp(&other.get_pipeline_type()))
            .then(self.get_bind_group_ids().cmp(&other.get_bind_group_ids()))
    }

    fn bind_and_draw(&self, render_pass: &mut wgpu::RenderPass);
}

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
    GlobalCamera,
    Custom
}

pub enum ComputeOrRenderPipeline
{
    Compute(wgpu::ComputePipeline),
    Render(wgpu::RenderPipeline)
}

pub struct RenderCache
{
    pipeline_layout_cache: HashMap<PipelineType, wgpu::PipelineLayout>,

    graphics_pipeline_cache: HashMap<PipelineType, wgpu::RenderPipeline>
}

impl RenderCache
{
    // pub fn new() -> Self {}

    pub fn lookup_graphics_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline
    {
        &self.graphics_pipeline_cache.get(&pipeline_type).unwrap()
    }
}

use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use strum::EnumIter;

use crate::render_cache::GenericPass;
use crate::{Camera, GenericPipeline, Renderer, Transform};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct DrawId(pub(crate) u32);

impl Deref for DrawId
{
    type Target = u32;

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct RenderPassId(pub util::Uuid);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PassStage
{
    VoxelDiscovery,
    PostVoxelDiscoveryCompute,
    VoxelColorTransfer,
    SimpleColor,
    MenuRender
}

pub trait Recordable: Debug + Send + Sync
{
    /// Your state functions
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;

    /// Called for all registered Recordable s
    fn pre_record_update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        camera: &Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> RecordInfo;

    fn record<'s>(&'s self, render_pass: &mut GenericPass<'s>, maybe_id: Option<DrawId>);
}

pub enum RecordInfo
{
    NoRecord,
    Record
    {
        render_pass: RenderPassId,
        pipeline:    Arc<GenericPipeline>,
        bind_groups: [Option<Arc<wgpu::BindGroup>>; 4],
        transform:   Option<Transform>
    },
    RecordIsolated
    {
        render_pass: RenderPassId
    }
}

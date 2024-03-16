use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::Arc;

use strum::EnumIter;

use crate::render_cache::GenericPass;
use crate::{Camera, GenericPipeline, Renderer, Transform};

pub type DrawId = u32;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PassStage
{
    GraphicsSimpleColor,
    MenuRender
}

pub trait Recordable: Debug + Send + Sync
{
    /// Your state functions
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;
    fn get_pass_stage(&self) -> PassStage;
    fn get_pipeline(&self) -> Option<&GenericPipeline>;

    /// Called for all registered Recordable s
    fn pre_record_update(
        &self,
        renderer: &Renderer,
        camera: &Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> RecordInfo;

    fn record<'s>(&'s self, render_pass: &mut GenericPass<'s>, maybe_id: Option<DrawId>);
}

pub(crate) fn recordable_ord(
    this: &dyn Recordable,
    other: &dyn Recordable,
    this_bind_groups: &[Option<Arc<wgpu::BindGroup>>; 4],
    other_bind_groups: &[Option<Arc<wgpu::BindGroup>>; 4]
) -> Ordering
{
    Equal
        .then(this.get_pass_stage().cmp(&other.get_pass_stage()))
        .then(this.get_pipeline().cmp(&other.get_pipeline()))
        .then(get_bind_group_ids(this_bind_groups).cmp(&get_bind_group_ids(other_bind_groups)))
}

#[derive(Default)]
pub struct RecordInfo
{
    pub should_draw: bool,
    pub transform:   Option<Transform>,
    pub bind_groups: [Option<Arc<wgpu::BindGroup>>; 4]
}

fn get_bind_group_ids(bind_groups: &[Option<Arc<wgpu::BindGroup>>; 4]) -> [Option<NonZeroU64>; 4]
{
    std::array::from_fn(|i| {
        bind_groups[i]
            .as_ref()
            .map(|group| NonZeroU64::new(group.global_id().inner()).unwrap())
    })
}

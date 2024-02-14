pub mod flat_textured;
pub mod lit_textured;
pub mod parallax_raymarched;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::fmt::Debug;
use std::num::NonZeroU64;

pub type DrawId = u32;

pub trait Recordable: Debug + Send + Sync
{
    /// Your state functions
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;
    fn get_pass_stage(&self) -> super::PassStage;
    fn get_pipeline_type(&self) -> super::PipelineType;

    /// Called for all registered Recordable s
    fn pre_record_update(&self, renderer: &crate::Renderer, camera: &crate::Camera) -> RecordInfo;

    fn get_bind_groups<'s>(
        &'s self,
        global_bind_group: &'s wgpu::BindGroup
    ) -> [Option<&'s wgpu::BindGroup>; 4];

    fn record<'s>(&'s self, render_pass: &mut super::GenericPass<'s>, maybe_id: Option<DrawId>);

    fn ord(&self, other: &dyn Recordable, global_bind_group: &wgpu::BindGroup) -> Ordering
    {
        Equal
            .then(self.get_pass_stage().cmp(&other.get_pass_stage()))
            .then(self.get_pipeline_type().cmp(&other.get_pipeline_type()))
            .then(
                get_bind_group_ids(&self.get_bind_groups(global_bind_group)).cmp(
                    &get_bind_group_ids(&other.get_bind_groups(global_bind_group))
                )
            )
    }
}

pub struct RecordInfo
{
    pub should_draw: bool,
    pub transform:   Option<crate::Transform>
}

fn get_bind_group_ids(bind_groups: &[Option<&'_ wgpu::BindGroup>; 4]) -> [Option<NonZeroU64>; 4]
{
    std::array::from_fn(|i| {
        bind_groups[i].map(|group| NonZeroU64::new(group.global_id().inner()).unwrap())
    })
}

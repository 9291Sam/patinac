pub mod flat_textured;
pub mod lit_textured;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::fmt::Debug;
use std::num::NonZeroU64;

use strum::EnumIter;

use crate::renderer::{GenericPass, GenericPipeline};
use crate::{Camera, Renderer, Transform};

pub type DrawId = u32;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PassStage
{
    GraphicsSimpleColor
}

pub trait Recordable: Debug + Send + Sync
{
    /// Your state functions
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;
    fn get_pass_stage(&self) -> PassStage;
    fn get_pipeline(&self) -> &GenericPipeline;

    /// Called for all registered Recordable s
    fn pre_record_update(&self, renderer: &Renderer, camera: &Camera) -> RecordInfo;

    fn get_bind_groups<'s>(
        &'s self,
        global_bind_group: &'s wgpu::BindGroup
    ) -> [Option<&'s wgpu::BindGroup>; 4];

    fn record<'s>(&'s self, render_pass: &mut GenericPass<'s>, maybe_id: Option<DrawId>);

    fn ord(&self, other: &dyn Recordable, global_bind_group: &wgpu::BindGroup) -> Ordering
    {
        Equal
            .then(self.get_pass_stage().cmp(&other.get_pass_stage()))
            .then(self.get_pipeline().cmp(&&other.get_pipeline()))
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
    pub transform:   Option<Transform>
}

fn get_bind_group_ids(bind_groups: &[Option<&'_ wgpu::BindGroup>; 4]) -> [Option<NonZeroU64>; 4]
{
    std::array::from_fn(|i| {
        bind_groups[i].map(|group| NonZeroU64::new(group.global_id().inner()).unwrap())
    })
}

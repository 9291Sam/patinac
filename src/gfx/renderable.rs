use std::cmp::Ordering;
use std::cmp::Ordering::*;
use std::fmt::Debug;
use std::num::NonZeroU64;

pub trait Renderable: Debug + Send + Sync
{
    fn get_pass_stage(&self) -> super::PassStage;
    fn get_pipeline_type(&self) -> super::PipelineType;
    fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4];
    fn get_bind_group_ids(&self) -> [Option<NonZeroU64>; 4]
    {
        // TODO: replace with wgpu::Id<wgpu::BindGroup>'s Ord impl once that gets
        // stabilized
        std::array::from_fn(|i| {
            self.get_bind_groups()[i]
                .map(|group| NonZeroU64::new(group.global_id().inner()).unwrap())
        })
    }

    fn should_render(&self) -> bool;

    fn ord(&self, other: &dyn Renderable) -> Ordering
    {
        Equal
            .then(self.get_pass_stage().cmp(&other.get_pass_stage()))
            .then(self.get_pipeline_type().cmp(&other.get_pipeline_type()))
            .then(self.get_bind_group_ids().cmp(&other.get_bind_group_ids()))
    }

    /// Pipeline is already bound
    fn bind_and_draw<'s>(
        &'s self,
        render_pass: &mut wgpu::RenderPass<'s>,
        renderer: &super::Renderer,
        camera: &super::Camera
    );
}

// trait Sealed {}
// impl<T> Sealed for Arc<T> {}

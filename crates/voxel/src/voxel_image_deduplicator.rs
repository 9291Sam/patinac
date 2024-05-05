use std::borrow::Cow;

struct VoxelImageDeduplicator
{
    uuid: util::Uuid
}

impl gfx::Recordable for VoxelImageDeduplicator
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Voxel Image Deduplicator")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record { render_pass: (), pipeline: (), bind_groups: (), transform: () }{}
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        todo!()
    }
}

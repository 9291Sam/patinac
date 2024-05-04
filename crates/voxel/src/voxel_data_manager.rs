use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use gfx::wgpu;

use crate::VoxelColorTransferRecordable;

pub struct VoxelWorldDataManager
{
    game:          Arc<game::Game>,
    uuid:          util::Uuid,
    resize_pinger: util::PingReceiver,

    // color_transfer_bind_group_windows: (
    //     util::Window<Arc<wgpu::BindGroup>>,
    //     util::WindowUpdater<Arc<wgpu::BindGroup>>
    // ),
    color_transfer_recordable: Arc<super::VoxelColorTransferRecordable>
}

impl Debug for VoxelWorldDataManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Voxel World Data Manager")
    }
}

impl VoxelWorldDataManager
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        Arc::new(VoxelWorldDataManager {
            game:                      game.clone(),
            // color_transfer_bindgroup_windows: todo!(),
            color_transfer_recordable: VoxelColorTransferRecordable::new(game.clone()),
            uuid:                      util::Uuid::new(),
            resize_pinger:             game.get_renderer().get_resize_pinger()
            // color_transfer_bind_group_windows: util::Window::n
        })
    }

    pub fn create_new_chunk() {}

    fn generate_bind_group(rec: &VoxelColorTransferRecordable) {}
}

impl game::EntityCastDepot for VoxelWorldDataManager
{
    fn as_entity(&self) -> Option<&dyn game::Entity>
    {
        todo!()
    }

    fn as_positionable(&self) -> Option<&dyn game::Positionable>
    {
        todo!()
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        todo!()
    }
}

impl game::Entity for VoxelWorldDataManager
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("VoxelWorldDataManager")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn tick(&self, game: &game::Game, _: game::TickTag)
    {
        if self.resize_pinger.recv_all()
        {
            // self.color_transfer_bind_group_windows.1.update(todo!())
        }
    }
}

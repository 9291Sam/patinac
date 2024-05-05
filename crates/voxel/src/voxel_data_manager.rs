use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use gfx::wgpu;

use crate::{VoxelColorTransferRecordable, VoxelImageDeduplicator};

// Stages:
// VoxelDiscovery            | rendering all chunks
// PostVoxelDiscoveryCompute | deduplication + rt
// VoxelColorTransfer        | recoalesce
pub struct VoxelWorldDataManager
{
    game:          Arc<game::Game>,
    uuid:          util::Uuid,
    resize_pinger: util::PingReceiver,

    // storage bufferss for data
    // WHACK ASS IDEA:
    // in the sets, store the voxel's index in the global brick map set
    // this means that you can have 2^32 voxel on screen at once
    voxel_lighting_bind_group: (
        util::Window<Arc<wgpu::BindGroup>>,
        util::WindowUpdater<Arc<wgpu::BindGroup>>
    ),
    voxel_lighting_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // set of chunks
    duplicator_recordable:     Arc<super::VoxelImageDeduplicator>,
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
        let transfer_layout = game.get_renderer().render_cache.cache_bind_group_layout(
            wgpu::BindGroupLayoutDescriptor {
                label:   Some("Global Discovery Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false
                    },
                    count:      None
                }]
            }
        );

        let color_transfer_bind_group =
            Self::generate_discovery_bind_group(&game, &transfer_layout);

        let (window, updater) = util::Window::new(color_transfer_bind_group.clone());

        let this = Arc::new(VoxelWorldDataManager {
            game:                             game.clone(),
            uuid:                             util::Uuid::new(),
            resize_pinger:                    game.get_renderer().get_resize_pinger(),
            voxel_lighting_bind_group:        (window.clone(), updater),
            voxel_lighting_bind_group_layout: transfer_layout.clone(),
            duplicator_recordable:            VoxelImageDeduplicator::new(
                game.clone(),
                transfer_layout.clone(),
                window.clone()
            ),
            color_transfer_recordable:        VoxelColorTransferRecordable::new(
                game.clone(),
                transfer_layout,
                window
            )
        });

        game.register(this.clone());

        this
    }

    fn generate_discovery_bind_group(
        game: &game::Game,
        color_transfer_bind_group_layout: &wgpu::BindGroupLayout
    ) -> Arc<wgpu::BindGroup>
    {
        Arc::new(
            game.get_renderer()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label:   Some("Global Discovery Bind Group"),
                    layout:  color_transfer_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::TextureView(
                            &game
                                .get_renderpass_manager()
                                .get_voxel_discovery_texture()
                                .get_view()
                        )
                    }]
                })
        )
    }
}

impl game::EntityCastDepot for VoxelWorldDataManager
{
    fn as_entity(&self) -> Option<&dyn game::Entity>
    {
        Some(self)
    }

    fn as_positionable(&self) -> Option<&dyn game::Positionable>
    {
        None
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
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
            self.voxel_lighting_bind_group
                .1
                .update(Self::generate_discovery_bind_group(
                    game,
                    &self.voxel_lighting_bind_group_layout
                ))
        }
    }
}

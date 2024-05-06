use std::borrow::Cow;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use bytemuck::bytes_of;
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt, DownloadBuffer};
use gfx::{glm, wgpu};

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

    voxel_lighting_buffers_critical_section: Mutex<Option<VoxelLightingBuffersCriticalSection>>,
    // storage buffers for data
    // WHACK ASS IDEA:
    // in the sets, store the voxel's index in the global brick map set
    // this means that you can have 2^32 voxel on screen at once
    voxel_lighting_bind_group: (
        util::Window<Arc<wgpu::BindGroup>>,
        util::WindowUpdater<Arc<wgpu::BindGroup>>
    ),
    voxel_lighting_bind_group_layout:        Arc<wgpu::BindGroupLayout>,

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
                entries: const {
                    &[
                        wgpu::BindGroupLayoutEntry {
                            binding:    0,
                            visibility: wgpu::ShaderStages::FRAGMENT
                                .union(wgpu::ShaderStages::COMPUTE),
                            ty:         wgpu::BindingType::Texture {
                                sample_type:    wgpu::TextureSampleType::Uint,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled:   false
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: false
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   NonZero::new(12)
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: true
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: false
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: false
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: false
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        }
                    ]
                }
            }
        );

        let (voxel_lighting_bind_group, buffers) =
            Self::generate_voxel_lighting_bind_group(&game, &transfer_layout, None);

        let (window, updater) = util::Window::new(voxel_lighting_bind_group);

        let this = Arc::new(VoxelWorldDataManager {
            game: game.clone(),
            uuid: util::Uuid::new(),
            resize_pinger: game.get_renderer().get_resize_pinger(),
            voxel_lighting_buffers_critical_section: Mutex::new(Some(buffers)),
            voxel_lighting_bind_group: (window.clone(), updater),
            voxel_lighting_bind_group_layout: transfer_layout.clone(),
            duplicator_recordable: VoxelImageDeduplicator::new(
                game.clone(),
                transfer_layout.clone(),
                window.clone()
            ),
            color_transfer_recordable: VoxelColorTransferRecordable::new(
                game.clone(),
                transfer_layout,
                window
            )
        });

        game.get_renderer().register(this.clone());

        this
    }

    fn generate_voxel_lighting_bind_group(
        game: &game::Game,
        voxel_lighting_bind_group_layout: &wgpu::BindGroupLayout,
        maybe_old_buffers: Option<VoxelLightingBuffersCriticalSection>
    ) -> (Arc<wgpu::BindGroup>, VoxelLightingBuffersCriticalSection)
    {
        let renderer = game.get_renderer().clone();
        let size = renderer.get_framebuffer_size();

        let elements: u32 = size.x * size.y;

        let buffers = if let Some(old_buffers) = maybe_old_buffers
        {
            let VoxelLightingBuffersCriticalSection {
                indirect_rt_workgroups_buffer,
                storage_set_len_buffer,
                storage_set_buffer: _,
                unique_voxel_len_buffer,
                unique_voxel_buffer: _
            } = old_buffers;

            VoxelLightingBuffersCriticalSection {
                indirect_rt_workgroups_buffer,
                storage_set_len_buffer,
                storage_set_buffer: renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Storage Set Buffer"),
                    size:               elements as u64 * std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false
                }),
                unique_voxel_len_buffer,
                // TODO: remove copy_src
                unique_voxel_buffer: renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Unique Voxel Buffer"),
                    size:               elements as u64 * std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false
                })
            }
        }
        else
        {
            VoxelLightingBuffersCriticalSection {
                indirect_rt_workgroups_buffer: renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Indirect RT Workgroups Buffer"),
                    size:               std::mem::size_of::<glm::U32Vec3>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                storage_set_len_buffer:        renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Set Storage Len Buffer"),
                    size:               std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                storage_set_buffer:            renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Storage Set Buffer"),
                    size:               elements as u64 * std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false
                }),
                unique_voxel_len_buffer:       renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Unique Voxel Len Buffer"),
                    size:               std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                unique_voxel_buffer:           renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Unique Voxel Buffer"),
                    size:               elements as u64 * std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false
                })
            }
        };

        // write storage_set_len
        renderer
            .queue
            .write_buffer(&buffers.storage_set_len_buffer, 0, bytes_of(&elements));

        (
            Arc::new(
                game.get_renderer()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label:   Some("Voxel Lighting Bind Group"),
                        layout:  voxel_lighting_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding:  0,
                                resource: wgpu::BindingResource::TextureView(
                                    &game
                                        .get_renderpass_manager()
                                        .get_voxel_discovery_texture()
                                        .get_view()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  1,
                                resource: wgpu::BindingResource::Buffer(
                                    buffers
                                        .indirect_rt_workgroups_buffer
                                        .as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  2,
                                resource: wgpu::BindingResource::Buffer(
                                    buffers.storage_set_len_buffer.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  3,
                                resource: wgpu::BindingResource::Buffer(
                                    buffers.storage_set_buffer.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  4,
                                resource: wgpu::BindingResource::Buffer(
                                    buffers.unique_voxel_len_buffer.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  5,
                                resource: wgpu::BindingResource::Buffer(
                                    buffers.unique_voxel_buffer.as_entire_buffer_binding()
                                )
                            }
                        ]
                    })
            ),
            buffers
        )
    }
}

impl gfx::Recordable for VoxelWorldDataManager
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("VoxelWorldDataManager")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        _: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let mut buffers = self.voxel_lighting_buffers_critical_section.lock().unwrap();

        if self.resize_pinger.recv_all()
        {
            let (new_group, new_buffers) = Self::generate_voxel_lighting_bind_group(
                &self.game,
                &self.voxel_lighting_bind_group_layout,
                buffers.take()
            );

            self.voxel_lighting_bind_group.1.update(new_group);

            *buffers = Some(new_buffers);
        }

        if let Some(buffers) = &mut *buffers
        {
            static ITERS: AtomicU32 = AtomicU32::new(0);

            DownloadBuffer::read_buffer(
                &renderer.device,
                &renderer.queue,
                &buffers.storage_set_buffer.slice(..),
                |res| {
                    let data: &[u8] = &res.unwrap();
                    let u32_data: &[u32] = bytemuck::cast_slice(data);

                    if ITERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 2000
                    {
                        let mut string = format!("{:?}", &u32_data[0..]);
                        string.remove_matches("4294967295, ");

                        let elems = log::trace!("{:?}", u32_data);

                        // panic!("op");
                    }
                }
            );

            renderer
                .queue
                .write_buffer(&buffers.indirect_rt_workgroups_buffer, 0, &[0; 12]);

            renderer
                .queue
                .write_buffer(&buffers.unique_voxel_len_buffer, 0, &[0; 4]);
        }
        else
        {
            unreachable!()
        }

        gfx::RecordInfo::NoRecord {}
    }

    fn record<'s>(&'s self, _: &mut gfx::GenericPass<'s>, _: Option<gfx::DrawId>) {}
}

struct VoxelLightingBuffersCriticalSection
{
    indirect_rt_workgroups_buffer: wgpu::Buffer,
    storage_set_len_buffer:        wgpu::Buffer,
    storage_set_buffer:            wgpu::Buffer,
    unique_voxel_len_buffer:       wgpu::Buffer,
    unique_voxel_buffer:           wgpu::Buffer
}

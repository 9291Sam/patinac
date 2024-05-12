use std::borrow::Cow;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex, Weak};

use bytemuck::{cast_slice, Contiguous};
use gfx::wgpu::util::DownloadBuffer;
use gfx::{glm, wgpu};
use itertools::Itertools;
use rand::Rng;

use crate::{FaceId, FaceInfo, VisibilityMarker, VisibilityUnMarker, VoxelColorTransferRecordable};

const TEMPORARY_FACE_ID_LIMIT: u64 = 2u64.pow(24);

/// The Pipeline
/// chunks render into Image<u64>
/// compute pass to mark visible faces and list their ids into a Buffer<FaceId>
/// compute pass over Buffer<FaceId> calculate colors && reset visibility
/// Raster pass over Image<u64> -> Screen Texture

pub struct VoxelChunkManager
{
    uuid: util::Uuid,
    game: Arc<game::Game>,
    this: Weak<VoxelChunkManager>,

    face_id_allocator:            Mutex<util::FreelistAllocator>,
    buffer_critical_section:      Mutex<Option<BufferCriticalSection>>,
    voxel_data_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    bind_group_windows: (
        util::Window<Arc<wgpu::BindGroup>>,
        util::WindowUpdater<Arc<wgpu::BindGroup>>
    ),
    resize_pinger:                util::PingReceiver,

    visibility_marker:   Arc<VisibilityMarker>,
    color_transfer:      Arc<VoxelColorTransferRecordable>,
    visibility_unmarker: Arc<VisibilityUnMarker>
}

impl Debug for VoxelChunkManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "VoxelChunkManager")
    }
}

impl VoxelChunkManager
{
    pub fn new(game: Arc<game::Game>) -> Arc<VoxelChunkManager>
    {
        let bind_group_layout = game.get_renderer().render_cache.cache_bind_group_layout(
            wgpu::BindGroupLayoutDescriptor {
                label:   Some("VoxelData Bind Group Layout"),
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
                            visibility: wgpu::ShaderStages::FRAGMENT
                                .union(wgpu::ShaderStages::COMPUTE),
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
                        }
                    ]
                }
            }
        );

        let (voxel_data_bind_group, buffers) =
            Self::generate_voxel_bind_group(&game, &bind_group_layout, None);

        let (bind_group_window, bind_group_window_updater) =
            util::Window::new(voxel_data_bind_group);

        let this = Arc::new_cyclic(|weak_this| {
            VoxelChunkManager {
                uuid:                         util::Uuid::new(),
                game:                         game.clone(),
                this:                         weak_this.clone(),
                face_id_allocator:            Mutex::new(util::FreelistAllocator::new(
                    (NonZero::new(TEMPORARY_FACE_ID_LIMIT)
                        .unwrap()
                        .into_integer() as usize)
                        .try_into()
                        .unwrap()
                )),
                buffer_critical_section:      Mutex::new(Some(buffers)),
                voxel_data_bind_group_layout: bind_group_layout.clone(),
                bind_group_windows:           (
                    bind_group_window.clone(),
                    bind_group_window_updater
                ),
                resize_pinger:                game.get_renderer().get_resize_pinger(),
                visibility_marker:            VisibilityMarker::new(
                    game.clone(),
                    bind_group_layout.clone(),
                    bind_group_window.clone()
                ),
                color_transfer:               VoxelColorTransferRecordable::new(
                    game.clone(),
                    bind_group_layout.clone(),
                    bind_group_window.clone()
                ),
                visibility_unmarker:          VisibilityUnMarker::new(
                    game.clone(),
                    bind_group_layout.clone(),
                    bind_group_window.clone()
                )
            }
        });

        game.get_renderer().register(this.clone());

        this
    }

    pub fn get_bind_group_window(&self) -> util::Window<Arc<wgpu::BindGroup>>
    {
        self.bind_group_windows.0.clone()
    }

    pub(crate) unsafe fn alloc_face_id(&self) -> FaceId
    {
        FaceId(
            self.face_id_allocator
                .lock()
                .unwrap()
                .allocate()
                .expect("Tried to allocate too many FaceId")
                .into_integer() as u32
        )
    }

    pub(crate) unsafe fn alloc_many_face_id(&self, amount: usize) -> Vec<FaceId>
    {
        let mut allocator = self.face_id_allocator.lock().unwrap();

        (0..amount)
            .map(|_| {
                FaceId(
                    allocator
                        .allocate()
                        .expect("Tried to allocate too many FaceId")
                        .into_integer() as u32
                )
            })
            .collect_vec()
    }

    pub(crate) unsafe fn dealloc_face_id(&self, id: FaceId)
    {
        self.face_id_allocator
            .lock()
            .unwrap()
            .free(NonZero::new(id.0 as usize).unwrap())
    }

    pub(crate) unsafe fn dealloc_many_face_id(&self, id: impl IntoIterator<Item = FaceId>)
    {
        let mut allocator = self.face_id_allocator.lock().unwrap();

        id.into_iter()
            .for_each(|i| allocator.free(NonZero::new(i.0 as usize).unwrap()))
    }

    fn generate_voxel_bind_group(
        game: &game::Game,
        bind_group_layout: &wgpu::BindGroupLayout,
        maybe_old_buffers: Option<BufferCriticalSection>
    ) -> (Arc<wgpu::BindGroup>, BufferCriticalSection)
    {
        let buffers: BufferCriticalSection = if let Some(old_buffers) = maybe_old_buffers
        {
            let BufferCriticalSection {
                indirect_color_calc_buffer,
                face_id_buffer,
                number_of_unique_voxels_buffer,
                unique_voxel_buffer
            } = old_buffers;

            BufferCriticalSection {
                indirect_color_calc_buffer,
                face_id_buffer,
                number_of_unique_voxels_buffer,
                unique_voxel_buffer
            }
        }
        else
        {
            let renderer = game.get_renderer();
            let screen_size = renderer.get_framebuffer_size();
            let screen_px = screen_size.x * screen_size.y;

            BufferCriticalSection {
                indirect_color_calc_buffer:     renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("VoxelDataBindGroup IndirectColorCalcBuffer"),
                    size:               std::mem::size_of::<glm::U32Vec3>() as u64,
                    usage:              wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                face_id_buffer:                 renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("VoxelDataBindGroup FaceIdBuffer"),
                    size:               std::mem::size_of::<FaceInfo>() as u64
                        * TEMPORARY_FACE_ID_LIMIT,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                number_of_unique_voxels_buffer: renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("VoxelDataBindGroup NumberOfUniqueVoxelsBuffer"),
                    size:               std::mem::size_of::<u32>() as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                }),
                unique_voxel_buffer:            renderer.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("VoxelDataBindGroup UniqueVoxelBuffer"),
                    size:               std::mem::size_of::<FaceInfo>() as u64 * screen_px as u64,
                    usage:              wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false
                })
            }
        };

        let face_data = (0..TEMPORARY_FACE_ID_LIMIT)
            .map(|_| {
                FaceInfo {
                    low:  rand::thread_rng().gen_range(0..=12u32) << 16,
                    high: 0
                }
            })
            .collect_vec();

        game.get_renderer().queue.write_buffer(
            &buffers.face_id_buffer,
            0,
            cast_slice(&face_data[..])
        );

        let bind_group = game
            .get_renderer()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("VoxelData BindGroup"),
                layout:  bind_group_layout,
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
                                .indirect_color_calc_buffer
                                .as_entire_buffer_binding()
                        )
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: wgpu::BindingResource::Buffer(
                            buffers.face_id_buffer.as_entire_buffer_binding()
                        )
                    },
                    wgpu::BindGroupEntry {
                        binding:  3,
                        resource: wgpu::BindingResource::Buffer(
                            buffers
                                .number_of_unique_voxels_buffer
                                .as_entire_buffer_binding()
                        )
                    },
                    wgpu::BindGroupEntry {
                        binding:  4,
                        resource: wgpu::BindingResource::Buffer(
                            buffers.unique_voxel_buffer.as_entire_buffer_binding()
                        )
                    }
                ]
            });

        (Arc::new(bind_group), buffers)
    }
}

impl gfx::Recordable for VoxelChunkManager
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Voxel Chunk Manager")
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
        let mut buffers = self.buffer_critical_section.lock().unwrap();

        if self.resize_pinger.recv_all()
        {
            let (new_group, new_buffers) = Self::generate_voxel_bind_group(
                &self.game,
                &self.voxel_data_bind_group_layout,
                buffers.take()
            );

            self.bind_group_windows.1.update(new_group);

            *buffers = Some(new_buffers);
        }

        if let Some(buffers) = &mut *buffers
        {
            static ITERS: AtomicU32 = AtomicU32::new(0);

            DownloadBuffer::read_buffer(
                &renderer.device,
                &renderer.queue,
                &buffers.indirect_color_calc_buffer.slice(..),
                |res| {
                    let data: &[u8] = &res.unwrap();
                    let u32_data: &[u32] = bytemuck::cast_slice(data);

                    if ITERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0
                    {
                        let v = u32_data.to_owned();

                        log::trace!("{:?}", v);

                        // panic!("done");
                    }
                }
            );

            renderer
                .queue
                .write_buffer(&buffers.indirect_color_calc_buffer, 0, &[0; 12]);

            renderer
                .queue
                .write_buffer(&buffers.number_of_unique_voxels_buffer, 0, &[0; 4]);
        }
        else
        {
            unreachable!()
        }

        gfx::RecordInfo::NoRecord {}
    }

    fn record<'s>(&'s self, _: &mut gfx::GenericPass<'s>, _: Option<gfx::DrawId>)
    {
        unreachable!()
    }
}

struct BufferCriticalSection
{
    indirect_color_calc_buffer:     wgpu::Buffer,
    face_id_buffer:                 wgpu::Buffer,
    number_of_unique_voxels_buffer: wgpu::Buffer,
    unique_voxel_buffer:            wgpu::Buffer
}

use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, cast_slice};
use gfx::{glm, wgpu, CacheablePipelineLayoutDescriptor};

use crate::cpu::{self, VoxelFaceDirection};
use crate::{gpu, BufferAllocation, ChunkLocalPosition, SubAllocatedCpuTrackedBuffer};

pub struct ChunkManager
{
    game:               Arc<game::Game>,
    uuid:               util::Uuid,
    pipeline:           Arc<gfx::GenericPipeline>,
    indirect_buffer:    wgpu::Buffer,
    face_id_bind_group: Arc<wgpu::BindGroup>,

    global_face_storage: Pin<Box<SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>>>,
    chunk:               Mutex<Chunk>,

    number_of_indirect_calls_flushed: AtomicU32
}

unsafe impl Sync for ChunkManager {}
unsafe impl Send for ChunkManager {}

impl Debug for ChunkManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "ChunkManagerIndirect")
    }
}

impl ChunkManager
{
    pub fn new(game: Arc<game::Game>) -> Arc<ChunkManager>
    {
        let renderer = game.get_renderer().clone();

        let mut allocator = Box::pin(SubAllocatedCpuTrackedBuffer::new(
            renderer.clone(),
            780974,
            "ChunkFacesSubBuffer",
            wgpu::BufferUsages::STORAGE
        ));

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Storage {
                                read_only: true
                            },
                            has_dynamic_offset: false,
                            min_binding_size:   None
                        },
                        count:      None
                    }],
                    label:   Some("FaceIdBindGroupIndirectLayout")
                });

        let face_ids_bind_group =
            Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                layout:  &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: allocator.access_buffer(),
                        offset: 0,
                        size:   Some(allocator.get_buffer_size_bytes())
                    })
                }],
                label:   Some("FaceIdBindGroupIndirect")
            }));

        let layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("ChunkManagerIndirect PipelineLayout"),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout.clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..4u32
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("chunk_manager_indirect.wgsl"));

        let this = Arc::new(ChunkManager {
            chunk:                            Mutex::new(Chunk::new(&mut allocator)),
            global_face_storage:              allocator,
            game:                             game.clone(),
            uuid:                             util::Uuid::new(),
            face_id_bind_group:               face_ids_bind_group,
            pipeline:                         renderer.render_cache.cache_render_pipeline(
                gfx::CacheableRenderPipelineDescriptor {
                    label: "ChunkManager Pipeline Indirect".into(),
                    layout: Some(layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    fragment_state: Some(gfx::CacheableFragmentState {
                        module:                           shader,
                        entry_point:                      "fs_main".into(),
                        targets:                          vec![Some(wgpu::ColorTargetState {
                            format:     gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                            blend:      Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL
                        })],
                        constants:                        None,
                        zero_initialize_workgroup_memory: false
                    }),
                    primitive_state: wgpu::PrimitiveState {
                        topology:           wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Ccw,
                        cull_mode:          None,
                        polygon_mode:       wgpu::PolygonMode::Fill,
                        unclipped_depth:    false,
                        conservative:       false
                    },
                    depth_stencil_state: Some(gfx::Renderer::get_default_depth_state()),
                    multisample_state: wgpu::MultisampleState {
                        count:                     1,
                        mask:                      !0,
                        alpha_to_coverage_enabled: false
                    },
                    multiview: None,
                    vertex_specialization: None,
                    zero_initalize_vertex_workgroup_memory: false
                }
            ),
            indirect_buffer:                  renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("ChunkManagerIndirectChunkDirBuffer"),
                size:               std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64
                    * 65535,
                usage:              wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }),
            number_of_indirect_calls_flushed: AtomicU32::new(0)
        });

        renderer.register(this.clone());

        this
    }

    pub fn insert_voxel(&self, pos: ChunkLocalPosition)
    {
        self.chunk.lock().unwrap().insert_voxel(pos);
    }
}

impl gfx::Recordable for ChunkManager
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("ChunkManagerIndirect")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let mut r: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();

        self.chunk
            .lock()
            .unwrap()
            .get_draw_ranges()
            .into_iter()
            .filter_map(|f| f)
            .for_each(|v: (Range<u32>, VoxelFaceDirection)| {
                r.push(wgpu::util::DrawIndirectArgs {
                    vertex_count:   (v.0.end - v.0.start) * 6,
                    instance_count: 1,
                    first_vertex:   v.0.start,
                    first_instance: v.1 as u32
                })
            });

        log::trace!("indirect calls: {:?}", &r[..]);

        fn draw_args_as_bytes(args: &[wgpu::util::DrawIndirectArgs]) -> &[u8]
        {
            unsafe {
                std::slice::from_raw_parts(
                    args.as_ptr() as *const u8,
                    args.len() * std::mem::size_of::<wgpu::util::DrawIndirectArgs>()
                )
            }
        }

        self.number_of_indirect_calls_flushed
            .store(r.len() as u32, Ordering::SeqCst);

        renderer
            .queue
            .write_buffer(&self.indirect_buffer, 0, draw_args_as_bytes(&r[..]));

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.face_id_bind_group.clone()),
                None,
                None
            ],
            transform:   Some(gfx::Transform::new())
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.multi_draw_indirect(
            &self.indirect_buffer,
            0,
            self.number_of_indirect_calls_flushed.load(Ordering::SeqCst)
        );
    }
}

struct DirectionalFaceData
{
    owning_allocator: *mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>,
    dir:              cpu::VoxelFaceDirection,
    faces_allocation: BufferAllocation,
    faces_stored:     u32
}

impl DirectionalFaceData
{
    pub fn new(
        allocator: &mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>,
        dir: cpu::VoxelFaceDirection
    ) -> DirectionalFaceData
    {
        let alloc = allocator.allocate(96000);

        Self {
            owning_allocator: allocator as *mut _,
            dir,
            faces_allocation: alloc,
            faces_stored: 0
        }
    }

    pub fn insert_face(&mut self, face: gpu::VoxelFace)
    {
        if self.faces_allocation.get_length() > self.faces_stored
        {
            self.faces_stored += 1;

            unsafe {
                self.owning_allocator.as_mut_unchecked().write(
                    &self.faces_allocation,
                    self.faces_stored..(self.faces_stored + 1),
                    &[face]
                )
            }
        }
        else
        {
            panic!()
        }
    }
}

struct Chunk
{
    drawable_faces: [Option<DirectionalFaceData>; 6]
}

impl Chunk
{
    pub fn new(allocator: &mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>) -> Chunk
    {
        Chunk {
            drawable_faces: std::array::from_fn(|i| {
                Some(DirectionalFaceData::new(
                    allocator,
                    VoxelFaceDirection::try_from(i as u8).unwrap()
                ))
            })
        }
    }

    pub fn insert_voxel(&mut self, local_pos: ChunkLocalPosition)
    {
        for d in VoxelFaceDirection::iterate()
        {
            self.drawable_faces[d as usize]
                .as_mut()
                .unwrap()
                .insert_face(gpu::VoxelFace::new(local_pos, glm::U8Vec2::new(1, 1)));
        }
    }

    pub fn get_draw_ranges(&self) -> [Option<(Range<u32>, VoxelFaceDirection)>; 6]
    {
        std::array::from_fn(|i| {
            unsafe {
                self.drawable_faces.get_unchecked(i).as_ref().map(|d| {
                    let start = d.faces_allocation.to_global_valid_range().start;

                    (start..(start + d.faces_stored), d.dir)
                })
            }
        })
    }
}
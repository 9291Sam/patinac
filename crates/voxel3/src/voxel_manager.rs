use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, AnyBitPattern, NoUninit};
use gfx::wgpu::{self, include_wgsl};
use gfx::{glm, CacheablePipelineLayoutDescriptor, CacheableRenderPipelineDescriptor};

use crate::chunk_manager::{get_chunk_position_from_world, ChunkManager};
use crate::material::MaterialManager;
use crate::CpuTrackedDenseSet;

pub struct VoxelManager
{
    game:              Arc<game::Game>,
    pipeline:          Arc<gfx::GenericPipeline>,
    bind_group:        Mutex<Arc<wgpu::BindGroup>>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    uuid: util::Uuid,

    face_id_allocator: Mutex<util::FreelistAllocator>,
    face_id_buffer:    Mutex<Arc<super::CpuTrackedDenseSet<u32>>>,
    face_data_buffer:  gfx::CpuTrackedBuffer<GpuFaceData>,
    chunk_manager:     Mutex<ChunkManager>,
    material_manager:  MaterialManager
}

impl Debug for VoxelManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Voxel Manager")
    }
}

impl VoxelManager
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        const INITIAL_SIZE: usize = 1024;
        let renderer = game.get_renderer();

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Voxel Manager Bind Group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
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
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    1,
                            visibility: wgpu::ShaderStages::VERTEX,
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
                            binding:    2,
                            visibility: wgpu::ShaderStages::VERTEX,
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
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: true
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        }
                    ]
                });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("Voxel Manager Pipeline Layout"),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout.clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..4
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(include_wgsl!("voxel_manager.wgsl"));

        let id_buffer = CpuTrackedDenseSet::new(
            renderer.clone(),
            INITIAL_SIZE,
            String::from("Face Id Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let data_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            INITIAL_SIZE,
            String::from("Face Data Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let chunk_manager = ChunkManager::new(renderer.clone());

        let mat_manager = MaterialManager::new(renderer);

        let combined_bind_group = Self::generate_bind_group(
            &renderer,
            &bind_group_layout,
            &id_buffer,
            &data_buffer,
            &chunk_manager,
            &mat_manager
        );

        let this = Arc::new(VoxelManager {
            game:              game.clone(),
            pipeline:          renderer.render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label: Cow::Borrowed("Voxel Manager Pipeline"),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    vertex_specialization: None,
                    zero_initalize_vertex_workgroup_memory: false,
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
                        topology:           wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Cw,
                        // TODO: test disabling backface culling because you're doing it on the CPU
                        // side!
                        cull_mode:          Some(wgpu::Face::Back),
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
                    multiview: None
                }
            ),
            bind_group:        Mutex::new(combined_bind_group),
            bind_group_layout: bind_group_layout.clone(),
            uuid:              util::Uuid::new(),
            face_id_allocator: Mutex::new(util::FreelistAllocator::new(INITIAL_SIZE)),
            face_id_buffer:    Mutex::new(id_buffer),
            face_data_buffer:  data_buffer,
            chunk_manager:     Mutex::new(chunk_manager),
            material_manager:  mat_manager
        });

        renderer.register(this.clone());

        this
    }

    // no chunks for now, just one global chunk

    pub fn insert_face(&self, face: VoxelFace)
    {
        let mut face_id_allocator = self.face_id_allocator.lock().unwrap();

        let new_face_id = if let Ok(id) = face_id_allocator.allocate()
        {
            id
        }
        else
        {
            let realloc_size = face_id_allocator.get_total_blocks() * 3 / 2;
            face_id_allocator.extend_size(realloc_size);
            self.face_data_buffer.realloc(realloc_size);

            face_id_allocator.allocate().unwrap()
        };

        self.face_id_buffer
            .lock()
            .unwrap()
            .insert(new_face_id as u32);

        let (chunk_world_pos, face_in_chunk_pos) = get_chunk_position_from_world(face.position);

        self.face_data_buffer.write(
            new_face_id,
            GpuFaceData::new(
                face.material,
                self.chunk_manager
                    .lock()
                    .unwrap()
                    .get_or_insert_chunk(chunk_world_pos),
                face_in_chunk_pos,
                face.direction
            )
        );
    }

    fn generate_bind_group(
        renderer: &gfx::Renderer,
        bind_group_layout: &wgpu::BindGroupLayout,
        face_id_buffer: &super::CpuTrackedDenseSet<u32>,
        face_data_buffer: &gfx::CpuTrackedBuffer<GpuFaceData>,
        chunk_manager: &ChunkManager,
        material_manager: &MaterialManager
    ) -> Arc<wgpu::BindGroup>
    {
        face_id_buffer.get_buffer(|raw_id_buf| {
            face_data_buffer.get_buffer(|raw_data_buf| {
                chunk_manager.get_buffer(|raw_chunk_buf| {
                    let material_buffer = material_manager.get_material_buffer();

                    Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                        label:   Some("Voxel Manager Bind Group"),
                        layout:  bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding:  0,
                                resource: wgpu::BindingResource::Buffer(
                                    raw_id_buf.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  1,
                                resource: wgpu::BindingResource::Buffer(
                                    raw_data_buf.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  2,
                                resource: wgpu::BindingResource::Buffer(
                                    raw_chunk_buf.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  3,
                                resource: wgpu::BindingResource::Buffer(
                                    material_buffer.as_entire_buffer_binding()
                                )
                            }
                        ]
                    }))
                })
            })
        })
    }
}

impl gfx::Recordable for VoxelManager
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Voxel Manager")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let mut needs_resize = false;

        let face_id_buffer = self.face_id_buffer.lock().unwrap();
        let chunk_manager = self.chunk_manager.lock().unwrap();

        needs_resize |= face_id_buffer.flush_to_gpu();
        needs_resize |= self.face_data_buffer.replicate_to_gpu();
        needs_resize |= chunk_manager.replicate_to_gpu();

        let mut bind_group = self.bind_group.lock().unwrap();

        if needs_resize
        {
            *bind_group = Self::generate_bind_group(
                renderer,
                &self.bind_group_layout,
                &face_id_buffer,
                &self.face_data_buffer,
                &chunk_manager,
                &self.material_manager
            );
        }

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(bind_group.clone()),
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
            unreachable!()
        };

        let elements = self.face_id_buffer.lock().unwrap().get_number_of_elements() * 6;

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw(0..elements as u32, 0..1);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoxelFaceDirection
{
    Top    = 0,
    Bottom = 1,
    Left   = 2,
    Right  = 3,
    Front  = 4,
    Back   = 5
}

impl TryFrom<u8> for VoxelFaceDirection
{
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error>
    {
        use VoxelFaceDirection::*;

        match value
        {
            0 => Ok(Top),
            1 => Ok(Bottom),
            2 => Ok(Left),
            3 => Ok(Right),
            4 => Ok(Front),
            5 => Ok(Back),
            _ => Err(value)
        }
    }
}

impl VoxelFaceDirection
{
    pub fn to_bits(self) -> u8
    {
        self as u8
    }

    pub fn iterate() -> impl Iterator<Item = VoxelFaceDirection>
    {
        [
            VoxelFaceDirection::Top,
            VoxelFaceDirection::Bottom,
            VoxelFaceDirection::Left,
            VoxelFaceDirection::Right,
            VoxelFaceDirection::Front,
            VoxelFaceDirection::Back
        ]
        .into_iter()
    }

    pub fn get_axis(self) -> glm::I16Vec3
    {
        match self
        {
            VoxelFaceDirection::Top => glm::I16Vec3::new(0, 1, 0),
            VoxelFaceDirection::Bottom => glm::I16Vec3::new(0, -1, 0),
            VoxelFaceDirection::Left => glm::I16Vec3::new(-1, 0, 0),
            VoxelFaceDirection::Right => glm::I16Vec3::new(1, 0, 0),
            VoxelFaceDirection::Front => glm::I16Vec3::new(0, 0, -1),
            VoxelFaceDirection::Back => glm::I16Vec3::new(0, 0, 1)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug)]
struct GpuFaceData
// is allocated at a specific index
{
    material_and_chunk_id: u32,
    // 9 bits x
    // 9 bits y
    // 9 bits z
    // 3 bits normal
    // 1 bit visibility
    // 1 bit unused
    location_within_chunk: u32
}

impl GpuFaceData
{
    pub fn new(material: u16, chunk_id: u16, pos: glm::U16Vec3, dir: VoxelFaceDirection) -> Self
    {
        assert!(pos.x < 2u16.pow(9), "{:?}", pos.x);
        assert!(pos.y < 2u16.pow(9), "{:?}", pos.y);
        assert!(pos.z < 2u16.pow(9), "{:?}", pos.z);

        GpuFaceData {
            material_and_chunk_id: (material as u32) | ((chunk_id as u32) << 16),
            location_within_chunk: (pos.x as u32)
                | ((pos.y as u32) << 9)
                | ((pos.z as u32) << 18)
                | ((dir.to_bits() as u32) << 27)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub struct VoxelFace
{
    pub direction: VoxelFaceDirection,
    pub position:  glm::I32Vec3,
    pub material:  u16
}

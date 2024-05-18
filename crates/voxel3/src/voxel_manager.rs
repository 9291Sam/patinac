use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, AnyBitPattern, NoUninit};
use gfx::wgpu::core::id;
use gfx::wgpu::{self, include_wgsl};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};

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
    face_data_buffer:  gfx::CpuTrackedBuffer<GpuFaceData>
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

        let combined_bind_group =
            Self::generate_bind_group(&renderer, &bind_group_layout, &id_buffer, &data_buffer);

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
            face_data_buffer:  data_buffer
        });

        renderer.register(this.clone());

        this
    }

    // no chunks for now, just one global chunk

    pub fn insert_face(&self, face: VoxelFace)
    {
        let new_face_id = self.face_id_allocator.lock().unwrap().allocate().unwrap();
        self.face_id_buffer
            .lock()
            .unwrap()
            .insert(new_face_id as u32);
        self.face_data_buffer.write(
            new_face_id,
            GpuFaceData::new(4, 0, face.position, face.direction)
        );
    }

    fn generate_bind_group(
        renderer: &gfx::Renderer,
        bind_group_layout: &wgpu::BindGroupLayout,
        face_id_buffer: &super::CpuTrackedDenseSet<u32>,
        face_data_buffer: &gfx::CpuTrackedBuffer<GpuFaceData>
    ) -> Arc<wgpu::BindGroup>
    {
        log::trace!("regen bind group");
        face_id_buffer.get_buffer(|raw_id_buf| {
            face_data_buffer.get_buffer(|raw_data_buf| {
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
                        }
                    ]
                }))
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
        needs_resize |= self.face_id_buffer.lock().unwrap().flush_to_gpu();
        needs_resize |= self.face_data_buffer.replicate_to_gpu();

        let mut bind_group = self.bind_group.lock().unwrap();

        if needs_resize
        {
            *bind_group = Self::generate_bind_group(
                renderer,
                &self.bind_group_layout,
                &self.face_id_buffer.lock().unwrap(),
                &self.face_data_buffer
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

// make an algorithm that finds all of the ranges of things that need to be
// drawn upload that list of ranges into a
// TODO: use draw_indrect

// no vertex buffer each 6 looks at a different range

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
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug)]
struct GpuFaceData
// is allocated at a specific index
{
    material:              u16,
    chunk_id:              u16,
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
        assert!(pos.x < 2u16.pow(9) - 1);
        assert!(pos.y < 2u16.pow(9) - 1);
        assert!(pos.z < 2u16.pow(9) - 1);

        GpuFaceData {
            material,
            chunk_id,
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
    pub voxel:     u16,
    pub position:  glm::U16Vec3
}

// one massive draw
// % 6
// [face_id_buffer] // use the data here to lookup everything else in the
// FaceData buffer

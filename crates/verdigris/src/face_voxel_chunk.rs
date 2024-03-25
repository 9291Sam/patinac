use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bytemuck::{cast_slice, Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};

#[derive(Debug)]
pub struct FaceVoxelChunk
{
    renderer: Arc<gfx::Renderer>,
    uuid:     util::Uuid,

    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,

    voxel_positions: Vec<FaceVoxelChunkVoxelInstance>,
    instance_buffer: wgpu::Buffer,
    pipeline:        Arc<gfx::GenericPipeline>,

    transform: Mutex<gfx::Transform>
}

impl FaceVoxelChunk
{
    pub fn new(game: &game::Game, transform: gfx::Transform) -> Arc<FaceVoxelChunk>
    {
        let uuid = util::Uuid::new();

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("face_voxel_chunk.wgsl"));

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "FaceVoxelChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![renderer.global_bind_group_layout.clone()],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let pipeline = game.get_renderer().render_cache.cache_render_pipeline(
            CacheableRenderPipelineDescriptor {
                label:                 "FaceVoxelChunk Pipeline".into(),
                layout:                Some(pipeline_layout),
                vertex_module:         shader.clone(),
                vertex_entry_point:    "vs_main".into(),
                vertex_buffer_layouts: vec![
                    VoxelVertex::describe(),
                    FaceVoxelChunkVoxelInstance::describe(),
                ],
                fragment_state:        Some(CacheableFragmentState {
                    module:      shader,
                    entry_point: "fs_main".into(),
                    targets:     vec![Some(wgpu::ColorTargetState {
                        format:     gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                        blend:      Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL
                    })]
                }),
                primitive_state:       wgpu::PrimitiveState {
                    topology:           wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face:         wgpu::FrontFace::Cw,
                    cull_mode:          Some(wgpu::Face::Back),
                    polygon_mode:       wgpu::PolygonMode::Fill,
                    unclipped_depth:    false,
                    conservative:       false
                },
                depth_stencil_state:   Some(gfx::Renderer::get_default_depth_state()),
                multisample_state:     wgpu::MultisampleState {
                    count:                     1,
                    mask:                      !0,
                    alpha_to_coverage_enabled: false
                },
                multiview:             None
            }
        );

        let instances: &[FaceVoxelChunkVoxelInstance] = &[];

        let this = Arc::new(FaceVoxelChunk {
            uuid,
            vertex_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("FaceVoxelChunk Vertex Buffer"),
                contents: cast_slice(&VOXEL_FACE_VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            index_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("FaceVoxelChunk Index Buffer"),
                contents: cast_slice(&VOXEL_FACE_INDICES),
                usage:    wgpu::BufferUsages::INDEX
            }),
            instance_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("FaceVoxelChunk Instance Buffer"),
                contents: cast_slice(instances),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            voxel_positions: Vec::from_iter(instances.iter().cloned()),
            pipeline,
            transform: Mutex::new(transform),
            renderer: game.get_renderer().clone()
        });

        renderer.register(this.clone());

        this
    }

    pub fn update_voxels(
        &mut self,
        positions: impl IntoIterator<Item = FaceVoxelChunkVoxelInstance>
    )
    {
        let instances: Vec<FaceVoxelChunkVoxelInstance> = positions.into_iter().collect();

        self.instance_buffer = self.renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Raster Instance Buffer"),
            contents: bytemuck::cast_slice(&instances[..]),
            usage:    wgpu::BufferUsages::VERTEX
        });

        self.voxel_positions = instances;
    }
}

impl gfx::Recordable for FaceVoxelChunk
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline(&self) -> Option<&gfx::GenericPipeline>
    {
        Some(&self.pipeline)
    }

    fn pre_record_update(
        &self,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo {
            should_draw: true,
            transform:   Some(self.transform.lock().unwrap().clone()),
            bind_groups: [Some(global_bind_group.clone()), None, None, None]
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw_indexed(
            0..VOXEL_FACE_INDICES.len() as u32,
            0,
            0..self.voxel_positions.len() as u32
        );
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct FaceVoxelChunkVoxelInstance
{
    data: u32
}

impl FaceVoxelChunkVoxelInstance
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Uint32];

    pub fn new(x: u32, y: u32, z: u32, face: VoxelFace, voxel: u32) -> Self
    {
        let five_bit_mask: u32 = 0b1_1111;
        let three_bit_mask: u32 = 0b111;

        let face = face as u32;

        assert!(x <= five_bit_mask);
        assert!(y <= five_bit_mask);
        assert!(z <= five_bit_mask);
        assert!(face <= three_bit_mask);
        assert!(voxel <= five_bit_mask);

        let x_data = five_bit_mask & x; // [0, 4]
        let y_data = (five_bit_mask & y) << 5; // [5, 9]
        let z_data = (five_bit_mask & z) << 10; // [10, 14]
        let f_data = (three_bit_mask & face) << 15; // [15, 17]
        let v_data = (five_bit_mask & voxel) << 18; // [18, 22]

        Self {
            data: x_data | y_data | z_data | f_data | v_data
        }
    }

    pub fn destructure(self) -> (u32, u32, VoxelFace, u32)
    {
        let five_bit_mask: u32 = 0b1_1111;
        let three_bit_mask: u32 = 0b111;

        let x = five_bit_mask & self.data;
        let y = five_bit_mask & (self.data >> 5);
        let z = five_bit_mask & (self.data >> 10);
        let f = three_bit_mask & (self.data >> 15);
        let v = five_bit_mask & (self.data >> 18);

        (x, y, VoxelFace::try_from(f).unwrap(), v)
    }

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Instance,
            attributes:   &Self::ATTRS
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct VoxelVertex
{
    p: glm::Vec2
}

impl VoxelVertex
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRS
        }
    }
}

#[repr(u32)]
pub enum VoxelFace
{
    Front  = 0,
    Back   = 1,
    Top    = 2,
    Bottom = 3,
    Left   = 4,
    Right  = 5
}

impl TryFrom<u32> for VoxelFace
{
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error>
    {
        match value
        {
            0 => Ok(VoxelFace::Front),
            1 => Ok(VoxelFace::Back),
            2 => Ok(VoxelFace::Top),
            3 => Ok(VoxelFace::Bottom),
            4 => Ok(VoxelFace::Left),
            5 => Ok(VoxelFace::Right),
            _ => Err(value)
        }
    }
}

const VOXEL_FACE_VERTICES: [VoxelVertex; 4] = [
    VoxelVertex {
        p: glm::Vec2::new(0.0, 0.0)
    },
    VoxelVertex {
        p: glm::Vec2::new(0.0, 1.0)
    },
    VoxelVertex {
        p: glm::Vec2::new(1.0, 0.0)
    },
    VoxelVertex {
        p: glm::Vec2::new(1.0, 1.0)
    }
];

const VOXEL_FACE_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

//

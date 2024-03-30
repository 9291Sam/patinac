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
pub struct RasterChunk
{
    renderer: Arc<gfx::Renderer>,
    uuid:     util::Uuid,

    vertex_buffer:      wgpu::Buffer,
    number_of_vertices: u32,

    pipeline: Arc<gfx::GenericPipeline>,

    transform: Mutex<gfx::Transform>
}

impl RasterChunk
{
    pub fn new(
        game: &game::Game,
        transform: gfx::Transform,
        faces: impl IntoIterator<Item = VoxelFace>
    ) -> Arc<RasterChunk>
    {
        let uuid = util::Uuid::new();

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("raster.wgsl"));

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "RasterChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![renderer.global_bind_group_layout.clone()],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let pipeline = game.get_renderer().render_cache.cache_render_pipeline(
            CacheableRenderPipelineDescriptor {
                label:                 "RasterChunk Pipeline".into(),
                layout:                Some(pipeline_layout),
                vertex_module:         shader.clone(),
                vertex_entry_point:    "vs_main".into(),
                vertex_buffer_layouts: vec![RasterChunkVoxelPoint::describe()],
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

        let (vertex_buffer, number_of_vertices) = Self::create_voxel_buffer(renderer, faces);

        log::trace!("Created RasterChunk with {} vertices", number_of_vertices);

        let this = Arc::new(RasterChunk {
            uuid,
            vertex_buffer,
            number_of_vertices: number_of_vertices as u32,
            pipeline,
            transform: Mutex::new(transform),
            renderer: game.get_renderer().clone()
        });

        renderer.register(this.clone());

        this
    }

    /// returns the number of vertices in the buffer
    fn create_voxel_buffer(
        renderer: &gfx::Renderer,
        faces: impl IntoIterator<Item = VoxelFace>
    ) -> (wgpu::Buffer, usize)
    {
        let instances: Vec<RasterChunkVoxelPoint> = faces
            .into_iter()
            .flat_map(|face| face.direction.to_face_points(face.position, face.voxel))
            .collect();

        let buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Raster Vertex Buffer {}"),
            contents: bytemuck::cast_slice(&instances[..]),
            usage:    wgpu::BufferUsages::VERTEX
        });

        (buffer, instances.len())
    }
}

impl gfx::Recordable for RasterChunk
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
        let t = self.transform.lock().unwrap().clone();

        gfx::RecordInfo {
            should_draw: true,
            transform:   Some(t),
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
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw(0..self.number_of_vertices, 0..1);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct RasterChunkVoxelPoint
{
    data: u32
}

impl RasterChunkVoxelPoint
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32];

    pub fn new(x: u32, y: u32, z: u32, voxel: u32) -> Self
    {
        let two_bit_mask: u32 = 0b11;
        let ten_bit_mask: u32 = 0b11_1111_1111;

        assert!(x <= ten_bit_mask, "{x}");
        assert!(y <= ten_bit_mask, "{y}");
        assert!(z <= ten_bit_mask, "{z}");
        assert!(voxel <= two_bit_mask, "{voxel}");

        let x_data = ten_bit_mask & x;
        let y_data = (ten_bit_mask & y) << 10;
        let z_data = (ten_bit_mask & z) << 20;
        let v_data = (two_bit_mask & voxel) << 30;

        Self {
            data: x_data | y_data | z_data | v_data
        }
    }

    pub fn destructure(self) -> (u32, u32, u32, u32)
    {
        let two_bit_mask: u32 = 0b11;
        let ten_bit_mask: u32 = 0b11_1111_1111;

        let x = ten_bit_mask & self.data;
        let y = ten_bit_mask & (self.data >> 10);
        let z = ten_bit_mask & (self.data >> 20);
        let v = two_bit_mask & (self.data >> 30);

        (x, y, z, v)
    }

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRS
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub struct VoxelFace
{
    pub direction: VoxelFaceDirection,
    pub voxel:     u32,
    pub position:  glm::U16Vec3
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoxelFaceDirection
{
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back
}

impl VoxelFaceDirection
{
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

    #[allow(clippy::just_underscores_and_digits)]
    pub fn to_face_points(self, pos: glm::U16Vec3, voxel: u32) -> [RasterChunkVoxelPoint; 6]
    {
        let (_0, _1, _2, _3) = match self
        {
            VoxelFaceDirection::Top =>
            {
                (
                    glm::U16Vec3::new(0, 1, 0),
                    glm::U16Vec3::new(0, 1, 1),
                    glm::U16Vec3::new(1, 1, 0),
                    glm::U16Vec3::new(1, 1, 1)
                )
            }
            VoxelFaceDirection::Bottom =>
            {
                (
                    glm::U16Vec3::new(0, 0, 1),
                    glm::U16Vec3::new(0, 0, 0),
                    glm::U16Vec3::new(1, 0, 1),
                    glm::U16Vec3::new(1, 0, 0)
                )
            }
            VoxelFaceDirection::Left =>
            {
                (
                    glm::U16Vec3::new(0, 0, 1),
                    glm::U16Vec3::new(0, 1, 1),
                    glm::U16Vec3::new(0, 0, 0),
                    glm::U16Vec3::new(0, 1, 0)
                )
            }
            VoxelFaceDirection::Right =>
            {
                (
                    glm::U16Vec3::new(1, 0, 0),
                    glm::U16Vec3::new(1, 1, 0),
                    glm::U16Vec3::new(1, 0, 1),
                    glm::U16Vec3::new(1, 1, 1)
                )
            }
            VoxelFaceDirection::Front =>
            {
                (
                    glm::U16Vec3::new(0, 0, 0),
                    glm::U16Vec3::new(0, 1, 0),
                    glm::U16Vec3::new(1, 0, 0),
                    glm::U16Vec3::new(1, 1, 0)
                )
            }
            VoxelFaceDirection::Back =>
            {
                (
                    glm::U16Vec3::new(1, 0, 1),
                    glm::U16Vec3::new(1, 1, 1),
                    glm::U16Vec3::new(0, 0, 1),
                    glm::U16Vec3::new(0, 1, 1)
                )
            }
        };

        // 0, 1, 2, 2, 1, 3

        let x = pos.x;
        let y = pos.y;
        let z = pos.z;

        let vertices: [RasterChunkVoxelPoint; 4] = [
            RasterChunkVoxelPoint::new(
                (x + _0.x).into(),
                (y + _0.y).into(),
                (z + _0.z).into(),
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _1.x).into(),
                (y + _1.y).into(),
                (z + _1.z).into(),
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _2.x).into(),
                (y + _2.y).into(),
                (z + _2.z).into(),
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _3.x).into(),
                (y + _3.y).into(),
                (z + _3.z).into(),
                voxel
            )
        ];

        [
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[2],
            vertices[1],
            vertices[3]
        ]
    }
}

const VOXEL_STRIP_VERTICES: [glm::IVec3; 8] = [
    glm::IVec3::new(1, 1, 0),
    glm::IVec3::new(0, 1, 0),
    glm::IVec3::new(1, 0, 0),
    glm::IVec3::new(0, 0, 0),
    glm::IVec3::new(1, 1, 1),
    glm::IVec3::new(0, 1, 1),
    glm::IVec3::new(0, 0, 1),
    glm::IVec3::new(1, 0, 1)
];
const VOXEL_STRIP_INDICES: [u16; 14] = [3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0];

const VOXEL_LIST_VERTICES: [glm::IVec3; 8] = [
    glm::IVec3::new(0, 0, 0),
    glm::IVec3::new(0, 0, 1),
    glm::IVec3::new(0, 1, 0),
    glm::IVec3::new(0, 1, 1),
    glm::IVec3::new(1, 0, 0),
    glm::IVec3::new(1, 0, 1),
    glm::IVec3::new(1, 1, 0),
    glm::IVec3::new(1, 1, 1)
];

#[rustfmt::skip]
const VOXEL_LIST_INDICES: [u16; 36] = [
    6, 2, 7,
    2, 3, 7,
    0, 4, 5,
    1, 0, 5,
    0, 2, 6,
    4, 0, 6,
    3, 1, 7,
    1, 5, 7,
    2, 0, 3,
    0, 1, 3,
    4, 6, 7,
    5, 4, 7
];

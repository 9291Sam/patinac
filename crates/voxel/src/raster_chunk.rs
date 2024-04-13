use std::borrow::Cow;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};
use util::AtomicF32;

#[derive(Debug)]
pub struct RasterChunk
{
    uuid: util::Uuid,

    vertex_buffer:      wgpu::Buffer,
    number_of_vertices: u32,
    time_alive:         AtomicF32,

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
                        stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        range:  0..(std::mem::size_of::<PC>() as u32)
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
                        format:     wgpu::TextureFormat::Rg32Uint,
                        blend:      None,
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

        // log::trace!(
        //     "Created RasterChunk with {} triangles",
        //     number_of_vertices / 3
        // );

        let this = Arc::new(RasterChunk {
            uuid,
            vertex_buffer,
            number_of_vertices: number_of_vertices as u32,
            pipeline,
            transform: Mutex::new(transform),
            time_alive: AtomicF32::new(0.0)
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
        gfx::PassStage::VoxelDiscovery
    }

    fn get_pipeline(&self) -> Option<&gfx::GenericPipeline>
    {
        Some(&self.pipeline)
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let t = self.transform.lock().unwrap().clone();
        self.time_alive.store(
            self.time_alive.load(SeqCst) + renderer.get_delta_time(),
            SeqCst
        );

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
        pass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0,
            bytemuck::bytes_of(&PC {
                id,
                time_alive: self.time_alive.load(SeqCst)
            })
        );
        pass.draw(0..self.number_of_vertices, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PC
{
    id:         u32,
    time_alive: f32
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct RasterChunkVoxelPoint
{
    ///   x [0]    y [1]
    /// [0,   8] [      ] | 9 + 0 bits  | x_pos
    /// [9,  17] [      ] | 9 + 0 bits  | y_pos
    /// [18, 26] [      ] | 9 + 0 bits  | z_pos
    /// [27, 31] [0,   3] | 5 + 4 bits  | l_width
    /// [      ] [4,  12] | 0 + 9 bits  | w_width
    /// [      ] [13, 15] | 0 + 3 bits  | face id
    /// [      ] [16, 31] | 0 + 16 bits | voxel id
    data: glm::U32Vec2
}

impl RasterChunkVoxelPoint
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32x2];

    #[inline(always)]
    pub fn new(
        x_pos: u32,
        y_pos: u32,
        z_pos: u32,
        l_width: u32,
        w_width: u32,
        face_id: VoxelFaceDirection,
        voxel: u16
    ) -> Self
    {
        let sixteen_bit_mask: u32 = 0b1111_1111_1111_1111;
        let nine_bit_mask: u32 = 0b1_1111_1111;
        let three_bit_mask: u32 = 0b111;

        let face = face_id as u32;
        let voxel = voxel as u32;

        assert!(x_pos <= nine_bit_mask, "{x_pos}");
        assert!(y_pos <= nine_bit_mask, "{y_pos}");
        assert!(z_pos <= nine_bit_mask, "{z_pos}");
        assert!(l_width <= nine_bit_mask, "{l_width}");
        assert!(w_width <= nine_bit_mask, "{w_width}");
        assert!(face <= three_bit_mask, "{face}");
        // don't need to assert voxel as its already a u16

        let x_data = nine_bit_mask & x_pos;
        let y_data = (nine_bit_mask & y_pos) << 9;
        let z_data = (nine_bit_mask & z_pos) << 18;
        let l_data_lo = (nine_bit_mask & l_width) << 27;

        let l_data_hi = (nine_bit_mask & l_width) >> 5;
        let w_data = (nine_bit_mask & w_width) << 4;
        let f_data = (three_bit_mask & face) << 13;
        let v_data = (sixteen_bit_mask & voxel) << 16;

        Self {
            data: glm::U32Vec2::new(
                x_data | y_data | z_data | l_data_lo,
                l_data_hi | w_data | f_data | v_data
            )
        }
    }

    pub fn destructure(self) -> (u32, u32, u32, u32, u32, VoxelFaceDirection, u16)
    {
        let nine_bit_mask: u32 = 0b1_1111_1111;
        let three_bit_mask: u32 = 0b111;

        let x_pos = self.data[0] & nine_bit_mask;
        let y_pos = (self.data[0] >> 9) & nine_bit_mask;
        let z_pos = (self.data[0] >> 18) & nine_bit_mask;

        let l_width_lo = (self.data[0] >> 27) & 0b11111;
        let l_width_hi = (self.data[1] & 0b1111) << 5;

        let l_width = l_width_lo | l_width_hi;
        let w_width = (self.data[1] >> 4) & nine_bit_mask;
        let face_id = VoxelFaceDirection::try_from((self.data[1] >> 13) & three_bit_mask).unwrap();
        let voxel_id = (self.data[1] >> 16) as u16;

        (x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id)
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
    pub voxel:     u16,
    pub position:  glm::U16Vec3,
    pub lw_size:   glm::U16Vec2
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

impl TryFrom<u32> for VoxelFaceDirection
{
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error>
    {
        match value
        {
            0 => Ok(VoxelFaceDirection::Top),
            1 => Ok(VoxelFaceDirection::Bottom),
            2 => Ok(VoxelFaceDirection::Left),
            3 => Ok(VoxelFaceDirection::Right),
            4 => Ok(VoxelFaceDirection::Front),
            5 => Ok(VoxelFaceDirection::Back),
            _ => Err(())
        }
    }
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
    pub fn to_face_points(self, pos: glm::U16Vec3, voxel: u16) -> [RasterChunkVoxelPoint; 6]
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
                1,
                1,
                self,
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _1.x).into(),
                (y + _1.y).into(),
                (z + _1.z).into(),
                1,
                1,
                self,
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _2.x).into(),
                (y + _2.y).into(),
                (z + _2.z).into(),
                1,
                1,
                self,
                voxel
            ),
            RasterChunkVoxelPoint::new(
                (x + _3.x).into(),
                (y + _3.y).into(),
                (z + _3.z).into(),
                1,
                1,
                self,
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

// const VOXEL_STRIP_VERTICES: [glm::IVec3; 8] = [
//     glm::IVec3::new(1, 1, 0),
//     glm::IVec3::new(0, 1, 0),
//     glm::IVec3::new(1, 0, 0),
//     glm::IVec3::new(0, 0, 0),
//     glm::IVec3::new(1, 1, 1),
//     glm::IVec3::new(0, 1, 1),
//     glm::IVec3::new(0, 0, 1),
//     glm::IVec3::new(1, 0, 1)
// ];
// const VOXEL_STRIP_INDICES: [u16; 14] = [3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4,
// 1, 0];

// const VOXEL_LIST_VERTICES: [glm::IVec3; 8] = [
//     glm::IVec3::new(0, 0, 0),
//     glm::IVec3::new(0, 0, 1),
//     glm::IVec3::new(0, 1, 0),
//     glm::IVec3::new(0, 1, 1),
//     glm::IVec3::new(1, 0, 0),
//     glm::IVec3::new(1, 0, 1),
//     glm::IVec3::new(1, 1, 0),
//     glm::IVec3::new(1, 1, 1)
// ];

// #[rustfmt::skip]
// const VOXEL_LIST_INDICES: [u16; 36] = [
//     6, 2, 7,
//     2, 3, 7,
//     0, 4, 5,
//     1, 0, 5,
//     0, 2, 6,
//     4, 0, 6,
//     3, 1, 7,
//     1, 5, 7,
//     2, 0, 3,
//     0, 1, 3,
//     4, 6, 7,
//     5, 4, 7
// ];

#[cfg(test)]
mod tests
{
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_new_and_destructure()
    {
        // Initialize the random number generator with a fixed seed for reproducibility
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..10_000
        {
            // Generate random values within appropriate ranges
            let x_pos: u32 = rng.gen_range(0..=0b1_1111_1111);
            let y_pos: u32 = rng.gen_range(0..=0b1_1111_1111);
            let z_pos: u32 = rng.gen_range(0..=0b1_1111_1111);
            let l_width: u32 = rng.gen_range(0..=0b1_1111_1111);
            let w_width: u32 = rng.gen_range(0..=0b1_1111_1111);
            let face_id: VoxelFaceDirection = rng.gen_range(0..=5).try_into().unwrap();
            let voxel_id: u16 = rng.gen();

            let voxel_point = RasterChunkVoxelPoint::new(
                x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id
            );
            let (x, y, z, l, w, f, v) = voxel_point.destructure();

            assert_eq!(x, x_pos);
            assert_eq!(y, y_pos);
            assert_eq!(z, z_pos);
            assert_eq!(l, l_width);
            assert_eq!(w, w_width);
            assert_eq!(f, face_id);
            assert_eq!(v, voxel_id);
        }
    }

    // Add more tests as needed
}

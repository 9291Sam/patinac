use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use bytemuck::{cast_slice, Pod, Zeroable};
use compile_warning::compile_warning;
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self, BindGroupDescriptor, BindGroupLayoutDescriptor};
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

    voxel_positions:   Vec<FaceVoxelChunkVoxelInstance>,
    voxel_data_buffer: wgpu::Buffer,

    voxel_data_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    voxel_data_bind_group:        Arc<wgpu::BindGroup>,
    pipeline:                     Arc<gfx::GenericPipeline>,

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

        static BINDINGS: &[wgpu::BindGroupLayoutEntry] = &[wgpu::BindGroupLayoutEntry {
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
        }];

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(BindGroupLayoutDescriptor {
                    label:   Some("Face Voxel Chunk Bind Group"),
                    entries: BINDINGS
                });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "FaceVoxelChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout.clone(),
                    ],
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
                vertex_buffer_layouts: vec![],
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

        let instances: [FaceVoxelChunkVoxelInstance; 1] = [FaceVoxelChunkVoxelInstance::new(
            0,
            0,
            0,
            1,
            1,
            VoxelFace::Top,
            0
        )];

        let (instances, buffer, bind_group) =
            Self::create_voxel_buffer_and_bind_group(renderer, &bind_group_layout, instances);

        let this = Arc::new(FaceVoxelChunk {
            uuid,
            voxel_positions: Vec::from_iter(instances.iter().cloned()),
            pipeline,
            transform: Mutex::new(transform),
            renderer: game.get_renderer().clone(),
            voxel_data_buffer: buffer,
            voxel_data_bind_group: bind_group,
            voxel_data_bind_group_layout: bind_group_layout
        });

        renderer.register(this.clone());

        this
    }

    pub fn update_voxels(
        &mut self,
        positions: impl IntoIterator<Item = FaceVoxelChunkVoxelInstance>
    )
    {
        let (instances, buffer, bind_group) = Self::create_voxel_buffer_and_bind_group(
            &self.renderer,
            &self.voxel_data_bind_group_layout,
            positions
        );

        self.voxel_positions = instances;
        self.voxel_data_buffer = buffer;
        self.voxel_data_bind_group = bind_group;
    }

    fn create_voxel_buffer_and_bind_group(
        renderer: &gfx::Renderer,
        layout: &wgpu::BindGroupLayout,
        positions: impl IntoIterator<Item = FaceVoxelChunkVoxelInstance>
    ) -> (
        Vec<FaceVoxelChunkVoxelInstance>,
        wgpu::Buffer,
        Arc<wgpu::BindGroup>
    )
    {
        let instances: Vec<FaceVoxelChunkVoxelInstance> = positions.into_iter().collect();

        let voxel_data_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Raster Face Instance Buffer"),
            contents: bytemuck::cast_slice(&instances[..]),
            usage:    wgpu::BufferUsages::STORAGE
        });

        let bind_group = renderer.create_bind_group(&BindGroupDescriptor {
            label: Some("Voxel Face data bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: voxel_data_buffer.as_entire_binding()
            }]
        });

        (instances, voxel_data_buffer, Arc::new(bind_group))
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
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.voxel_data_bind_group.clone()),
                None,
                None
            ]
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw(0..(self.voxel_positions.len() as u32 * 6 as u32), 0..1);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct FaceVoxelChunkVoxelInstance
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

impl FaceVoxelChunkVoxelInstance
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Uint32x2];

    #[inline(always)]
    pub fn new(
        x_pos: u32,
        y_pos: u32,
        z_pos: u32,
        l_width: u32,
        w_width: u32,
        face_id: VoxelFace,
        voxel: u16
    ) -> Self
    {
        let sixteen_bit_mask: u32 = 0b1111_1111_1111_1111;
        let nine_bit_mask: u32 = 0b1_1111_1111;
        let three_bit_mask: u32 = 0b111;

        let face = face_id as u32;
        let voxel = voxel as u32;

        assert!(x_pos <= nine_bit_mask);
        assert!(y_pos <= nine_bit_mask);
        assert!(z_pos <= nine_bit_mask);
        assert!(l_width <= nine_bit_mask);
        assert!(w_width <= nine_bit_mask);
        assert!(face <= three_bit_mask);
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

    pub fn destructure(self) -> (u32, u32, u32, u32, u32, VoxelFace, u16)
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
        let face_id = VoxelFace::try_from((self.data[1] >> 13) & three_bit_mask).unwrap();
        let voxel_id = (self.data[1] >> 16) as u16;

        (x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id)
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

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoxelFace
{
    Front  = 0,
    Back   = 1,
    Top    = 2,
    Bottom = 3,
    Left   = 4,
    Right  = 5
}

impl VoxelFace
{
    pub fn iter() -> impl Iterator<Item = VoxelFace>
    {
        [
            VoxelFace::Front,
            VoxelFace::Back,
            VoxelFace::Top,
            VoxelFace::Bottom,
            VoxelFace::Left,
            VoxelFace::Right
        ]
        .into_iter()
    }

    pub fn get_axis(self) -> glm::I32Vec3
    {
        match self
        {
            VoxelFace::Front => glm::I32Vec3::new(0, 0, -1),
            VoxelFace::Back => glm::I32Vec3::new(0, 0, 1),
            VoxelFace::Top => glm::I32Vec3::new(0, 1, 0),
            VoxelFace::Bottom => glm::I32Vec3::new(0, -1, 0),
            VoxelFace::Left => glm::I32Vec3::new(-1, 0, 0),
            VoxelFace::Right => glm::I32Vec3::new(1, 0, 0)
        }
    }
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

#[cfg(test)]
mod tests
{
    use rand::Rng;

    use super::*;

    #[test]
    fn test_new_instance()
    {
        let instance = FaceVoxelChunkVoxelInstance::new(5, 6, 7, 8, 9, VoxelFace::Front, 123);
        let (x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id) = instance.destructure();
        assert_eq!(x_pos, 5);
        assert_eq!(y_pos, 6);
        assert_eq!(z_pos, 7);
        assert_eq!(l_width, 8);
        assert_eq!(w_width, 9);
        assert_eq!(face_id, VoxelFace::Front);
        assert_eq!(voxel_id, 123);
    }

    #[test]
    #[should_panic]
    fn test_new_instance_overflow_x_pos()
    {
        let _ = FaceVoxelChunkVoxelInstance::new(512, 6, 7, 8, 9, VoxelFace::Front, 123);
    }

    #[test]
    fn test_destructure_instance()
    {
        let instance = FaceVoxelChunkVoxelInstance::new(5, 6, 7, 8, 9, VoxelFace::Front, 123);
        let (x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id) = instance.destructure();
        assert_eq!(x_pos, 5);
        assert_eq!(y_pos, 6);
        assert_eq!(z_pos, 7);
        assert_eq!(l_width, 8);
        assert_eq!(w_width, 9);
        assert_eq!(face_id, VoxelFace::Front);
        assert_eq!(voxel_id, 123);
    }

    #[test]
    fn test_new_and_destructure_instance()
    {
        let mut rng = rand::thread_rng();

        for _ in 0..10000
        {
            let x_pos = rng.gen_range(0..=0x1FF); // 9 bits
            let y_pos = rng.gen_range(0..=0x1FF); // 9 bits
            let z_pos = rng.gen_range(0..=0x1FF); // 9 bits
            let l_width = rng.gen_range(0..=0x1FF); // 9 bits
            let w_width = rng.gen_range(0..=0x1FF); // 9 bits
            let face_id = VoxelFace::try_from(rng.gen_range(0..=5)).unwrap(); // 3 bits
            let voxel_id = rng.gen_range(0..=0xFFFF); // 16 bits

            let instance = FaceVoxelChunkVoxelInstance::new(
                x_pos, y_pos, z_pos, l_width, w_width, face_id, voxel_id
            );
            let (x_pos_d, y_pos_d, z_pos_d, l_width_d, w_width_d, face_id_d, voxel_id_d) =
                instance.destructure();

            assert_eq!(x_pos, x_pos_d);
            assert_eq!(y_pos, y_pos_d);
            assert_eq!(z_pos, z_pos_d);
            assert_eq!(l_width, l_width_d);
            assert_eq!(w_width, w_width_d);
            assert_eq!(face_id, face_id_d);
            assert_eq!(voxel_id, voxel_id_d);
        }
    }

    #[test]
    fn test_vertex_buffer_layout()
    {
        let layout = FaceVoxelChunkVoxelInstance::describe();
        assert_eq!(
            layout.array_stride,
            std::mem::size_of::<FaceVoxelChunkVoxelInstance>() as wgpu::BufferAddress
        );
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 1);
        assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Uint32x2);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[0].shader_location, 1);
    }
}

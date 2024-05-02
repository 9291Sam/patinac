use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bytemuck::{cast_slice, Pod, Zeroable};
use game::{Entity, Positionable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};

use crate::gpu_data::{Voxel, VoxelChunkDataManager, CHUNK_VOXEL_SIZE};

#[derive(Debug)]
pub struct BrickMapChunk
{
    uuid:     util::Uuid,
    name:     String,
    position: Mutex<glm::Vec3>,

    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    number_of_indices: u32,

    voxel_chunk_data: Arc<VoxelChunkDataManager>,
    pipeline:         Arc<gfx::GenericPipeline>
}

impl BrickMapChunk
{
    pub fn new(
        game: &game::Game,
        center_position: glm::Vec3,
        data_manager: Arc<VoxelChunkDataManager>
    ) -> Arc<Self>
    {
        let uuid = util::Uuid::new();

        let vertex_buffer_label = format!("BrickMapChunk {} Vertex Buffer", uuid);
        let index_buffer_label = format!("BrickMapChunk {} Index Buffer", uuid);

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("brick_map_chunk.wgsl"));

        let voxel_data_manager = data_manager;

        log::trace!(
            "used bricks empty {:?}",
            voxel_data_manager.peek_allocated_bricks(),
        );

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "Voxel BrickMapChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        voxel_data_manager.get_bind_group_layout().clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let this = Arc::new(Self {
            voxel_chunk_data: voxel_data_manager,
            uuid,
            name: "Voxel BrickMapChunk".into(),
            position: Mutex::new(center_position),
            vertex_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some(&vertex_buffer_label),
                contents: cast_slice(&CUBE_VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            index_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some(&index_buffer_label),
                contents: cast_slice(&CUBE_INDICES),
                usage:    wgpu::BufferUsages::INDEX
            }),
            number_of_indices: CUBE_INDICES.len() as u32,
            pipeline: game.get_renderer().render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label: "Voxel BrickMapChunk Pipeline".into(),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![Vertex::describe_layout()],
                    vertex_specialization: None,
                    zero_initalize_vertex_workgroup_memory: false,
                    fragment_state: Some(CacheableFragmentState {
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
                    multiview: None
                }
            )
        });

        game.register(this.clone());
        game.register_chunk(Arc::downgrade(&(this.clone() as Arc<dyn game::World>)));
        renderer.register(this.clone());

        this
    }

    pub fn access_data_manager(&self) -> &Arc<VoxelChunkDataManager>
    {
        &self.voxel_chunk_data
    }
}

impl game::World for BrickMapChunk
{
    fn get_height(&self, pos: glm::Vec3) -> f32
    {
        let me_pos = *self.position.lock().unwrap();

        let (x, z) = (
            pos.x.floor() + me_pos.x + CHUNK_VOXEL_SIZE as f32,
            pos.z.floor() + me_pos.z + CHUNK_VOXEL_SIZE as f32
        );

        let x = x.clamp(0.0, CHUNK_VOXEL_SIZE as f32 - 1.0) as usize;
        let z = z.clamp(0.0, CHUNK_VOXEL_SIZE as f32 - 1.0) as usize;

        let mut height = 0.0;

        for y in 0..CHUNK_VOXEL_SIZE
        {
            let sample_pos = glm::U16Vec3::new(x as u16, y as u16, z as u16);

            if self.voxel_chunk_data.read_voxel(sample_pos) != Voxel::Air
            {
                height = y as f32 - me_pos.y - CHUNK_VOXEL_SIZE as f32 + 1.0;
            }
        }

        height
    }
}

impl gfx::Recordable for BrickMapChunk
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed(&self.name)
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        todo!()
        // gfx::RecordInfo {
        //     should_draw: true,
        //     transform:   Some(gfx::Transform {
        //         translation: *self.position.lock().unwrap(),
        //         ..Default::default()
        //     }),
        //     bind_groups: [
        //         Some(global_bind_group.clone()),
        //         Some(
        //             // TODO: modify the manager to be thread safe
        //             self.voxel_chunk_data.get_bind_group()
        //         ),
        //         None,
        //         None
        //     ]
        // }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}

impl game::EntityCastDepot for BrickMapChunk
{
    fn as_entity(&self) -> Option<&dyn Entity>
    {
        Some(self)
    }

    fn as_positionable(&self) -> Option<&dyn Positionable>
    {
        Some(self)
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
    }
}

impl game::Entity for BrickMapChunk
{
    fn get_name(&self) -> Cow<'_, str>
    {
        gfx::Recordable::get_name(self)
    }

    fn get_uuid(&self) -> util::Uuid
    {
        gfx::Recordable::get_uuid(self)
    }

    fn tick(&self, _: &game::Game, _: game::TickTag) {}
}

impl game::Positionable for BrickMapChunk
{
    fn get_position(&self) -> glm::Vec3
    {
        *self.position.lock().unwrap()
    }

    fn get_position_mut(&self, func: &dyn Fn(&mut glm::Vec3))
    {
        func(&mut self.position.lock().unwrap())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Vertex
{
    position: glm::Vec3
}

impl Vertex
{
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    pub fn describe_layout() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBUTES
        }
    }
}

const VOXELS_EDGE_LENGTH: f32 = CHUNK_VOXEL_SIZE as f32;

const CUBE_VERTICES: [Vertex; 8] = [
    Vertex {
        position: glm::Vec3::new(0.0, 0.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(0.0, 0.0, VOXELS_EDGE_LENGTH)
    },
    Vertex {
        position: glm::Vec3::new(0.0, VOXELS_EDGE_LENGTH, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(0.0, VOXELS_EDGE_LENGTH, VOXELS_EDGE_LENGTH)
    },
    Vertex {
        position: glm::Vec3::new(VOXELS_EDGE_LENGTH, 0.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(VOXELS_EDGE_LENGTH, 0.0, VOXELS_EDGE_LENGTH)
    },
    Vertex {
        position: glm::Vec3::new(VOXELS_EDGE_LENGTH, VOXELS_EDGE_LENGTH, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(VOXELS_EDGE_LENGTH, VOXELS_EDGE_LENGTH, VOXELS_EDGE_LENGTH)
    }
];

#[rustfmt::skip]
const CUBE_INDICES: [u16; 36] = [
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

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use fnv::{FnvHashMap, FnvHashSet};
use gfx::{glm, wgpu, CacheablePipelineLayoutDescriptor};
use itertools::Itertools;
use rapier3d::dynamics::{RigidBody, RigidBodyBuilder};
use rapier3d::geometry::{Collider, ColliderBuilder, SharedShape};
use rapier3d::math::{Isometry, Real};

use crate::data::{self, VoxelFace, VoxelFaceDirection};
use crate::suballocated_buffer::SubAllocatedCpuTrackedDenseSet;
use crate::{
    get_world_offset_of_chunk,
    world_position_to_chunk_position,
    ChunkCoordinate,
    ChunkLocalPosition,
    SubAllocatedCpuTrackedBuffer,
    WorldPosition,
    CHUNK_EDGE_LEN_VOXELS
};

#[no_mangle]
static NUMBER_OF_CHUNKS: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
static NUMBER_OF_VISIBLE_FACES: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
static NUMBER_OF_TOTAL_FACES: AtomicUsize = AtomicUsize::new(0);

pub struct ChunkManager
{
    game:               Arc<game::Game>,
    uuid:               util::Uuid,
    pipeline:           Arc<gfx::GenericPipeline>,
    indirect_buffer:    wgpu::Buffer,
    instance_data:      wgpu::Buffer,
    face_id_bind_group: Arc<wgpu::BindGroup>,

    global_face_data: Mutex<SubAllocatedCpuTrackedBuffer<VoxelFace>>,
    chunks:           Mutex<FnvHashMap<ChunkCoordinate, Chunk>>,

    number_of_indirect_calls_flushed: AtomicU32
}
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

        let mut allocator = Mutex::new(SubAllocatedCpuTrackedBuffer::new(
            renderer.clone(),
            1048576 * 96,
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

        let buffer_len_bytes = allocator.get_mut().unwrap().get_buffer_size_bytes();

        let face_ids_bind_group =
            Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                layout:  &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: allocator.get_mut().unwrap().access_buffer(),
                        offset: 0,
                        size:   Some(buffer_len_bytes)
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
            chunks:                           Mutex::new(FnvHashMap::default()),
            global_face_data:                 allocator,
            game:                             game.clone(),
            uuid:                             util::Uuid::new(),
            face_id_bind_group:               face_ids_bind_group,
            instance_data:                    renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Indirect Instance Data Buffer"),
                size:               std::mem::size_of::<PackedInstanceData>() as u64 * 32768,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }),
            pipeline:                         renderer.render_cache.cache_render_pipeline(
                gfx::CacheableRenderPipelineDescriptor {
                    label: "ChunkManager Pipeline Indirect".into(),
                    layout: Some(layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![PackedInstanceData::desc()],
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
                    zero_initialize_vertex_workgroup_memory: false,
                    fragment_specialization: None,
                    zero_initialize_fragment_workgroup_memory: false
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

    pub fn insert_many_voxel(&self, it: impl IntoIterator<Item = WorldPosition>)
    {
        let iterator = it.into_iter();

        let mut chunks = iterator.array_chunks::<3072>();

        for i in chunks.by_ref()
        {
            self.insert_many_voxel_deadlocking(i);
        }

        if let Some(remainder) = chunks.into_remainder()
        {
            self.insert_many_voxel_deadlocking(remainder);
        }
    }

    // fn new_insert_many_voxel_deadlocking(&)

    fn insert_many_voxel_deadlocking(&self, pos: impl IntoIterator<Item = WorldPosition>)
    {
        let mut allocator = self.global_face_data.lock().unwrap();
        let mut chunks = self.chunks.lock().unwrap();

        for p in pos
        {
            let (chunk_coordinate, local_pos) = world_position_to_chunk_position(p);

            match chunks.entry(chunk_coordinate)
            {
                Entry::Occupied(mut e) => e.get_mut().insert_voxel(local_pos, &mut allocator),
                Entry::Vacant(e) =>
                {
                    e.insert(Chunk::new(&mut allocator))
                        .insert_voxel(local_pos, &mut allocator)
                }
            }
        }
    }

    // pub fn does_many_voxel_exist(&self, positions: &[WorldPosition]) -> Vec<bool>
    // {
    //     let (coordinate, local_pos) = world_position_to_chunk_position(pos);

    //     self.chunks.lock()
    // }

    pub fn build_collision_info(&self) -> Arc<ChunkCollider>
    {
        ChunkCollider::new(self.game.clone(), &self.chunks.lock().unwrap())
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
        camera: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let mut indirect_args: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();
        let mut indirect_data: Vec<PackedInstanceData> = Vec::new();

        let mut face_allocator = self.global_face_data.lock().unwrap();
        let chunks = self.chunks.lock().unwrap();

        face_allocator.replicate_to_gpu();

        // TODO: tomorrow 0.5 for shaded axis chunks and 0.0 elsewhere /shrug
        let mut idx = 0;
        let mut total_number_of_faces = 0;
        let mut rendered_faces = 0;

        let chunk_offsets: [glm::I32Vec3; 8] = std::array::from_fn(|idx| {
            glm::I32Vec3::new(
                (CHUNK_EDGE_LEN_VOXELS * (idx / 4)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * ((idx / 2) % 2)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * (idx % 2)) as i32
            )
        });

        for (coordinate, chunk) in chunks.iter()
        {
            let draw_ranges = chunk.get_draw_ranges(&mut face_allocator);

            for range in draw_ranges
            {
                if let Some((r, dir)) = range
                {
                    total_number_of_faces += r.end + 1 - r.start;

                    let camera_world_position = camera.get_position();

                    if r.is_empty()
                    {
                        continue;
                    }

                    'outer: for offset in chunk_offsets.iter().map(|o| {
                        (o + get_world_offset_of_chunk(*coordinate).0).cast()
                            - camera_world_position
                    })
                    {
                        let is_camera_in_chunk = offset.magnitude() < CHUNK_EDGE_LEN_VOXELS as f32;

                        let is_chunk_visible = {
                            let is_chunk_in_camera_view =
                                offset.normalize().dot(&camera.get_forward_vector())
                                    > (renderer.get_fov().max() / 2.0).cos();

                            is_chunk_in_camera_view || is_camera_in_chunk
                        };

                        let is_chunk_direction_visible =
                            offset.normalize().dot(&dir.get_axis().cast()) < 0.0;

                        if is_camera_in_chunk || (is_chunk_visible && is_chunk_direction_visible)
                        {
                            indirect_args.push(wgpu::util::DrawIndirectArgs {
                                vertex_count:   (r.end + 1 - r.start) * 6,
                                instance_count: 1,
                                first_vertex:   r.start * 6,
                                first_instance: idx as u32
                            });

                            indirect_data.push(PackedInstanceData {
                                chunk_world_offset: get_world_offset_of_chunk(*coordinate).0.cast(),
                                normal_id:          dir as u32
                            });

                            idx += 1;

                            rendered_faces += r.end + 1 - r.start;
                            break 'outer;
                        }
                    }
                }
            }
        }

        fn draw_args_as_bytes(args: &[wgpu::util::DrawIndirectArgs]) -> &[u8]
        {
            unsafe {
                std::slice::from_raw_parts(
                    args.as_ptr() as *const u8,
                    args.len() * std::mem::size_of::<wgpu::util::DrawIndirectArgs>()
                )
            }
        }

        NUMBER_OF_CHUNKS.store(chunks.len(), Ordering::Relaxed);
        NUMBER_OF_VISIBLE_FACES.store(rendered_faces as usize, Ordering::Relaxed);
        NUMBER_OF_TOTAL_FACES.store(total_number_of_faces as usize, Ordering::Relaxed);

        self.number_of_indirect_calls_flushed
            .store(indirect_args.len() as u32, Ordering::SeqCst);

        renderer.queue.write_buffer(
            &self.indirect_buffer,
            0,
            draw_args_as_bytes(&indirect_args[..])
        );

        renderer
            .queue
            .write_buffer(&self.instance_data, 0, cast_slice(&indirect_data[..]));

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

        pass.set_vertex_buffer(0, self.instance_data.slice(..));
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.multi_draw_indirect(
            &self.indirect_buffer,
            0,
            self.number_of_indirect_calls_flushed.load(Ordering::SeqCst)
        );
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PackedInstanceData
{
    pub chunk_world_offset: glm::Vec3,
    pub normal_id:          u32 // TODO: add a scale factor
}

impl PackedInstanceData
{
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Instance,
            attributes:   &Self::ATTRIBS
        }
    }
}

struct DirectionalFaceData
{
    dir:             data::VoxelFaceDirection,
    faces_dense_set: SubAllocatedCpuTrackedDenseSet<VoxelFace>
}

impl DirectionalFaceData
{
    pub fn new(
        allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>,
        dir: data::VoxelFaceDirection
    ) -> DirectionalFaceData
    {
        Self {
            dir:             dir.clone(),
            faces_dense_set: SubAllocatedCpuTrackedDenseSet::new(1024, allocator)
        }
    }

    pub fn insert_face(
        &mut self,
        face: VoxelFace,
        allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>
    )
    {
        self.faces_dense_set.insert(face, allocator);
    }

    pub fn remove_face(
        &mut self,
        face: VoxelFace,
        allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>
    )
    {
        // TODO: deal with duplicates properly
        let _ = self.faces_dense_set.remove(face, allocator);
    }
}

struct Chunk
{
    drawable_faces: [Option<DirectionalFaceData>; 6],
    voxel_exists:   FnvHashSet<ChunkLocalPosition>
}

impl Chunk
{
    pub fn new(allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>) -> Chunk
    {
        Chunk {
            drawable_faces: std::array::from_fn(|i| {
                Some(DirectionalFaceData::new(
                    allocator,
                    VoxelFaceDirection::try_from(i as u8).unwrap()
                ))
            }),
            voxel_exists:   FnvHashSet::default()
        }
    }

    pub fn insert_voxel(
        &mut self,
        local_pos: ChunkLocalPosition,
        allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>
    )
    {
        self.voxel_exists.insert(local_pos);

        for d in VoxelFaceDirection::iterate()
        {
            if let Some(adj_pos) = (local_pos.0.cast() + d.get_axis())
                .try_cast()
                .map(ChunkLocalPosition)
            {
                if !self.voxel_exists.contains(&adj_pos)
                // other is empty
                {
                    self.drawable_faces[d as usize]
                        .as_mut()
                        .unwrap()
                        .insert_face(VoxelFace::new(local_pos, glm::U8Vec2::new(1, 1)), allocator);
                }
                else
                {
                    self.drawable_faces[d.opposite() as usize]
                        .as_mut()
                        .unwrap()
                        .remove_face(VoxelFace::new(adj_pos, glm::U8Vec2::new(1, 1)), allocator);
                }
            }
        }
    }

    pub fn get_draw_ranges(
        &self,
        allocator: &mut SubAllocatedCpuTrackedBuffer<VoxelFace>
    ) -> [Option<(Range<u32>, VoxelFaceDirection)>; 6]
    {
        std::array::from_fn(|i| {
            unsafe {
                self.drawable_faces
                    .get_unchecked(i)
                    .as_ref()
                    .map(|d| (d.faces_dense_set.get_global_range(allocator), d.dir))
            }
        })
    }
}

pub struct ChunkCollider
{
    uuid: util::Uuid,

    data: Mutex<Option<(RigidBody, Vec<Collider>)>>
}

impl ChunkCollider
{
    fn new(game: Arc<game::Game>, voxels: &FnvHashMap<ChunkCoordinate, Chunk>) -> Arc<Self>
    {
        let voxel_aabb: SharedShape = SharedShape::cuboid(0.5, 0.5, 0.5);

        let it = voxels
            .iter()
            .map(|(coord, chunk)| {
                chunk
                    .voxel_exists
                    .iter()
                    .map((|pos| pos.0.cast() + get_world_offset_of_chunk(*coord).0))
            })
            .flatten()
            .map(|world_voxel_pos| {
                (
                    Isometry::<Real>::new(
                        world_voxel_pos.cast() + glm::Vec3::repeat(0.5),
                        glm::Vec3::identity()
                    ),
                    voxel_aabb.clone()
                )
            })
            .collect_vec();

        let a: (RigidBody, Vec<Collider>) = (
            RigidBodyBuilder::fixed().build(),
            vec![ColliderBuilder::compound(it).build()]
        );

        let this = Arc::new(ChunkCollider {
            uuid: util::Uuid::new(),
            data: Mutex::new(Some(a))
        });

        game.register(this.clone());

        this
    }
}

impl game::EntityCastDepot for ChunkCollider
{
    fn as_self_managed(self: Arc<Self>) -> Option<Arc<dyn game::SelfManagedEntity>>
    {
        None
    }

    fn as_positionalable(&self) -> Option<&dyn game::Positionalable>
    {
        Some(self)
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        Some(self)
    }

    fn as_collideable(&self) -> Option<&dyn game::Collideable>
    {
        Some(self)
    }
}

impl game::Entity for ChunkCollider
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Chunk Collider")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn tick(&self, _: &game::Game, _: game::TickTag) {}
}

impl game::Positionalable for ChunkCollider
{
    fn get_position(&self) -> glm::Vec3
    {
        glm::Vec3::zeros()
    }
}

impl game::Transformable for ChunkCollider
{
    fn get_transform(&self) -> gfx::Transform
    {
        gfx::Transform::new()
    }
}

impl game::Collideable for ChunkCollider
{
    fn init_collideable(&self) -> (RigidBody, Vec<Collider>)
    {
        self.data.lock().unwrap().take().unwrap()
    }

    fn physics_tick(&self, _: &game::Game, _: glm::Vec3, _: &mut RigidBody, _: game::TickTag) {}
}

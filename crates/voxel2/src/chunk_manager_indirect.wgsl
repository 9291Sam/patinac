
struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

const NumberOfModels: u32 = 1024;

alias GlobalMatricies = array<mat4x4<f32>, NumberOfModels>;
alias GlobalPositions = array<vec3<f32>, NumberOfModels>;

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: GlobalMatricies;
@group(0) @binding(2) var<uniform> global_model: GlobalMatricies;

@group(1) @binding(0) var<storage, read> face_id_buffer: array<u32>;

var<push_constant> pc_id: u32;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput
{
    var FACE_LOOKUP_TABLE: array<array<vec3<u32>, 4>, 6> =
    array<array<vec3<u32>, 4>, 6>(
        TOP_FACE_POINTS,
        BOTTOM_FACE_POINTS,
        LEFT_FACE_POINTS,
        RIGHT_FACE_POINTS,
        FRONT_FACE_POINTS,
        BACK_FACE_POINTS
    );

    var IDX_TO_VTX_TABLE: array<u32, 6> = array<u32, 6>(0, 1, 2, 2, 1, 3);

    let point_within_face: u32 = vertex_index % 6;
    let face_number: u32 = vertex_index / 6;

    let face_id: u32 = face_id_buffer[face_number];
    // let face_data: FaceData = face_data_buffer[face_id];

    let eight_bit_mask: u32 = u32(255);
    let three_bit_mask: u32 = u32(7);
    
    let x_pos: u32 = face_id & eight_bit_mask;
    let y_pos: u32 = (face_id >> 8) & eight_bit_mask;
    let z_pos: u32 = (face_id >> 16) & eight_bit_mask;

    let face_voxel_pos = vec3<u32>(x_pos, y_pos, z_pos);
    let face_normal_id = instance_index & three_bit_mask;

    // var material = face_data.mat_and_chunk_id & sixteen_bit_mask;
    // let chunk_id = (face_data.mat_and_chunk_id >> 16) & sixteen_bit_mask;
    // let chunk_data = chunk_data_buffer[chunk_id];
    // let chunk_pos = chunk_data.position;
    let chunk_pos = vec3<f32>(0.0);
    // let chunk_scale = chunk_data.scale;
    let chunk_scale = vec3<f32>(1.0);

    // if (face_id == 0 )
    // {
    //     material = 0u;
    // }

    let face_point_local: vec3<f32> = vec3<f32>(FACE_LOOKUP_TABLE[face_normal_id][IDX_TO_VTX_TABLE[point_within_face]]) * chunk_scale.xyz;
    let face_point_world = vec4<f32>(face_point_local + vec3<f32>(face_voxel_pos) + chunk_pos.xyz, 1.0);

    return VertexOutput(
        global_model_view_projection[pc_id] * face_point_world,
        face_normal_id, 
    );
}

struct VertexOutput
{
    @builtin(position) position: vec4<f32>,
    @location(0) material: u32,
    // @location(1) uv: vec2<f32>,
}

@fragment
fn fs_main(@location(0) m: u32) -> @location(0) vec4<f32>
{
    return vec4<f32>(sin(f32(m) * 3784.0 + 3843.21) / 7, cos(f32(m) - 484.0 ), sin(f32(m) * 3034.33), 1.0); // material_buffer[voxel_material_id].diffuse_color;
}

const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

struct MaterialData
{
    diffuse_color:             vec4<f32>,
    subsurface_color:          vec4<f32>,
    diffuse_subsurface_weight: f32,

    specular_color: vec4<f32>,
    specular:       f32,
    roughness:      f32,
    metallic:       f32,

    emissive_color_and_power: vec4<f32>,
    coat_color_and_power:     vec4<f32>,

    special: u32
}

struct FaceData
{
    // bottom u16 is material
    // top u16 is chunk_id
    mat_and_chunk_id: u32,

    // 9 bits x
    // 9 bits y
    // 9 bits z
    // 3 bits normal
    // 1 bit visibility
    // 1 bit unused
    data: u32,
}

struct ChunkData
{
    position: vec4<f32>,
    scale: vec4<f32>,
}


const TOP_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0, 1, 0),
    vec3<u32>(0, 1, 1),
    vec3<u32>(1, 1, 0),
    vec3<u32>(1, 1, 1)
);

const BOTTOM_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0, 0, 1),
    vec3<u32>(0, 0, 0),
    vec3<u32>(1, 0, 1),
    vec3<u32>(1, 0, 0)
);


const LEFT_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0, 0, 1),
    vec3<u32>(0, 1, 1),
    vec3<u32>(0, 0, 0),
    vec3<u32>(0, 1, 0)
);


const RIGHT_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1, 0, 0),
    vec3<u32>(1, 1, 0),
    vec3<u32>(1, 0, 1),
    vec3<u32>(1, 1, 1)
);


const FRONT_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0, 0, 0),
    vec3<u32>(0, 1, 0),
    vec3<u32>(1, 0, 0),
    vec3<u32>(1, 1, 0)
);


const BACK_FACE_POINTS: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(1, 0, 1),
    vec3<u32>(1, 1, 1),
    vec3<u32>(0, 0, 1),
    vec3<u32>(0, 1, 1)
);
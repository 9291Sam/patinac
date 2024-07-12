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

@group(1) @binding(0) var<storage, read> face_data_buffer: array<u32>;
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>;
// @group(1) @binding(3) var<storage, read> visiblity_bricks: array<u32>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;

var<push_constant> pc_id: u32;

struct VertexInput {
    @location(0) chunk_position: vec3<f32>,
    @location(1) normal_id: u32,
    @location(2) chunk_id: u32,
}

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput
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

    let face_data: u32 = face_data_buffer[face_number];

    let eight_bit_mask: u32 = u32(255);
    let three_bit_mask: u32 = u32(7);
    
    let x_pos: u32 = face_data & eight_bit_mask;
    let y_pos: u32 = (face_data >> 8) & eight_bit_mask;
    let z_pos: u32 = (face_data >> 16) & eight_bit_mask;

    let face_voxel_pos = vec3<u32>(x_pos, y_pos, z_pos);
    let face_normal_id = in.normal_id;

    let face_point_local: vec3<f32> = vec3<f32>(FACE_LOOKUP_TABLE[face_normal_id][IDX_TO_VTX_TABLE[point_within_face]]);
    let face_point_world = vec4<f32>(face_point_local + vec3<f32>(face_voxel_pos) + in.chunk_position, 1.0);
    
    return VertexOutput(
        global_model_view_projection[pc_id] * face_point_world,
        in.chunk_id | (in.normal_id << 27),
        face_number 
    );
}

fn rand(s: u32) -> f32
{
    return fract(sin(f32(s) / 473.489484));
}

fn randvec3(s: vec3<u32>) -> vec3<f32>
{
    return vec3<f32>(
        rand(s.x),
        rand(s.y),
        rand(s.z),
    );
}


struct VertexOutput
{
    @builtin(position) position: vec4<f32>,
    @location(0) chunk_and_dir: u32,
    @location(1) face_number: u32,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32>
{
    return vec2<u32>(in.chunk_and_dir, in.face_number);
}

const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

struct MaterialData
{
    // special: u32,
    diffuse_color:    vec4<f32>,
    subsurface_color: vec4<f32>,
    specular_color:   vec4<f32>,

    diffuse_subsurface_weight: f32,
    specular:                  f32,
    roughness:                 f32,
    metallic:                  f32,

    emissive_color_and_power: vec4<f32>,
    coat_color_and_power:     vec4<f32>,

}

struct BrickMap
{
    map: array<array<array<u32, 32>, 32>, 32>
}

struct MateralBrick
{
    voxels: array<array<array<u32, 4>, 8>, 8>,
}

fn material_bricks_load(brick_ptr: u32, pos: vec3<u32>) -> u32
{
    let val = material_bricks[brick_ptr].voxels[pos.x][pos.y][pos.z / 2];

    if (pos.z % 2 == 0) // low u16
    {
        return val & u32(65535);
    }
    else // high u16
    {
        return val >> 16;
    }
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
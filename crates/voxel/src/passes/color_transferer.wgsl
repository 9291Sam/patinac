
// Texture<FromFragmentShader>
@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
// [number_of_image_workgroups, 1, 1];
// @group(0) @binding(1) var<storage, read_write> indirect_color_calc_buffer: array<atomic<u32>, 3>;
// // [FaceIdInfo, ...]
// @group(0) @binding(2) var<storage, read_write> face_id_buffer: array<FaceInfo>;
// // number_of_voxels_in_buffer
// @group(0) @binding(3) var<storage, read_write> number_of_unique_voxels: atomic<u32>;
// // [FaceId, ...]
// @group(0) @binding(4) var<storage, read_write> unique_voxel_buffer: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> face_data_buffer: array<u32>;
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>;
// @group(1) @binding(3) var<storage, read> visiblity_bricks: array<u32>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

@group(2) @binding(0) var<uniform> global_info: GlobalInfo;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32>
{
    switch (index)
    {
        case 0u:      {return vec4<f32>(-1.0, 3.0, 0.5, 1.0); }
        case 1u:      {return vec4<f32>(3.0, -1.0, 0.5, 1.0); }
        case 2u:      {return vec4<f32>(-1.0, -1.0, 0.5, 1.0); }
        case default: {return vec4<f32>(0.0); }
    }
}

@fragment
fn fs_main(@builtin(position) in: vec4<f32>) -> @location(0) vec4<f32>
{
    let voxel_data: vec2<u32> = textureLoad(voxel_discovery_image, vec2<u32>(u32(in.x), u32(in.y)), 0).xy;

    if (all(voxel_data == vec2<u32>(0)))
    {
        discard;
    }
    
    let eight_bit_mask: u32 = u32(255);

    let chunk_id = voxel_data.x;

    let x_pos: u32 = voxel_data[1] & eight_bit_mask;
    let y_pos: u32 = (voxel_data[1] >> 8) & eight_bit_mask;
    let z_pos: u32 = (voxel_data[1] >> 16) & eight_bit_mask;
    let normal_id: u32 = (voxel_data[1] >> 27) & u32(7);


    var normal: vec3<f32>;

    switch (normal_id)
    {
        case 0u: {normal = vec3<f32>(0.0, 1.0, 0.0); }
        case 1u: {normal = vec3<f32>(0.0, -1.0, 0.0); }     
        case 2u: {normal = vec3<f32>(-1.0, 0.0, 0.0); }       
        case 3u: {normal = vec3<f32>(1.0, 0.0, 0.0); }       
        case 4u: {normal = vec3<f32>(0.0, 0.0, -1.0); }      
        case 5u: {normal = vec3<f32>(0.0, 0.0, 1.0); }
        case default: {normal = vec3<f32>(0.0); }
    }


    let face_voxel_pos = vec3<u32>(x_pos, y_pos, z_pos);
    let global_face_voxel_position = gpu_chunk_data[chunk_id].xyz + vec3<f32>(face_voxel_pos);
    
    let brick_coordinate = face_voxel_pos / 8u;
    let brick_local_coordinate = face_voxel_pos % 8u;

    let brick_ptr = brick_map[chunk_id].map[brick_coordinate.x][brick_coordinate.y][brick_coordinate.z];
    let voxel = material_bricks_load(brick_ptr, brick_local_coordinate);

    let ambient_strength = 0.025;
    let ambient = ambient_strength * vec3<f32>(1.0);

    let light_color = vec4<f32>(.7, 0.9, 0.5, 32.0);
    let light_pos = vec3<f32>(120.0, 15.0, -40.0);

    var light_dir = light_pos - global_face_voxel_position;
    let distance = length(light_dir);
    light_dir /= distance;

    // intensity of diffuse
    let diffuse_intensity = saturate(dot(normal, light_dir));
    let diffuse = diffuse_intensity * light_color.xyz * material_buffer[voxel].diffuse_color.xyz * light_color.w / pow(distance, 1.55);

    let view_vector = normalize(global_info.camera_pos.xyz - global_face_voxel_position);
    let h = normalize(light_dir + view_vector);
    let specular_intensity = saturate(pow(dot(normal, h), material_buffer[voxel].specular));

    let specular = specular_intensity * light_color.xyz * material_buffer[voxel].specular_color.xyz * light_color.w / pow(distance, 1.55);



    let result = saturate(ambient + diffuse + specular);
    // FragColor = vec4(result, 1.0);

    return vec4<f32>(result, 1.0);
    // return vec4<f32>(global_face_voxel_position / 512.0, 1.0);

    // return vec4<f32>(material_buffer[voxel].diffuse_color.xyz, 1.0);
    // return vec4<f32>((normal + 1) / 2, 1.0);
    // return vec4<f32>(vec3<f32>(face_voxel_pos) / 255.0, 1.0);
}

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

fn get_voxel_color(voxel: u32) -> vec4<f32>
{
    switch voxel
    {
        case 0u:  {return ERROR_COLOR;}
        case 1u:  {return vec4<f32>(0.5, 0.5, 0.5, 1.0);}
        case 2u:  {return vec4<f32>(0.6, 0.6, 0.6, 1.0);}
        case 3u:  {return vec4<f32>(0.7, 0.7, 0.7, 1.0);}
        case 4u:  {return vec4<f32>(0.8, 0.8, 0.8, 1.0);}
        case 5u:  {return vec4<f32>(0.9, 0.9, 0.9, 1.0);}
        case 6u:  {return vec4<f32>(1.0, 1.0, 1.0, 1.0);}
        case 7u:  {return vec4<f32>(0.0, 0.5, 0.0, 1.0);}
        case 8u:  {return vec4<f32>(0.0, 0.6, 0.0, 1.0);}
        case 9u:  {return vec4<f32>(0.0, 0.7, 0.0, 1.0);}
        case 10u: {return vec4<f32>(0.0, 0.8, 0.0, 1.0);}
        case 11u: {return vec4<f32>(0.0, 0.9, 0.0, 1.0);}
        case 12u: {return vec4<f32>(0.2, 0.5, 0.2, 1.0);}
        default:  {return ERROR_COLOR;}
    }
}

fn rand(i: u32) -> u32
{
    let index = (i << 13) ^ i;

    return (index * (index * index * 15731 + 789221) + 1376312589);
}

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
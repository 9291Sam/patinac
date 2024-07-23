
@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;

@group(1) @binding(0) var<storage, read> face_data_buffer: array<VoxelFaceData>; 
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>; 
@group(1) @binding(3) var<storage, read> visiblity_bricks: array<VisibilityBrick>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> is_face_number_visible_bool: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> face_numbers_to_face_ids: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> next_face_id: atomic<u32>;
@group(1) @binding(9) var<storage, read_write> renderered_face_info: array<RenderedFaceInfo>;

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

    if (all(voxel_data == vec2<u32>(0))) // TODO: better null sentienl
    {
        discard;
    }
    
    let face_number = voxel_data.y;
    let face_id = face_numbers_to_face_ids[face_number];
    let color = renderered_face_info[face_id].color;
  
    return color;
}


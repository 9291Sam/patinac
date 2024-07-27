@group(1) @binding(0) var<storage, read> face_data_buffer: array<VoxelFaceData>; 
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>; 
@group(1) @binding(3) var<storage, read> visiblity_bricks: array<VisibilityBrick>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> is_face_number_visible_bits: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> face_numbers_to_face_ids: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> next_face_id: u32;
@group(1) @binding(9) var<storage, read_write> renderered_face_info: array<RenderedFaceInfo>;
@group(1) @binding(10) var<storage, read> point_lights: array<PointLight>;
@group(1) @binding(11) var<storage, read> point_light_number: u32;

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;


@compute @workgroup_size(256)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
){ 
    let global_invocation_index = global_invocation_id.x;
    
    if (global_invocation_index < next_face_id)
    {
        let chunk_id = renderered_face_info[global_invocation_index].chunk_id;
        let combined_dir_and_pos = renderered_face_info[global_invocation_index].combined_dir_and_pos;

        let eight_bit_mask: u32 = u32(255);

        let x_pos: u32 = combined_dir_and_pos & eight_bit_mask;
        let y_pos: u32 = (combined_dir_and_pos >> 8) & eight_bit_mask;
        let z_pos: u32 = (combined_dir_and_pos >> 16) & eight_bit_mask;
        let dir:   u32 = (combined_dir_and_pos >> 24) & u32(7);

        let face_voxel_pos = vec3<u32>(x_pos, y_pos, z_pos);

        var normal: vec3<f32>;

        switch (dir)
        {
            case 0u: {normal = vec3<f32>(0.0, 1.0, 0.0); }
            case 1u: {normal = vec3<f32>(0.0, -1.0, 0.0); }     
            case 2u: {normal = vec3<f32>(-1.0, 0.0, 0.0); }       
            case 3u: {normal = vec3<f32>(1.0, 0.0, 0.0); }       
            case 4u: {normal = vec3<f32>(0.0, 0.0, -1.0); }      
            case 5u: {normal = vec3<f32>(0.0, 0.0, 1.0); }
            case default: {normal = vec3<f32>(0.0); }
        }

        let global_face_voxel_position = gpu_chunk_data[chunk_id].xyz + vec3<f32>(face_voxel_pos);
        
        let brick_coordinate = face_voxel_pos / 8u;
        let brick_local_coordinate = face_voxel_pos % 8u;

        let brick_ptr = brick_map_load(chunk_id, brick_coordinate);
        let voxel = material_bricks_load(brick_ptr, brick_local_coordinate);

        var sum_diffuse_strength = vec3<f32>(0.0);
        var sum_specular_strength = vec3<f32>(0.0);

        for (var i: u32 = 0; i < point_light_number; i++)
        {
            let res = calculate_light_power(
                global_info.camera_pos.xyz, 
                global_face_voxel_position,
                normal,
                point_lights[i],
                material_buffer[voxel].specular
            );
            
            sum_diffuse_strength += res.diffuse_strength;
            sum_specular_strength += res.specular_strength;
        }

        let real_diffuse_color = exp_strength_falloff(sum_diffuse_strength) * material_buffer[voxel].diffuse_color.xyz;
        let real_specular_color = exp_strength_falloff(sum_specular_strength) * material_buffer[voxel].specular_color.xyz;
        let real_ambient_color = 0.005 * vec3<f32>(1.0);

        let real_max_color = saturate(real_ambient_color + real_diffuse_color + real_specular_color); 
        // max(
        //     real_ambient_color,
        //     max(
        //         real_diffuse_color,
        //         real_specular_color
        //     )
        // );
        
        renderered_face_info[global_invocation_index].color = vec4<f32>(real_max_color, 1.0);
    
    }
}

fn exp_strength_falloff(strength: vec3<f32>) -> vec3<f32>
{
    return 1.0 - exp2(-0.25 * strength);
}
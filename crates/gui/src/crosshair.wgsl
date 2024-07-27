@group(0) @binding(0) var screen_texture: texture_storage_2d<r32uint, read_write>;

@compute @workgroup_size(32, 32)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3u)
{
    let global_size: vec2u = textureDimensions(screen_texture).xy;
    
    let x_width: u32 = select(2u, 3u, global_size.x % 2 == 0);
    let y_width: u32 = select(2u, 3u, global_size.y % 2 == 0);

    let x_band_low: u32 = 16u - x_width + 1u;
    let y_band_low: u32 = 16u - y_width + 1u;

    let global_texture_center: vec2u = global_size / 2;
    let compute_top_left_start: vec2u = global_texture_center - 16;

    let this_screen_px_global: vec2u = compute_top_left_start + global_invocation_id.xy;

    if (
        global_invocation_id.x >= x_band_low && global_invocation_id.x < (x_band_low + x_width) ||
        global_invocation_id.y >= y_band_low && global_invocation_id.y < (y_band_low + y_width)
    )
    {
        let color: vec4<f32> = unpack4x8unorm(textureLoad(screen_texture, this_screen_px_global).x);

        textureStore(screen_texture, this_screen_px_global, vec4<u32>(pack4x8unorm(1.0 - color), 0u, 0u, 0u));
    }

}
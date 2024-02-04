


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
}

struct PushConstants
{
    mvp: mat4x4<f32>,
    model: mat4x4<f32>
}

var<push_constant> push_constants: PushConstants;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput
{
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = push_constants.mvp * vec4<f32>(model.position, 1.0);
    let world_pos_intercalc = push_constants.model * vec4<f32>(model.position, 1.0);
    out.world_pos = world_pos_intercalc.xyz / world_pos_intercalc.w;

    return out;
}
 
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

const LightPosition: vec3<f32> = vec3<f32>(1.0, 6.6, 7.2);
const LightColor: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 4.0);

const Ambient: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 0.02);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let normal: vec3<f32> = normalize(abs(textureSample(t_normal, s_diffuse, in.tex_coords))).xyz;

    let direction_non_norm: vec3<f32> = (LightPosition - in.world_pos);
    let direction_norm: vec3<f32> = normalize(direction_non_norm);

    let attenuation = LightColor.w / dot(direction_non_norm, direction_non_norm);
    let angle = max(dot(normal, direction_norm), 0.0);
    	
    return vec4<f32>(LightColor.xyz * attenuation * angle, 1.0) * color;
}
 
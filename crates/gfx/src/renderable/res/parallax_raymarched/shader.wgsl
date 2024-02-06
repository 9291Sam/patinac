struct VertexInput {
    @location(0) color: vec3<f32>,
    @location(1) position: vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec3<f32>
}

struct PushConstants
{
    mvp: mat4x4<f32>,
    model: mat4x4<f32>
}

var<push_constant> push_constants: PushConstants;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput
{
    var out: VertexOutput;

    out.clip_position = push_constants.mvp * vec4<f32>(input.position, 1.0);

    let world_pos_intercalc = push_constants.model * vec4<f32>(input.position, 1.0);
    out.world_pos = world_pos_intercalc.xyz / world_pos_intercalc.w;

    out.color = input.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>
{
    return vec4<f32>(in.color, 1.0);
}
 
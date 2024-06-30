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

@group(1) @binding(0) var skybox_texture: texture_cube<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

var<push_constant> pc_id: u32;


@vertex
fn vs_main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = global_model_view_projection[pc_id] * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}

struct FragmentOutput
{
  @location(0) color: vec4f,
  @builtin(frag_depth) frag_depth: f32
}

@fragment
fn fs_main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f,

) -> FragmentOutput {
  // Our camera and the skybox cube are both centered at (0, 0, 0)
  // so we can use the cube geometry position to get viewing vector to sample
  // the cube texture. The magnitude of the vector doesn't matter.
  var cubemapVec = fragPosition.xyz - vec3(0.5);
  // When viewed from the inside, cubemaps are left-handed (z away from viewer),
  // but common camera matrix convention results in a right-handed world space
  // (z toward viewer), so we have to flip it.
  cubemapVec.z *= -1.0;

  // frag_depth = 1.0;
  return FragmentOutput(textureSample(skybox_texture, skybox_sampler, cubemapVec), 0.9999999);
}

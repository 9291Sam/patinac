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
    model: mat4x4<f32>,
    camera_pos: vec3<f32>
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
    let rayDir: vec3<f32> = (in.world_pos - push_constants.camera_pos);
    var rayPos: vec3<f32> = in.world_pos;

    var mapPos: vec3<i32> = vec3<i32>(floor(rayPos + vec3<f32>(0.0)));

    let deltaDist: vec3<f32> = abs(vec3<f32>(length(rayDir)) / rayDir);

    let rayStep: vec3<i32> = vec3<i32>(sign(rayDir));

    var sideDist: vec3<f32> =
        (sign(rayDir) * (vec3<f32>(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;

    var mask: vec3<bool> = vec3<bool>(false, false, false);

    for (var i: i32 = 0; i < MAX_RAY_STEPS; i = i + 1) {
        if (getVoxel(mapPos)) {
            break;
        }

        mask = sideDist.xyz <= min(sideDist.yzx, sideDist.zxy);
        sideDist = sideDist + vec3<f32>(mask) * deltaDist;
        mapPos = mapPos + vec3<i32>(vec3<f32>(mask)) * rayStep;
    }

    let scale: f32 = 0.35;

    let multiplier = ((vec3<f32>(mask) * scale) + (1 - scale));

    let color = get_spherical_coords(vec3<f32>(mapPos)) * multiplier;

    return vec4<f32>(color, 1.0);
}
 
 // The raycasting code is somewhat based around a 2D raycasting tutorial found here: 
// http://lodev.org/cgtutor/raycasting.html

// Constants
const USE_BRANCHLESS_DDA : bool = true;
const MAX_RAY_STEPS : i32 = 192;

// Sphere distance function
fn sdSphere(p: vec3<f32>, d: f32) -> f32 {
    return length(p) - d;
}

// Box distance function
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d: vec3<f32> = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3<f32>(0.0)));
}

// Function to check if a voxel exists at a given position
fn getVoxel(c: vec3<i32>) -> bool {
    let p: vec3<f32> = vec3<f32>(c) + vec3<f32>(0.5);
    let d: f32 = min(max(-sdSphere(p, 7.5), sdBox(p, vec3<f32>(6.0))), -sdSphere(p, 25.0));
    return d < 0.0;
}

// 2D rotation function
// fn rotate2d(v: vec2<f32>, a: f32) -> vec2<f32> {
//     let sinA: f32 = sin(a);
//     let cosA: f32 = cos(a);
//     return vec2<f32>(v.x * cosA - v.y * sinA, v.y * cosA + v.x * sinA);
// }

// Main function
// [[stage(fragment)]]
fn mainImage(fragCoord: vec2<f32>) {
    
}


fn triple32(i_x: u32) -> u32
{
    var x: u32 = i_x;
    x ^= x >> 17;
    x *= 0xed5ad4bbu;
    x ^= x >> 11;
    x *= 0xac4c1b51u;
    x ^= x >> 15;
    x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

fn get_spherical_coords(point: vec3<f32>) -> vec3<f32>
{
    let rho:   f32 = sqrt(dot(point, point));
    let theta: f32 = atan2(point.y, point.x);
    let phi:   f32 = acos(point.z / rho);

    return vec3<f32>(
        // map(rho, 0.0, 64.0, 0.0, 1.0),
        0.0,
        map(theta, -PI, PI, 0.0, 1.0),
        map(phi, 0.0, PI, 0.0, 1.0),
    );
}

const TAU: f32 = 6.2831853071795862;
const PI: f32 = 3.141592653589793;

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32)
    -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
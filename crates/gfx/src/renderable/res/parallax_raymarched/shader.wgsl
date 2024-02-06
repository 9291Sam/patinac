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

    var color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    if (mask.x) {
        color = vec3<f32>(0.5);
    }
    if (mask.y) {
        color = vec3<f32>(1.0);
    }
    if (mask.z) {
        color = vec3<f32>(0.75);
    }

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


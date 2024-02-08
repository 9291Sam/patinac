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
    vp: mat4x4<f32>,
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

struct FragmentOutput
{
   @builtin(frag_depth) depth: f32,
   @location(0) color: vec4<f32>
}

struct Brick
{
    data: array<array<array<u32, 8>, 8>, 8>,
}

@group(0) @binding(0) var<storage> tracking_array: array<u32>;
@group(0) @binding(1) var<storage> brick_array: array<Brick>;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput
{
    // return vec4<f32>(0.0);
    // let ro = in.world_pos + 0.5;
    // let rd = normalize(in.world_pos - push_constants.camera_pos);
    // // ro.xz = rot * ro.xz;
    // // rd.xz = rot * rd.xz;
    
    // var pos = ro;
    // var cell = floor(ro);
    
    // var nextEdge = vec3<f32>(rd > vec3<f32>(0.0));
    // var steps = (nextEdge - fract(pos)) / rd;
    // var originalStepSizes = abs(1.0 / rd);
    // var rdSign = sign(rd);
    
    // var i = 0;
    // for (; i < MAX_RAY_STEPS; i++) {
    //     let stepSize = min(steps.x, min(steps.y, steps.z));
    //     pos += rd * stepSize;
    //     let stepAxis = vec3<f32>(steps <= vec3(stepSize));
    //     cell += stepAxis * rdSign;
    //     steps += originalStepSizes * stepAxis - stepSize;
    
    //     if (getVoxel(vec3<i32>(cell))) {
    //         // return vec4<f32>(get_random_color(cell), 1.0);
    //         // // vec3 normal = stepAxis * rdSign;
    //         // // fragColor.rgb = vec3(dot(normal, normalize(vec3(1, 3, 2)))) * 0.5 + 0.5;
    //         // // return;
    //         break;
    //     }
    // }

    // if (i == MAX_RAY_STEPS)
    // {
    //     discard;
    // }

    
    let rayDir: vec3<f32> = (in.world_pos - push_constants.camera_pos );
    var rayPos: vec3<f32> = in.world_pos + 0.5;

    var mapPos: vec3<i32> = vec3<i32>(floor(rayPos + vec3<f32>(0.0)));

    let deltaDist: vec3<f32> = abs(vec3<f32>(length(rayDir)) / rayDir);

    let rayStep: vec3<i32> = vec3<i32>(sign(rayDir));

    var sideDist: vec3<f32> =
        (sign(rayDir) * (vec3<f32>(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;

    var mask: vec3<bool> = vec3<bool>(false, false, false);

    var i: i32 = 0;
    for (; i < MAX_RAY_STEPS; i = i + 1) {
        if (getVoxel(mapPos)) {
            break;
        }

        mask = sideDist.xyz <= min(sideDist.yzx, sideDist.zxy);
        sideDist = sideDist + vec3<f32>(mask) * deltaDist;
        mapPos = mapPos + vec3<i32>(vec3<f32>(mask)) * rayStep;
    }

    if (i == MAX_RAY_STEPS)
    {
        discard;
    }

    var out: FragmentOutput;

    out.color = vec4<f32>(get_random_color(vec3<f32>(mapPos)), 1.0);
    // let strike = pos; // + (-rd * 0.5);

    // // out.color = vec4<f32>(pos - 0.5, 1.0);
    // let depth_intercalc = push_constants.vp * vec4<f32>(strike, 1.0);
    // out.depth = depth_intercalc.z / depth_intercalc.w;




    // out.depth 

    var c: Cube;
    c.center = vec3<f32>(mapPos) + 0.5;
    c.edge_length = 1.0;

    var r: Ray;
    r.origin = rayPos;
    r.direction = rayDir;
    
    let res = Cube_tryIntersect(c, r);

    if (!res.intersection_occurred)
    {
        out.color = ERROR_COLOR;
        return out;
    }

    let strike_pos_world: vec3<f32> = res.maybe_hit_point - 0.5;
    
    let depth_intercalc = push_constants.vp * vec4<f32>(strike_pos_world, 1.0);
    let maybe_depth = depth_intercalc.z / depth_intercalc.w;

    if (maybe_depth > 1.0)
    {
        out.color = ERROR_COLOR;
        return out;
    }

    out.depth = maybe_depth;

    return out;
}

// Constants
const USE_BRANCHLESS_DDA : bool = true;
const MAX_RAY_STEPS : i32 = 192;
const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

// Sphere distance function
fn sdSphere(p: vec3<f32>, d: f32) -> f32
{
    return length(p) - d;
}

// Box distance function
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d: vec3<f32> = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3<f32>(0.0)));
}

// Function to check if a voxel exists at a given position
fn getVoxel(c: vec3<i32>) -> bool
{
    // if (any(c < vec3<i32>(-64)) || any(c > vec3<i32>(63)))
    // {
    //     return false;
    // }

    // let b_idx: vec3<u32> = vec3<u32>((c / vec3<i32>(8)) + vec3<i32>(8));

    // let maybe_brick_ptr = tracking_array[64 * b_idx.x + 8 * b_idx.y + b_idx.z];

    // if (maybe_brick_ptr == 0)
    // {
    //     return false;
    // }

    // if (brick_array[maybe_brick_ptr].data[c.x % 8][c.y % 8][c.z % 8] != 0)
    // {
    //     return true;
    // }

    // return false;

    if (sqrt(dot(vec3<f32>(c), vec3<f32>(c))) > 27.0)
    {
        return false;
    }

    let mask = vec3<i32>(c == vec3<i32>(0, 0, 0));
    if (dot(mask, mask) == 2)
    {
        return true;
    }

    let p: vec3<f32> = vec3<f32>(c) + vec3<f32>(0.5);
    let d: f32 = min(max(-sdSphere(p, 7.5), sdBox(p, vec3<f32>(6.0))), -sdSphere(p, 25.0));
    return d < 0.0;
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

fn get_random_color(point: vec3<f32>) -> vec3<f32>
{
    var out: vec3<f32>;
    out.x = rand(point.xy * point.xz);
    out.y = rand(point.yz * point.yx);
    out.z = rand(point.zx * point.zy);

    return out;
}

const TAU: f32 = 6.2831853071795862;
const PI: f32 = 3.141592653589793;

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32)
    -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

struct Cube {
    center: vec3<f32>,
    edge_length: f32,
};

fn Cube_contains(me: Cube, point: vec3<f32>) -> bool {
    let p0 = me.center - (me.edge_length / 2.0);
    let p1 = me.center + (me.edge_length / 2.0);

    if (all(p0 < point) && all(point < p1)) {
        return true;
    } else {
        return false;
    }
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct IntersectionResult {
    intersection_occurred: bool,
    maybe_distance: f32,
    maybe_hit_point: vec3<f32>,
    maybe_normal: vec3<f32>,
    maybe_color: vec4<f32>,
};

fn Cube_tryIntersect(me: Cube, ray: Ray) -> IntersectionResult {
    let p0 = me.center - (me.edge_length / 2.0);
    let p1 = me.center + (me.edge_length / 2.0);

    if (Cube_contains(me, ray.origin)) {
        var res: IntersectionResult;
        
        res.intersection_occurred = true;
        res.maybe_distance = 0.0;
        res.maybe_hit_point = ray.origin;
        res.maybe_normal = vec3<f32>(0.0);
        res.maybe_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        return res; 
    }

    let t1 = (p0 - ray.origin) / ray.direction;
    let t2 = (p1 - ray.origin) / ray.direction;

    let vMax = max(t1, t2);
    let vMin = min(t1, t2);

    let tMax = min(min(vMax.x, vMax.y), vMax.z);
    let tMin = max(max(vMin.x, vMin.y), vMin.z);

    let hit = tMin <= tMax && tMax > 0.0;

    if (!hit) {

        var res: IntersectionResult;
        
        res.intersection_occurred = false;
        
        return res;
    }

    let hitPoint = ray.origin + tMin * ray.direction;

    var normal: vec3<f32>;

    if (abs(hitPoint.x - p1.x) < 0.0001) {
        normal = vec3<f32>(-1.0, 0.0, 0.0); // Hit right face
    } else if (abs(hitPoint.x - p0.x) < 0.0001) {
        normal = vec3<f32>(1.0, 0.0, 0.0); // Hit left face
    } else if (abs(hitPoint.y - p1.y) < 0.0001) {
        normal = vec3<f32>(0.0, -1.0, 0.0); // Hit top face
    } else if (abs(hitPoint.y - p0.y) < 0.0001) {
        normal = vec3<f32>(0.0, 1.0, 0.0); // Hit bottom face
    } else if (abs(hitPoint.z - p1.z) < 0.0001) {
        normal = vec3<f32>(0.0, 0.0, -1.0); // Hit front face
    } else if (abs(hitPoint.z - p0.z) < 0.0001) {
        normal = vec3<f32>(0.0, 0.0, 1.0); // Hit back face
    } else {
        normal = vec3<f32>(0.0, 0.0, 0.0); // null case
    }

    var res: IntersectionResult;
        
    res.intersection_occurred = true;
    res.maybe_distance = length(ray.origin - hitPoint);
    res.maybe_hit_point = hitPoint;
    res.maybe_normal = normal;
    res.maybe_color = vec4<f32>(0.0, 1.0, 1.0, 1.0);

    return res; 
}

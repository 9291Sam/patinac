struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

alias Matricies = array<mat4x4<f32>, 1024>;
alias MaybeBrickPointer = u32;
alias BrickPointer = u32;

const BrickEdgeLength: u32 = 8;
const BrickMapEdgeLength: u32 = 128;
const VoxelBricku32Length = (BrickMapEdgeLength * BrickMapEdgeLength * BrickMapEdgeLength / 2);

struct Brick
{
    u16_brick_data: array<u32, VoxelBricku32Length>,
}

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;

@group(1) @binding(0) var<storage> brick_map: array<array<array<MaybeBrickPointer, BrickMapEdgeLength>, BrickMapEdgeLength>, BrickMapEdgeLength>;
@group(1) @binding(1) var<storage> brick_buffer: array<Brick>;

var<push_constant> id: u32;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput
{
    var out: VertexOutput;

    out.clip_position = global_model_view_projection[id] * vec4<f32>(input.position, 1.0);

    out.local_pos = input.position + vec3<f32>(512.0);

    let world_pos_intercalc = global_model[id] * vec4<f32>(input.position, 1.0);
    out.world_pos = world_pos_intercalc.xyz / world_pos_intercalc.w;

    return out;
}

struct FragmentOutput
{
   @builtin(frag_depth) depth: f32,
   @location(0) color: vec4<f32>
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput
{    
    let rayDir: vec3<f32> = normalize(in.world_pos - global_info.camera_pos.xyz); //normalize(in.world_pos - global_info.camera_pos.xyz );
    var rayPos: vec3<f32> =  in.local_pos + 0.5;

    // TODO: convert to u32
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


        // TODO: if out of this bounds destroy

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

    // let strike_pos_world: vec3<f32> = res.maybe_hit_point - 0.5;
    
    // let depth_intercalc = global_info.view_projection * vec4<f32>(strike_pos_world, 1.0);
    // let maybe_depth = depth_intercalc.z / depth_intercalc.w;

    // if (maybe_depth > 1.0 || maybe_depth < 0.0)
    // {
    //     out.color = ERROR_COLOR;
    //     return out;
    // }

    // // out.color = vec4<f32>(1.0);
    // out.depth = maybe_depth;
    
    // out.color = vec4<f32>(rayPos.xyz / 8, 1.0); //  vec4<f32>(rand(in.world_pos.xy), rand(in.world_pos.yz), rand(in.world_pos.zx), 1.0);
    return out;
}

// Constants
const USE_BRANCHLESS_DDA : bool = true;
const MAX_RAY_STEPS : i32 = 384;
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

fn getVoxel(c: vec3<i32>) -> bool
{
    if (any(c == vec3<i32>(0)))
    {
        return false;
    }

    if (length(vec3<f32>(c)) > 27.5)
    {
        return false;
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

fn mod_euc(l: i32, r: i32) -> i32
{
    let res = l % r;
    if res < 0 
    {
        if r > 0
        {
            return res + r;
        }
        else
        {
            return res - r;
        }
    }

    return res;
}

fn div_euc(l: i32, r: i32) -> i32
{
    let q = l / r;

    if l % r < 0
    {
        if r > 0
        {
            return q - 1;
        }
        else
        {
            return q + 1;
        }
    }

    return q;
}
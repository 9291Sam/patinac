struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) local_to_world_offset_pos: vec3<f32>,
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

struct Brick
{
    u16_brick_data: array<array<array<u32, 4>, 8>, 8>,
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

    let ltw_offset_intercalc = global_model[id] * vec4<f32>(vec3<f32>(-512.0), 1.0);
    out.local_to_world_offset_pos = ltw_offset_intercalc.xyz / ltw_offset_intercalc.w;

    return out;
}

struct FragmentOutput
{
   @builtin(frag_depth) depth: f32,
   @location(0) color: vec4<f32>
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front_face: bool) -> FragmentOutput
{    
    var ray: Ray;
    ray.direction = normalize(in.world_pos - global_info.camera_pos.xyz);

    let camera_pos_local: vec3<f32> = global_info.camera_pos.xyz - in.local_to_world_offset_pos;

    let camera_in_chunk: bool = all(camera_pos_local >= vec3<f32>(0.0)) && all(camera_pos_local < vec3<f32>(1024.0));

    if ((camera_in_chunk && is_front_face) || (!camera_in_chunk && !is_front_face))
    {
        discard;
    }

    if camera_in_chunk
    {
        ray.origin = camera_pos_local;
    }
    else
    {
        ray.origin = in.local_pos;
    }

    let mapPos: vec3<i32> = simple_dda_traversal_bricks(ray);

    var out: FragmentOutput;

    out.color = vec4<f32>(get_random_color(vec3<f32>(mapPos)), 1.0);

    var c: Cube;
    c.center = vec3<f32>(mapPos) + 0.5;
    c.edge_length = 1.0;
    
    let res = Cube_tryIntersect(c, ray);

    var strike_pos_world: vec3<f32>; 

    let cube_contains_ray = Cube_contains(c, camera_pos_local);

    if (!res.intersection_occurred && !cube_contains_ray)
    {
        out.color = ERROR_COLOR;
        return out;
    }

    // TODO: fix this shit
    if (cube_contains_ray)
    {
        strike_pos_world = global_info.camera_pos.xyz + ray.direction * 0.001;
    }
    else
    {
        strike_pos_world = res.maybe_hit_point + in.local_to_world_offset_pos;
    }


    let depth_intercalc = global_info.view_projection * vec4<f32>(strike_pos_world, 1.0);
    let maybe_depth = depth_intercalc.z / depth_intercalc.w;

    // if (maybe_depth > 1.0 || maybe_depth < 0.0)
    // {
    //     out.color = ERROR_COLOR;
    //     return out;
    // }

    out.depth = maybe_depth;

    // out.color = vec4<f32>(res.maybe_hit_point.xyz / 8, 1.0); //  vec4<f32>(rand(in.world_pos.xy), rand(in.world_pos.yz), rand(in.world_pos.zx), 1.0);
    return out;
}

const SIMPLE_DDA_ITER_BRICKS_STEPS: i32 = 128 * 3 + 1;
fn simple_dda_traversal_bricks(unadjusted_ray: Ray) -> vec3<i32>
{
    var adjusted_ray: Ray;
    adjusted_ray.origin = unadjusted_ray.origin / 8;
    adjusted_ray.direction = unadjusted_ray.direction;


    // TODO: why the FUCK is this a float
    var voxelPos: vec3<f32> = floor(adjusted_ray.origin);
    var distance: f32;
    var normal: vec3<f32>;
    let rayDirectionSign: vec3<f32> = sign(adjusted_ray.direction);

    let rdi: vec3<f32> = 1.0 / (2.0 * adjusted_ray.direction);


    var i: i32;

    for (i = 0; i < SIMPLE_DDA_ITER_BRICKS_STEPS; i += 1)
    {
        let mapPos: vec3<i32> = vec3<i32>(floor(voxelPos));
        
        if (any(mapPos < vec3<i32>(-1)) || any(mapPos > vec3<i32>(129)))
        {
            discard;
        }
        
        if (!(any(mapPos < vec3<i32>(0)) || any(mapPos >= vec3<i32>(128))))
        {
            let maybe_brick_pointer = brick_map[mapPos.x][mapPos.y][mapPos.z];

            if (maybe_brick_pointer != 0)
            {
                var brick_cube: Cube;
                brick_cube.center = 8.0 * vec3<f32>(mapPos) + 4;
                brick_cube.edge_length = 8.0;

                let res = Cube_tryIntersect(brick_cube, unadjusted_ray);

                if (res.intersection_occurred)
                {
                    var brick_ray: Ray;
                    brick_ray.origin = res.maybe_hit_point - 8 * vec3<f32>(mapPos);
                    brick_ray.direction = adjusted_ray.direction;
                    let maybe_local_brick_pos = traverse_brick_dda(maybe_brick_pointer, brick_ray);

                    if (all(maybe_local_brick_pos != InvalidBrickTraversalSentinel))
                    {
                        return maybe_local_brick_pos + mapPos * 8;
                    }


                }


                

                // if (res.intersection_occurred)
                // {
                //     // TODO: scale inward
                //     return vec3<i32>(res.maybe_hit_point + 0.0001);
                // }
            }
        }
        let plain: vec3<f32> = ((vec3<f32>(1.0) + rayDirectionSign - vec3<f32>(2.0) * (adjusted_ray.origin - voxelPos)) * rdi);

        distance = min(plain.x, min(plain.y, plain.z));
        // normal = vec3(equal(vec3(distance), plain)) * rayDirectionSign;
        normal = vec3<f32>(vec3<f32>(distance) == plain) * rayDirectionSign;
        voxelPos += normal;
    }

    discard;

    // if (ii == ITERSTEPS) discard;

    // vec3 position = ro+rd*dist;
    // return hit(normal, dist, position);
}

const BRICK_TRAVERSAL_STEPS: i32 = 8 * 3 + 1;
// traverses as if the brick is from 000 -> 888
const InvalidBrickTraversalSentinel: vec3<i32> = vec3<i32>(-1);
fn traverse_brick_dda(brick: BrickPointer, ray: Ray) -> vec3<i32>
{
    var voxelPos: vec3<f32> = floor(ray.origin);
    var distance: f32;
    var normal: vec3<f32>;
    let rayDirectionSign: vec3<f32> = sign(ray.direction);

    let rdi: vec3<f32> = 1.0 / (2.0 * ray.direction);

    for (var i = 0; i < BRICK_TRAVERSAL_STEPS; i += 1)
    {
        let mapPos = vec3<i32>(floor(voxelPos));

        if (any(mapPos < vec3<i32>(0)) || any(mapPos > vec3<i32>(8)))
        {
            return InvalidBrickTraversalSentinel;
        }
        
        if (Brick_access(brick, vec3<u32>(mapPos)) != 0)
        {
            return vec3<i32>(floor(voxelPos));
        }

        let plain: vec3<f32> = ((vec3<f32>(1.0) + rayDirectionSign - vec3<f32>(2.0) * (ray.origin - voxelPos)) * rdi);

        distance = min(plain.x, min(plain.y, plain.z));
        normal = vec3<f32>(vec3<f32>(distance) == plain) * rayDirectionSign;
        voxelPos += normal;
    }

    return InvalidBrickTraversalSentinel;
}


const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

fn Brick_access(me: BrickPointer, pos: vec3<u32>) -> u32
{
    // return select(0u, 1u, pos.x == pos.y);
    let last: u32 = pos.z % 2;

    let val: u32 = brick_buffer[me].u16_brick_data[pos.x][pos.y][pos.z / 2];

    switch (last)
    {
        case 0u: { return extractBits(val, 0u, 16u);  }
        case 1u: { return extractBits(val, 16u, 16u); }
        default: { return 0u; }
    }
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

fn rand(co: vec2<f32>) -> f32
{   
    return fract(sin(dot(co / (2.71 * log(length(co))), vec2(12.9898, 78.233))) * 43758.5453);
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
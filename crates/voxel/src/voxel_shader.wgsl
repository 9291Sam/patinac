struct VertexInput {
    @location(0) position: vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    // @location(2) color: vec3<f32>
}

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

alias Matricies = array<mat4x4<f32>, 1024>;

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;

var<push_constant> push_constant_id: u32;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput
{
    var out: VertexOutput;

    out.clip_position = global_model_view_projection[push_constant_id] * vec4<f32>(input.position, 1.0);

    let world_pos_intercalc = global_model[push_constant_id] * vec4<f32>(input.position, 1.0);
    out.world_pos = world_pos_intercalc.xyz / world_pos_intercalc.w;

    // out.color = input.color;

    return out;
}

struct FragmentOutput
{
   @builtin(frag_depth) depth: f32,
   @location(0) color: vec4<f32>
}

// const TRACKING_ARRAY_SIZE: u32 = 16;
// const BRICK_SIDE_LENGTH: u32 = 8;

// struct Brick
// {
//     data: array<array<array<u32, BRICK_SIDE_LENGTH>, BRICK_SIDE_LENGTH>, BRICK_SIDE_LENGTH>,
// }


// @group(0) @binding(0) var<storage> tracking_array: array<array<array<u32, TRACKING_ARRAY_SIZE>, TRACKING_ARRAY_SIZE>, TRACKING_ARRAY_SIZE>;
// @group(0) @binding(1) var<storage> brick_array: array<Brick>;

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

    
    let rayDir: vec3<f32> = (in.world_pos - global_info.camera_pos.xyz );
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
    
    let depth_intercalc = global_info.view_projection * vec4<f32>(strike_pos_world, 1.0);
    let maybe_depth = depth_intercalc.z / depth_intercalc.w;

    if (maybe_depth > 1.0 || maybe_depth < 0.0)
    {
        out.color = ERROR_COLOR;
        return out;
    }

    out.depth = maybe_depth;

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
	let p: vec3<f32> = vec3<f32>(c) + vec3<f32>(0.5);
	let d: f32 = min(max(-sdSphere(p, 7.5), sdBox(p, vec3<f32>(6.0))), -sdSphere(p, 25.0));
	return d < 0.0;
}

// Function to check if a voxel exists at a given position
// fn getVoxel(c: vec3<i32>) -> bool
// {
    // let bound: i32 = i32(BRICK_SIDE_LENGTH) * i32(TRACKING_ARRAY_SIZE);

    // if (any(c >= vec3<i32>(bound)) || any(c < vec3<i32>(-bound)))
    // {
    //     return false;
    // }

    // let brick_idx = vec3<i32>(div_euc(c.x, i32(BRICK_SIDE_LENGTH)), div_euc(c.y, i32(BRICK_SIDE_LENGTH)),div_euc(c.z, i32(BRICK_SIDE_LENGTH))) + vec3<i32>(TRACKING_ARRAY_SIZE / 2);
    // let maybe_brick_ptr = tracking_array[brick_idx.x][brick_idx.y][brick_idx.z];


    // if maybe_brick_ptr == 0
    // {
    //     return false;
    // }

    // let voxel_idx = vec3<i32>(mod_euc(c.x, i32(BRICK_SIDE_LENGTH)), mod_euc(c.y, i32(BRICK_SIDE_LENGTH)),mod_euc(c.z, i32(BRICK_SIDE_LENGTH)));

    // if brick_array[maybe_brick_ptr].data[voxel_idx.x][voxel_idx.y][voxel_idx.z] != 0
    // {
    //     return true;
    // }

    // return false;
// }

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

// struct VertexInput
// {
//     @location(0) position: vec3<f32>
// }

// struct VertexOutput
// {
//     @builtin(position) clip_position: vec4<f32>,
//     @location(0)       world_pos:     vec3<f32>,
//     @location(1)       camera_dir:    vec3<f32>
// }



// @vertex fn vs_main(in: VertexInput) -> VertexOutput
// {
//     var out: VertexOutput;

//     out.clip_position =
//         global_model_view_projection.data[push_constant_id]
//         * vec4<f32>(in.position, 1.0);

//     let vert_world_pos_intercalc: vec4<f32> =
//         global_model.data[push_constant_id]
//         * vec4<f32>(in.position, 1.0);
//     let vert_world_pos = vert_world_pos_intercalc.xyz / vert_world_pos_intercalc.w;

//     out.world_pos = vert_world_pos;
//     out.camera_dir = (vert_world_pos - global_info.camera_pos.xyz);

//     return out;
// }

// struct FragmentOutput
// {
//     @builtin(frag_depth) depth: f32,
//     @location(0) color: vec4<f32>
// }

// @fragment fn fs_main(in: VertexOutput) -> FragmentOutput
// {
    
// 	let rayDir: vec3<f32> = normalize(in.camera_dir);
// 	let rayPos: vec3<f32>= in.world_pos;
    
// 	var mapPos: vec3<i32> = vec3<i32>(floor(rayPos + 0.));

// 	let deltaDist: vec3<f32> = abs(vec3<f32>(length(rayDir)) / rayDir);
	
// 	let rayStep: vec3<i32> = vec3<i32>(sign(rayDir));

// 	var sideDist : vec3<f32>= (sign(rayDir) * (vec3<f32>(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist; 
	
// 	var mask: vec3<bool>;
	
// 	for (var i = 0; i < MAX_RAY_STEPS; i++) {
// 		if (getVoxel(mapPos)) {break;}
// 		// if (USE_BRANCHLESS_DDA) {
//             //Thanks kzy for the suggestion!
//             mask = sideDist.xyz <= min(sideDist.yzx, sideDist.zxy);
// 			/*bvec3 b1 = lessThan(sideDist.xyz, sideDist.yzx);
// 			bvec3 b2 = lessThanEqual(sideDist.xyz, sideDist.zxy);
// 			mask.x = b1.x && b2.x;
// 			mask.y = b1.y && b2.y;
// 			mask.z = b1.z && b2.z;*/
// 			//Would've done mask = b1 && b2 but the compiler is making me do it component wise.
			
// 			//All components of mask are false except for the corresponding largest component
// 			//of sideDist, which is the axis along which the ray should be incremented.			
			
// 			sideDist += vec3<f32>(mask) * deltaDist;
// 			mapPos += vec3<i32>(vec3<f32>(mask)) * rayStep;
// 		// }
// 		// else {
// 			// if (sideDist.x < sideDist.y) {
// 			// 	if (sideDist.x < sideDist.z) {
// 			// 		sideDist.x += deltaDist.x;
// 			// 		mapPos.x += rayStep.x;
// 			// 		mask = bvec3(true, false, false);
// 			// 	}
// 			// 	else {
// 			// 		sideDist.z += deltaDist.z;
// 			// 		mapPos.z += rayStep.z;
// 			// 		mask = bvec3(false, false, true);
// 			// 	}
// 			// }
// 			// else {
// 			// 	if (sideDist.y < sideDist.z) {
// 			// 		sideDist.y += deltaDist.y;
// 			// 		mapPos.y += rayStep.y;
// 			// 		mask = bvec3(false, true, false);
// 			// 	}
// 			// 	else {
// 			// 		sideDist.z += deltaDist.z;
// 			// 		mapPos.z += rayStep.z;
// 			// 		mask = bvec3(false, false, true);
// 			// 	}
// 			// }
// 		// }
// 	}
	
// 	var color: vec3<f32>;
// 	if (mask.x) {
// 		color = vec3<f32>(0.5);
// 	}
// 	if (mask.y) {
// 		color = vec3<f32>(1.0);
// 	}
// 	if (mask.z) {
// 		color = vec3<f32>(0.75);
// 	}
// 	// fragColor.rgb = color;

//     var out: FragmentOutput;
//     out.color = vec4<f32>(color, 1.0);

//     var c: Cube;
//     c.center = vec3<f32>(mapPos) + 0.5;
//     c.edge_length = 1.0;

//     var r: Ray;
//     r.origin = rayPos;
//     r.direction = rayDir;

//     let strike_pos_world: vec3<f32> = Cube_tryIntersect(c, r).maybe_hit_point -0.5;
    
//     // TODO: this is wrong, as of now this is all in the local 0->8 space when this needs to be in world space
//     let depth_intercalc = global_info.view_projection * vec4<f32>(strike_pos_world, 1.0);
//     out.depth = depth_intercalc.z / depth_intercalc.w;

//     return out;
// }

// //The raycasting code is somewhat based around a 2D raycasting toutorial found here: 
// //http://lodev.org/cgtutor/raycasting.html

// const USE_BRANCHLESS_DDA: bool = true;
// const MAX_RAY_STEPS: i32 = 64;

// fn sdSphere(p: vec3<f32>,d: f32) -> f32 { return length(p) - d; } 

// fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
//     let d: vec3<f32> = abs(p) - b;
//     return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3<f32>(0.0)));
// }
	
// fn getVoxel(c: vec3<i32>) -> bool
// {
// 	let p: vec3<f32> = vec3<f32>(c) + vec3<f32>(0.5);
// 	let d: f32 = min(max(-sdSphere(p, 7.5), sdBox(p, vec3<f32>(6.0))), -sdSphere(p, 25.0));
// 	return d < 0.0;
// }

// // void mainImage( out vec4 fragColor, in vec2 fragCoord )
// // {
// // 	vec2 screenPos = (fragCoord.xy / iResolution.xy) * 2.0 - 1.0;
// // 	vec3 cameraDir = vec3(0.0, 0.0, 0.8);
// // 	vec3 cameraPlaneU = vec3(1.0, 0.0, 0.0);
// // 	vec3 cameraPlaneV = vec3(0.0, 1.0, 0.0) * iResolution.y / iResolution.x;
// // 	vec3 rayDir = cameraDir + screenPos.x * cameraPlaneU + screenPos.y * cameraPlaneV;
// // 	vec3 rayPos = vec3(0.0, 2.0 * sin(iTime * 2.7), -12.0);
		
// // 	rayPos.xz = rotate2d(rayPos.xz, iTime);
// // 	rayDir.xz = rotate2d(rayDir.xz, iTime);
	
// // 	ivec3 mapPos = ivec3(floor(rayPos + 0.));

// // 	vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
	
// // 	ivec3 rayStep = ivec3(sign(rayDir));

// // 	vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist; 
	
// // 	bvec3 mask;
	
// // 	for (int i = 0; i < MAX_RAY_STEPS; i++) {
// // 		if (getVoxel(mapPos)) continue;
// // 		if (USE_BRANCHLESS_DDA) {
// //             //Thanks kzy for the suggestion!
// //             mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
// // 			/*bvec3 b1 = lessThan(sideDist.xyz, sideDist.yzx);
// // 			bvec3 b2 = lessThanEqual(sideDist.xyz, sideDist.zxy);
// // 			mask.x = b1.x && b2.x;
// // 			mask.y = b1.y && b2.y;
// // 			mask.z = b1.z && b2.z;*/
// // 			//Would've done mask = b1 && b2 but the compiler is making me do it component wise.
			
// // 			//All components of mask are false except for the corresponding largest component
// // 			//of sideDist, which is the axis along which the ray should be incremented.			
			
// // 			sideDist += vec3(mask) * deltaDist;
// // 			mapPos += ivec3(vec3(mask)) * rayStep;
// // 		}
// // 		else {
// // 			if (sideDist.x < sideDist.y) {
// // 				if (sideDist.x < sideDist.z) {
// // 					sideDist.x += deltaDist.x;
// // 					mapPos.x += rayStep.x;
// // 					mask = bvec3(true, false, false);
// // 				}
// // 				else {
// // 					sideDist.z += deltaDist.z;
// // 					mapPos.z += rayStep.z;
// // 					mask = bvec3(false, false, true);
// // 				}
// // 			}
// // 			else {
// // 				if (sideDist.y < sideDist.z) {
// // 					sideDist.y += deltaDist.y;
// // 					mapPos.y += rayStep.y;
// // 					mask = bvec3(false, true, false);
// // 				}
// // 				else {
// // 					sideDist.z += deltaDist.z;
// // 					mapPos.z += rayStep.z;
// // 					mask = bvec3(false, false, true);
// // 				}
// // 			}
// // 		}
// // 	}
	
// // 	vec3 color;
// // 	if (mask.x) {
// // 		color = vec3(0.5);
// // 	}
// // 	if (mask.y) {
// // 		color = vec3(1.0);
// // 	}
// // 	if (mask.z) {
// // 		color = vec3(0.75);
// // 	}
// // 	fragColor.rgb = color;
// // 	//fragColor.rgb = vec3(0.1 * noiseDeriv);
// // }


// struct Cube {
//     center: vec3<f32>,
//     edge_length: f32,
// };

// fn Cube_contains(me: Cube, point: vec3<f32>) -> bool {
//     let p0 = me.center - (me.edge_length / 2.0);
//     let p1 = me.center + (me.edge_length / 2.0);

//     if (all(p0 < point) && all(point < p1)) {
//         return true;
//     } else {
//         return false;
//     }
// }

// struct Ray {
//     origin: vec3<f32>,
//     direction: vec3<f32>,
// };

// struct IntersectionResult {
//     intersection_occurred: bool,
//     maybe_distance: f32,
//     maybe_hit_point: vec3<f32>,
//     maybe_normal: vec3<f32>,
//     maybe_color: vec4<f32>,
// };

// fn Cube_tryIntersect(me: Cube, ray: Ray) -> IntersectionResult {
//     let p0 = me.center - (me.edge_length / 2.0);
//     let p1 = me.center + (me.edge_length / 2.0);

//     if (Cube_contains(me, ray.origin)) {
//         var res: IntersectionResult;
        
//         res.intersection_occurred = true;
//         res.maybe_distance = 0.0;
//         res.maybe_hit_point = ray.origin;
//         res.maybe_normal = vec3<f32>(0.0);
//         res.maybe_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

//         return res; 
//     }

//     let t1 = (p0 - ray.origin) / ray.direction;
//     let t2 = (p1 - ray.origin) / ray.direction;

//     let vMax = max(t1, t2);
//     let vMin = min(t1, t2);

//     let tMax = min(min(vMax.x, vMax.y), vMax.z);
//     let tMin = max(max(vMin.x, vMin.y), vMin.z);

//     let hit = tMin <= tMax && tMax > 0.0;

//     if (!hit) {

//         var res: IntersectionResult;
        
//         res.intersection_occurred = false;
        
//         return res;
//     }

//     let hitPoint = ray.origin + tMin * ray.direction;

//     var normal: vec3<f32>;

//     if (abs(hitPoint.x - p1.x) < 0.0001) {
//         normal = vec3<f32>(-1.0, 0.0, 0.0); // Hit right face
//     } else if (abs(hitPoint.x - p0.x) < 0.0001) {
//         normal = vec3<f32>(1.0, 0.0, 0.0); // Hit left face
//     } else if (abs(hitPoint.y - p1.y) < 0.0001) {
//         normal = vec3<f32>(0.0, -1.0, 0.0); // Hit top face
//     } else if (abs(hitPoint.y - p0.y) < 0.0001) {
//         normal = vec3<f32>(0.0, 1.0, 0.0); // Hit bottom face
//     } else if (abs(hitPoint.z - p1.z) < 0.0001) {
//         normal = vec3<f32>(0.0, 0.0, -1.0); // Hit front face
//     } else if (abs(hitPoint.z - p0.z) < 0.0001) {
//         normal = vec3<f32>(0.0, 0.0, 1.0); // Hit back face
//     } else {
//         normal = vec3<f32>(0.0, 0.0, 0.0); // null case
//     }

//     var res: IntersectionResult;
        
//     res.intersection_occurred = true;
//     res.maybe_distance = length(ray.origin - hitPoint);
//     res.maybe_hit_point = hitPoint;
//     res.maybe_normal = normal;
//     res.maybe_color = vec4<f32>(0.0, 1.0, 1.0, 1.0);

//     return res; 
// }
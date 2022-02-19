struct NoiseParams {
    global_dim: vec3<u32>;
    local_dim: vec3<u32>;
    time: u32;
};

struct Output {
    output: array<f32>;
};

@group(0)
@binding(0)
var<uniform> noise_params: NoiseParams;

@group(0)
@binding(1)
var<storage, read_write> noise_output: Output;

fn mod(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {
  return x - y * floor(x/y); 
}
fn permute(x: vec4<f32>) -> vec4<f32> {return mod(((x*34.0)+vec4<f32>(1.0))*x, vec4<f32>(289.0));}
fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {return 1.79284291400159 * vec4<f32>(1.0) - 0.85373472095314 * r;}
fn fade(t: vec4<f32>) -> vec4<f32> {return t*t*t*(t*(t*6.0-vec4<f32>(15.0))+vec4<f32>(10.0));}

fn cnoise(P: vec4<f32>) -> f32 {
  var Pi0 = floor(P); // Integer part for indexing
  var Pi1 = Pi0 + 1.0; // Integer part + 1
  Pi0 = mod(Pi0, vec4<f32>(289.0));
  Pi1 = mod(Pi1, vec4<f32>(289.0));
  let Pf0 = fract(P); // Fractional part for interpolation
  let Pf1 = Pf0 - vec4<f32>(1.0); // Fractional part - 1.0
  let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4<f32>(Pi0.yy, Pi1.yy);
  let iz0 = vec4<f32>(Pi0.zzzz);
  let iz1 = vec4<f32>(Pi1.zzzz);
  let iw0 = vec4<f32>(Pi0.wwww);
  let iw1 = vec4<f32>(Pi1.wwww);

  let ixy = permute(permute(ix) + iy);
  let ixy0 = permute(ixy + iz0);
  let ixy1 = permute(ixy + iz1);
  let ixy00 = permute(ixy0 + iw0);
  let ixy01 = permute(ixy0 + iw1);
  let ixy10 = permute(ixy1 + iw0);
  let ixy11 = permute(ixy1 + iw1);

  var gx00 = ixy00 / 7.0;
  var gy00 = floor(gx00) / 7.0;
  var gz00 = floor(gy00) / 6.0;
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  var gw00 = vec4<f32>(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  var sw00 = step(gw00, vec4<f32>(0.0));
  gx00 = gx00 - sw00 * (step(vec4<f32>(0.0), gx00) - 0.5);
  gy00 = gy00 - sw00 * (step(vec4<f32>(0.0), gy00) - 0.5);

  var gx01 = ixy01 / 7.0;
  var gy01 = floor(gx01) / 7.0;
  var gz01 = floor(gy01) / 6.0;
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  var gw01 = vec4<f32>(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  var sw01 = step(gw01, vec4<f32>(0.0));
  gx01 = gx01 - sw01 * (step(vec4<f32>(0.0), gx01) - 0.5);
  gy01 = gy01 - sw01 * (step(vec4<f32>(0.0), gy01) - 0.5);

  var gx10 = ixy10 / 7.0;
  var gy10 = floor(gx10) / 7.0;
  var gz10 = floor(gy10) / 6.0;
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  var gw10 = vec4<f32>(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  var sw10 = step(gw10, vec4<f32>(0.0));
  gx10 = gx10 - sw10 * (step(vec4<f32>(0.0), gx10) - 0.5);
  gy10 = gy10 - sw10 * (step(vec4<f32>(0.0), gy10) - 0.5);

  var gx11 = ixy11 / 7.0;
  var gy11 = floor(gx11) / 7.0;
  var gz11 = floor(gy11) / 6.0;
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  var gw11 = vec4<f32>(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  var sw11 = step(gw11, vec4<f32>(0.0));
  gx11 = gx11 - sw11 * (step(vec4<f32>(0.0), gx11) - 0.5);
  gy11 = gy11 - sw11 * (step(vec4<f32>(0.0), gy11) - 0.5);

  var g0000 = vec4<f32>(gx00.x,gy00.x,gz00.x,gw00.x);
  var g1000 = vec4<f32>(gx00.y,gy00.y,gz00.y,gw00.y);
  var g0100 = vec4<f32>(gx00.z,gy00.z,gz00.z,gw00.z);
  var g1100 = vec4<f32>(gx00.w,gy00.w,gz00.w,gw00.w);
  var g0010 = vec4<f32>(gx10.x,gy10.x,gz10.x,gw10.x);
  var g1010 = vec4<f32>(gx10.y,gy10.y,gz10.y,gw10.y);
  var g0110 = vec4<f32>(gx10.z,gy10.z,gz10.z,gw10.z);
  var g1110 = vec4<f32>(gx10.w,gy10.w,gz10.w,gw10.w);
  var g0001 = vec4<f32>(gx01.x,gy01.x,gz01.x,gw01.x);
  var g1001 = vec4<f32>(gx01.y,gy01.y,gz01.y,gw01.y);
  var g0101 = vec4<f32>(gx01.z,gy01.z,gz01.z,gw01.z);
  var g1101 = vec4<f32>(gx01.w,gy01.w,gz01.w,gw01.w);
  var g0011 = vec4<f32>(gx11.x,gy11.x,gz11.x,gw11.x);
  var g1011 = vec4<f32>(gx11.y,gy11.y,gz11.y,gw11.y);
  var g0111 = vec4<f32>(gx11.z,gy11.z,gz11.z,gw11.z);
  var g1111 = vec4<f32>(gx11.w,gy11.w,gz11.w,gw11.w);

  let norm00: vec4<f32> = taylorInvSqrt(vec4<f32>(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 = g0000 * norm00.x;
  g0100 = g0100 * norm00.y;
  g1000 = g1000 * norm00.z;
  g1100 = g1100 * norm00.w;

  let norm01 = taylorInvSqrt(vec4<f32>(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 = g0001 * norm01.x;
  g0101 = g0101 * norm01.y;
  g1001 = g1001 * norm01.z;
  g1101 = g1101 * norm01.w;

  let norm10 = taylorInvSqrt(vec4<f32>(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 = g0010 * norm10.x;
  g0110 = g0110 * norm10.y;
  g1010 = g1010 * norm10.z;
  g1110 = g1110 * norm10.w;

  let norm11 = taylorInvSqrt(vec4<f32>(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 = g0011 * norm11.x;
  g0111 = g0111 * norm11.y;
  g1011 = g1011 * norm11.z;
  g1111 = g1111 * norm11.w;

  let n0000 = dot(g0000, Pf0);
  let n1000 = dot(g1000, vec4<f32>(Pf1.x, Pf0.yzw));
  let n0100 = dot(g0100, vec4<f32>(Pf0.x, Pf1.y, Pf0.zw));
  let n1100 = dot(g1100, vec4<f32>(Pf1.xy, Pf0.zw));
  let n0010 = dot(g0010, vec4<f32>(Pf0.xy, Pf1.z, Pf0.w));
  let n1010 = dot(g1010, vec4<f32>(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  let n0110 = dot(g0110, vec4<f32>(Pf0.x, Pf1.yz, Pf0.w));
  let n1110 = dot(g1110, vec4<f32>(Pf1.xyz, Pf0.w));
  let n0001 = dot(g0001, vec4<f32>(Pf0.xyz, Pf1.w));
  let n1001 = dot(g1001, vec4<f32>(Pf1.x, Pf0.yz, Pf1.w));
  let n0101 = dot(g0101, vec4<f32>(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  let n1101 = dot(g1101, vec4<f32>(Pf1.xy, Pf0.z, Pf1.w));
  let n0011 = dot(g0011, vec4<f32>(Pf0.xy, Pf1.zw));
  let n1011 = dot(g1011, vec4<f32>(Pf1.x, Pf0.y, Pf1.zw));
  let n0111 = dot(g0111, vec4<f32>(Pf0.x, Pf1.yzw));
  let n1111 = dot(g1111, Pf1);

  let fade_xyzw = fade(Pf0);
  let n_0w = mix(vec4<f32>(n0000, n1000, n0100, n1100), vec4<f32>(n0001, n1001, n0101, n1101), fade_xyzw.w);
  let n_1w = mix(vec4<f32>(n0010, n1010, n0110, n1110), vec4<f32>(n0011, n1011, n0111, n1111), fade_xyzw.w);
  let n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  let n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  let n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}

// Noise functions copied from https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83 and converted to wgsl.

//++ fn hash(n: f32) -> f32 {
//++     return fract(sin(n) * 10000.0);
//++ }
//++ 
//++ fn hash_v2(p: vec2<f32>) -> f32 {
//++     return fract(10000.0 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
//++ }
//++ 
//++ fn noise(x: f32) -> f32 {
//++     let i: f32 = floor(x);
//++     let f: f32 = fract(x);
//++     let u: f32 = f * f * (3.0 - 2.0 * f);
//++     return mix(hash(i), hash(i + 1.0), u);
//++ }
//++ 
//++ fn noise2(x: vec2<f32>) -> f32 {
//++ 
//++ 	let i: vec2<f32> = floor(x);
//++ 	let f: vec2<f32> = fract(x);
//++ 
//++ 	// Four corners in 2D of a tile
//++ 	let a: f32 = hash_v2(i);
//++ 	let b: f32 = hash_v2(i + vec2<f32>(1.0, 0.0));
//++ 	let c: f32 = hash_v2(i + vec2<f32>(0.0, 1.0));
//++ 	let d: f32 = hash_v2(i + vec2<f32>(1.0, 1.0));
//++ 
//++ 	let u: vec2<f32> = f * f * (3.0 - 2.0 * f);
//++ 	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
//++ }
//++ 
//++ fn noise3(x: vec3<f32>) -> f32 {
//++ 
//++ 	let st = vec3<f32>(110.0, 241.0, 171.0);
//++ 
//++ 	let i = floor(x);
//++ 	let f = fract(x);
//++ 
//++     	let n = dot(i, st);
//++ 
//++ 
//++ 	let u = f * f * (3.0 - 2.0 * f);
//++ 	return mix(mix(mix( hash(n + dot(st, vec3<f32>(0.0, 0.0, 0.0))), hash(n + dot(st, vec3<f32>(1.0, 0.0, 0.0))), u.x),
//++                    mix( hash(n + dot(st, vec3<f32>(0.0, 1.0, 0.0))), hash(n + dot(st, vec3<f32>(1.0, 1.0, 0.0))), u.x), u.y),
//++                mix(mix( hash(n + dot(st, vec3<f32>(0.0, 0.0, 1.0))), hash(n + dot(st, vec3<f32>(1.0, 0.0, 1.0))), u.x),
//++                    mix( hash(n + dot(st, vec3<f32>(0.0, 1.0, 1.0))), hash(n + dot(st, vec3<f32>(1.0, 1.0, 1.0))), u.x), u.y), u.z);
//++ }
//++ 
//++ let NUM_OCTAVES: u32 = 5u;
//++ 
//++ fn fbm(x: f32) -> f32 {
//++ 
//++     var v: f32 = 0.0;
//++     var a: f32 = 0.5;
//++     var xx: f32 = x; 
//++     let shift: f32 = 100.0;
//++     for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
//++     	v = a + a * noise(xx);
//++     	xx = xx * 2.0 + shift;
//++     	a = a * 0.5;
//++     }
//++     return v;
//++ }
//++ 
//++ 
//++ fn fbm2(x: vec2<f32>) -> f32 {
//++ 
//++     let shift = vec2<f32>(100.0);
//++     let rot = mat2x2<f32>(vec2<f32>(cos(0.5), sin(0.5)), vec2<f32>(-sin(0.5), cos(0.50)));
//++     
//++     var v: f32 = 0.0;
//++     var a: f32 = 0.5;
//++     var xx: vec2<f32> = x; 
//++     
//++     for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
//++         v = v + a * noise2(xx);
//++         xx = rot * xx * 2.0 + shift;
//++         a = a * 0.5;
//++     }
//++     return v;
//++ }
//++ 
//++ fn fbm3(x: vec3<f32>) -> f32 {
//++ 
//++     let shift: f32 = 100.0;
//++ 
//++     var v: f32 = 0.0;
//++     var a: f32 = 0.5;
//++     var xx: vec3<f32> = x; 
//++ 
//++     for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
//++     	v = a + a * noise3(xx);
//++     	xx = xx * 2.0 + shift;
//++     	a = a * 0.5;
//++     }
//++     return v;
//++ }

fn index1D_to_index3D(global_index: vec3<u32>, x_dim: u32, y_dim: u32) -> vec3<u32> {
	var index: u32 = global_index.x;
	var wh: u32 = x_dim * y_dim;
	let z: u32 = index / wh;
	index = index - z * wh;
	let y: u32 = index / x_dim;
	index = index - y * x_dim;
	let x: u32 = index;
	return vec3<u32>(x, y, z);	
}

fn encode3Dmorton32(x: u32, y: u32, z: u32) -> u32 {
    var x_temp = (x      | (x      << 16u)) & 0x030000FFu;
        x_temp = (x_temp | (x_temp <<  8u)) & 0x0300F00Fu;
        x_temp = (x_temp | (x_temp <<  4u)) & 0x030C30C3u;
        x_temp = (x_temp | (x_temp <<  2u)) & 0x09249249u;

    var y_temp = (y      | (y      << 16u)) & 0x030000FFu;
        y_temp = (y_temp | (y_temp <<  8u)) & 0x0300F00Fu;
        y_temp = (y_temp | (y_temp <<  4u)) & 0x030C30C3u;
        y_temp = (y_temp | (y_temp <<  2u)) & 0x09249249u;

    var z_temp = (z      | (z      << 16u)) & 0x030000FFu;
        z_temp = (z_temp | (z_temp <<  8u)) & 0x0300F00Fu;
        z_temp = (z_temp | (z_temp <<  4u)) & 0x030C30C3u;
        z_temp = (z_temp | (z_temp <<  2u)) & 0x09249249u;
    return x_temp | (y_temp << 1u) | (z_temp << 2u);
}

fn get_third_bits32(m: u32) -> u32 {
    var x = m & 0x9249249u;
    x = (x ^ (x >> 2u))  & 0x30c30c3u;
    x = (x ^ (x >> 4u))  & 0x0300f00fu;
    x = (x ^ (x >> 8u))  & 0x30000ffu;
    x = (x ^ (x >> 16u)) & 0x000003ffu;

    return x;
}

fn decode3Dmorton32(m: u32) -> vec3<u32> {
    return vec3<u32>(
        get_third_bits32(m),
        get_third_bits32(m >> 1u),
        get_third_bits32(m >> 2u)
   );
}

@stage(compute)
@workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32) {

    let scene_center = vec4<f32>(vec3<f32>(noise_params.global_dim * noise_params.local_dim), 0.0) * 0.5;

    let offset = 256u;

    let noise_velocity = 2.4;
    let wave_height_factor = 0.2;

    let actual_global_id = local_id.x + offset * 4u * work_group_id.x; 

    let c0 = vec4<f32>(vec3<f32>(decode3Dmorton32(actual_global_id))       , 1.0);
    let c1 = vec4<f32>(vec3<f32>(decode3Dmorton32(actual_global_id + offset)) , 1.0);
    let c2 = vec4<f32>(vec3<f32>(decode3Dmorton32(actual_global_id + offset * 2u)) , 1.0);
    let c3 = vec4<f32>(vec3<f32>(decode3Dmorton32(actual_global_id + offset * 3u)) , 1.0);

    let ball0 = pow(c0.x - scene_center.x, 2.0) + pow(c0.y - scene_center.y, 2.0) + pow(c0.z - scene_center.z, 2.0) - pow(50.0, 2.0); 
    let ball1 = pow(c1.x - scene_center.x, 2.0) + pow(c1.y - scene_center.y, 2.0) + pow(c1.z - scene_center.z, 2.0) - pow(50.0, 2.0); 
    let ball2 = pow(c2.x - scene_center.x, 2.0) + pow(c2.y - scene_center.y, 2.0) + pow(c2.z - scene_center.z, 2.0) - pow(50.0, 2.0); 
    let ball3 = pow(c3.x - scene_center.x, 2.0) + pow(c3.y - scene_center.y, 2.0) + pow(c3.z - scene_center.z, 2.0) - pow(50.0, 2.0); 
 
    noise_output.output[actual_global_id]               = ball0 + cnoise(c0 * 0.2) * 1300.0;
    noise_output.output[actual_global_id + offset]      = ball1 + cnoise(c1 * 0.2) * 1300.0;
    noise_output.output[actual_global_id + offset * 2u] = ball2 + cnoise(c2 * 0.2) * 1300.0;
    noise_output.output[actual_global_id + offset * 3u] = ball3 + cnoise(c3 * 0.2) * 1300.0;
}


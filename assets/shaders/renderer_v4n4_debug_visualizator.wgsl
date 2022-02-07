struct Camera {
    u_view_proj: mat4x4<f32>;
    camera_pos: vec4<f32>;
};

@group(0)
@binding(0)
var<uniform> camerauniform: Camera;

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), vec4<f32>(step(c.b, c.g)));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), vec4<f32>(step(p.x, c.r)));

    let d: f32 = q.x - min(q.w, q.y);
    let e: f32 = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32>  {
    //vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let K: vec4<f32> = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    //vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    let p: vec3<f32> = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.yyy);
    //return vec3<f32>(1.0, 2.0, 3.0);
}

struct VertexOutput {
    @builtin(position) my_pos: vec4<f32>;
    @location(0) pos: vec4<f32>;
    @location(1) nor: vec4<f32>;
};

@stage(vertex)
fn vs_main(@location(0) pos: vec4<f32>, @location(1) nor: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.my_pos = camerauniform.u_view_proj * pos;
    out.pos = pos;
    out.nor = nor;
    return out;
}

// Ligth/material properties.
let light_pos: vec3<f32> = vec3<f32>(3.0, 28.0, 3.0);
let light_color: vec3<f32> = vec3<f32>(0.8, 0.3, 0.3);
let material_spec_color: vec3<f32> = vec3<f32>(0.5, 0.1, 0.1);
let material_shininess: f32 = 55.0;
let ambient_coeffience: f32 = 0.15;
let attentuation_factor: f32 = 0.013;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var light_dir: vec3<f32> = normalize(light_pos - in.pos.xyz);
    var normal: vec3<f32> = normalize(in.nor).xyz; // is this necessery? 
    var diff_coeffient: f32 = max(0.0, dot(normal, light_dir));
    var reflection_vector: vec3<f32> = reflect(-light_dir, normal);
    var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - in.pos.xyz);
    
    var cosAngle: f32 = max(0.0, dot(camera_dir, reflection_vector));
    var specular_coeffient: f32 = 0.0;

    if (diff_coeffient > 0.0) {
        specular_coeffient = pow(cosAngle, material_shininess);
    }

    var offset_factor: f32 = 1.5;
    
    var coord1: vec2<f32> = in.pos.xy*offset_factor;
    var coord2: vec2<f32> = in.pos.xz*offset_factor;
    var coord3: vec2<f32> = in.pos.yz*offset_factor + in.pos.xz*offset_factor*offset_factor;
    
    var surface_color: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);

    var specular_component: vec3<f32> = specular_coeffient * material_spec_color * light_color;
    var ambient_component:  vec3<f32> = ambient_coeffience * light_color * surface_color.xyz;
    var diffuse_component:  vec3<f32> = diff_coeffient * light_color * surface_color.xyz;
    
    var distance_to_light: f32 = distance(in.pos.xyz, light_pos); 
    var attentuation: f32 = 1.0 / (1.0 + attentuation_factor * pow(distance_to_light,2.0));
    
    var final_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);

    var the_color: vec3<f32> = rgb2hsv(final_color.xyz);

    var dist_to_frag: f32 = distance(camerauniform.camera_pos.xyz, in.pos.xyz);
    var blah: f32 = 1.0 / (1.0 + 0.0005 * pow(dist_to_frag,2.0));

    the_color = the_color + vec3<f32>(0.0, 0.2, 0.0);
    var the_color2: vec3<f32> = hsv2rgb(the_color);

    return vec4<f32>(mix(vec3<f32>(0.5, 0.0, 0.0), the_color2, vec3<f32>(blah)), 1.0);
}

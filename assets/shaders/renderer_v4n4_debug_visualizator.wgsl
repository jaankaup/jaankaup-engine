struct Camera {
    u_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct Light {
    light_pos: vec3<f32>,
    material_shininess: f32,
    material_spec_color: vec3<f32>,
    ambient_coeffience: f32,
    light_color: vec3<f32>,
    attentuation_factor: f32,
};

struct OtherParams {
    scale_factor: f32,
};

struct VertexOutput {
    @builtin(position) my_pos: vec4<f32>,
    @location(0) pos: vec4<f32>,
    @location(1) nor: vec4<f32>,
    // @location(2) diff_coeffient: f32,
    // @location(3) reflection_vector: vec3<f32>,
    // @location(4) camera_dir: vec3<f32>,
    @location(2) @interpolate(flat) col: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camerauniform: Camera;

@group(0) @binding(1)
var<uniform> light: Light;

@group(0) @binding(2)
var<uniform> other_params: OtherParams;

fn decode_color(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 256u) / 255.0;
  let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
  let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
  let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

@vertex
fn vs_main(@location(0) pos: vec4<f32>, @location(1) nor: vec4<f32>) -> VertexOutput {

    let color: vec3<f32> = decode_color(bitcast<u32>(pos.w)).xyz;

    // var light_dir: vec3<f32> = normalize(light.light_pos - pos.xyz);
    // var diff_coeffient: f32 = max(0.0, dot(nor.xyz, light_dir));
    // var reflection_vector: vec3<f32> = reflect(-light_dir, nor.xyz);
    // var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - pos.xyz);

    var out_next_stage: VertexOutput;
    out_next_stage.my_pos = camerauniform.u_view_proj * vec4<f32>(other_params.scale_factor * pos.xyz, 1.0);
    out_next_stage.pos = pos;
    out_next_stage.nor = nor;
    //++ out.diff_coeffient = diff_coeffient;
    //++ out.reflection_vector = reflection_vector;
    //++ out.camera_dir = camera_dir;
    out_next_stage.col = color;
    return out_next_stage;
}

@fragment
fn fs_main(in_data: VertexOutput) -> @location(0) vec4<f32> {

    var light_dir: vec3<f32> = normalize(light.light_pos - in_data.pos.xyz);
    var normal: vec3<f32> = normalize(in_data.nor).xyz; // is this necessery? 
    var diff_coeffient: f32 = max(0.0, dot(normal, light_dir));
    var reflection_vector: vec3<f32> = reflect(-light_dir, normal);
    var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - in_data.pos.xyz);
    
    var cosAngle: f32 = max(0.0, dot(camera_dir, reflection_vector));
    var specular_coeffient: f32 = 0.0;

    if (diff_coeffient > 0.0) {
        specular_coeffient = pow(cosAngle, light.material_shininess);
    }

    var offset_factor: f32 = 0.5;
    
    var surface_color: vec3<f32> = in_data.col; 

    var specular_component: vec3<f32> = specular_coeffient * light.material_spec_color * light.light_color;
    var ambient_component:  vec3<f32> = light.ambient_coeffience * light.light_color * surface_color.xyz;
    var diffuse_component:  vec3<f32> = diff_coeffient * light.light_color * surface_color.xyz;
    
    var distance_to_light: f32 = distance(in_data.pos.xyz, light.light_pos); 
    var attentuation: f32 = 1.0 / (1.0 + light.attentuation_factor * pow(distance_to_light,2.0));
    
    var final_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);

    return final_color;

//++    var light_dir: vec3<f32> = normalize(light.light_pos - in.pos.xyz);
//++    var normal: vec3<f32> = normalize(in.nor).xyz;
//++    var diff_coeffient: f32 = max(0.0, dot(normal, light_dir));
//++    var reflection_vector: vec3<f32> = reflect(-light_dir, normal);
//++    var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - in.pos.xyz);
//++
//++    var cosAngle: f32 = max(0.0, dot(camera_dir, reflection_vector));
//++
//++    //++ var cosAngle: f32 = max(0.0, dot(in.camera_dir, in.reflection_vector));
//++
//++    var specular_coeffient: f32 = 0.0;
//++
//++    if (diff_coeffient > 0.0) {
//++        specular_coeffient = pow(cosAngle, light.material_shininess);
//++    }
//++
//++    var surface_color: vec3<f32> = in.col; 
//++
//++    var specular_component: vec3<f32> = specular_coeffient * light.material_spec_color * light.light_color;
//++    var ambient_component:  vec3<f32> = light.ambient_coeffience * light.light_color * surface_color.xyz;
//++    var diffuse_component:  vec3<f32> = diff_coeffient * light.light_color * surface_color.xyz;
//++
//++    //++ var specular_component: vec3<f32> = specular_coeffient * light.material_spec_color * light.light_color;
//++    //++ var ambient_component:  vec3<f32> = light.ambient_coeffience * light.light_color * surface_color.xyz;
//++    //++ var diffuse_component:  vec3<f32> = in.diff_coeffient * light.light_color * surface_color.xyz;
//++    
//++    var distance_to_light: f32 = distance(in.pos.xyz, light.light_pos); 
//++    var attentuation: f32 = 1.0 / (1.0 + light.attentuation_factor * pow(distance_to_light,2.0));
//++    
//++    var final_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);
//++
//++    return final_color;
}

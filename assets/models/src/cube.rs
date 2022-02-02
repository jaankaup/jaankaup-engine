use jaankaup_core::buffer::*;
use jaankaup_core::wgpu;

/// Data for textured cube. vvvttnnn vvvttnnn vvvttnnn ...
//#[allow(dead_code)]
//pub fn create_cube(texture_coordinates: bool) -> Vec<f32> {
pub fn create_cube(device: &wgpu::Device, texture_coordinates: bool) -> wgpu::Buffer {

    // let v_data = [
    //     [1.0 , -1.0, -1.0], [1.0 , -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
    //     [1.0 , 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
    // ];

    // 4-component version
    let v_data = [
        [1.0 , -1.0, -1.0, 1.0], [1.0 , -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
        [1.0 , 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0],
    ];

    let t_data = [
        [0.748573,0.750412], [0.749279,0.501284], [0.999110,0.501077], [0.999455,0.750380], [0.250471,0.500702],
        [0.249682,0.749677], [0.001085,0.750380], [0.001517,0.499994], [0.499422,0.500239],
        [0.500149,0.750166], [0.748355,0.998230], [0.500193,0.998728], [0.498993,0.250415], [0.748953,0.250920],
    ];
    
    // let n_data = [
    //     [0.0 , 0.0 , -1.0], [-1.0, -0.0, 0.0], [0.0, -0.0, 1.0], [0.0, 0.0 , 1.0], [1.0 , -0.0, 0.0],
    //     [1.0 , 0.0 , 0.0], [0.0 , 1.0 , 0.0], [0.0, -1.0, 0.0],
    // ];

    // 4-component version
    let n_data = [
        [0.0 , 0.0 , -1.0, 0.0], [-1.0, -0.0, 0.0, 0.0], [0.0, -0.0, 1.0, 0.0], [0.0, 0.0 , 1.0, 0.0], [1.0 , -0.0, 0.0, 0.0],
        [1.0 , 0.0 , 0.0, 0.0], [0.0 , 1.0 , 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
    ];

    // let mut vs: Vec<[f32; 3]> = Vec::new();
    // let mut ts: Vec<[f32; 2]> = Vec::new();
    // let mut vn: Vec<[f32; 3]> = Vec::new();

    // 4-component version
    let mut vs: Vec<[f32; 4]> = Vec::new();
    let mut ts: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 4]> = Vec::new();

    vs.push(v_data[4]); ts.push(t_data[0]); vn.push(n_data[0]);
    vs.push(v_data[0]); ts.push(t_data[1]); vn.push(n_data[0]);
    vs.push(v_data[3]); ts.push(t_data[2]); vn.push(n_data[0]);
    vs.push(v_data[4]); ts.push(t_data[0]); vn.push(n_data[0]);
    vs.push(v_data[3]); ts.push(t_data[2]); vn.push(n_data[0]);
    vs.push(v_data[7]); ts.push(t_data[3]); vn.push(n_data[0]);
    vs.push(v_data[2]); ts.push(t_data[4]); vn.push(n_data[1]);
    vs.push(v_data[6]); ts.push(t_data[5]); vn.push(n_data[1]);
    vs.push(v_data[7]); ts.push(t_data[6]); vn.push(n_data[1]);
    vs.push(v_data[2]); ts.push(t_data[4]); vn.push(n_data[1]);
    vs.push(v_data[7]); ts.push(t_data[6]); vn.push(n_data[1]);
    vs.push(v_data[3]); ts.push(t_data[7]); vn.push(n_data[1]);
    vs.push(v_data[1]); ts.push(t_data[8]); vn.push(n_data[2]);
    vs.push(v_data[5]); ts.push(t_data[9]); vn.push(n_data[2]);
    vs.push(v_data[2]); ts.push(t_data[4]); vn.push(n_data[2]);
    vs.push(v_data[5]); ts.push(t_data[9]); vn.push(n_data[3]);
    vs.push(v_data[6]); ts.push(t_data[5]); vn.push(n_data[3]);
    vs.push(v_data[2]); ts.push(t_data[4]); vn.push(n_data[3]);
    vs.push(v_data[0]); ts.push(t_data[1]); vn.push(n_data[4]);
    vs.push(v_data[4]); ts.push(t_data[0]); vn.push(n_data[4]);
    vs.push(v_data[1]); ts.push(t_data[8]); vn.push(n_data[4]);
    vs.push(v_data[4]); ts.push(t_data[0]); vn.push(n_data[5]);
    vs.push(v_data[5]); ts.push(t_data[9]); vn.push(n_data[5]);
    vs.push(v_data[1]); ts.push(t_data[8]); vn.push(n_data[5]);
    vs.push(v_data[4]); ts.push(t_data[0]); vn.push(n_data[6]);
    vs.push(v_data[7]); ts.push(t_data[10]); vn.push(n_data[6]);
    vs.push(v_data[5]); ts.push(t_data[9]); vn.push(n_data[6]);
    vs.push(v_data[7]); ts.push(t_data[10]); vn.push(n_data[6]);
    vs.push(v_data[6]); ts.push(t_data[11]); vn.push(n_data[6]);
    vs.push(v_data[5]); ts.push(t_data[9]); vn.push(n_data[6]);
    vs.push(v_data[0]); ts.push(t_data[1]); vn.push(n_data[7]);
    vs.push(v_data[1]); ts.push(t_data[8]); vn.push(n_data[7]);
    vs.push(v_data[2]); ts.push(t_data[12]); vn.push(n_data[7]);
    vs.push(v_data[0]); ts.push(t_data[1]); vn.push(n_data[7]);
    vs.push(v_data[2]); ts.push(t_data[12]); vn.push(n_data[7]);
    vs.push(v_data[3]); ts.push(t_data[13]); vn.push(n_data[7]);

    let mut p_data: Vec<f32> = Vec::new();

    for i in 0..vs.len() {
        p_data.push(vs[i][0]); p_data.push(vs[i][1]); p_data.push(vs[i][2]); p_data.push(vs[i][3]);


        if texture_coordinates { 
            p_data.push(ts[i][0]); p_data.push(ts[i][1]);
        }

        p_data.push(vn[i][0]); p_data.push(vn[i][1]); p_data.push(vn[i][2]); p_data.push(vn[i][3]); 
    }

    buffer_from_data::<f32>(
        device,
        &p_data,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        None
    )
}

/// Creates a buffer for screen filling texture.
pub fn create_screen_texture_buffer(device: &wgpu::Device) -> wgpu::Buffer {

    buffer_from_data::<f32>(
        device,
        // gl_Position     |    point_pos
        &[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
          -1.0,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
          -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        None
    )
}

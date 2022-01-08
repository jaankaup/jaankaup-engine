// pub mod wgpu_system;
// pub mod input;
// pub mod shader;
// pub mod misc;
// pub mod buffer;
// pub mod texture;
// pub mod assets;
// pub mod camera;
// pub mod two_triangles;
// pub mod mc;
// pub mod temp;
// pub mod render_pipelines;
// pub mod noise3d;
// pub mod compute;
pub mod template;
pub mod input;
pub mod misc;
pub mod texture;
pub mod screen;
pub use wgpu;
pub use winit;
//pub use bytemuck;
//pub use bytemuck::{Pod, Zeroable};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

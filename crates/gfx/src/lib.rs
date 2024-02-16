#![feature(new_uninit)]
#![feature(hasher_prefixfree_extras)]
#![feature(map_try_insert)]

mod linalg;
mod recordables;
mod render_cache;
mod renderer;

pub use linalg::*;
pub use nalgebra::UnitQuaternion;
pub use recordables::lit_textured::LitTextured;
pub use recordables::Recordable;
pub use renderer::Renderer;
pub use {nalgebra_glm as glm, wgpu};

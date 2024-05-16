#![feature(lazy_cell)]
#![feature(hasher_prefixfree_extras)]
#![feature(map_try_insert)]
#![allow(clippy::type_complexity)]

mod cpu_tracked_buffer;
mod input_manager;
mod linalg;
mod recordables;
mod render_cache;
mod renderer;
mod screen_sized_texture;

pub use cpu_tracked_buffer::CpuTrackedBuffer;
pub use input_manager::*;
pub use linalg::*;
pub use recordables::{DrawId, PassStage, RecordInfo, Recordable, RenderPassId};
pub use render_cache::{
    CacheableComputePipelineDescriptor,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor,
    GenericPass,
    GenericPipeline
};
pub use renderer::{EncoderToPassFn, RenderPassSendFunction, Renderer};
pub use screen_sized_texture::*;
pub use wgpu;
pub use winit::keyboard::KeyCode;

pub mod glm
{
    pub use nalgebra::UnitQuaternion;
    pub use nalgebra_glm::*;
}

pub use nalgebra as nal;

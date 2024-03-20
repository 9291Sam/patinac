#![feature(hasher_prefixfree_extras)]
#![feature(map_try_insert)]

mod input_manager;
mod linalg;
mod recordables;
mod render_cache;
mod renderer;

pub use input_manager::*;
pub use linalg::*;
pub use recordables::{DrawId, PassStage, RecordInfo, Recordable};
pub use render_cache::{
    CacheableComputePipelineDescriptor,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor,
    GenericPass,
    GenericPipeline
};
pub use renderer::Renderer;
pub use wgpu;

pub mod glm
{
    pub use nalgebra::UnitQuaternion;
    pub use nalgebra_glm::*;
}

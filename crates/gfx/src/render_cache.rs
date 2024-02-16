use std::borrow::Borrow;
use std::sync::Mutex;
use std::{collections::HashMap, sync::Arc};
use std::mem::size_of;
use std::hash::Hash;
use std::num::NonZeroU64;

use nalgebra_glm as glm;
use strum::{EnumIter, IntoEnumIterator};

use crate::renderer::{ShaderGlobalInfo, ShaderMatrices, DEPTH_FORMAT, SURFACE_TEXTURE_FORMAT};



// #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
// pub enum PipelineType
// {
//     FlatTextured,
//     LitTextured
// }

// #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
// pub enum BindGroupType
// {
//     GlobalData,
//     FlatSimpleTexture,
//     LitSimpleTexture
// }

#[derive(Debug)]
pub struct RenderCache
{
    device: Arc<wgpu::Device>,
    bind_group_layouts: Mutex<HashMap<CacheableBindGroupLayoutDescriptor, Arc<wgpu::BindGroupLayout>>>
}

impl Drop for RenderCache
{
    fn drop(&mut self) {
        self.trim();
    }
}

impl RenderCache
{
    pub(crate) fn new(device: Arc<wgpu::Device>) -> Self
    {
        Self
        {
            device,
            bind_group_layouts: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_init_bind_group_layout(&self, layout: wgpu::BindGroupLayoutDescriptor<'static>)
    -> Arc<wgpu::BindGroupLayout>

    {
        let mut guard = self.bind_group_layouts.lock().unwrap();
        let cache_val = CacheableBindGroupLayoutDescriptor(layout);

        guard.entry(cache_val).or_insert_with_key(|k| {
            Arc::new(self.device.create_bind_group_layout(&k.0))
        }).clone()

    }

    pub(crate) fn trim(&self)
    {


    }
}


#[derive(Debug)]
struct CacheableBindGroupLayoutDescriptor(wgpu::BindGroupLayoutDescriptor<'static>);

impl PartialEq for CacheableBindGroupLayoutDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = self.0;
        let r = other.0;

        l.label == r.label
        && l.entries.len() == r.entries.len()
        && l.entries.iter().zip(r.entries.iter()).fold(true, |acc, (l, r)|  l == r && acc)
    }
}

impl Eq for CacheableBindGroupLayoutDescriptor {}

impl Hash for CacheableBindGroupLayoutDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        state.write_str(self.0.label.unwrap_or(""));

        self.0.entries.iter().for_each(|e| e.hash(state));
    }
}

// #[derive(Debug)]
// pub struct RenderCache
// {
//     bind_group_layout_cache: HashMap<BindGroupType, wgpu::BindGroupLayout>,
//     pipeline_cache:          HashMap<PipelineType, GenericPipeline>
// }

// impl RenderCache
// {
//     // pub fn cache_new()
//     pub fn new(device: &wgpu::Device) -> Self
//     {
//         let bind_group_layout_cache: HashMap<_, _> = BindGroupType::iter()
//             .map(|bind_group_type| {
//                 let new_bind_group_layout =
//                     match bind_group_type
//                     {
//                         BindGroupType::GlobalData =>
//                         {
//                             device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                                 label:   Some("GlobalData"),
//                                 // camera
//                                 // projection matricies
//                                 // depth buffer
//                                 
//                             })
//                         }
//                         BindGroupType::FlatSimpleTexture =>
//                         {
//                             device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                                 entries: &[
//                                     wgpu::BindGroupLayoutEntry {
//                                         binding:    0,
//                                         visibility: wgpu::ShaderStages::FRAGMENT,
//                                         ty:         wgpu::BindingType::Texture {
//                                             multisampled:   false,
//                                             view_dimension: wgpu::TextureViewDimension::D2,
//                                             sample_type:    wgpu::TextureSampleType::Float {
//                                                 filterable: true
//                                             }
//                                         },
//                                         count:      None
//                                     },
//                                     wgpu::BindGroupLayoutEntry {
//                                         binding:    1,
//                                         visibility: wgpu::ShaderStages::FRAGMENT,
//                                         ty:         wgpu::BindingType::Sampler(
//                                             wgpu::SamplerBindingType::Filtering
//                                         ),
//                                         count:      None
//                                     }
//                                 ],
//                                 label:   Some("texture_bind_group_layout")
//                             })
//                         }
//                         BindGroupType::LitSimpleTexture =>
//                         {
//                             device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                                 entries: &[
//                                     wgpu::BindGroupLayoutEntry {
//                                         binding:    0,
//                                         visibility: wgpu::ShaderStages::FRAGMENT,
//                                         ty:         wgpu::BindingType::Texture {
//                                             multisampled:   false,
//                                             view_dimension: wgpu::TextureViewDimension::D2,
//                                             sample_type:    wgpu::TextureSampleType::Float {
//                                                 filterable: true
//                                             }
//                                         },
//                                         count:      None
//                                     },
//                                     wgpu::BindGroupLayoutEntry {
//                                         binding:    1,
//                                         visibility: wgpu::ShaderStages::FRAGMENT,
//                                         ty:         wgpu::BindingType::Texture {
//                                             multisampled:   false,
//                                             view_dimension: wgpu::TextureViewDimension::D2,
//                                             sample_type:    wgpu::TextureSampleType::Float {
//                                                 filterable: true
//                                             }
//                                         },
//                                         count:      None
//                                     },
//                                     wgpu::BindGroupLayoutEntry {
//                                         binding:    2,
//                                         visibility: wgpu::ShaderStages::FRAGMENT,
//                                         ty:         wgpu::BindingType::Sampler(
//                                             wgpu::SamplerBindingType::Filtering
//                                         ),
//                                         count:      None
//                                     }
//                                 ],
//                                 label:   Some("texture_bind_group_layout")
//                             })
//                         }
//                     };

//                 (bind_group_type, new_bind_group_layout)
//             })
//             .collect::<HashMap<BindGroupType, wgpu::BindGroupLayout>>();

//         let pipeline_layout_cache: HashMap<_, _> = PipelineType::iter()
//             .map(|pipeline_type| {
//                 let new_pipeline_layout = match pipeline_type
//                 {
//                     PipelineType::FlatTextured =>
//                     {
//                         device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                             label:                Some("FlatTextured"),
//                             bind_group_layouts:   &[
//                                 bind_group_layout_cache
//                                     .get(&BindGroupType::GlobalData)
//                                     .unwrap(),
//                                 bind_group_layout_cache
//                                     .get(&BindGroupType::FlatSimpleTexture)
//                                     .unwrap()
//                             ],
//                             push_constant_ranges: &[wgpu::PushConstantRange {
//                                 stages: wgpu::ShaderStages::VERTEX,
//                                 range:  0..(std::mem::size_of::<glm::Mat4>() as u32)
//                             }]
//                         })
//                     }
//                     PipelineType::LitTextured =>
//                     {
//                         device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                             label:                Some("LitTextured"),
//                             bind_group_layouts:   &[
//                                 bind_group_layout_cache
//                                     .get(&BindGroupType::GlobalData)
//                                     .unwrap(),
//                                 bind_group_layout_cache
//                                     .get(&BindGroupType::LitSimpleTexture)
//                                     .unwrap()
//                             ],
//                             push_constant_ranges: &[wgpu::PushConstantRange {
//                                 stages: wgpu::ShaderStages::VERTEX,
//                                 range:  0..(std::mem::size_of::<u32>() as u32)
//                             }]
//                         })
//                     }
//                 };

//                 (pipeline_type, new_pipeline_layout)
//             })
//             .collect();

//         let pipeline_cache: HashMap<_, _> = PipelineType::iter()
//             .map(|pipeline_type| {
//                 let default_depth_state = Some(wgpu::DepthStencilState {
//                     format:              DEPTH_FORMAT,
//                     depth_write_enabled: true,
//                     depth_compare:       wgpu::CompareFunction::Less,
//                     stencil:             wgpu::StencilState::default(),
//                     bias:                wgpu::DepthBiasState::default()
//                 });

//                 let default_multisample_state = wgpu::MultisampleState {
//                     count:                     1,
//                     mask:                      !0,
//                     alpha_to_coverage_enabled: false
//                 };

//                 let new_pipeline = match pipeline_type
//                 {
//                     PipelineType::FlatTextured =>
//                     {
//                         let shader = device.create_shader_module(wgpu::include_wgsl!(
//                             "renderable/res/flat_textured/flat_textured.wgsl"
//                         ));

//                         GenericPipeline::Render(device.create_render_pipeline(
//                             &wgpu::RenderPipelineDescriptor {
//                                 label:         Some("FlatTextured"),
//                                 layout:
//                                     pipeline_layout_cache.get(&PipelineType::FlatTextured),
//                                 vertex:        wgpu::VertexState {
//                                     module:      &shader,
//                                     entry_point: "vs_main",
//                                     buffers:     &[
//                                         super::recordables::flat_textured::Vertex::desc()
//                                     ]
//                                 },
//                                 fragment:      Some(wgpu::FragmentState {
//                                     module:      &shader,
//                                     entry_point: "fs_main",
//                                     targets:     &[Some(wgpu::ColorTargetState {
//                                         format:     SURFACE_TEXTURE_FORMAT,
//                                         blend:      Some(wgpu::BlendState::REPLACE),
//                                         write_mask: wgpu::ColorWrites::ALL
//                                     })]
//                                 }),
//                                 primitive:     wgpu::PrimitiveState {
//                                     topology:           wgpu::PrimitiveTopology::TriangleStrip,
//                                     strip_index_format: None,
//                                     front_face:         wgpu::FrontFace::Ccw,
//                                     cull_mode:          None,
//                                     polygon_mode:       wgpu::PolygonMode::Fill,
//                                     unclipped_depth:    false,
//                                     conservative:       false
//                                 },
//                                 depth_stencil: default_depth_state,
//                                 multisample:   default_multisample_state,
//                                 multiview:     None
//                             }
//                         ))
//                     }
//                     PipelineType::LitTextured =>
//                     {
//                         let shader = device.create_shader_module(wgpu::include_wgsl!(
//                             "renderable/res/lit_textured/lit_textured.wgsl"
//                         ));

//                         GenericPipeline::Render(device.create_render_pipeline(
//                             &wgpu::RenderPipelineDescriptor {
//                                 label:         Some("LitTextured"),
//                                 layout:
//                                     pipeline_layout_cache.get(&PipelineType::LitTextured),
//                                 vertex:        wgpu::VertexState {
//                                     module:      &shader,
//                                     entry_point: "vs_main",
//                                     buffers:     &[super::recordables::lit_textured::Vertex::desc()]
//                                 },
//                                 fragment:      Some(wgpu::FragmentState {
//                                     module:      &shader,
//                                     entry_point: "fs_main",
//                                     targets:     &[Some(wgpu::ColorTargetState {
//                                         format:     SURFACE_TEXTURE_FORMAT,
//                                         blend:      Some(wgpu::BlendState::REPLACE),
//                                         write_mask: wgpu::ColorWrites::ALL
//                                     })]
//                                 }),
//                                 primitive:     wgpu::PrimitiveState {
//                                     topology:           wgpu::PrimitiveTopology::TriangleList,
//                                     strip_index_format: None,
//                                     front_face:         wgpu::FrontFace::Cw,
//                                     cull_mode:          Some(wgpu::Face::Back),
//                                     polygon_mode:       wgpu::PolygonMode::Fill,
//                                     unclipped_depth:    false,
//                                     conservative:       false
//                                 },
//                                 depth_stencil: default_depth_state,
//                                 multisample:   default_multisample_state,
//                                 multiview:     None
//                             }
//                         ))
//                     }
//                 };

//                 (pipeline_type, new_pipeline)
//             })
//             .collect();

//         RenderCache {
//             bind_group_layout_cache,
//             pipeline_cache
//         }
//     }

//     pub fn lookup_bind_group_layout(&self, bind_group_type: BindGroupType)
//     -> &wgpu::BindGroupLayout
//     {
//         self.bind_group_layout_cache.get(&bind_group_type).unwrap()
//     }

//     pub fn lookup_pipeline(&self, pipeline_type: PipelineType) -> &GenericPipeline
//     {
//         self.pipeline_cache.get(&pipeline_type).unwrap()
//     }
// }

pub enum GenericPass<'p>
{
    Compute(wgpu::ComputePass<'p>),
    Render(wgpu::RenderPass<'p>)
}

#[derive(Debug)]
pub enum GenericPipeline
{
    Compute(wgpu::ComputePipeline),
    Render(wgpu::RenderPipeline)
}

impl PartialEq for GenericPipeline
{
    fn eq(&self, other: &Self) -> bool
    {
        self.global_id() == other.global_id()
    }
}

impl Eq for GenericPipeline {}

impl PartialOrd for GenericPipeline
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.global_id().partial_cmp(&other.global_id())
    }
}

impl Ord for GenericPipeline
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.global_id().cmp(&other.global_id())
    }
}

impl GenericPipeline
{
     fn global_id(&self) -> u64
    {
        match self
        {
            GenericPipeline::Compute(p) => p.global_id().inner(),
            GenericPipeline::Render(p) => p.global_id().inner()
        }
    }
}

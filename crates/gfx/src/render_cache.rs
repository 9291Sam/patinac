// use std::sync::Mutex;
// use std::{collections::HashMap, sync::Arc};
// use std::hash::Hash;

// #[derive(Debug)]
// pub struct RenderCache
// {
//     device: Arc<wgpu::Device>,
//     bind_group_layouts: Mutex<HashMap<CacheableBindGroupLayoutDescriptor, Arc<wgpu::BindGroupLayout>>>,
//     pipeline_layout_cache: Mutex<HashMap<CacheablePipelineLayoutDescriptor, Arc<wgpu::PipelineLayout>>>
// }

// impl Drop for RenderCache
// {
//     fn drop(&mut self) {
//         self.trim();
//     }
// }

// impl RenderCache
// {
//     pub(crate) fn new(device: Arc<wgpu::Device>) -> Self
//     {
//         Self
//         {
//             device,
//             bind_group_layouts: Mutex::new(HashMap::new()),
//             pipeline_layout_cache: Mutex::new(HashMap::new())
//         }
//     }

//     pub fn get_or_init_bind_group_layout(&self, layout: wgpu::BindGroupLayoutDescriptor<'static>)
//         -> Arc<wgpu::BindGroupLayout>
//     {
//         self.bind_group_layouts.lock().unwrap().entry(CacheableBindGroupLayoutDescriptor(layout)).or_insert_with_key(|k| {
//             Arc::new(self.device.create_bind_group_layout(&k.0))
//         }).clone()

//     }

//     pub fn get_or_init_pipeline_layout(&self, layout: wgpu::PipelineLayoutDescriptor<'static>)
//         -> Arc<wgpu::PipelineLayout>
//     {

//         self.pipeline_layout_cache.lock().unwrap().entry(CacheablePipelineLayoutDescriptor(layout)).or_insert_with_key(|k| {
//             Arc::new(self.device.create_pipeline_layout(&k.0))
//         }).clone()
//     }

//     pub(crate) fn trim(&self)
//     {


//     }
// }

// fn get_or_init()



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
//                                
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
//                         
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
//                         
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


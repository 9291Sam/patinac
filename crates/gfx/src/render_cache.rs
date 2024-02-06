use std::collections::HashMap;

use nalgebra_glm as glm;
use strum::{EnumIter, IntoEnumIterator};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PassStage
{
    GraphicsSimpleColor
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum PipelineType
{
    FlatTextured,
    LitTextured,
    ParallaxRaymarched
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum BindGroupType
{
    GlobalCamera,
    FlatSimpleTexture,
    LitSimpleTexture
}

pub struct RenderCache
{
    bind_group_layout_cache: HashMap<BindGroupType, wgpu::BindGroupLayout>,
    pipeline_cache:          HashMap<PipelineType, GenericPipeline>
}

impl RenderCache
{
    pub fn new(device: &wgpu::Device) -> Self
    {
        let bind_group_layout_cache: HashMap<_, _> = BindGroupType::iter()
            .map(|bind_group_type| {
                let new_bind_group_layout = match bind_group_type
                {
                    BindGroupType::GlobalCamera =>
                    {
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label:   Some("GlobalCamera"),
                            entries: &[]
                        })
                    }
                    BindGroupType::FlatSimpleTexture =>
                    {
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding:    0,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty:         wgpu::BindingType::Texture {
                                        multisampled:   false,
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        sample_type:    wgpu::TextureSampleType::Float {
                                            filterable: true
                                        }
                                    },
                                    count:      None
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding:    1,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty:         wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Filtering
                                    ),
                                    count:      None
                                }
                            ],
                            label:   Some("texture_bind_group_layout")
                        })
                    }
                    BindGroupType::LitSimpleTexture =>
                    {
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding:    0,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty:         wgpu::BindingType::Texture {
                                        multisampled:   false,
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        sample_type:    wgpu::TextureSampleType::Float {
                                            filterable: true
                                        }
                                    },
                                    count:      None
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding:    1,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty:         wgpu::BindingType::Texture {
                                        multisampled:   false,
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        sample_type:    wgpu::TextureSampleType::Float {
                                            filterable: true
                                        }
                                    },
                                    count:      None
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding:    2,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty:         wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Filtering
                                    ),
                                    count:      None
                                }
                            ],
                            label:   Some("texture_bind_group_layout")
                        })
                    }
                };

                (bind_group_type, new_bind_group_layout)
            })
            .collect::<HashMap<BindGroupType, wgpu::BindGroupLayout>>();

        let pipeline_layout_cache: HashMap<_, _> = PipelineType::iter()
            .map(|pipeline_type| {
                let new_pipeline_layout = match pipeline_type
                {
                    PipelineType::FlatTextured =>
                    {
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:                Some("FlatTextured"),
                            bind_group_layouts:   &[bind_group_layout_cache
                                .get(&BindGroupType::FlatSimpleTexture)
                                .unwrap()],
                            push_constant_ranges: &[wgpu::PushConstantRange {
                                stages: wgpu::ShaderStages::VERTEX,
                                range:  0..(std::mem::size_of::<glm::Mat4>() as u32)
                            }]
                        })
                    }
                    PipelineType::LitTextured =>
                    {
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:                Some("LitTextured"),
                            bind_group_layouts:   &[bind_group_layout_cache
                                .get(&BindGroupType::LitSimpleTexture)
                                .unwrap()],
                            push_constant_ranges: &[wgpu::PushConstantRange {
                                stages: wgpu::ShaderStages::VERTEX,
                                range:  0..(std::mem::size_of::<glm::Mat4>() as u32 * 2)
                            }]
                        })
                    }
                    PipelineType::ParallaxRaymarched =>
                    {
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:                Some("Parallax Raymarched Pipeline Layout"),
                            bind_group_layouts:   &[],
                            push_constant_ranges: &[wgpu::PushConstantRange {
                                stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                range:  0..(std::mem::size_of::<
                                    crate::renderable::parallax_raymarched::PushConstants
                                >() as u32)
                            }]
                        })
                    }
                };

                (pipeline_type, new_pipeline_layout)
            })
            .collect();

        let pipeline_cache: HashMap<_, _> = PipelineType::iter()
            .map(|pipeline_type| {
                let default_depth_state = Some(wgpu::DepthStencilState {
                    format:              super::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare:       wgpu::CompareFunction::Less,
                    stencil:             wgpu::StencilState::default(),
                    bias:                wgpu::DepthBiasState::default()
                });

                let default_multisample_state = wgpu::MultisampleState {
                    count:                     1,
                    mask:                      !0,
                    alpha_to_coverage_enabled: false
                };

                let new_pipeline = match pipeline_type
                {
                    PipelineType::FlatTextured =>
                    {
                        let shader = device.create_shader_module(wgpu::include_wgsl!(
                            "renderable/res/flat_textured/flat_textured.wgsl"
                        ));

                        GenericPipeline::Render(device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label:         Some("FlatTextured"),
                                layout:
                                    pipeline_layout_cache.get(&PipelineType::FlatTextured),
                                vertex:        wgpu::VertexState {
                                    module:      &shader,
                                    entry_point: "vs_main",
                                    buffers:     &[super::renderable::flat_textured::Vertex::desc()]
                                },
                                fragment:      Some(wgpu::FragmentState {
                                    module:      &shader,
                                    entry_point: "fs_main",
                                    targets:     &[Some(wgpu::ColorTargetState {
                                        format:     super::SURFACE_TEXTURE_FORMAT,
                                        blend:      Some(wgpu::BlendState::REPLACE),
                                        write_mask: wgpu::ColorWrites::ALL
                                    })]
                                }),
                                primitive:     wgpu::PrimitiveState {
                                    topology:           wgpu::PrimitiveTopology::TriangleStrip,
                                    strip_index_format: None,
                                    front_face:         wgpu::FrontFace::Ccw,
                                    cull_mode:          None,
                                    polygon_mode:       wgpu::PolygonMode::Fill,
                                    unclipped_depth:    false,
                                    conservative:       false
                                },
                                depth_stencil: default_depth_state,
                                multisample:   default_multisample_state,
                                multiview:     None
                            }
                        ))
                    }
                    PipelineType::LitTextured =>
                    {
                        let shader = device.create_shader_module(wgpu::include_wgsl!(
                            "renderable/res/lit_textured/lit_textured.wgsl"
                        ));

                        GenericPipeline::Render(device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label:         Some("LitTextured"),
                                layout:
                                    pipeline_layout_cache.get(&PipelineType::LitTextured),
                                vertex:        wgpu::VertexState {
                                    module:      &shader,
                                    entry_point: "vs_main",
                                    buffers:     &[super::renderable::lit_textured::Vertex::desc()]
                                },
                                fragment:      Some(wgpu::FragmentState {
                                    module:      &shader,
                                    entry_point: "fs_main",
                                    targets:     &[Some(wgpu::ColorTargetState {
                                        format:     super::SURFACE_TEXTURE_FORMAT,
                                        blend:      Some(wgpu::BlendState::REPLACE),
                                        write_mask: wgpu::ColorWrites::ALL
                                    })]
                                }),
                                primitive:     wgpu::PrimitiveState {
                                    topology:           wgpu::PrimitiveTopology::TriangleList,
                                    strip_index_format: None,
                                    front_face:         wgpu::FrontFace::Cw,
                                    cull_mode:          Some(wgpu::Face::Back),
                                    polygon_mode:       wgpu::PolygonMode::Fill,
                                    unclipped_depth:    false,
                                    conservative:       false
                                },
                                depth_stencil: default_depth_state,
                                multisample:   default_multisample_state,
                                multiview:     None
                            }
                        ))
                    }
                    PipelineType::ParallaxRaymarched =>
                    {
                        let shader = device.create_shader_module(wgpu::include_wgsl!(
                            "renderable/res/parallax_raymarched/shader.wgsl"
                        ));

                        GenericPipeline::Render(device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label:         Some("Parallax Raymarched Pipeline"),
                                layout:
                                    pipeline_layout_cache.get(&PipelineType::ParallaxRaymarched),
                                vertex:        wgpu::VertexState {
                                    module:      &shader,
                                    entry_point: "vs_main",
                                    buffers:     &[
                                        super::renderable::parallax_raymarched::Vertex::desc()
                                    ]
                                },
                                fragment:      Some(wgpu::FragmentState {
                                    module:      &shader,
                                    entry_point: "fs_main",
                                    targets:     &[Some(wgpu::ColorTargetState {
                                        format:     super::SURFACE_TEXTURE_FORMAT,
                                        blend:      Some(wgpu::BlendState::REPLACE),
                                        write_mask: wgpu::ColorWrites::ALL
                                    })]
                                }),
                                primitive:     wgpu::PrimitiveState {
                                    topology:           wgpu::PrimitiveTopology::TriangleList,
                                    strip_index_format: None,
                                    front_face:         wgpu::FrontFace::Cw,
                                    cull_mode:          Some(wgpu::Face::Back),
                                    polygon_mode:       wgpu::PolygonMode::Fill,
                                    unclipped_depth:    false,
                                    conservative:       false
                                },
                                depth_stencil: default_depth_state,
                                multisample:   default_multisample_state,
                                multiview:     None
                            }
                        ))
                    }
                };

                (pipeline_type, new_pipeline)
            })
            .collect();

        RenderCache {
            bind_group_layout_cache,
            pipeline_cache
        }
    }

    pub fn lookup_bind_group_layout(&self, bind_group_type: BindGroupType)
    -> &wgpu::BindGroupLayout
    {
        self.bind_group_layout_cache.get(&bind_group_type).unwrap()
    }

    pub fn lookup_pipeline(&self, pipeline_type: PipelineType) -> &GenericPipeline
    {
        self.pipeline_cache.get(&pipeline_type).unwrap()
    }
}

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

impl GenericPipeline
{
    pub fn global_id(&self) -> u64
    {
        match self
        {
            GenericPipeline::Compute(p) => p.global_id().inner(),
            GenericPipeline::Render(p) => p.global_id().inner()
        }
    }
}

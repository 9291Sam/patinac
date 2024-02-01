use std::collections::HashMap;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
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
    TestSample
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, EnumIter)]
pub enum BindGroupType
{
    GlobalCamera,
    TestSimpleTexture
}

pub struct RenderCache
{
    bind_group_layout_cache: HashMap<BindGroupType, wgpu::BindGroupLayout>,

    // pipeline_layout_cache:   HashMap<PipelineType, wgpu::PipelineLayout>,
    render_pipeline_cache:  HashMap<PipelineType, wgpu::RenderPipeline>,
    compute_pipeline_cache: HashMap<PipelineType, wgpu::ComputePipeline>
}

impl RenderCache
{
    pub fn new(device: &wgpu::Device) -> Self
    {
        let bind_group_layout_cache = BindGroupType::iter()
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
                    BindGroupType::TestSimpleTexture =>
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
                };

                (bind_group_type, new_bind_group_layout)
            })
            .collect::<HashMap<BindGroupType, wgpu::BindGroupLayout>>();

        let pipeline_layout_cache = PipelineType::iter()
            .map(|pipeline_type| {
                let new_pipeline_layout = match pipeline_type
                {
                    PipelineType::TestSample =>
                    {
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:                Some("GraphicsFlat"),
                            bind_group_layouts:   &[bind_group_layout_cache
                                .get(&BindGroupType::TestSimpleTexture)
                                .unwrap()],
                            push_constant_ranges: &[wgpu::PushConstantRange {
                                stages: wgpu::ShaderStages::VERTEX,
                                range:  0..(std::mem::size_of::<glm::Mat4>() as u32)
                            }]
                        })
                    }
                };

                (pipeline_type, new_pipeline_layout)
            })
            .collect::<HashMap<PipelineType, wgpu::PipelineLayout>>();

        let render_pipeline_cache = PipelineType::iter()
            .filter_map(|pipeline_type| {
                let maybe_new_pipeline = match pipeline_type
                {
                    PipelineType::TestSample =>
                    {
                        let shader =
                            device.create_shader_module(wgpu::include_wgsl!("shaders/foo.wgsl"));

                        Some(
                            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label:         Some("GraphicsFlat"),
                                layout:        pipeline_layout_cache.get(&PipelineType::TestSample),
                                vertex:        wgpu::VertexState {
                                    module:      &shader,
                                    entry_point: "vs_main",
                                    buffers:     &[Vertex::desc()]
                                },
                                fragment:      Some(wgpu::FragmentState {
                                    // 3.
                                    module:      &shader,
                                    entry_point: "fs_main",
                                    targets:     &[Some(wgpu::ColorTargetState {
                                        // 4.
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
                                depth_stencil: None,
                                multisample:   wgpu::MultisampleState {
                                    count:                     1,
                                    mask:                      !0,
                                    alpha_to_coverage_enabled: false
                                },
                                multiview:     None
                            })
                        )
                    }
                };

                maybe_new_pipeline.map(|p| (pipeline_type, p))
            })
            .collect();
        let compute_pipeline_cache = HashMap::new();

        RenderCache {
            bind_group_layout_cache,
            // pipeline_layout_cache,
            render_pipeline_cache,
            compute_pipeline_cache
        }
    }

    pub fn lookup_bind_group_layout(&self, bind_group_type: BindGroupType)
    -> &wgpu::BindGroupLayout
    {
        self.bind_group_layout_cache.get(&bind_group_type).unwrap()
    }

    pub fn lookup_render_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline
    {
        self.render_pipeline_cache.get(&pipeline_type).unwrap()
    }

    pub fn lookup_compute_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::ComputePipeline
    {
        self.compute_pipeline_cache.get(&pipeline_type).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum PipelinePass
{
    Compute,
    Render
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex
{
    pub position:   glm::Vec3,
    pub tex_coords: glm::Vec2
}

impl Vertex
{
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBS
        }
    }
}

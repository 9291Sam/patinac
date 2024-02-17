use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use wgpu::PipelineLayout;

use crate::renderer::GenericPipeline;

#[derive(Debug)]
pub struct RenderCache
{
    device: Arc<wgpu::Device>,

    bind_group_layout_cache:
        Mutex<HashMap<CacheableBindGroupLayoutDescriptor, Arc<wgpu::BindGroupLayout>>>,
    pipeline_layout_cache:
        Mutex<HashMap<CacheablePipelineLayoutDescriptor, Arc<wgpu::PipelineLayout>>>,
    shader_module_cache: Mutex<HashMap<CacheableShaderModuleDescriptor, Arc<wgpu::ShaderModule>>>,
    render_pipeline_cache: Mutex<HashMap<CacheableRenderPipelineDescriptor, Arc<GenericPipeline>>>,
    compute_pipeline_cache:
        Mutex<HashMap<CacheableComputePipelineDescriptor, Arc<GenericPipeline>>>
}

impl RenderCache
{
    pub(crate) fn new(device: Arc<wgpu::Device>) -> Self
    {
        Self {
            device,
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            pipeline_layout_cache: Mutex::new(HashMap::new()),
            shader_module_cache: Mutex::new(HashMap::new()),
            render_pipeline_cache: Mutex::new(HashMap::new()),
            compute_pipeline_cache: Mutex::new(HashMap::new())
        }
    }

    pub(crate) fn trim(&self)
    {
        self.bind_group_layout_cache
            .lock()
            .unwrap()
            .retain(|_, arc| Arc::strong_count(&arc) > 1);

        self.pipeline_layout_cache
            .lock()
            .unwrap()
            .retain(|_, arc| Arc::strong_count(&arc) > 1);

        self.shader_module_cache
            .lock()
            .unwrap()
            .retain(|_, arc| Arc::strong_count(&arc) > 1);

        self.render_pipeline_cache
            .lock()
            .unwrap()
            .retain(|_, arc| Arc::strong_count(&arc) > 1);

        self.compute_pipeline_cache
            .lock()
            .unwrap()
            .retain(|_, arc| Arc::strong_count(&arc) > 1);
    }

    pub fn cache_bind_group_layout(
        &self,
        descriptor: wgpu::BindGroupLayoutDescriptor<'static>
    ) -> Arc<wgpu::BindGroupLayout>
    {
        self.bind_group_layout_cache
            .lock()
            .unwrap()
            .entry(CacheableBindGroupLayoutDescriptor(descriptor))
            .or_insert_with_key(|k| Arc::new(self.device.create_bind_group_layout(&k.0)))
            .clone()
    }

    pub fn cache_pipeline_layout(
        &self,
        descriptor: CacheablePipelineLayoutDescriptor
    ) -> Arc<wgpu::PipelineLayout>
    {
        self.pipeline_layout_cache
            .lock()
            .unwrap()
            .entry(descriptor)
            .or_insert_with_key(|k| {
                Arc::new(
                    k.access(|raw_descriptor| self.device.create_pipeline_layout(raw_descriptor))
                )
            })
            .clone()
    }

    pub fn cache_shader_module(
        &self,
        descriptor: wgpu::ShaderModuleDescriptor<'static>
    ) -> Arc<wgpu::ShaderModule>
    {
        self.shader_module_cache
            .lock()
            .unwrap()
            .entry(CacheableShaderModuleDescriptor(descriptor))
            .or_insert_with_key(|k| Arc::new(self.device.create_shader_module(k.0.clone())))
            .clone()
    }

    pub fn cache_render_pipeline(
        &self,
        descriptor: CacheableRenderPipelineDescriptor
    ) -> Arc<GenericPipeline>
    {
        self.render_pipeline_cache
            .lock()
            .unwrap()
            .entry(descriptor)
            .or_insert_with_key(|k| {
                Arc::new(GenericPipeline::Render(k.access(|raw_descriptor| {
                    self.device.create_render_pipeline(raw_descriptor)
                })))
            })
            .clone()
    }

    pub fn cache_compute_pipeline(
        &self,
        descriptor: CacheableComputePipelineDescriptor
    ) -> Arc<GenericPipeline>
    {
        self.compute_pipeline_cache
            .lock()
            .unwrap()
            .entry(descriptor)
            .or_insert_with_key(|k| {
                Arc::new(GenericPipeline::Compute(k.access(|raw_descriptor| {
                    self.device.create_compute_pipeline(raw_descriptor)
                })))
            })
            .clone()
    }
}

#[derive(Debug)]
struct CacheableBindGroupLayoutDescriptor(wgpu::BindGroupLayoutDescriptor<'static>);

impl PartialEq for CacheableBindGroupLayoutDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = &self.0;
        let r = &other.0;

        l.label == r.label
            && l.entries.len() == r.entries.len()
            && l.entries
                .iter()
                .zip(r.entries.iter())
                .fold(true, |acc, (l, r)| l == r && acc)
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

#[derive(Debug, Clone)]
pub struct CacheablePipelineLayoutDescriptor
{
    pub label:                Cow<'static, str>,
    pub bind_group_layouts:   Vec<Arc<wgpu::BindGroupLayout>>,
    pub push_constant_ranges: Vec<wgpu::PushConstantRange>
}

impl CacheablePipelineLayoutDescriptor
{
    fn access<R>(&self, access_func: impl FnOnce(&wgpu::PipelineLayoutDescriptor<'_>) -> R) -> R
    {
        let ref_vec: Vec<&wgpu::BindGroupLayout> =
            self.bind_group_layouts.iter().map(|l| &**l).collect();

        let descriptor = wgpu::PipelineLayoutDescriptor {
            label:                Some(&self.label),
            bind_group_layouts:   &ref_vec,
            push_constant_ranges: &self.push_constant_ranges
        };

        access_func(&descriptor)
    }
}

impl PartialEq for CacheablePipelineLayoutDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        self.label == other.label
            && self.bind_group_layouts.len() == other.bind_group_layouts.len()
            && self
                .bind_group_layouts
                .iter()
                .zip(other.bind_group_layouts.iter())
                .fold(true, |acc, (l, r)| {
                    acc && Arc::as_ptr(&l) == Arc::as_ptr(&r)
                })
            && self.push_constant_ranges == other.push_constant_ranges
    }
}

impl Eq for CacheablePipelineLayoutDescriptor {}

impl Hash for CacheablePipelineLayoutDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.label.hash(state);
        self.bind_group_layouts
            .iter()
            .for_each(|l| Arc::as_ptr(&l).hash(state));
        self.push_constant_ranges.hash(state);
    }
}

#[derive(Debug)]
struct CacheableShaderModuleDescriptor(wgpu::ShaderModuleDescriptor<'static>);

impl PartialEq for CacheableShaderModuleDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = &self.0;
        let r = &other.0;

        l.label == r.label
            && match (&l.source, &r.source)
            {
                (wgpu::ShaderSource::Wgsl(l_s), wgpu::ShaderSource::Wgsl(r_s)) => l_s == r_s,
                _ => unimplemented!()
            }
    }
}

impl Eq for CacheableShaderModuleDescriptor {}

impl Hash for CacheableShaderModuleDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.0.label.hash(state);

        match &self.0.source
        {
            wgpu::ShaderSource::Wgsl(s) => s.hash(state),
            _ => unimplemented!()
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheableFragmentState
{
    pub module:      Arc<wgpu::ShaderModule>,
    pub entry_point: Cow<'static, str>,
    pub targets:     Vec<Option<wgpu::ColorTargetState>>
}

impl PartialEq for CacheableFragmentState
{
    fn eq(&self, other: &Self) -> bool
    {
        Arc::as_ptr(&self.module) == Arc::as_ptr(&other.module)
            && self.entry_point == other.entry_point
            && self.targets == other.targets
    }
}

impl Eq for CacheableFragmentState {}

impl Hash for CacheableFragmentState
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        Arc::as_ptr(&self.module).hash(state);
        self.entry_point.hash(state);
        self.targets.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct CacheableRenderPipelineDescriptor
{
    pub label:                 Cow<'static, str>,
    pub layout:                Option<Arc<PipelineLayout>>,
    pub vertex_module:         Arc<wgpu::ShaderModule>,
    pub vertex_entry_point:    Cow<'static, str>,
    pub vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout<'static>>,
    pub fragment_state:        Option<CacheableFragmentState>,
    pub primitive_state:       wgpu::PrimitiveState,
    pub depth_stencil_state:   Option<wgpu::DepthStencilState>,
    pub multisample_state:     wgpu::MultisampleState,
    pub multiview:             Option<NonZeroU32>
}

impl CacheableRenderPipelineDescriptor
{
    fn access<R>(&self, access_func: impl FnOnce(&wgpu::RenderPipelineDescriptor<'_>) -> R) -> R
    {
        let ref_layout = self.layout.as_ref().map(|l| &**l);

        let ref_fragment_state: Option<wgpu::FragmentState> = match &self.fragment_state
        {
            Some(c) =>
            {
                Some(wgpu::FragmentState {
                    module:      &c.module,
                    entry_point: &c.entry_point,
                    targets:     &c.targets
                })
            }
            None => None
        };

        let descriptor = wgpu::RenderPipelineDescriptor {
            label:         Some(&self.label),
            layout:        ref_layout,
            vertex:        wgpu::VertexState {
                module:      &self.vertex_module,
                entry_point: &self.vertex_entry_point,
                buffers:     &self.vertex_buffer_layouts
            },
            primitive:     self.primitive_state,
            depth_stencil: self.depth_stencil_state.clone(),
            multisample:   self.multisample_state,
            fragment:      ref_fragment_state,
            multiview:     self.multiview
        };

        access_func(&descriptor)
    }
}

impl PartialEq for CacheableRenderPipelineDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        self.label == other.label
            && self.layout.as_ref().map(|l| Arc::as_ptr(&l))
                == other.layout.as_ref().map(|r| Arc::as_ptr(&r))
            && Arc::as_ptr(&self.vertex_module) == Arc::as_ptr(&other.vertex_module)
            && self.vertex_entry_point == other.vertex_entry_point
            && self.vertex_buffer_layouts == other.vertex_buffer_layouts
            && self.fragment_state == other.fragment_state
            && self.primitive_state == other.primitive_state
            && self.depth_stencil_state == other.depth_stencil_state
            && self.multisample_state == other.multisample_state
            && self.multiview == other.multiview
    }
}

impl Eq for CacheableRenderPipelineDescriptor {}

impl Hash for CacheableRenderPipelineDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.label.hash(state);
        self.layout.as_ref().map(|l| Arc::as_ptr(&l)).hash(state);
        Arc::as_ptr(&self.vertex_module).hash(state);
        self.vertex_entry_point.hash(state);
        self.vertex_buffer_layouts.hash(state);
        self.fragment_state.hash(state);
        self.primitive_state.hash(state);
        self.depth_stencil_state.hash(state);
        self.multisample_state.hash(state);
        self.multiview.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct CacheableComputePipelineDescriptor
{
    label:       Cow<'static, str>,
    layout:      Option<Arc<wgpu::PipelineLayout>>,
    module:      Arc<wgpu::ShaderModule>,
    entry_point: Cow<'static, str>
}

impl CacheableComputePipelineDescriptor
{
    pub fn access<R>(
        &self,
        access_func: impl FnOnce(&wgpu::ComputePipelineDescriptor<'_>) -> R
    ) -> R
    {
        let descriptor = wgpu::ComputePipelineDescriptor {
            label:       Some(&self.label),
            layout:      self.layout.as_ref().map(|l| &**l),
            module:      &self.module,
            entry_point: &self.entry_point
        };

        access_func(&descriptor)
    }
}

impl PartialEq for CacheableComputePipelineDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        self.label == other.label
            && self.layout.as_ref().map(|l| Arc::as_ptr(&*l))
                == other.layout.as_ref().map(|l| Arc::as_ptr(&*l))
            && Arc::as_ptr(&self.module) == Arc::as_ptr(&other.module)
            && self.entry_point == other.entry_point
    }
}

impl Eq for CacheableComputePipelineDescriptor {}

impl Hash for CacheableComputePipelineDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.label.hash(state);
        self.layout.as_ref().map(|l| Arc::as_ptr(&l).hash(state));
        Arc::as_ptr(&self.module).hash(state);
        self.entry_point.hash(state);
    }
}

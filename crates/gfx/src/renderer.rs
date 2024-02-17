use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::atomic::Ordering::{self, *};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use bytemuck::{bytes_of, Pod, Zeroable};
use nalgebra_glm as glm;
use pollster::FutureExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use strum::IntoEnumIterator;
use util::{Registrar, SendSyncMutPtr};
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::{EventLoop, EventLoopWindowTarget};
use winit::keyboard::KeyCode;
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{CursorGrabMode, Window, WindowBuilder};
use winit_input_helper::WinitInputHelper;

use crate::recordables::{DrawId, PassStage, RecordInfo, Recordable};
use crate::{Camera, Transform};

pub const SURFACE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug)]
pub struct Renderer
{
    pub queue:                    wgpu::Queue,
    pub global_bind_group_layout: wgpu::BindGroupLayout,

    // Render Cache
    bind_group_layout_cache:
        Mutex<HashMap<CacheableBindGroupLayoutDescriptor, Arc<wgpu::BindGroupLayout>>>,
    pipeline_layout_cache:
        Mutex<HashMap<CacheablePipelineLayoutDescriptor, Arc<wgpu::PipelineLayout>>>,
    shader_module_cache: Mutex<HashMap<CacheableShaderModuleDescriptor, Arc<wgpu::ShaderModule>>>,
    compute_pipeline_cache:
        Mutex<HashMap<CacheableComputePipelineDescriptor, Arc<GenericPipeline>>>,
    render_pipeline_cache: Mutex<HashMap<CacheableRenderPipelineDescriptor, Arc<GenericPipeline>>>,

    device:      Arc<wgpu::Device>,
    renderables: util::Registrar<util::Uuid, Weak<dyn Recordable>>,

    // Rendering views
    window_size_x:            AtomicU32,
    window_size_y:            AtomicU32,
    float_delta_frame_time_s: AtomicU32,

    // Rendering
    thread_id:        ThreadId,
    critical_section: Mutex<CriticalSection>
}

impl Drop for Renderer
{
    fn drop(&mut self)
    {
        if std::thread::current().id() != self.thread_id
        {
            eprintln!("Dropping Renderer from a thread it was not created on!")
        }

        if Arc::strong_count(&self.device) != 1
        {
            log::warn!("Retained device! {}", Arc::strong_count(&self.device))
        }
    }
}

impl Deref for Renderer
{
    type Target = wgpu::Device;

    fn deref(&self) -> &Self::Target
    {
        &self.device
    }
}

// SAFETY: You must not receive any EventLoop events outside of the thread that
// created it, you can't even do this as its behind a mutex!
// We also verify this with a threadID
unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

impl Renderer
{
    #[allow(clippy::new_without_default)]

    /// # Safety
    ///
    /// The Self returned must be dropped on the same thread that it was created
    /// on
    pub unsafe fn new() -> Self
    {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_inner_size(PhysicalSize {
                width:  1920,
                height: 1080
            })
            .with_position(winit::dpi::PhysicalPosition {
                x: 100, y: 100
            })
            .build(&event_loop)
            .unwrap();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            #[cfg(debug_assertions)]
            flags: wgpu::InstanceFlags::debugging(),
            #[cfg(not(debug_assertions))]
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        // SAFETY: The window must outlive the surface, this is guarantee by Renderer's
        // Drop order
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
                .unwrap()
        };

        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false
            })
            .block_on()
            .expect("Failed to find a wgpu Adapter!");

        if !adapter.get_downlevel_capabilities().is_webgpu_compliant()
        {
            log::warn!("Device is not fully supported!");
        }

        let maybe_driver_version = adapter.get_info().driver_info;

        let version_string: String = if maybe_driver_version.is_empty()
        {
            "".into()
        }
        else
        {
            format!("with version {} ", maybe_driver_version)
        };

        log::info!(
            "Selected Device {} {}using backend {:?}",
            adapter.get_info().name,
            version_string,
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("Device"),
                    required_features: wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::TEXTURE_BINDING_ARRAY
                        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    required_limits:   adapter.limits()
                },
                None
            )
            .block_on()
            .unwrap();
        let device = Arc::new(device);

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .into_iter()
            .find(|f| *f == SURFACE_TEXTURE_FORMAT)
            .unwrap();

        let desired_present_modes = [
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo
        ];

        let selected_mode = desired_present_modes
            .into_iter()
            .find_map(|mode| {
                surface_caps
                    .present_modes
                    .iter()
                    .find(|other_mode| mode == **other_mode)
            })
            .map(|m| m.to_owned());

        log::info!("Selected present mode {:?}", selected_mode.unwrap());

        let config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:                        surface_format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  selected_mode.unwrap(),
            alpha_mode:                    surface_caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2
        };
        surface.configure(&device, &config);

        let critical_section = CriticalSection {
            thread_id: std::thread::current().id(),
            surface,
            config,
            window,
            event_loop,
            camera: RefCell::new(super::Camera::new(
                glm::Vec3::new(0.0, 0.0, -10.0),
                0.0,
                0.0
            ))
        };

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("Global Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   NonZeroU64::new(
                                std::mem::size_of::<ShaderGlobalInfo>() as u64
                            )
                        },
                        count:      None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   NonZeroU64::new(
                                std::mem::size_of::<ShaderMatrices>() as u64
                            )
                        },
                        count:      None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   NonZeroU64::new(
                                std::mem::size_of::<ShaderMatrices>() as u64
                            )
                        },
                        count:      None
                    }
                ]
            });

        Renderer {
            thread_id: std::thread::current().id(),
            renderables: Registrar::new(),
            queue,
            device,
            critical_section: Mutex::new(critical_section),
            global_bind_group_layout,
            window_size_x: AtomicU32::new(size.width),
            window_size_y: AtomicU32::new(size.height),
            float_delta_frame_time_s: AtomicU32::new(0.0f32.to_bits()),
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            pipeline_layout_cache: Mutex::new(HashMap::new()),
            shader_module_cache: Mutex::new(HashMap::new()),
            compute_pipeline_cache: Mutex::new(HashMap::new()),
            render_pipeline_cache: Mutex::new(HashMap::new())
        }
    }

    pub fn register(&self, renderable: Arc<dyn Recordable>)
    {
        self.renderables
            .insert(renderable.get_uuid(), Arc::downgrade(&renderable));
    }

    pub fn get_fov(&self) -> glm::Vec2
    {
        let y_rads: f32 = glm::radians(&glm::Vec1::new(70.0)).x;
        let x_rads: f32 = y_rads * self.get_aspect_ratio();

        glm::Vec2::new(x_rads, y_rads)
    }

    pub fn get_aspect_ratio(&self) -> f32
    {
        let width_height = self.get_framebuffer_size().xy();

        width_height.x as f32 / width_height.y as f32
    }

    pub fn get_framebuffer_size(&self) -> glm::UVec2
    {
        let loaded_x = self.window_size_x.load(Ordering::SeqCst);
        let loaded_y = self.window_size_y.load(Ordering::SeqCst);

        glm::UVec2::new(loaded_x, loaded_y)
    }

    pub fn get_delta_time(&self) -> f32
    {
        f32::from_bits(self.float_delta_frame_time_s.load(Ordering::Acquire))
    }

    pub fn cache_bind_group_layout(
        &self,
        create_info: wgpu::BindGroupLayoutDescriptor<'static>
    ) -> Arc<wgpu::BindGroupLayout>
    {
        self.bind_group_layout_cache
            .lock()
            .unwrap()
            .entry(CacheableBindGroupLayoutDescriptor(create_info))
            .or_insert_with_key(|k| Arc::new(self.device.create_bind_group_layout(&k.0)))
            .clone()
    }

    pub fn cache_pipeline_layout(
        &self,
        create_info: wgpu::PipelineLayoutDescriptor<'_>
    ) -> Arc<wgpu::PipelineLayout>
    {
        self.pipeline_layout_cache
            .lock()
            .unwrap()
            .entry(CacheablePipelineLayoutDescriptor(create_info))
            .or_insert_with_key(|k| Arc::new(self.device.create_pipeline_layout(&k.0)))
            .clone()
    }

    pub fn cache_shader_module(
        &self,
        create_info: wgpu::ShaderModuleDescriptor<'static>
    ) -> Arc<wgpu::ShaderModule>
    {
        self.shader_module_cache
            .lock()
            .unwrap()
            .entry(CacheableShaderModuleDescriptor(create_info))
            .or_insert_with_key(|k| Arc::new(self.device.create_shader_module(k.0)))
            .clone()
    }

    pub fn cache_compute_pipeline(
        &self,
        create_info: wgpu::ComputePipelineDescriptor<'static>
    ) -> Arc<GenericPipeline>
    {
        self.compute_pipeline_cache
            .lock()
            .unwrap()
            .entry(CacheableComputePipelineDescriptor(create_info))
            .or_insert_with_key(|k| {
                Arc::new(GenericPipeline::Compute(
                    self.device.create_compute_pipeline(&k.0)
                ))
            })
            .clone()
    }

    pub fn cache_render_pipeline(
        &self,
        create_info: wgpu::RenderPipelineDescriptor<'static>
    ) -> Arc<GenericPipeline>
    {
        self.render_pipeline_cache
            .lock()
            .unwrap()
            .entry(CacheableRenderPipelineDescriptor(create_info))
            .or_insert_with_key(|k| {
                Arc::new(GenericPipeline::Render(
                    self.device.create_render_pipeline(&k.0)
                ))
            })
            .clone()
    }

    fn set_framebuffer_size(&self, new_size: glm::UVec2)
    {
        self.window_size_x.store(new_size.x, Ordering::SeqCst);
        self.window_size_y.store(new_size.y, Ordering::SeqCst);
    }

    pub fn enter_gfx_loop(&self, should_stop: &AtomicBool)
    {
        let mut guard = self.critical_section.lock().unwrap();
        let CriticalSection {
            thread_id,
            surface,
            config,
            window,
            event_loop,
            camera
        } = &mut *guard;

        assert!(
            *thread_id == std::thread::current().id(),
            "Renderer::enter_gfx_loop() must be called on the same thread that Renderer::new() \
             was called from!"
        );

        // Helper Structures
        let input_helper = RefCell::new(WinitInputHelper::new());

        let depth_buffer = RefCell::new(create_depth_buffer(&self.device, config));

        let global_info_uniform_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               std::mem::size_of::<ShaderGlobalInfo>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_mvp_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               std::mem::size_of::<ShaderMatrices>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_model_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               std::mem::size_of::<ShaderMatrices>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_bind_group = self.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Global Info Bind Group"),
            layout:  &self.global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: global_info_uniform_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: global_mvp_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: global_model_buffer.as_entire_binding()
                }
            ]
        });

        // Because of a bug in winit, the first resize command that comes in is borked
        // on Windows https://github.com/rust-windowing/winit/issues/2094
        // we want to skip the first resize event
        #[cfg(target_os = "windows")]
        let mut is_first_resize = true;
        #[cfg(not(target_os = "windows"))]
        let mut is_first_resize = false;

        let mut resize_func = |maybe_new_size: Option<PhysicalSize<u32>>| {
            if is_first_resize
            {
                is_first_resize = false;
                return;
            }

            let new_size = maybe_new_size.unwrap_or_else(|| {
                let old_size = self.get_framebuffer_size();

                PhysicalSize {
                    width:  old_size.x,
                    height: old_size.y
                }
            });

            if new_size.width > 0 && new_size.height > 0
            {
                self.set_framebuffer_size(glm::UVec2::new(new_size.width, new_size.height));

                config.width = new_size.width;
                config.height = new_size.height;

                surface.configure(&self.device, config);

                *depth_buffer.borrow_mut() = create_depth_buffer(&self.device, config);
            }
        };

        let mut previous_frame_time = std::time::Instant::now();

        let mut render_func = || -> Result<(), wgpu::SurfaceError> {
            let output = surface.get_current_texture()?;
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut renderables_map: HashMap<
                PassStage,
                Vec<(Arc<dyn Recordable>, Option<DrawId>)>
            > = PassStage::iter().map(|s| (s, Vec::new())).collect();

            let mut shader_mvps = ShaderMatrices {
                ..Default::default()
            };
            let mut shader_models = ShaderMatrices {
                ..Default::default()
            };

            let shader_mvp_ptr: SendSyncMutPtr<glm::Mat4> =
                shader_mvps.matrices.as_mut_ptr().into();

            let shader_model_ptr: SendSyncMutPtr<glm::Mat4> =
                shader_models.matrices.as_mut_ptr().into();

            let id_counter = AtomicU32::new(0);
            let get_next_id = || {
                let maybe_new_id = id_counter.fetch_add(1, Ordering::Relaxed);

                if maybe_new_id >= SHADER_MATRICES_SIZE as u32
                {
                    panic!("Too many draw calls with matrices!");
                }

                maybe_new_id
            };

            {
                let closure_camera = camera.borrow().clone();

                self.renderables
                    .access()
                    .into_par_iter()
                    .filter_map(|(ptr, weak_renderable)| {
                        match weak_renderable.upgrade()
                        {
                            Some(r) =>
                            {
                                let record_info = r.pre_record_update(self, &closure_camera);

                                match record_info
                                {
                                    RecordInfo {
                                        should_draw: false, ..
                                    } => None,
                                    RecordInfo {
                                        should_draw: true,
                                        transform: Some(t)
                                    } =>
                                    {
                                        let this_id = get_next_id();

                                        unsafe {
                                            shader_mvp_ptr
                                                .add(this_id as usize)
                                                .write(closure_camera.get_perspective(self, &t));

                                            shader_model_ptr
                                                .add(this_id as usize)
                                                .write(t.as_model_matrix())
                                        };

                                        Some((r, Some(this_id)))
                                    }
                                    RecordInfo {
                                        should_draw: true,
                                        transform: None
                                    } => Some((r, None))
                                }
                            }
                            None =>
                            {
                                self.renderables.delete(ptr);
                                None
                            }
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .for_each(|(r, r_i)| {
                        renderables_map
                            .get_mut(&r.get_pass_stage())
                            .unwrap()
                            .push((r, r_i));
                    });
            }

            renderables_map.iter_mut().for_each(|(_, renderable_vec)| {
                renderable_vec.sort_by(|l, r| l.0.ord(&*r.0, &global_bind_group))
            });

            let global_info = {
                let guard = camera.borrow();

                ShaderGlobalInfo {
                    camera_pos:      guard.get_position(),
                    _padding:        0.0,
                    view_projection: guard.get_perspective(
                        self,
                        &crate::Transform {
                            ..Default::default()
                        }
                    )
                }
            };

            self.queue
                .write_buffer(&global_info_uniform_buffer, 0, bytes_of(&global_info));
            self.queue
                .write_buffer(&global_mvp_buffer, 0, bytes_of(&shader_mvps));
            self.queue
                .write_buffer(&global_model_buffer, 0, bytes_of(&shader_models));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                });

            let mut active_pipeline: Option<&GenericPipeline> = None;
            let mut active_bind_groups: [Option<wgpu::Id<wgpu::BindGroup>>; 4] = [None; 4];

            for pass_type in PassStage::iter()
            {
                let (_, ref depth_view, _) = *depth_buffer.borrow();

                let mut render_pass: GenericPass = match pass_type
                {
                    PassStage::GraphicsSimpleColor =>
                    {
                        GenericPass::Render(encoder.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label:                    Some("Render Pass"),
                                color_attachments:        &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view:           &view,
                                        resolve_target: None,
                                        ops:            wgpu::Operations {
                                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.1,
                                                g: 0.2,
                                                b: 0.3,
                                                a: 1.0
                                            }),
                                            store: wgpu::StoreOp::Store
                                        }
                                    }
                                )],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view:        depth_view,
                                        depth_ops:   Some(wgpu::Operations {
                                            load:  wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store
                                        }),
                                        stencil_ops: None
                                    }
                                ),
                                occlusion_query_set:      None,
                                timestamp_writes:         None
                            }
                        ))
                    }
                };

                for (renderable, maybe_id) in renderables_map.get(&pass_type).unwrap()
                {
                    let desired_pipeline = renderable.get_pipeline();

                    if active_pipeline != Some(desired_pipeline)
                    {
                        match (desired_pipeline, &mut render_pass)
                        {
                            (GenericPipeline::Compute(p), GenericPass::Compute(pass)) =>
                            {
                                pass.set_pipeline(p);
                            }
                            (GenericPipeline::Render(p), GenericPass::Render(pass)) =>
                            {
                                pass.set_pipeline(p)
                            }
                            (_, _) => panic!("Pass Pipeline Invariant Violated!")
                        }

                        active_pipeline = Some(desired_pipeline);
                    }

                    for (idx, (active_bind_group_id, maybe_new_bind_group)) in active_bind_groups
                        .iter_mut()
                        .zip(renderable.get_bind_groups(&global_bind_group))
                        .enumerate()
                    {
                        if *active_bind_group_id != maybe_new_bind_group.map(|g| g.global_id())
                        {
                            if let Some(new_bind_group) = maybe_new_bind_group
                            {
                                match render_pass
                                {
                                    GenericPass::Compute(ref mut p) =>
                                    {
                                        p.set_bind_group(idx as u32, new_bind_group, &[])
                                    }
                                    GenericPass::Render(ref mut p) =>
                                    {
                                        p.set_bind_group(idx as u32, new_bind_group, &[])
                                    }
                                }
                                *active_bind_group_id = Some(new_bind_group.global_id());
                            }
                        }
                    }

                    renderable.record(&mut render_pass, *maybe_id);
                }

                active_pipeline = None;
                active_bind_groups = [None; 4];
            }

            window.pre_present_notify();

            self.queue.submit([encoder.finish()]);
            output.present();

            let now = std::time::Instant::now();

            self.float_delta_frame_time_s.store(
                (now - previous_frame_time).as_secs_f32().to_bits(),
                Ordering::Release
            );

            previous_frame_time = now;

            Ok(())
        };

        let handle_input = |control_flow: &EventLoopWindowTarget<()>| {
            // TODO: do the trig thing so that diagonal isn't faster!

            let input_helper = input_helper.borrow();

            let move_scale = 10.0
                * if input_helper.key_held(KeyCode::ShiftLeft)
                {
                    5.0
                }
                else
                {
                    1.0
                };
            let rotate_scale = 1000.0;
            if input_helper.key_held(KeyCode::KeyK)
            {
                log::info!("position: {:?}", camera.borrow().get_position())
            }

            if input_helper.key_held(KeyCode::KeyW)
            {
                let v = camera.borrow().get_forward_vector() * move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::KeyS)
            {
                let v = camera.borrow().get_forward_vector() * -move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::KeyD)
            {
                let v = camera.borrow().get_right_vector() * move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::KeyA)
            {
                let v = camera.borrow().get_right_vector() * -move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::Space)
            {
                let v = *Transform::global_up_vector() * move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::ControlLeft)
            {
                let v = *Transform::global_up_vector() * -move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            let mouse_diff_px: glm::Vec2 = {
                let mouse_cords_diff_px_f32: (f32, f32) = input_helper.mouse_diff();

                glm::Vec2::new(mouse_cords_diff_px_f32.0, mouse_cords_diff_px_f32.1)
            };

            let screen_size_px: glm::Vec2 = {
                let screen_size_u32 = self.get_framebuffer_size();

                glm::Vec2::new(screen_size_u32.x as f32, screen_size_u32.y as f32)
            };

            // delta over the whole screen -1 -> 1
            let normalized_delta = mouse_diff_px.component_div(&screen_size_px);

            let delta_rads = normalized_delta
                .component_div(&glm::Vec2::repeat(2.0))
                .component_mul(&self.get_fov());

            {
                let mut camera_guard = camera.borrow_mut();

                camera_guard.add_yaw(delta_rads.x * rotate_scale * self.get_delta_time());
                camera_guard.add_pitch(delta_rads.y * rotate_scale * self.get_delta_time());
            }

            if input_helper.key_pressed(KeyCode::Escape)
            {
                control_flow.exit();
            }
        };

        window.set_cursor_visible(false);
        #[cfg(target_os = "macos")]
        window.set_cursor_grab(CursorGrabMode::Locked).unwrap();
        #[cfg(not(target_os = "macos"))]
        window.set_cursor_grab(CursorGrabMode::Confined).unwrap();

        let _ = event_loop.run_on_demand(|event, control_flow| {
            if should_stop.load(Acquire)
            {
                control_flow.exit();
            }

            input_helper.borrow_mut().update(&event);

            match event
            {
                Event::WindowEvent {
                    window_id,
                    ref event
                } if window_id == window.id() =>
                {
                    match event
                    {
                        WindowEvent::Resized(new_size) => resize_func(Some(*new_size)),
                        WindowEvent::CloseRequested => control_flow.exit(),
                        WindowEvent::RedrawRequested =>
                        {
                            use wgpu::SurfaceError::*;

                            handle_input(control_flow);

                            match render_func()
                            {
                                Ok(_) => (),
                                Err(Timeout) => log::warn!("render timeout!"),
                                Err(Outdated) => (),
                                Err(Lost) => resize_func(None),
                                Err(OutOfMemory) => todo!()
                            }

                            window.request_redraw();
                        }
                        _ => ()
                    }
                }

                _ => ()
            }

            if control_flow.exiting()
            {
                should_stop.store(true, Release);
            }
        });

        log::info!("event loop returned");
    }

    pub fn get_default_depth_state() -> wgpu::DepthStencilState
    {
        wgpu::DepthStencilState {
            format:              DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare:       wgpu::CompareFunction::Less,
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default()
        }
    }

    pub fn get_surface_format() -> wgpu::TextureFormat
    {
        SURFACE_TEXTURE_FORMAT
    }
}

#[derive(Debug)]
struct CriticalSection
{
    thread_id:  ThreadId,
    surface:    wgpu::Surface<'static>,
    config:     wgpu::SurfaceConfiguration,
    window:     Window,
    event_loop: EventLoop<()>,

    camera: RefCell<Camera>
}

unsafe impl Sync for CriticalSection {}

fn create_depth_buffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler)
{
    let size = wgpu::Extent3d {
        width:                 config.width,
        height:                config.height,
        depth_or_array_layers: 1
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("Depth Buffer"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[]
    };
    let texture = device.create_texture(&desc);

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        // 4.
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual), // 5.
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });

    (texture, view, sampler)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub(crate) struct ShaderGlobalInfo
{
    camera_pos:      glm::Vec3,
    _padding:        f32,
    view_projection: glm::Mat4
}

const SHADER_MATRICES_SIZE: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub(crate) struct ShaderMatrices
{
    matrices: [glm::Mat4; SHADER_MATRICES_SIZE]
}

impl Default for ShaderMatrices
{
    fn default() -> Self
    {
        Self {
            matrices: [Default::default(); SHADER_MATRICES_SIZE]
        }
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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>
    {
        self.global_id().partial_cmp(&other.global_id())
    }
}

impl Ord for GenericPipeline
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering
    {
        self.global_id().cmp(&other.global_id())
    }
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

#[derive(Debug)]
struct CacheablePipelineLayoutDescriptor(wgpu::PipelineLayoutDescriptor<'static>);

impl PartialEq for CacheablePipelineLayoutDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = self.0;
        let r = other.0;

        l.label == r.label
            && l.bind_group_layouts.len() == r.bind_group_layouts.len()
            && l.bind_group_layouts
                .iter()
                .zip(r.bind_group_layouts.iter())
                .fold(true, |acc, (l, r)| l.global_id() == r.global_id() && acc)
            && l.push_constant_ranges.len() == r.push_constant_ranges.len()
            && l.push_constant_ranges
                .iter()
                .zip(r.push_constant_ranges.iter())
                .fold(true, |acc, (l, r)| l == r && acc)
    }
}

impl Eq for CacheablePipelineLayoutDescriptor {}

impl Hash for CacheablePipelineLayoutDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.0.label.hash(state);

        self.0
            .bind_group_layouts
            .iter()
            .for_each(|l| l.global_id().hash(state));

        self.0
            .push_constant_ranges
            .iter()
            .for_each(|r| r.hash(state));
    }
}

#[derive(Debug)]
struct CacheableShaderModuleDescriptor(wgpu::ShaderModuleDescriptor<'static>);

impl PartialEq for CacheableShaderModuleDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = self.0;
        let r = other.0;

        l.label == r.label
            && match (l.source, r.source)
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

        match self.0.source
        {
            wgpu::ShaderSource::Wgsl(s) => s.hash(state),
            _ => unimplemented!()
        }
    }
}

#[derive(Debug)]
struct CacheableComputePipelineDescriptor(wgpu::ComputePipelineDescriptor<'static>);

impl PartialEq for CacheableComputePipelineDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = self.0;
        let r = other.0;

        l.label == r.label
            && l.layout.map(|l| l.global_id()) == r.layout.map(|l| l.global_id())
            && l.module.global_id() == r.module.global_id()
            && l.entry_point == r.entry_point
    }
}

impl Eq for CacheableComputePipelineDescriptor {}

impl Hash for CacheableComputePipelineDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.0.label.hash(state);

        match self.0.layout
        {
            Some(l) => l.global_id().hash(state),
            None => state.write_i8(-34) // value, fresh from my ass!
        }

        self.0.module.global_id().hash(state);

        self.0.entry_point.hash(state);
    }
}

#[derive(Debug)]
struct CacheableRenderPipelineDescriptor(wgpu::RenderPipelineDescriptor<'static>);

impl PartialEq for CacheableRenderPipelineDescriptor
{
    fn eq(&self, other: &Self) -> bool
    {
        let l = self.0;
        let r = other.0;

        let eq = l.label == r.label
            && l.layout.map(|l| l.global_id()) == r.layout.map(|l| l.global_id())
            && l.vertex.module.global_id() == r.vertex.module.global_id()
            && l.vertex.entry_point == r.vertex.entry_point
            && l.vertex.buffers.len() == r.vertex.buffers.len()
            && l.vertex
                .buffers
                .iter()
                .zip(r.vertex.buffers.iter())
                .fold(true, |acc, (l, r)| l == r && acc)
            && l.primitive == r.primitive
            && l.depth_stencil == r.depth_stencil
            && l.multisample == r.multisample
            && l.fragment.map(|s| s.module.global_id()) == r.fragment.map(|s| s.module.global_id())
            && l.fragment.map(|s| s.entry_point) == r.fragment.map(|s| s.entry_point)
            && l.fragment.map(|s| s.targets.len()) == r.fragment.map(|s| s.targets.len())
            && l.multiview == r.multiview;

        if !eq
        {
            return false;
        }

        match (l.fragment, r.fragment)
        {
            (None, None) => true,
            (Some(l), Some(r)) =>
            {
                for (s_l, s_r) in l.targets.iter().zip(r.targets.iter())
                {
                    if s_l != s_r
                    {
                        return false;
                    }
                }

                true
            }
            _ => false
        }
    }
}

impl Eq for CacheableRenderPipelineDescriptor {}

impl Hash for CacheableRenderPipelineDescriptor
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.0.label.hash(state);

        match self.0.layout
        {
            Some(l) => l.global_id().hash(state),
            None => state.write_i8(-34) // value, fresh from my ass!
        }

        self.0.vertex.module.global_id().hash(state);
        self.0.vertex.entry_point.hash(state);
        self.0.vertex.buffers.hash(state);

        self.0.primitive.hash(state);
        self.0.depth_stencil.hash(state);
        self.0.multisample.hash(state);
        self.0.fragment.map(|f| {
            f.entry_point.hash(state);
            f.module.global_id().hash(state);
            f.targets.hash(state);
        });

        self.0.multiview.hash(state);
    }
}

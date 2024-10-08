use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::num::NonZero;
use std::ops::Deref;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::Ordering::{self};
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use bytemuck::{bytes_of, Pod, Zeroable};
use nalgebra_glm as glm;
use pollster::FutureExt;
use util::{AtomicF32, AtomicU32U32, Registrar};
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};

use crate::recordables::{RecordInfo, Recordable, RenderPassId};
use crate::render_cache::{GenericPass, RenderCache};
use crate::{Camera, DrawId, GenericPipeline, InputManager};

#[derive(Debug)]
pub struct Renderer
{
    pub queue:                    wgpu::Queue,
    pub device:                   Arc<wgpu::Device>,
    pub render_cache:             RenderCache,
    pub global_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    renderables:           util::Registrar<util::Uuid, Weak<dyn Recordable>>,
    screen_sized_textures: util::Registrar<util::Uuid, Weak<super::ScreenSizedTexture>>,

    resize_pingers: (util::PingSender, util::PingReceiver),
    window_size:    AtomicU32U32,
    delta_time:     AtomicF32,
    limits:         wgpu::Limits,
    input_manager:  InputManager,

    // Rendering
    camera_window:    util::Window<Camera>,
    thread_id:        ThreadId,
    critical_section: Mutex<CriticalSection>
}

impl Drop for Renderer
{
    fn drop(&mut self)
    {
        if std::thread::current().id() != self.thread_id
        {
            log::error!(
                "Dropping Renderer from a thread ({}) it was not created on!",
                std::thread::current().name().unwrap_or("???")
            )
        }

        self.display_holdover_recordables_info();
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
unsafe impl Send for Renderer {}

// SAFETY: you must not pass any events via the window channel as, for some
// ungodly reason, they use an Rc rather than an Arc on the message channel
unsafe impl Sync for Renderer {}

impl UnwindSafe for Renderer {}
impl RefUnwindSafe for Renderer {}

impl Renderer
{
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const SURFACE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    #[allow(clippy::new_without_default)]

    /// # Safety
    ///
    /// The Self returned must be dropped on the same thread that it was created
    /// on
    pub unsafe fn new(
        window_title: impl Into<String>
    ) -> (
        Self,
        util::WindowUpdater<RenderPassSendFunction>,
        util::WindowUpdater<Camera>
    )
    {
        let event_loop = EventLoop::new().unwrap();
        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(PhysicalSize {
                    width:  1920,
                    height: 1080
                })
                .with_title(window_title)
                .with_position(winit::dpi::PhysicalPosition {
                    x: 100, y: 100
                })
                .build(&event_loop)
                .unwrap()
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends:             wgpu::Backends::all(),
            flags:                wgpu::InstanceFlags::from_build_config(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version:   Default::default()
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

        log::info!(
            "Selected Device {} {}using backend {:?}",
            adapter.get_info().name,
            if maybe_driver_version.is_empty()
            {
                Cow::Borrowed("")
            }
            else
            {
                format!("with version {} ", maybe_driver_version).into()
            },
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("Device"),
                    required_features: wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::MULTI_DRAW_INDIRECT
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE
                        | wgpu::Features::SUBGROUP
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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
            .find(|f| *f == Self::SURFACE_TEXTURE_FORMAT)
            .unwrap();

        let desired_present_modes = [
            // wgpu::PresentMode::Mailbox,
            // #[cfg(not(debug_assertions))]
            // wgpu::PresentMode::FifoRelaxed,
            // #[cfg(not(debug_assertions))]
            // wgpu::PresentMode::Fifo
            wgpu::PresentMode::Immediate
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

        let input_manager = InputManager::new(
            window.clone(),
            PhysicalSize {
                width:  config.width,
                height: config.height
            }
        );

        let (renderpass_func_window, renderpass_updater) =
            util::Window::<RenderPassSendFunction>::new(RenderPassSendFunction::new(Vec::new()));

        let (camera_window, camera_window_updater) =
            util::Window::new(Camera::new(glm::Vec3::repeat(0.0), 0.0, 0.0));

        let critical_section = CriticalSection {
            thread_id: std::thread::current().id(),
            surface,
            config,
            window,
            event_loop,
            renderpass_func_window
        };

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("Global Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::all(),
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   None
                        },
                        count:      None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::all(),
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   None
                        },
                        count:      None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::all(),
                        ty:         wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   None
                        },
                        count:      None
                    }
                ]
            });

        let render_cache = RenderCache::new(device.clone());

        (
            Renderer {
                thread_id: std::thread::current().id(),
                renderables: Registrar::new(),
                screen_sized_textures: Registrar::new(),
                queue,
                device,
                critical_section: Mutex::new(critical_section),
                global_bind_group_layout: Arc::new(global_bind_group_layout),
                resize_pingers: util::create_pingers(),
                window_size: AtomicU32U32::new((size.width, size.height)),
                delta_time: AtomicF32::new(0.0f32),
                render_cache,
                limits: adapter.limits(),
                input_manager,
                camera_window
            },
            renderpass_updater,
            camera_window_updater
        )
    }

    pub fn register(&self, renderable: Arc<dyn Recordable>)
    {
        self.renderables
            .insert(renderable.get_uuid(), Arc::downgrade(&renderable));
    }

    pub fn get_resize_pinger(&self) -> util::PingReceiver
    {
        self.resize_pingers.1.clone()
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
        let (loaded_x, loaded_y) = self.window_size.load(Ordering::SeqCst);

        glm::UVec2::new(loaded_x, loaded_y)
    }

    pub fn get_delta_time(&self) -> f32
    {
        self.delta_time.load(Ordering::SeqCst)
    }

    pub fn enter_gfx_loop(&self, should_continue: &dyn Fn() -> bool, request_terminate: &dyn Fn())
    {
        let mut guard = self.critical_section.lock().unwrap();
        let CriticalSection {
            thread_id,
            surface,
            config,
            window,
            event_loop,
            renderpass_func_window
        } = &mut *guard;

        assert!(
            *thread_id == std::thread::current().id(),
            "Renderer::enter_gfx_loop() must be called on the same thread that Renderer::new() \
             was called from!"
        );

        let max_shader_matrices = usize::min(
            self.limits.max_uniform_buffer_binding_size as usize / std::mem::size_of::<glm::Mat4>(),
            4096
        );

        let global_info_uniform_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               std::mem::size_of::<ShaderGlobalInfo>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_mvp_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               (max_shader_matrices * std::mem::size_of::<glm::Mat4>()) as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_model_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Global Uniform Buffer"),
            size:               (max_shader_matrices * std::mem::size_of::<glm::Mat4>()) as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let global_bind_group = Arc::new(self.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }));

        window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

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
                self.window_size
                    .store((new_size.width, new_size.height), Ordering::SeqCst);

                config.width = new_size.width;
                config.height = new_size.height;

                surface.configure(&self.device, config);

                self.screen_sized_textures
                    .access()
                    .into_iter()
                    .for_each(|(id, weak_texture)| {
                        match weak_texture.upgrade()
                        {
                            Some(texture) =>
                            {
                                texture.resize_to_screen_size(self);
                            }
                            None => self.screen_sized_textures.delete(id)
                        }
                    });

                self.resize_pingers.0.ping();
            }
        };

        let shader_mvps = RefCell::new(ShaderMatrices::new(max_shader_matrices));
        let shader_models = RefCell::new(ShaderMatrices::new(max_shader_matrices));

        let render_func = || -> Result<(), wgpu::SurfaceError> {
            let screen_texture = surface.get_current_texture()?;
            let screen_texture_view = screen_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let id_counter = AtomicU32::new(0);
            let get_next_id = || {
                let maybe_new_id = id_counter.fetch_add(1, Ordering::Relaxed);

                if maybe_new_id >= max_shader_matrices as u32
                {
                    panic!("Too many draw calls with matrices!");
                }

                maybe_new_id
            };

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Patinac Main Command Encoder")
                });

            // log::trace!("before calling all pre_record_update s");
            let camera = self.camera_window.get();

            let strong_renderable_record_info: Vec<(RecordInfo, Arc<dyn Recordable>)> = self
                .renderables
                .access()
                .into_iter()
                .filter_map(|(id, maybe_recordable)| {
                    match maybe_recordable.upgrade()
                    {
                        Some(strong) =>
                        {
                            Some((
                                strong.clone().pre_record_update(
                                    &mut encoder,
                                    self,
                                    &camera,
                                    &global_bind_group,
                                    &screen_texture
                                ),
                                strong
                            ))
                        }
                        None =>
                        {
                            self.renderables.delete(id);
                            None
                        }
                    }
                })
                .collect();

            // log::trace!("after calling all pre_record_update s");

            #[no_mangle]
            static RECORDABLES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

            RECORDABLES_ALLOCATED.store(strong_renderable_record_info.len(), Ordering::Relaxed);

            let mut renderpass_order_map: HashMap<RenderPassId, usize> = HashMap::new();

            let mut renderpass_data: HashMap<
                RenderPassId,
                (
                    EncoderToPassFn,
                    Vec<(
                        Option<Arc<GenericPipeline>>,
                        [Option<Arc<wgpu::BindGroup>>; 4],
                        Option<DrawId>,
                        Arc<dyn Recordable>
                    )>
                )
            > = renderpass_func_window
                .get()
                .func_array
                .into_iter()
                .enumerate()
                .map(|(idx, f)| {
                    let (id, f) = f();

                    renderpass_order_map.insert(id, idx);

                    (id, (f, Vec::new()))
                })
                .collect();

            let mut number_of_recorded = 0;

            for (info, recordable) in strong_renderable_record_info
            {
                if !matches!(info, RecordInfo::NoRecord)
                {
                    number_of_recorded += 1;
                }

                match info
                {
                    RecordInfo::NoRecord =>
                    {
                        // Do Nothing
                    }
                    RecordInfo::Record {
                        render_pass,
                        pipeline,
                        bind_groups,
                        transform
                    } =>
                    {
                        let id: Option<DrawId> = if let Some(t) = transform
                        {
                            let raw_id = get_next_id();

                            shader_models
                                .borrow_mut()
                                .write_at(raw_id as usize, t.as_model_matrix());
                            shader_mvps
                                .borrow_mut()
                                .write_at(raw_id as usize, camera.get_perspective(self, &t));
                            Some(DrawId(raw_id))
                        }
                        else
                        {
                            None
                        };

                        renderpass_data.get_mut(&render_pass).unwrap().1.push((
                            Some(pipeline),
                            bind_groups,
                            id,
                            recordable
                        ))
                    }
                    RecordInfo::RecordIsolated {
                        render_pass
                    } =>
                    {
                        renderpass_data.get_mut(&render_pass).unwrap().1.push((
                            None,
                            [None, None, None, None],
                            None,
                            recordable
                        ))
                    }
                }
            }

            #[no_mangle]
            static RECORDABLES_ACTIVE: AtomicUsize = AtomicUsize::new(0);

            RECORDABLES_ACTIVE.store(number_of_recorded, Ordering::Relaxed);

            let mut order_of_passes: Vec<(RenderPassId, usize)> = renderpass_order_map
                .into_iter()
                .collect::<Vec<(RenderPassId, usize)>>(
            );

            order_of_passes.sort_by(|l, r| l.1.cmp(&r.1));

            let mut final_renderpass_drawcalls: Vec<(
                EncoderToPassFn,
                Vec<(
                    Option<Arc<GenericPipeline>>,
                    [Option<Arc<wgpu::BindGroup>>; 4],
                    Option<DrawId>,
                    Arc<dyn Recordable>
                )>
            )> = Vec::new();

            fn cmp_bind_groups(
                l: &[Option<Arc<wgpu::BindGroup>>; 4],
                r: &[Option<Arc<wgpu::BindGroup>>; 4]
            ) -> std::cmp::Ordering
            {
                std::cmp::Ordering::Equal
                    .then(
                        l[0].as_ref()
                            .map(|g| g.global_id())
                            .cmp(&r[0].as_ref().map(|g| g.global_id()))
                    )
                    .then(
                        l[1].as_ref()
                            .map(|g| g.global_id())
                            .cmp(&r[1].as_ref().map(|g| g.global_id()))
                    )
                    .then(
                        l[2].as_ref()
                            .map(|g| g.global_id())
                            .cmp(&r[2].as_ref().map(|g| g.global_id()))
                    )
                    .then(
                        l[3].as_ref()
                            .map(|g| g.global_id())
                            .cmp(&r[3].as_ref().map(|g| g.global_id()))
                    )
            }

            for pass_id in order_of_passes
            {
                let (func, data_vec) = renderpass_data.remove(&pass_id.0).unwrap();

                final_renderpass_drawcalls.push((func, data_vec));

                final_renderpass_drawcalls
                    .last_mut()
                    .unwrap()
                    .1
                    .sort_by(|l, r| {
                        std::cmp::Ordering::Equal
                            .then(l.0.cmp(&r.0))
                            .then(cmp_bind_groups(&l.1, &r.1))
                            .then(l.2.cmp(&r.2))
                            .then(
                                (l.3.as_ref() as *const _ as *const ())
                                    .cmp(&(r.3.as_ref() as *const _ as *const ()))
                            )
                    })
            }

            let global_info = ShaderGlobalInfo {
                camera_pos:      camera.get_position(),
                _padding:        0.0,
                view_projection: camera.get_perspective(
                    self,
                    &crate::Transform {
                        ..Default::default()
                    }
                )
            };

            let data: &[u8] = bytes_of(&global_info);

            self.queue
                .write_buffer_with(
                    &global_info_uniform_buffer,
                    0,
                    NonZero::new(data.len() as u64).unwrap()
                )
                .unwrap()
                .copy_from_slice(data);

            let number_of_used_ids = id_counter.load(Ordering::Acquire) as usize;

            {
                let m = shader_mvps.borrow();
                let slice =
                    &m.as_mat_bytes()[0..number_of_used_ids * std::mem::size_of::<glm::Mat4>()];

                if let Some(slice_len) = NonZero::new(slice.len() as u64)
                {
                    self.queue
                        .write_buffer_with(&global_mvp_buffer, 0, slice_len)
                        .unwrap()
                        .copy_from_slice(slice);
                }
            }

            {
                let m = shader_models.borrow();
                let slice =
                    &m.as_mat_bytes()[0..number_of_used_ids * std::mem::size_of::<glm::Mat4>()];

                if let Some(slice_len) = NonZero::new(slice.len() as u64)
                {
                    self.queue
                        .write_buffer_with(&global_model_buffer, 0, slice_len)
                        .unwrap()
                        .copy_from_slice(slice);
                }
            }

            let mut active_pipeline: Option<&GenericPipeline> = None;
            let mut active_bind_groups: [Option<wgpu::Id<wgpu::BindGroup>>; 4] = [None; 4];

            for (pass_func, raw_recordables) in final_renderpass_drawcalls.iter()
            {
                // idk how to write the lifetimes here, it's fine the variables are dropped in
                // the right order
                let recordables: &Vec<(
                    Option<Arc<GenericPipeline>>,
                    [Option<Arc<wgpu::BindGroup>>; 4],
                    Option<DrawId>,
                    Arc<dyn Recordable>
                )> = unsafe { util::modify_lifetime(raw_recordables) };

                pass_func(
                    &mut encoder,
                    &screen_texture_view,
                    &mut |render_pass: &mut GenericPass| {
                        for (maybe_desired_pipeline, desired_bind_groups, draw_id, recordable) in
                            recordables.iter()
                        {
                            if let Some(desired_pipeline) = maybe_desired_pipeline
                            {
                                if active_pipeline != Some(desired_pipeline)
                                {
                                    match (&**desired_pipeline, &mut *render_pass)
                                    {
                                        (
                                            GenericPipeline::Compute(p),
                                            GenericPass::Compute(pass)
                                        ) =>
                                        {
                                            pass.set_pipeline(p);
                                        }
                                        (GenericPipeline::Render(p), GenericPass::Render(pass)) =>
                                        {
                                            pass.set_pipeline(p)
                                        }
                                        (_, _) =>
                                        {
                                            panic!(
                                                "Pass Pipeline Invariant Violated! {:?}",
                                                &**recordable
                                            )
                                        }
                                    }

                                    active_pipeline = Some(desired_pipeline);
                                }
                            }

                            for (idx, (active_bind_group_id, maybe_new_bind_group)) in
                                active_bind_groups
                                    .iter_mut()
                                    .zip(desired_bind_groups)
                                    .enumerate()
                            {
                                if *active_bind_group_id
                                    != maybe_new_bind_group.as_ref().map(|g| g.global_id())
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

                            recordable.record(render_pass, *draw_id);

                            number_of_recorded += 1;
                        }

                        active_pipeline = None;
                        active_bind_groups = [None; 4];
                    }
                );
            }

            window.pre_present_notify();

            self.queue.submit([encoder.finish()]);
            screen_texture.present();

            self.delta_time
                .store(self.input_manager.get_delta_time(), Ordering::Release);

            std::mem::drop(final_renderpass_drawcalls);

            Ok(())
        };

        self.input_manager.attach_cursor();

        let _ = event_loop.run_on_demand(|event, control_flow| {
            if !should_continue()
            {
                control_flow.exit();
            }

            self.input_manager.update_with_event(&event);

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

                            if self.input_manager.is_key_pressed(KeyCode::Escape)
                            {
                                control_flow.exit();
                            }

                            match render_func()
                            {
                                Ok(_) => (),
                                Err(Timeout) => log::warn!("render timeout!"),
                                Err(Outdated) => (),
                                Err(Lost) => resize_func(None),
                                Err(OutOfMemory) => panic!("Not enough memory for next frame!")
                            }

                            window.request_redraw();
                        }
                        _ => ()
                    }
                }
                Event::MemoryWarning => self.render_cache.trim(),
                _ => ()
            }

            if control_flow.exiting()
            {
                request_terminate();
            }
        });

        log::info!("Event loop returned");
    }

    pub fn get_default_depth_state() -> wgpu::DepthStencilState
    {
        wgpu::DepthStencilState {
            format:              Self::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare:       wgpu::CompareFunction::Less,
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default()
        }
    }

    pub fn get_input_manager(&self) -> &InputManager
    {
        &self.input_manager
    }

    pub(crate) fn register_screen_sized_image(&self, img: Arc<super::ScreenSizedTexture>)
    {
        self.screen_sized_textures
            .insert(util::Uuid::new(), Arc::downgrade(&img))
    }

    pub fn display_holdover_recordables_info(&self)
    {
        self.renderables
            .access()
            .into_iter()
            .filter_map(|(_, weak)| weak.upgrade())
            .for_each(|strong| log::warn!("Retained Renderable! {:?}", &*strong));
    }
}

pub type EncoderToPassFn = Box<
    dyn for<'enc> Fn(
            &'enc mut wgpu::CommandEncoder,
            &'enc wgpu::TextureView,
            &'enc mut (dyn FnMut(&'_ mut GenericPass<'_>) + 'enc)
        ) + Send
        + Sync
>;

type FuncArray = Vec<Arc<dyn Fn() -> (RenderPassId, EncoderToPassFn) + Send + Sync>>;

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct RenderPassSendFunction
{
    func_array: FuncArray
}

impl From<FuncArray> for RenderPassSendFunction
{
    fn from(value: FuncArray) -> Self
    {
        Self {
            func_array: value
        }
    }
}

impl RenderPassSendFunction
{
    pub fn new(func_array: FuncArray) -> Self
    {
        RenderPassSendFunction {
            func_array
        }
    }
}

impl Debug for RenderPassSendFunction
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Render Pass Send Function")
    }
}

#[derive(Debug)]
struct CriticalSection
{
    thread_id:              ThreadId,
    renderpass_func_window: util::Window<RenderPassSendFunction>,
    surface:                wgpu::Surface<'static>,
    config:                 wgpu::SurfaceConfiguration,
    window:                 Arc<Window>,
    event_loop:             EventLoop<()>
}

unsafe impl Sync for CriticalSection {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub(crate) struct ShaderGlobalInfo
{
    camera_pos:      glm::Vec3,
    _padding:        f32,
    view_projection: glm::Mat4
}

#[repr(C)]
#[derive(Clone)]
pub(crate) struct ShaderMatrices
{
    matrices: Box<[glm::Mat4]>
}

impl Default for ShaderMatrices
{
    fn default() -> Self
    {
        Self {
            matrices: vec![].into_boxed_slice()
        }
    }
}

impl ShaderMatrices
{
    pub fn new(size: usize) -> ShaderMatrices
    {
        ShaderMatrices {
            matrices: vec![glm::Mat4::default(); size].into_boxed_slice()
        }
    }

    fn as_mat_bytes(&self) -> &[u8]
    {
        bytemuck::cast_slice(&self.matrices)
    }

    fn write_at(&mut self, idx: usize, mat: glm::Mat4)
    {
        self.matrices[idx] = mat;
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::ops::Deref;
use std::sync::atomic::Ordering::{self, *};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use nalgebra_glm as glm;
use pollster::FutureExt;
use strum::IntoEnumIterator;
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::{EventLoop, EventLoopWindowTarget};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};
use winit_input_helper::WinitInputHelper;

use crate::util;
use crate::util::Registrar;

pub const SURFACE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

pub struct Renderer
{
    pub queue:        wgpu::Queue,
    pub render_cache: super::RenderCache,

    // pub camera:       Mutex<super::Camera>,

    // gated publics
    device:      wgpu::Device,
    // TODO: replace with UUIDs
    renderables: util::Registrar<*const (), Weak<dyn super::Renderable>>,

    // Rendering views
    window_size_x:            AtomicU32,
    window_size_y:            AtomicU32,
    float_delta_frame_time_s: AtomicU32,

    // Rendering
    critical_section: Mutex<CriticalSection>
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

impl Renderer
{
    pub fn new() -> Self
    {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

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
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits:   wgpu::Limits {
                        max_push_constant_size: 128,
                        ..Default::default()
                    }
                },
                None // Trace path
            )
            .block_on()
            .unwrap();

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
            size,
            window,
            event_loop,
            camera: RefCell::new(super::Camera::new(glm::Vec3::repeat(0.0), 0.0, 0.0))
        };

        let render_cache = super::RenderCache::new(&device);

        Renderer {
            renderables: Registrar::new(),
            queue,
            device,
            critical_section: Mutex::new(critical_section),
            render_cache,
            window_size_x: AtomicU32::new(size.width),
            window_size_y: AtomicU32::new(size.height),
            // camera: Mutex::new(super::Camera::new(glm::Vec3::repeat(0.0), 0.0, 0.0)),
            float_delta_frame_time_s: AtomicU32::new(0.0f32.to_bits())
        }
    }

    pub fn register(&self, renderable: Weak<dyn super::Renderable>)
    {
        self.renderables
            .insert(renderable.as_ptr() as *const (), renderable);
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
            size,
            window,
            event_loop,
            camera
        } = &mut *guard;

        assert!(
            *thread_id == std::thread::current().id(),
            "Renderer::enter_gfx_loop() must be called on the same thread that Renderer::new() \
             was called from!"
        );

        let input_helper = RefCell::new(WinitInputHelper::new());

        // Because of a bug in winit, the first resize command that comes in is borked
        // https://github.com/rust-windowing/winit/issues/2094
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

            let new_size = maybe_new_size.unwrap_or(*size);

            if new_size.width > 0 && new_size.height > 0
            {
                // this seems excessive...
                *size = new_size;
                config.width = new_size.width;
                config.height = new_size.height;
                self.set_framebuffer_size(glm::UVec2::new(new_size.width, new_size.height));
                surface.configure(&self.device, config);
            }
        };

        let mut previous_frame_time = std::time::Instant::now();

        let mut render_func = || -> Result<(), wgpu::SurfaceError> {
            let now = std::time::Instant::now();

            self.float_delta_frame_time_s.store(
                (now - previous_frame_time).as_secs_f32().to_bits(),
                Ordering::Release
            );

            previous_frame_time = now;

            let output = surface.get_current_texture()?;
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut renderables_map: HashMap<super::PassStage, Vec<Arc<dyn super::Renderable>>> =
                super::PassStage::iter().map(|s| (s, Vec::new())).collect();

            self.renderables
                .access()
                .into_iter()
                .filter_map(|(ptr, weak_renderable)| {
                    match weak_renderable.upgrade()
                    {
                        Some(s) => Some(s),
                        None =>
                        {
                            self.renderables.delete(ptr);
                            None
                        }
                    }
                })
                .for_each(|r| {
                    renderables_map
                        .get_mut(&r.get_pass_stage())
                        .unwrap()
                        .push(r);
                });

            renderables_map
                .iter_mut()
                .for_each(|(_, renderable_vec)| renderable_vec.sort_by(|l, r| l.ord(&**r)));

            // let camera = self.camera.lock().unwrap().clone();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                });

            let mut active_pipeline: Option<wgpu::Id<wgpu::RenderPipeline>> = None;
            let mut active_bind_groups: [Option<wgpu::Id<wgpu::BindGroup>>; 4] = [None; 4];

            for pass_type in super::PassStage::iter()
            {
                let mut render_pass = match pass_type
                {
                    crate::gfx::PassStage::GraphicsSimpleColor =>
                    {
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label:                    Some("Render Pass"),
                            color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
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
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set:      None,
                            timestamp_writes:         None
                        })
                    }
                };

                for renderable in renderables_map.get(&pass_type).unwrap()
                {
                    let renderable_pipeline = self
                        .render_cache
                        .lookup_render_pipeline(renderable.get_pipeline_type());

                    if active_pipeline != Some(renderable_pipeline.global_id())
                    {
                        render_pass.set_pipeline(renderable_pipeline);

                        active_pipeline = Some(renderable_pipeline.global_id());
                    }

                    for (idx, (active_bind_group_id, maybe_new_bind_group)) in active_bind_groups
                        .iter_mut()
                        .zip(renderable.get_bind_groups())
                        .enumerate()
                    {
                        // if the bind group is different
                        if *active_bind_group_id != maybe_new_bind_group.map(|g| g.global_id())
                        {
                            // if the different group actually exists, sometimes there may be a
                            // bound one but we want None bound, so we can just leave it there
                            if let Some(new_bind_group) = maybe_new_bind_group
                            {
                                render_pass.set_bind_group(idx as u32, new_bind_group, &[]);
                                *active_bind_group_id = Some(new_bind_group.global_id());
                            }
                        }
                    }

                    renderable.bind_and_draw(&mut render_pass, self, &camera.borrow());
                }

                active_pipeline = None;
                active_bind_groups = [None; 4];
            }

            window.pre_present_notify();

            self.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            Ok(())
        };

        let handle_input = |control_flow: &EventLoopWindowTarget<()>| {
            // TODO: do the trig thing so that diagonal isn't faster!
            let move_scale = 10.0;
            let rotate_scale = 1000.0;

            let input_helper = input_helper.borrow();

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

            if input_helper.key_held(KeyCode::KeyE)
            {
                let v = camera.borrow().get_up_vector() * move_scale;

                camera.borrow_mut().add_position(v * self.get_delta_time());
            };

            if input_helper.key_held(KeyCode::KeyQ)
            {
                let v = camera.borrow().get_up_vector() * -move_scale;

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
}

struct CriticalSection
{
    thread_id:  ThreadId,
    surface:    wgpu::Surface<'static>,
    config:     wgpu::SurfaceConfiguration,
    size:       PhysicalSize<u32>,
    window:     Window,
    event_loop: EventLoop<()>,

    camera: RefCell<super::Camera>
}

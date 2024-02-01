use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use pollster::FutureExt;
use strum::IntoEnumIterator;
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};

use crate::util;
use crate::util::Registrar;

pub const SURFACE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

pub struct Renderer
{
    pub queue:        wgpu::Queue,
    pub render_cache: super::RenderCache,
    device:           wgpu::Device,
    renderables:      util::Registrar<*const (), Weak<dyn super::Renderable>>,
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
                    required_limits:   wgpu::Limits::default()
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
            event_loop
        };

        let render_cache = super::RenderCache::new(&device);

        Renderer {
            renderables: Registrar::new(),
            queue,
            device,
            critical_section: Mutex::new(critical_section),
            render_cache
        }
    }

    pub fn register(&self, renderable: Weak<dyn super::Renderable>)
    {
        self.renderables
            .insert(renderable.as_ptr() as *const (), renderable);
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
            event_loop
        } = &mut *guard;

        assert!(
            *thread_id == std::thread::current().id(),
            "Renderer::enter_gfx_loop() must be called on the same thread that Renderer::new() \
             was called from!"
        );

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
                *size = new_size;
                config.width = new_size.width;
                config.height = new_size.height;
                surface.configure(&self.device, config);
            }
        };

        let render_func = || -> Result<(), wgpu::SurfaceError> {
            let output = surface.get_current_texture()?;
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut renderables_map: HashMap<super::PassStage, Vec<Arc<dyn super::Renderable>>> =
                super::PassStage::iter().map(|s| (s, Vec::new())).collect();

            self.renderables
                .access()
                .into_iter()
                // Upgrade if possible, otherwise cull
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
                // put into the renderpass map
                .for_each(|r| {
                    renderables_map.get_mut(&r.get_pass_stage()).unwrap().push(r);
                });

            renderables_map
                .iter_mut()
                .for_each(|(_, renderable_vec)| renderable_vec.sort_by(|l, r| l.ord(&**r)));

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

                    if active_pipeline != Some(renderable_pipeline).map(|p| p.global_id())
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

                    renderable.bind_and_draw(&mut render_pass);
                }

                active_pipeline = None;
                active_bind_groups = [None; 4];
            }

            window.pre_present_notify();

            self.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            Ok(())
        };

        let _ = event_loop.run_on_demand(|event, control_flow| {
            if should_stop.load(Acquire)
            {
                control_flow.exit();
            }

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
                        WindowEvent::KeyboardInput {
                            device_id: _,
                            ref event,
                            is_synthetic: _
                        } =>
                        {
                            if let Key::Named(NamedKey::Escape) = event.logical_key
                            {
                                control_flow.exit()
                            }
                        }
                        WindowEvent::CloseRequested => control_flow.exit(),
                        WindowEvent::RedrawRequested =>
                        {
                            use wgpu::SurfaceError::*;

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
    event_loop: EventLoop<()>
}

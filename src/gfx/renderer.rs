use std::ops::Deref;
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, AtomicPtr};
use std::sync::{Mutex, RwLock, RwLockReadGuard};

use pollster::FutureExt;
use scopeguard::defer;
use wgpu::core::device;
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};

pub struct Renderer
{
    queue:      wgpu::Queue,
    device:     wgpu::Device,
    adapter:    wgpu::Adapter,
    surface:    wgpu::Surface<'static>,
    size:       Mutex<PhysicalSize<u32>>,
    window:     Window,
    event_loop: Mutex<EventLoop<()>>
}

unsafe impl Send for Renderer {}
unsafe impl Sync for Renderer {}

impl Renderer
{
    pub fn new() -> Self
    {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            #[cfg(debug_assertions)]
            flags: wgpu::InstanceFlags::debugging(),
            #[cfg(not(debug_assertions))]
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

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

        Renderer {
            queue,
            device,
            adapter,
            surface,
            size: Mutex::new(size),
            window,
            event_loop: Mutex::new(event_loop)
        }
    }

    pub fn get_device(&self) -> &wgpu::Device
    {
        &self.device
    }

    pub fn enter_gfx_loop(&self, should_stop: &AtomicBool)
    {
        let mut size_guard = self.size.lock().unwrap();
        let size: &mut PhysicalSize<u32> = &mut *size_guard;

        let surface_caps = self.surface.get_capabilities(&self.adapter);

        let surface_format = surface_caps
            .formats
            .into_iter()
            .find(|f| f.is_srgb())
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

        let mut config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:                        surface_format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  selected_mode.unwrap(),
            alpha_mode:                    surface_caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2
        };
        self.surface.configure(&self.device, &config);

        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/foo.wgsl"));
        let render_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label:                Some("Render Pipeline Layout"),
                    bind_group_layouts:   &[],
                    push_constant_ranges: &[]
                });

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:         Some("Render Pipeline"),
                layout:        Some(&render_pipeline_layout),
                vertex:        wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &[]
                },
                fragment:      Some(wgpu::FragmentState {
                    // 3.
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[Some(wgpu::ColorTargetState {
                        // 4.
                        format:     config.format,
                        blend:      Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL
                    })]
                }),
                primitive:     wgpu::PrimitiveState {
                    topology:           wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face:         wgpu::FrontFace::Ccw,
                    cull_mode:          Some(wgpu::Face::Back),
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
            });

        // Because of a bug in winit, the first resize command that comes in is borked
        // https://github.com/rust-windowing/winit/issues/2094
        #[cfg(target_os = "windows")]
        let mut is_first_resize = true;
        #[cfg(not(target_os = "windows"))]
        let mut is_first_resize = false;

        let mut resize_func = |maybe_new_size: Option<PhysicalSize<u32>>| {
            #[cfg(target_os = "windows")]
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
                self.surface.configure(&self.device, &config);
            }
        };

        let render_func = || -> Result<(), wgpu::SurfaceError> {
            let output = self.surface.get_current_texture()?;
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder")
                });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                });

                render_pass.set_pipeline(&render_pipeline);
                render_pass.draw(0..3, 0..1)
            }

            // submit will accept anything that implements IntoIter
            self.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            Ok(())
        };

        let _ = self
            .event_loop
            .lock()
            .unwrap()
            .run_on_demand(move |event, control_flow| {
                if should_stop.load(Acquire)
                {
                    control_flow.exit();
                }

                match event
                {
                    Event::WindowEvent {
                        window_id,
                        ref event
                    } if window_id == self.window.id() =>
                    {
                        match event
                        {
                            WindowEvent::Resized(new_size) => resize_func(Some(*new_size)),
                            WindowEvent::KeyboardInput {
                                device_id,
                                ref event,
                                is_synthetic
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
                            }
                            _ => ()
                        }
                    }
                    _ => ()
                }

                if (control_flow.exiting())
                {
                    should_stop.store(true, Release);
                }
            });

        log::info!("event loop returned");
    }
}

// pub struct DeviceAccessGuard<'a>
// {
//     stored_guard: RwLockReadGuard<'a, AtomicPtr<wgpu::Device>>
// }

// use std::sync::atomic::Ordering::*;

// impl<'a> DeviceAccessGuard<'a>
// {
//     pub fn new(adopted_lock: &'a RwLock<AtomicPtr<wgpu::Device>>) ->
// DeviceAccessGuard<'a>     {
//         let stored_guard = adopted_lock.read().unwrap();

//         assert!(!stored_guard.load(Acquire).is_null());

//         DeviceAccessGuard {
//             stored_guard
//         }
//     }
// }

// impl<'a> Deref for DeviceAccessGuard<'a>
// {
//     type Target = wgpu::Device;

//     fn deref(&self) -> &Self::Target
//     {
//         let loaded_ptr = self.stored_guard.load(Acquire);

//         debug_assert!(!loaded_ptr.is_null());

//         unsafe { &*(loaded_ptr as *const wgpu::Device) }
//     }
// }

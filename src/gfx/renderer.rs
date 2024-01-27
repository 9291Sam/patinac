use std::ops::Deref;
use std::sync::atomic::AtomicBool;

use winit::dpi::PhysicalSize;

pub struct Renderer {}

impl Renderer
{
    pub fn new() -> Self
    {
        Renderer {}
    }

    pub fn enter_gfx_loop(&self, should_stop: &AtomicBool)
    {
        use pollster::FutureExt;
        /// As much as I hate this design, we're forced into making a massive
        /// ol' state machine because of winit
        use winit::event::*;
        use winit::event_loop::{ControlFlow, EventLoop};
        use winit::keyboard::{Key, NamedKey};
        use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
        use winit::window::WindowBuilder;

        let mut event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

        let mut size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            #[cfg(debug_assertions)]
            flags: wgpu::InstanceFlags::VALIDATION,
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

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a
        // different one will result in all the colors coming out darker. If you
        // want to support non sRGB surfaces, you'll need to account for that
        // when drawing to the frame.
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
        surface.configure(&device, &config);

        let _ = event_loop.run_on_demand(move |event, control_flow| {
            match event
            {
                Event::WindowEvent {
                    window_id,
                    ref event
                } if window_id == window.id() =>
                {
                    match event
                    {
                        WindowEvent::Resized(new_size) =>
                        {
                            if new_size.width > 0 && new_size.height > 0
                            {
                                size = *new_size;
                                config.width = new_size.width;
                                config.height = new_size.height;
                                surface.configure(&device, &config);
                            }
                        }
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
                        WindowEvent::RedrawRequested => self.draw(),
                        _ => ()
                    }
                }
                _ => ()
            }
        });

        log::info!("event loop returned");
    }

    fn draw(&self) {}
}

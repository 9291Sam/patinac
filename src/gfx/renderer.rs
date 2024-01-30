use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::Mutex;
use std::thread::ThreadId;

use image::GenericImageView;
use pollster::FutureExt;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};

pub struct Renderer
{
    pub queue:        wgpu::Queue,
    device:           wgpu::Device,
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

        log::info!(
            "Selected Device {} with driver {} using backend {:?}",
            adapter.get_info().name,
            adapter.get_info().driver_info,
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

        Renderer {
            queue,
            device,
            critical_section: Mutex::new(critical_section)
        }
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

        const VERTICES: &[Vertex] = &[
            Vertex {
                position:   [-0.0868241, 0.49240386, 0.0],
                tex_coords: [0.4131759, 0.99240386]
            }, // A
            Vertex {
                position:   [-0.49513406, 0.06958647, 0.0],
                tex_coords: [0.0048659444, 0.56958647]
            }, // B
            Vertex {
                position:   [-0.21918549, -0.44939706, 0.0],
                tex_coords: [0.28081453, 0.05060294]
            }, // C
            Vertex {
                position:   [0.35966998, -0.3473291, 0.0],
                tex_coords: [0.85967, 0.1526709]
            }, // D
            Vertex {
                position:   [0.44147372, 0.2347359, 0.0],
                tex_coords: [0.9414737, 0.7347359]
            } // E
        ];

        const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4, /* padding */ 0];

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage:    wgpu::BufferUsages::INDEX
            });

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();
        let dimensions = diffuse_image.dimensions();

        let tree_texture_size = wgpu::Extent3d {
            width:                 dimensions.0,
            height:                dimensions.1,
            depth_or_array_layers: 1
        };

        let tree_texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label:           Some("tree texture"),
                size:            tree_texture_size,
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8UnormSrgb,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats:    &[]
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &diffuse_rgba
        );

        let tree_texture_view = tree_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let tree_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                });

        let diffuse_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout:  &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&tree_texture_view)
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&tree_sampler)
                }
            ],
            label:   Some("diffuse_bind_group")
        });

        let shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/foo.wgsl"));
        let render_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label:                Some("Render Pipeline Layout"),
                    bind_group_layouts:   &[&texture_bind_group_layout],
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
                    buffers:     &[Vertex::desc()]
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
                    topology:           wgpu::PrimitiveTopology::TriangleStrip,
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
                render_pass.set_bind_group(0, &diffuse_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
            }

            // submit will accept anything that implements IntoIter
            self.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            Ok(())
        };

        let _ = event_loop.run_on_demand(move |event, control_flow| {
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex
{
    position:   [f32; 3],
    tex_coords: [f32; 2]
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

struct CriticalSection
{
    thread_id:  ThreadId,
    surface:    wgpu::Surface<'static>,
    config:     wgpu::SurfaceConfiguration,
    size:       PhysicalSize<u32>,
    window:     Window,
    event_loop: EventLoop<()>
}

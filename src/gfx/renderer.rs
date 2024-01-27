use std::sync::atomic::AtomicBool;

use winit::dpi::PhysicalSize;

pub struct Renderer {}

impl Renderer
{
    pub fn new() -> Self
    {
        Renderer {}
    }

    pub fn enter_tick_loop(&self, should_stop: &AtomicBool)
    {
        use winit::event::*;
        use winit::event_loop::{ControlFlow, EventLoop};
        use winit::keyboard::{Key, NamedKey};
        use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
        use winit::window::WindowBuilder;

        let mut event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

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
                        // WindowEvent::ActivationTokenDone {
                        //     serial,
                        //     token
                        // } => todo!(),
                        WindowEvent::Resized(new_size) => self.resize(*new_size),
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
                _ => () /* Event::DeviceEvent {
                         *     device_id,
                         *     event
                         * } => todo!(),
                         * Event::UserEvent(_) => todo!(),
                         * Event::Suspended => (),
                         * Event::Resumed => (),
                         * Event::AboutToWait => (),
                         * Event::LoopExiting => log::info!("Shutdown Requested!"),
                         * Event::MemoryWarning => log::warn!("TODO: Fire GC!") */
            }
        });

        log::info!("event loop returned");
    }

    fn resize(&self, new_size: PhysicalSize<u32>) {}

    fn draw(&self) {}
}

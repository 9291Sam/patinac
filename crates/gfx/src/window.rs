use std::sync::Mutex;
use std::thread::ThreadId;

use winit::event_loop::EventLoop;
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::windows::EventLoopBuilderExtWindows;

pub struct Window
{
    event_loop:      Mutex<EventLoop<()>>,
    window_internal: Mutex<WindowInternal>
}

impl Window
{
    pub fn new() -> Self
    {
        let event_loop = EventLoop::new().unwrap();
        Self {
            event_loop:      Mutex::new(event_loop),
            window_internal: Mutex::new(WindowInternal {})
        }
    }

    pub fn handle_events_on_this_thread(&self)
    {
        self.event_loop
            .lock()
            .unwrap()
            .run_app_on_demand(&mut *self.window_internal.lock().unwrap());
    }
}

struct WindowInternal {}

impl winit::application::ApplicationHandler for WindowInternal
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop)
    {
        todo!()
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent
    )
    {
        match event
        {
            winit::event::WindowEvent::ActivationTokenDone {
                serial,
                token
            } => todo!(),
            winit::event::WindowEvent::Resized(_) => todo!(),
            winit::event::WindowEvent::Moved(_) => todo!(),
            winit::event::WindowEvent::CloseRequested => todo!(),
            winit::event::WindowEvent::Destroyed => todo!(),
            winit::event::WindowEvent::DroppedFile(_) => todo!(),
            winit::event::WindowEvent::HoveredFile(_) => todo!(),
            winit::event::WindowEvent::HoveredFileCancelled => todo!(),
            winit::event::WindowEvent::Focused(_) => todo!(),
            winit::event::WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic
            } => todo!(),
            winit::event::WindowEvent::ModifiersChanged(_) => todo!(),
            winit::event::WindowEvent::Ime(_) => todo!(),
            winit::event::WindowEvent::CursorMoved {
                device_id,
                position
            } => todo!(),
            winit::event::WindowEvent::CursorEntered {
                device_id
            } => todo!(),
            winit::event::WindowEvent::CursorLeft {
                device_id
            } => todo!(),
            winit::event::WindowEvent::MouseWheel {
                device_id,
                delta,
                phase
            } => todo!(),
            winit::event::WindowEvent::MouseInput {
                device_id,
                state,
                button
            } => todo!(),
            winit::event::WindowEvent::PinchGesture {
                device_id,
                delta,
                phase
            } => todo!(),
            winit::event::WindowEvent::PanGesture {
                device_id,
                delta,
                phase
            } => todo!(),
            winit::event::WindowEvent::DoubleTapGesture {
                device_id
            } => todo!(),
            winit::event::WindowEvent::RotationGesture {
                device_id,
                delta,
                phase
            } => todo!(),
            winit::event::WindowEvent::TouchpadPressure {
                device_id,
                pressure,
                stage
            } => todo!(),
            winit::event::WindowEvent::AxisMotion {
                device_id,
                axis,
                value
            } => todo!(),
            winit::event::WindowEvent::Touch(_) => todo!(),
            winit::event::WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer
            } => todo!(),
            winit::event::WindowEvent::ThemeChanged(_) => todo!(),
            winit::event::WindowEvent::Occluded(_) => todo!(),
            winit::event::WindowEvent::RedrawRequested => todo!()
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent
    )
    {
        match event
        {
            winit::event::DeviceEvent::Added => todo!(),
            winit::event::DeviceEvent::Removed => todo!(),
            winit::event::DeviceEvent::MouseMotion {
                delta
            } => todo!(),
            winit::event::DeviceEvent::MouseWheel {
                delta
            } => todo!(),
            winit::event::DeviceEvent::Motion {
                axis,
                value
            } => todo!(),
            winit::event::DeviceEvent::Button {
                button,
                state
            } => todo!(),
            winit::event::DeviceEvent::Key(_) => todo!()
        }
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause
    )
    {
        let _ = (event_loop, cause);
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: ())
    {
        let _ = (event_loop, event);
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop)
    {
        let _ = event_loop;
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop)
    {
        let _ = event_loop;
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop)
    {
        let _ = event_loop;
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop)
    {
        let _ = event_loop;
    }
}

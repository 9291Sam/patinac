use std::collections::HashMap;

use winit::keyboard::KeyCode;

struct InputManager
{
    is_key_pressed: HashMap<KeyCode, bool>
}

impl InputManager
{
    pub fn new() -> InputManager
    {
        InputManager {
            is_key_pressed: HashMap::new()
        }
    }

    pub fn update_with_event(event: &winit::event::Event<()>)
    {
        match event {
            winit::event::Event::NewEvents(_) => todo!(),
            winit::event::Event::WindowEvent { window_id, event } => todo!(),
            winit::event::Event::DeviceEvent { device_id, event } => todo!(),
            winit::event::Event::UserEvent(_) => todo!(),
            winit::event::Event::Suspended => todo!(),
            winit::event::Event::Resumed => todo!(),
            winit::event::Event::AboutToWait => todo!(),
            winit::event::Event::LoopExiting => todo!(),
            winit::event::Event::MemoryWarning => todo!(),
        }
    }

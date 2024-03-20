use std::collections::hash_map::Entry::*;
use std::collections::HashMap;

use winit::event::{DeviceId, ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowId;

struct InputManager
{
    is_key_pressed: HashMap<PhysicalKey, ElementState>
}

impl InputManager
{
    pub fn new() -> InputManager
    {
        InputManager {
            is_key_pressed: HashMap::new()
        }
    }

    pub fn update_with_event(
        &mut self,
        this_device_id: &DeviceId,
        this_window_id: &WindowId,
        in_event: &Event<()>
    )
    {
        if let winit::event::Event::WindowEvent {
            window_id,
            event
        } = in_event
        {
            if window_id == this_window_id
            {
                if let WindowEvent::KeyboardInput {
                    device_id,
                    event,
                    is_synthetic
                } = event
                {
                    if !is_synthetic && this_device_id == device_id
                    {
                        let KeyEvent {
                            physical_key,
                            state,
                            ..
                        } = event;

                        self.is_key_pressed.insert(*physical_key, *state);
                    }
                }
            }
        }
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool
    {
        match self.is_key_pressed.get(&PhysicalKey::Code(key))
        {
            Some(v) => *v == ElementState::Pressed,
            None => false
        }
    }
}

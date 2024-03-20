use std::collections::HashMap;

use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

pub(crate) struct InputManager
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

    pub fn update_with_event(&mut self, in_event: &Event<()>)
    {
        if let winit::event::Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state,
                            ..
                        },
                    is_synthetic,
                    ..
                },
            ..
        } = in_event
        {
            if !is_synthetic
            {
                self.is_key_pressed.insert(*physical_key, *state);
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

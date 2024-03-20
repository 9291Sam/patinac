use std::collections::HashMap;

use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

const ZERO_POS: PhysicalPosition<f64> = PhysicalPosition {
    x: 0.0, y: 0.0
};

pub(crate) struct InputManager<'w>
{
    window:                   &'w Window,
    is_key_pressed:           HashMap<PhysicalKey, ElementState>,
    previous_frame_time:      std::time::Instant,
    delta_frame_time:         f32,
    previous_frame_mouse_pos: PhysicalPosition<f64>,
    best_guess_mouse_pos:     PhysicalPosition<f64>,
    delta_mouse_pos:          PhysicalPosition<f64>,
    window_size:              PhysicalSize<u32>,
    is_cursor_attached:       bool
}

// TODO: display scaling

impl InputManager<'_>
{
    pub fn new(window: &Window, size: PhysicalSize<u32>) -> InputManager
    {
        let mut this = InputManager {
            window,
            is_key_pressed: HashMap::new(),
            previous_frame_time: std::time::Instant::now(),
            delta_frame_time: 0.0,
            previous_frame_mouse_pos: ZERO_POS,
            best_guess_mouse_pos: ZERO_POS,
            delta_mouse_pos: ZERO_POS,
            window_size: size,
            is_cursor_attached: true
        };

        this.attach_cursor();

        this
    }

    pub fn update_with_event(&mut self, event: &Event<()>)
    {
        if let winit::event::Event::WindowEvent {
            event: window_event,
            ..
        } = event
        {
            match window_event
            {
                WindowEvent::RedrawRequested =>
                {
                    let now = std::time::Instant::now();

                    self.delta_frame_time = (now - self.previous_frame_time).as_secs_f32();

                    self.previous_frame_time = now;

                    if self.is_cursor_attached
                    {
                        self.delta_mouse_pos = PhysicalPosition {
                            x: (self.best_guess_mouse_pos.x - self.previous_frame_mouse_pos.x),
                            y: (self.best_guess_mouse_pos.y - self.previous_frame_mouse_pos.y)
                        };

                        let c = get_center_screen_pos(self.window_size);

                        self.previous_frame_mouse_pos = c;

                        self.window.set_cursor_position(c).unwrap();

                        self.best_guess_mouse_pos = c;
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state,
                            ..
                        },
                    is_synthetic,
                    ..
                } if !is_synthetic =>
                {
                    self.is_key_pressed.insert(*physical_key, *state);
                }
                WindowEvent::CursorMoved {
                    position, ..
                } =>
                {
                    self.best_guess_mouse_pos = *position;
                }
                // WindowEvent::CursorEntered {
                //     device_id
                // } => todo!(),
                // WindowEvent::CursorLeft {
                //     device_id
                // } => todo!(),
                // WindowEvent::MouseWheel {
                //     device_id,
                //     delta,
                //     phase
                // } => todo!(),
                // WindowEvent::MouseInput {
                //     device_id,
                //     state,
                //     button
                // } => todo!(),
                _ => ()
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

    pub fn get_mouse_delta(&self) -> (f32, f32)
    {
        (self.delta_mouse_pos.x as f32, self.delta_mouse_pos.y as f32)
    }

    pub fn get_delta_time(&self) -> f32
    {
        self.delta_frame_time
    }

    pub fn attach_cursor(&mut self)
    {
        self.is_cursor_attached = true;

        self.previous_frame_mouse_pos = ZERO_POS;
        self.best_guess_mouse_pos = ZERO_POS;

        self.window.set_cursor_position(ZERO_POS).unwrap();
    }

    pub fn detach_cursor(&mut self)
    {
        self.is_cursor_attached = false;

        self.previous_frame_mouse_pos = ZERO_POS;
        self.best_guess_mouse_pos = ZERO_POS;
        self.delta_mouse_pos = ZERO_POS;

        self.window.set_cursor_position(ZERO_POS).unwrap();
    }
}

fn get_center_screen_pos(window_size: PhysicalSize<u32>) -> PhysicalPosition<f64>
{
    PhysicalPosition {
        x: window_size.width as f64 / 2.0,
        y: window_size.height as f64 / 2.0
    }
}

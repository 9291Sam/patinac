use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use dashmap::DashMap;
use util::{AtomicF32, AtomicF32F32};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

const ZERO_POS: PhysicalPosition<f64> = PhysicalPosition {
    x: 0.0, y: 0.0
};

pub(crate) struct InputManager<'w>
{
    window:           &'w Window,
    is_key_pressed:   DashMap<PhysicalKey, ElementState>,
    critical_section: Mutex<InputManagerCriticalSection>,

    delta_frame_time:   AtomicF32,
    delta_mouse_pos_px: AtomicF32F32
}

struct InputManagerCriticalSection
{
    previous_frame_time:      std::time::Instant,
    previous_frame_mouse_pos: PhysicalPosition<f64>,
    best_guess_mouse_pos:     PhysicalPosition<f64>,
    window_size:              PhysicalSize<u32>,
    is_cursor_attached:       bool
}

// TODO: display scaling

impl InputManager<'_>
{
    pub fn new(window: &Window, size: PhysicalSize<u32>) -> InputManager
    {
        let this = InputManager {
            window,
            is_key_pressed: DashMap::new(),
            critical_section: Mutex::new(InputManagerCriticalSection {
                previous_frame_time:      std::time::Instant::now(),
                previous_frame_mouse_pos: ZERO_POS,
                best_guess_mouse_pos:     ZERO_POS,
                window_size:              size,
                is_cursor_attached:       true
            }),
            delta_frame_time: AtomicF32::new(0.0),
            delta_mouse_pos_px: AtomicF32F32::new((0.0, 0.0))
        };

        this.attach_cursor();

        this
    }

    pub(crate) fn update_with_event(&self, event: &Event<()>)
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
                    let InputManagerCriticalSection {
                        previous_frame_time,
                        previous_frame_mouse_pos,
                        best_guess_mouse_pos,
                        window_size,
                        is_cursor_attached
                    } = &mut *self.critical_section.lock().unwrap();
                    let now = std::time::Instant::now();

                    self.delta_frame_time.store(
                        (now - *previous_frame_time).as_secs_f32(),
                        Ordering::Release
                    );

                    *previous_frame_time = now;

                    if *is_cursor_attached
                    {
                        self.delta_mouse_pos_px.store(
                            (
                                (best_guess_mouse_pos.x - previous_frame_mouse_pos.x) as f32,
                                (best_guess_mouse_pos.y - previous_frame_mouse_pos.y) as f32
                            ),
                            Ordering::Release
                        );

                        let c = get_center_screen_pos(*window_size);

                        *previous_frame_mouse_pos = c;

                        self.window.set_cursor_position(c).unwrap();

                        *best_guess_mouse_pos = c;
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
                    self.critical_section.lock().unwrap().best_guess_mouse_pos = *position;
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
        self.delta_mouse_pos_px.load(Ordering::Acquire)
    }

    pub(crate) fn get_delta_time(&self) -> f32
    {
        self.delta_frame_time.load(Ordering::Acquire)
    }

    pub fn attach_cursor(&self)
    {
        let InputManagerCriticalSection {
            previous_frame_mouse_pos,
            best_guess_mouse_pos,
            is_cursor_attached,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        *is_cursor_attached = true;

        *previous_frame_mouse_pos = ZERO_POS;
        *best_guess_mouse_pos = ZERO_POS;

        self.window.set_cursor_position(ZERO_POS).unwrap();
        self.window.set_cursor_visible(false);
    }

    pub fn detach_cursor(&self)
    {
        let InputManagerCriticalSection {
            previous_frame_mouse_pos,
            best_guess_mouse_pos,
            is_cursor_attached,
            window_size,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        *is_cursor_attached = false;

        *previous_frame_mouse_pos = ZERO_POS;
        *best_guess_mouse_pos = ZERO_POS;
        self.delta_mouse_pos_px.store((0.0, 0.0), Ordering::Release);

        self.window
            .set_cursor_position(get_center_screen_pos(*window_size))
            .unwrap();
        self.window.set_cursor_visible(true);
    }
}

fn get_center_screen_pos(window_size: PhysicalSize<u32>) -> PhysicalPosition<f64>
{
    PhysicalPosition {
        x: window_size.width as f64 / 2.0,
        y: window_size.height as f64 / 2.0
    }
}

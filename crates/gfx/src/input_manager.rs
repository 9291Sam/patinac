use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use util::{AtomicF32, AtomicF32F32};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

const ZERO_POS: PhysicalPosition<f64> = PhysicalPosition {
    x: 0.0, y: 0.0
};

#[derive(Debug)]
pub struct InputManager
{
    window:           Arc<Window>,
    is_key_pressed:   DashMap<PhysicalKey, ElementState>,
    critical_section: Mutex<InputManagerCriticalSection>,

    delta_frame_time:   AtomicF32,
    delta_mouse_pos_px: AtomicF32F32,
    ignore_frames:      AtomicU64
}

#[derive(Debug)]
struct InputManagerCriticalSection
{
    previous_frame_time:      std::time::Instant,
    previous_frame_mouse_pos: PhysicalPosition<f64>,
    best_guess_mouse_pos:     PhysicalPosition<f64>,
    window_size:              PhysicalSize<u32>,
    is_cursor_attached:       bool
}

// TODO: display scaling

impl InputManager
{
    pub fn new(window: Arc<Window>, size: PhysicalSize<u32>) -> InputManager
    {
        let this = InputManager {
            window,
            is_key_pressed: DashMap::new(),
            critical_section: Mutex::new(InputManagerCriticalSection {
                previous_frame_time:      std::time::Instant::now(),
                previous_frame_mouse_pos: ZERO_POS,
                best_guess_mouse_pos:     ZERO_POS,
                window_size:              size,
                is_cursor_attached:       false
            }),
            delta_frame_time: AtomicF32::new(0.0),
            delta_mouse_pos_px: AtomicF32F32::new((0.0, 0.0)),
            ignore_frames: AtomicU64::new(0)
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
                    let ignore: bool = if self.ignore_frames.load(Ordering::Acquire) != 0
                    {
                        self.ignore_frames.fetch_sub(1, Ordering::AcqRel);

                        true
                    }
                    else
                    {
                        false
                    };

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
                        let value_to_store = if ignore
                        {
                            (0.0, 0.0)
                        }
                        else
                        {
                            (
                                (best_guess_mouse_pos.x - previous_frame_mouse_pos.x) as f32,
                                (best_guess_mouse_pos.y - previous_frame_mouse_pos.y) as f32
                            )
                        };

                        let c = get_center_screen_pos(*window_size);

                        *previous_frame_mouse_pos = c;

                        match self.window.set_cursor_position(c)
                        {
                            Ok(_) =>
                            {
                                self.delta_mouse_pos_px
                                    .store(value_to_store, Ordering::Release)
                            }
                            Err(_) =>
                            {
                                log::trace!(
                                    "Failed to set cursor position to center of screen, another \
                                     did the sticky keys prompt appear?"
                                )
                            }
                        }

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
                WindowEvent::MouseInput {
                    state,
                    button,
                    ..
                } if *button == MouseButton::Left =>
                {
                    match state
                    {
                        ElementState::Pressed => self.attach_cursor(),
                        ElementState::Released => ()
                    }
                }
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
            ref mut previous_frame_mouse_pos,
            ref mut best_guess_mouse_pos,
            ref mut is_cursor_attached,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        if *is_cursor_attached
        {
            return;
        }

        *is_cursor_attached = true;

        *previous_frame_mouse_pos = ZERO_POS;
        *best_guess_mouse_pos = ZERO_POS;
        self.delta_mouse_pos_px.store((0.0, 0.0), Ordering::Release);
        self.ignore_frames.fetch_add(2, Ordering::AcqRel);

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

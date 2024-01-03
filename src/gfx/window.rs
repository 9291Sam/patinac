use std::collections::HashMap;
use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, UNIX_EPOCH};

use ash::vk;
use crossbeam::atomic::AtomicCell;
use glfw::ClientApiHint::NoApi;
use glfw::WindowMode::Windowed;
use glfw::{CursorMode, GlfwReceiver, WindowEvent, WindowHint};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, EnumIter)]
pub enum Interaction
{
    PlayerMoveForward,
    PlayerMoveBackward,
    PlayerMoveLeft,
    PlayerMoveRight,
    PlayerMoveUp,
    PlayerMoveDown,
    PlayerSprint,
    ToggleConsole,
    ToggleCursorAttachment
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, EnumIter)]
pub enum InteractionMethod
{
    /// Only fires for one frame, no matter how long you hold the button
    /// down for. Useful for a toggle switch,
    /// i.e opening the developer console
    /// opening an inventory menu
    SinglePress,
    /// Fires every frame, as long as the button is pressed
    /// Useful for movement keys
    EveryFrame
}

/// A very messy class that handles OS level window events and
// TODO: remove ignore_frames
// TODO: refactor this whole thing now so much irrelevant stuff is
pub struct Window
{
    glfw:           Mutex<glfw::Glfw>,
    window:         Mutex<glfw::PWindow>,
    events_channel: GlfwReceiver<(f64, glfw::WindowEvent)>,

    previous_frame_duration:     AtomicCell<f32>,
    previous_frame_timepoint_ns: AtomicU64,

    interaction_active_timepoint_ns_map: HashMap<Interaction, AtomicU64>,
    interaction_method_map:              HashMap<Interaction, InteractionMethod>,

    previous_frame_mouse_pos: AtomicCell<(f32, f32)>,
    mouse_delta_pixels:       AtomicCell<(i32, i32)>,

    framebuffer_size_pixels: AtomicCell<(u32, u32)>,

    is_cursor_attached: AtomicBool,
    key_actions_maps:   HashMap<glfw::Key, Interaction>,

    ignore_frames: AtomicU32
}

impl Window
{
    const NULL_TIME: u64 = !0u64;

    pub fn new(keybinds: HashMap<Interaction, (InteractionMethod, glfw::Key)>) -> Self
    {
        let mut glfw = glfw::init(window_callback).expect("Failed to initalize GLFW");
        glfw.set_error_callback(window_callback);

        glfw.window_hint(WindowHint::ClientApi(NoApi));
        glfw.window_hint(WindowHint::Resizable(true));

        let (mut window, events_channel) =
            glfw.create_window(1920, 1080, "patinac", Windowed).unwrap();

        window.set_pos(100, 100);
        window.set_all_polling(true);

        for e in Interaction::iter()
        {
            assert!(keybinds.contains_key(&e), "{e:?} was not in keybinds!");
        }

        let mut key_actions_maps: HashMap<glfw::Key, Interaction> = HashMap::new();
        let mut interaction_active_timepoint_ns_map: HashMap<Interaction, AtomicU64> =
            HashMap::new();
        let mut interaction_method_map: HashMap<Interaction, InteractionMethod> = HashMap::new();

        for (interaction, (method, key)) in keybinds.into_iter()
        {
            key_actions_maps.insert(key, interaction);
            interaction_active_timepoint_ns_map
                .insert(interaction, AtomicU64::new(Self::NULL_TIME));
            interaction_method_map.insert(interaction, method);
        }

        let framebuffer_size_pixelsi32 = window.get_framebuffer_size();
        let framebuffer_size_pixels = AtomicCell::new((
            framebuffer_size_pixelsi32.0 as u32,
            framebuffer_size_pixelsi32.1 as u32
        ));

        let previous_frame_duration = AtomicCell::<f32>::new(0.016);
        let previous_frame_timepoint_ns =
            AtomicU64::new(chrono::Utc::now().timestamp_nanos_opt().unwrap() as u64);

        let previous_frame_mouse_pos = AtomicCell::new((0.0f32, 0.0f32));
        let mouse_delta_pixels = AtomicCell::new((0i32, 0i32));

        let is_cursor_attached = AtomicBool::new(true);

        let ignore_frames = AtomicU32::new(3u32);

        Window {
            glfw: Mutex::new(glfw),
            window: Mutex::new(window),
            events_channel,
            previous_frame_duration,
            previous_frame_timepoint_ns,
            interaction_active_timepoint_ns_map,
            interaction_method_map,
            previous_frame_mouse_pos,
            mouse_delta_pixels,
            framebuffer_size_pixels,
            is_cursor_attached,
            key_actions_maps,
            ignore_frames
        }
    }

    pub fn is_action_active(&self, interaction: Interaction) -> bool
    {
        if !self.is_cursor_attached.load(Ordering::Acquire)
        {
            return false;
        }

        if let Some(maybe_time) = self.interaction_active_timepoint_ns_map.get(&interaction)
        {
            if *self.interaction_method_map.get(&interaction).unwrap()
                == InteractionMethod::SinglePress
            {
                // if its a single press interaction, we want to do an atomic
                // swap with NULL_TIME
                maybe_time.swap(Self::NULL_TIME, Ordering::Acquire) != Self::NULL_TIME
            }
            else
            {
                maybe_time.load(Ordering::Acquire) != Self::NULL_TIME
            }
        }
        else
        {
            panic!("{:?} was requested and not present!", interaction);
        }
    }

    // (width, height)
    pub fn get_screen_space_mouse_delta(&self) -> (f32, f32)
    {
        if !self.is_cursor_attached.load(Ordering::Acquire)
            || self.ignore_frames.load(Ordering::Acquire) != 0
        {
            return (0.0, 0.0);
        }

        let mouse_delta_pixels = self.mouse_delta_pixels.load();
        let framebuffer_size = self.framebuffer_size_pixels.load();

        (
            mouse_delta_pixels.0 as f32 / framebuffer_size.0 as f32,
            mouse_delta_pixels.1 as f32 / framebuffer_size.1 as f32
        )
    }

    pub fn get_delta_time_seconds(&self) -> f32
    {
        self.previous_frame_duration.load()
    }

    pub fn get_framebuffer_size(&self) -> (u32, u32)
    {
        let i32_tuple = self.window.lock().unwrap().get_framebuffer_size();

        (i32_tuple.0 as u32, i32_tuple.1 as u32)
    }

    pub fn create_surface(&self, instance: ash::vk::Instance) -> vk::SurfaceKHR
    {
        let mut surface: vk::SurfaceKHR = vk::SurfaceKHR::null();

        self.window
            .lock()
            .unwrap()
            .create_window_surface(instance, std::ptr::null(), addr_of_mut!(surface))
            .result()
            .expect("Failed to create window surface!");

        surface
    }

    pub fn should_close(&self) -> bool
    {
        self.window.lock().unwrap().should_close()
    }

    pub fn block_while_minimized(&self)
    {
        while let (0, 0) = self.get_framebuffer_size()
        {
            self.process_events();
            std::thread::yield_now();
        }
    }

    pub fn attach_cursor(&self)
    {
        self.window
            .lock()
            .unwrap()
            .set_cursor_mode(CursorMode::Disabled);
        self.is_cursor_attached.store(true, Ordering::Release);
    }

    pub fn detach_cursor(&self)
    {
        let size = self.get_framebuffer_size();

        self.window
            .lock()
            .unwrap()
            .set_cursor_mode(CursorMode::Normal);
        self.window
            .lock()
            .unwrap()
            .set_cursor_pos(size.0 as f64 / 2.0, size.1 as f64 / 2.0);

        self.is_cursor_attached.store(false, Ordering::Release);
    }

    // This function may be called with any other function, however this function
    // cannot execute twice concurrently
    pub unsafe fn end_frame(&self)
    {
        let start_end_time = chrono::Utc::now().timestamp_nanos_opt().unwrap() as u64;

        self.glfw.lock().unwrap().poll_events();

        // This is fine since we're already in an unsafe function and by that
        // precondition this is in a critical section
        if self.ignore_frames.load(Ordering::Acquire) > 0
        {
            self.ignore_frames.fetch_sub(1, Ordering::AcqRel);
        }

        // Mouse processing
        let (current_x_pos, current_y_pos) = self.window.lock().unwrap().get_cursor_pos();
        let (prev_x_pos, prev_y_pos) = self.previous_frame_mouse_pos.load();

        self.mouse_delta_pixels.store((
            (current_x_pos - prev_x_pos as f64) as i32,
            (current_y_pos - prev_y_pos as f64) as i32
        ));
        self.previous_frame_mouse_pos
            .store((current_x_pos as f32, current_y_pos as f32));

        // Delta time processing
        self.previous_frame_duration.store(
            Duration::from_nanos(
                start_end_time - self.previous_frame_timepoint_ns.load(Ordering::Acquire)
            )
            .as_secs_f32()
        );

        self.previous_frame_timepoint_ns
            .store(start_end_time, Ordering::Release);
    }

    fn process_events(&self)
    {
        while let Some((num, event)) = self.events_channel.receive()
        {
            match event
            {
                WindowEvent::Pos(_, _) => (),
                WindowEvent::Size(_, _) => (),
                WindowEvent::Close => (),
                WindowEvent::Refresh => (),
                WindowEvent::Focus(focused) =>
                {
                    if focused
                    {
                        self.attach_cursor()
                    }
                }
                WindowEvent::Iconify(_) => (),
                WindowEvent::FramebufferSize(new_width, new_height) =>
                {
                    self.framebuffer_size_pixels
                        .store((new_width as u32, new_height as u32))
                }
                WindowEvent::MouseButton(_, _, _) => (),
                WindowEvent::CursorPos(_, _) => (),
                WindowEvent::CursorEnter(_) => (),
                WindowEvent::Scroll(_, _) => (),
                WindowEvent::Key(key, _scancode, action, _modifiers) =>
                'key_block: {
                    if !self.key_actions_maps.contains_key(&key)
                    {
                        log::trace!("Unbound key pressed {:?}", key);
                        break 'key_block;
                    }

                    let action_from_key = self.key_actions_maps.get(&key).unwrap();

                    let action_timepoint = self
                        .interaction_active_timepoint_ns_map
                        .get(action_from_key)
                        .unwrap();

                    match action
                    {
                        glfw::Action::Release =>
                        {
                            action_timepoint.store(Self::NULL_TIME, Ordering::Release)
                        }
                        glfw::Action::Press =>
                        {
                            action_timepoint.store(
                                chrono::Utc::now().timestamp_nanos_opt().unwrap() as u64,
                                Ordering::Release
                            )
                        }
                        glfw::Action::Repeat => ()
                    }
                }
                WindowEvent::Char(_) => (),
                WindowEvent::CharModifiers(_, _) => (),
                WindowEvent::FileDrop(_) => (),
                WindowEvent::Maximize(_) => (),
                WindowEvent::ContentScale(_, _) => ()
            }
        }
    }
}

fn window_callback(error: glfw::Error, message: String)
{
    log::error!("GLFW error raised! {} | {}", error, message);
}

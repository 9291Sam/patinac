#![feature(stmt_expr_attributes)]

use std::collections::HashMap;
use std::sync::OnceLock;

mod gfx;
mod util;

static LOGGER: OnceLock<util::AsyncLogger> = OnceLock::new();

fn main()
{
    // Initialize logger
    LOGGER.set(util::AsyncLogger::new()).unwrap();
    log::set_logger(LOGGER.get().unwrap()).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    log::trace!("Hello, world!");
    log::debug!("Hello, world!");
    log::info!("Hello, world!");
    log::warn!("Hello, world!");
    log::error!("Hello, world!");

    let mut keybinds: HashMap<gfx::Interaction, (gfx::InteractionMethod, glfw::Key)> =
        HashMap::new();
    #[rustfmt::skip]
    {
        keybinds.insert(gfx::Interaction::PlayerMoveForward,      (gfx::InteractionMethod::EveryFrame, glfw::Key::W));
        keybinds.insert(gfx::Interaction::PlayerMoveBackward,     (gfx::InteractionMethod::EveryFrame, glfw::Key::S));
        keybinds.insert(gfx::Interaction::PlayerMoveLeft,         (gfx::InteractionMethod::EveryFrame, glfw::Key::A));
        keybinds.insert(gfx::Interaction::PlayerMoveRight,        (gfx::InteractionMethod::EveryFrame, glfw::Key::D));
        keybinds.insert(gfx::Interaction::PlayerMoveUp,           (gfx::InteractionMethod::EveryFrame, glfw::Key::Space));
        keybinds.insert(gfx::Interaction::PlayerMoveDown,         (gfx::InteractionMethod::EveryFrame, glfw::Key::LeftControl));
        keybinds.insert(gfx::Interaction::PlayerSprint,           (gfx::InteractionMethod::EveryFrame, glfw::Key::LeftShift));
        keybinds.insert(gfx::Interaction::ToggleConsole,          (gfx::InteractionMethod::SinglePress, glfw::Key::GraveAccent));
        keybinds.insert(gfx::Interaction::ToggleCursorAttachment, (gfx::InteractionMethod::SinglePress, glfw::Key::Backslash));
    }

    let window = gfx::Window::new(keybinds);

    while !window.should_close()
    {
        unsafe { window.end_frame() };
    }

    unsafe { LOGGER.get().unwrap().join_worker_thread() }
}

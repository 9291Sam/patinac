use std::collections::HashMap;

use winit::keyboard::KeyCode;

struct InputManager
{
    key_states: HashMap<KeyCode, KeyPressedState>
}

impl InputManager
{
    pub fn new() -> InputManager
    {
        InputManager {
            key_states: todo!()
        }
    }

    // pub fn update_key_state(&mut self, key: KeyCode, new_state: KeyPressedState)
    // {}

    // pub fn get_key_state(&self, key: KeyCode)
    // {
    //     self.key_states.get(&key).unwrap()
    // }

    // attach and detach cursor
}

///      ╔═════╗      Released
///      ║  ^  ║
/// ═════╝  |  ╚═════ Pressed
///   ^  ^  ^  ^  ^
///   \>          \> Pressed
///      \>          Releasing
///         \>       Released
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum KeyPressedState
{
    Pressed,
    Releasing,
    Released,
    Pressing
}

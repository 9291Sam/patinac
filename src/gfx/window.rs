pub struct Window
{
    glfw: glfw::Glfw
}

#[repr(u8)]
// #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, EnumIter)]
pub enum Action
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

enum InteractionMethod
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

// struct ActionInformation
// {
//     GlfwKeyType       key;
//     InteractionMethod method;
// }

impl Window
{
    pub fn new() -> Self
    {
        // glfw::w_hint(
        //     glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi)
        //         | glfw::WindowHint::Resizable(true)
        // );

        let glfw = glfw::init(window_callback).expect("Failed to initalize GLFW");

        Window {
            glfw
        }
    }
}

fn window_callback(error: glfw::Error, message: String) {}

///      x
///      ╔═════╗      Released
///      ║  ^  ║
/// ═════╝  |  ╚═════ Pressed
///   ^  ^  ^  ^  ^
///   \>          \> Pressed
///      \>          Releasing
///         \>       Released
///
///  |  |    | \- Released
///          \ - Depressed
///                \\-
enum KeyPressedState
{
    Pressed,
    Releasing,
    Released,
    Pressing
}

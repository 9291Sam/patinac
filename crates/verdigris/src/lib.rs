#![feature(stmt_expr_attributes)]
#![feature(new_uninit)]

mod demo_scene;
mod instanced_indirect;
mod player;
mod recordables;

pub use demo_scene::DemoScene;
pub(crate) use player::Player;

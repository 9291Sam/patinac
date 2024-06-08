#![feature(stmt_expr_attributes)]
#![feature(new_uninit)]
#![allow(soft_unstable)]
#![feature(test)]

mod demo_scene;
mod instanced_indirect;
mod player;
mod recordables;

pub use demo_scene::DemoScene;
pub(crate) use player::Player;

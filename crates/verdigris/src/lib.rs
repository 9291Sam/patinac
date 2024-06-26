#![feature(stmt_expr_attributes)]
#![feature(new_uninit)]
#![allow(soft_unstable)]
#![feature(test)]

mod demo_scene;
mod player;
mod recordables;
mod voxel_world;

pub use demo_scene::DemoScene;
pub(crate) use player::Player;
pub use recordables::flat_textured::FlatTextured;
pub use recordables::lit_textured::LitTextured;

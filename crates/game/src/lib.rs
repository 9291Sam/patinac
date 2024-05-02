#![feature(trait_upcasting)]
mod entity;
mod game;
mod renderpasses;

pub use entity::{Entity, EntityCastDepot, Positionable, SelfManagedEntity, Transformable};
pub use game::{Game, TickTag, World};
pub use renderpasses::{PassStage, RenderPassManager};

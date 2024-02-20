#![feature(trait_upcasting)]

mod entity;
mod game;

pub use entity::{DowncastEntity, Entity, EntityCast, Positionable, Transformable};
pub use game::{Game, TickTag};

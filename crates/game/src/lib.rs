#![feature(trait_upcasting)]

mod entity;
mod game;

pub use entity::{Entity, EntityCastDepot, Positionable, Transformable};
pub use game::{Game, TickTag};

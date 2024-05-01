#![feature(trait_upcasting)]
mod entity;
mod game;

pub use entity::{Entity, EntityCastDepot, Positionable, SelfManagedEntity, Transformable};
pub use game::{Game, TickTag, World};

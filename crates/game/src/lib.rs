#![feature(trait_upcasting)]
mod entity;
mod game;
mod renderpasses;

pub use entity::{
    Collideable,
    Entity,
    EntityCastDepot,
    Positionalable,
    SelfManagedEntity,
    Transformable
};
pub use game::{Game, TickTag};
pub use renderpasses::{PassStage, RenderPassManager};

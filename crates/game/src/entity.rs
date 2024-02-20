use core::fmt::Debug;
use std::any::Any;
use std::borrow::Cow;

use gfx::glm;

use crate::game::TickTag;

pub trait Entity: Debug + Send + Sync
{
    fn as_any(&self) -> &dyn Any;
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;

    fn tick(&self, game: &super::Game, _: TickTag);

    fn eq(&self, other: &dyn Entity) -> bool
    where
        Self: Sized
    {
        std::ptr::eq(self as *const dyn Entity, other as *const dyn Entity)
    }
}

pub trait Positionable: Entity
{
    fn get_position(&self, func: &dyn Fn(glm::Vec3));
    fn get_position_mut(&self, func: &dyn Fn(&mut glm::Vec3));
}
pub trait Transformable: Positionable
{
    fn get_transform(&self, func: &dyn Fn(&gfx::Transform));
    fn get_transform_mut(&self, func: &dyn Fn(&mut gfx::Transform));
}

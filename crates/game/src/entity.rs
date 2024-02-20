use core::fmt::Debug;
use std::any::Any;
use std::borrow::Cow;

use crate::game::TickTag;

pub trait Entity: Debug + Send + Sync + Any
{
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
    fn get_position<R>(&self, func: impl FnOnce(Option<&gfx::Transform>) -> R) -> R;
    fn get_position_mut<R>(&self, func: impl FnOnce(Option<&mut gfx::Transform>) -> R) -> R;
}
pub trait Transformable: Positionable
{
    fn get_position<R>(&self, func: impl FnOnce(Option<&gfx::Transform>) -> R) -> R;
    fn get_position_mut<R>(&self, func: impl FnOnce(Option<&mut gfx::Transform>) -> R) -> R;
}

pub trait EntityCastable
{
    fn cast<T: Entity>(&self) -> Option<&T>
    where
        Self: Sized + 'static
    {
        (self as &dyn Any).downcast_ref::<T>()
    }

    fn cast_mut<T: Entity>(&mut self) -> Option<&mut T>
    where
        Self: Sized + 'static
    {
        (self as &mut dyn Any).downcast_mut::<T>()
    }
}

impl<E: Entity> EntityCastable for E {}

use core::fmt::Debug;
use std::borrow::Cow;
use std::option::Option;

use gfx::glm;

use crate::game::TickTag;

pub trait Entity: Debug + Send + Sync + EntityCastDepot
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

impl<'a> dyn Entity + 'a
{
    pub fn cast<T: EntityCastTarget<'a> + ?Sized>(&'a self) -> Option<&'a T>
    {
        T::cast(self)
    }
}

pub trait EntityCastTarget<'a>
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>;
}
impl<'a> EntityCastTarget<'a> for dyn Entity + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_entity()
    }
}
impl<'a> EntityCastTarget<'a> for dyn Positionable + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_positionable()
    }
}
impl<'a> EntityCastTarget<'a> for dyn Transformable + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_transformable()
    }
}

pub trait EntityCastDepot
{
    fn as_entity(&self) -> Option<&dyn Entity>;
    fn as_positionable(&self) -> Option<&dyn Positionable>;
    fn as_transformable(&self) -> Option<&dyn Transformable>;
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

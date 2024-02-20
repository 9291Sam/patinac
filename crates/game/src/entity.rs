use core::fmt::Debug;
use std::any::Any;
// pub trait Entity: Debug + Send + Sync + EntityCast
// {
//     fn as_any(&self) -> &dyn Any;
//     fn get_name(&self) -> Cow<'_, str>;
//     fn get_uuid(&self) -> util::Uuid;

//     fn tick(&self, game: &super::Game, _: TickTag);

//     fn eq(&self, other: &dyn Entity) -> bool
//     where
//         Self: Sized
//     {
//         std::ptr::eq(self as *const dyn Entity, other as *const dyn Entity)
//     }
// }

// pub trait EntityCast
// {
//     fn as_entity(&self) -> Option<&dyn Entity>;
//     fn as_positionable(&self) -> Option<&dyn Positionable>;
//     fn as_transformable(&self) -> Option<&dyn Transformable>;
// }
use std::borrow::Cow;
use std::option::Option;

use gfx::glm;

use crate::game::TickTag;

pub trait Entity: Debug + Send + Sync + EntityCast
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

pub trait EntityCast
{
    fn as_entity(&self) -> Option<&dyn Entity>;
    fn as_positionable(&self) -> Option<&dyn Positionable>;
    fn as_transformable(&self) -> Option<&dyn Transformable>;
}

pub trait DowncastEntity: EntityCast
{
    fn downcast<T: EntityCast + ?Sized>(&self) -> Option<&T>
    where
        T: 'static
    {
        if let Some(entity) = self.as_entity()
        {
            entity.downcast_ref()
        }
        else if let Some(positionable) = self.as_positionable()
        {
            positionable.downcast_ref()
        }
        else if let Some(transformable) = self.as_transformable()
        {
            transformable.downcast_ref()
        }
        else
        {
            None
        }
    }

    fn downcast_ref<T: EntityCast + ?Sized>(&self) -> Option<&T>
    where
        T: 'static
    {
        self.downcast::<T>().map(|r| r as &T)
    }
}

impl<T: Entity + EntityCast + ?Sized> DowncastEntity for T {}

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
// https://discord.com/channels/273534239310479360/273541522815713281/1209533218689253436

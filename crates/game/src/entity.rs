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

// There are three parts to this scheme:

// 1: The convenience method. This is what lets you type `entity.cast::<dyn
// Positionable>()`. Conceptually, it corresponds to `Iterator::collect`
// or `str::parse`: its entire implementation is just a call to a helper trait
// (`FromIterator`, `FromStr`).
impl<'a> dyn Entity + 'a
{
    pub fn cast<T: EntityCastTarget<'a> + ?Sized>(&'a self) -> Option<&'a T>
    {
        T::cast(self)
    }
}

// 2: The helper trait. This is implemented by types who can produce an
// `Option<&Self>` from an `EntityCastDepot`. Each such type does so by calling
// a different method from the depot.

// There's a lifetime generic on this trait, for somewhat mistifying reason:
// `dyn Trait` is actually `dyn Trait + 'static`, so I wrote `dyn Entity + 'a`
// earlier. However, nothing says that the `T` you're casting to lives for at
// most as long as `'a`, which causes the compiler to complain that `'a` in that
// function needs to live for `'static`. To prevent that, we introduce a
// lifetime on `EntityCastTarget`. I'll be entirely honest, I'm not sure I fully
// understand _why_ this works, but it does seem to work.
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

// 3: The depot trait, where entities define the methods to cast themselves to
// different trait objects. Remember how I said `dyn Trait` is `dyn Trait +
// 'static`? Yeah, `&dyn Trait` is different. The elision rules for that one
// make it `&'a (dyn Trait + 'a)`
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

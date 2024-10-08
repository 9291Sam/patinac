use core::fmt::Debug;
use std::borrow::Cow;
use std::sync::Arc;

use gfx::glm;
use rapier3d::dynamics::{IslandManager, RigidBody, RigidBodyHandle, RigidBodySet};
use rapier3d::geometry::{Collider, ColliderSet};

use crate::game::TickTag;

pub trait Entity: Send + Sync + EntityCastDepot
{
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;

    fn tick(&self, game: &super::Game, _: TickTag);
}

impl Debug for dyn Entity
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Entity {} | Id: {}", self.get_name(), self.get_uuid())
    }
}

// ! If you add a new trait, don't forget to add it to `EntityCastDepot` and
// ! extend `EntityCastTarget`
pub trait EntityCastDepot
{
    fn as_self_managed(self: Arc<Self>) -> Option<Arc<dyn SelfManagedEntity>>;
    fn as_positionalable(&self) -> Option<&dyn Positionalable>;
    fn as_transformable(&self) -> Option<&dyn Transformable>;
    fn as_collideable(&self) -> Option<&dyn Collideable>;
}

pub trait SelfManagedEntity: Entity
{
    fn is_alive(&self) -> bool;
}
pub trait Positionalable: Entity
{
    fn get_position(&self) -> glm::Vec3;
}
pub trait Transformable: Positionalable
{
    fn get_transform(&self) -> gfx::Transform;
}
pub trait Collideable: Transformable
{
    fn init_collideable(&self) -> (RigidBody, Vec<Collider>);

    #[allow(clippy::too_many_arguments)]
    fn physics_tick(
        &self,
        game: &super::Game,
        gravity: glm::Vec3,
        this_rigid_body_handle: RigidBodyHandle,
        collider_set: &mut ColliderSet,
        rigid_body_set: &mut RigidBodySet,
        islands: &mut IslandManager,
        _: TickTag
    );
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
impl<'a> EntityCastTarget<'a> for dyn Positionalable + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_positionalable()
    }
}
impl<'a> EntityCastTarget<'a> for dyn Transformable + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_transformable()
    }
}
impl<'a> EntityCastTarget<'a> for dyn Collideable + 'a
{
    fn cast<T: EntityCastDepot + ?Sized>(this: &'a T) -> Option<&'a Self>
    {
        this.as_collideable()
    }
}

use core::fmt::Debug;
use std::borrow::Cow;

use gfx::glm;

use crate::game::TickTag;

pub trait Entity: Debug + Send + Sync
{
    fn get_name(&self) -> Cow<'_, str>;
    fn get_uuid(&self) -> util::Uuid;
    fn get_position(&self) -> Option<glm::Vec3>;

    fn tick(&self, game: &super::Game, _: TickTag);

    fn eq(&self, other: &dyn Entity) -> bool
    where
        Self: Sized
    {
        std::ptr::eq(self as *const dyn Entity, other as *const dyn Entity)
    }
}

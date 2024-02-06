use core::fmt::Debug;

use crate::TickTag;

pub trait Entity: Debug + Send + Sync
{
    fn get_name(&self) -> &str;
    fn get_uuid(&self) -> util::Uuid;
    fn get_position(&self) -> Option<gfx::Vec3>;

    fn tick(&self, game: &super::Game, _: TickTag);

    fn eq(&self, other: &dyn Entity) -> bool
    where
        Self: Sized
    {
        std::ptr::eq(self as *const dyn Entity, other as *const dyn Entity)
    }
}

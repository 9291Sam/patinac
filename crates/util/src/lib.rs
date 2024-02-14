#![feature(map_try_insert)]
#![feature(maybe_uninit_as_bytes)]

mod allocator;
mod log;
mod registrar;
mod uuid;

use std::ops::Deref;

pub use allocator::*;
pub use log::*;
pub use registrar::*;
pub use uuid::*;

pub struct SendSyncMutPtr<T>(pub *mut T);

unsafe impl<T> Send for SendSyncMutPtr<T> {}
unsafe impl<T> Sync for SendSyncMutPtr<T> {}

impl<T> Deref for SendSyncMutPtr<T>
{
    type Target = *mut T;

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl<T> From<*mut T> for SendSyncMutPtr<T>
{
    fn from(value: *mut T) -> Self
    {
        Self(value)
    }
}

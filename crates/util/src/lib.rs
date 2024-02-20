#![feature(map_try_insert)]
#![feature(maybe_uninit_as_bytes)]

mod allocator;
mod log;
mod registrar;
mod uuid;

use std::any::Any;
use std::ops::Deref;

pub use allocator::*;
pub use log::*;
pub use registrar::*;
pub use uuid::*;

pub fn hash_combine(a_seed: u64, bytes: &[u8]) -> u64
{
    let mut seed = a_seed;
    for b in bytes
    {
        seed = u64::from(*b) ^ 0x9E37_79B9_E377_9B9Eu64 ^ (seed << 12) ^ (seed >> 48);
    }
    seed
}

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

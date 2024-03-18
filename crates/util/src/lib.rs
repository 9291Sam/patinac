#![feature(adt_const_params)]
#![feature(map_try_insert)]
#![feature(maybe_uninit_as_bytes)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(trait_alias)]
#![feature(type_alias_impl_trait)]

mod allocator;
mod r#async;
mod crash_handler;
mod global_allocator;
mod log;
mod registrar;
mod uuid;
mod window;

use std::marker::ConstParamTy;
use std::ops::Deref;
use std::sync::Arc;

pub use allocator::*;
pub use r#async::*;
pub use crash_handler::*;
pub use global_allocator::*;
pub use log::*;
pub use registrar::*;
pub use uuid::*;
pub use window::*;

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

// boooo bad joke
pub enum Sow<'t, T>
{
    Strong(Arc<T>),
    Ref(&'t T)
}

impl<'t, T> Deref for Sow<'t, T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target
    {
        match self
        {
            Sow::Strong(arc) => arc,
            Sow::Ref(r) => r
        }
    }
}

impl<T> From<*mut T> for SendSyncMutPtr<T>
{
    fn from(value: *mut T) -> Self
    {
        Self(value)
    }
}

#[derive(ConstParamTy, PartialEq, Eq)]
pub enum SuffixType
{
    Short,
    Full
}

pub fn bytes_as_string<const SUFFIX: SuffixType>(bytes: f64) -> String
{
    const FULL_SUFFIX: &[&str] = &[
        "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB"
    ];

    const SHORT_SUFFIX: &[&str] = &["B", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"];

    const UNIT: f64 = 1024.0;

    let size: f64 = bytes;

    if size <= 0.0
    {
        return "0 B".to_string();
    }

    let base = size.log10() / UNIT.log10();

    let result = format!("{:.3}", UNIT.powf(base - base.floor()),)
        .trim_end_matches(".0")
        .to_owned();

    // Add suffix
    [
        &result,
        match SUFFIX
        {
            SuffixType::Short => SHORT_SUFFIX,
            SuffixType::Full => FULL_SUFFIX
        }[base.floor() as usize]
    ]
    .join(" ")
}

// struct StaticMutex<T: ?Sized>
// {
//     t: UnsafeCell<T>
// }

// unsafe impl<T: ?Sized> Sync for StaticMutex<T> {}
// unsafe impl<T: ?Sized + Send> Send for StaticMutex<T> {}

// impl<T: ?Sized> StaticMutex<T>
// {
//     pub fn get_mut(&mut self) -> &mut T
//     {
//         &mut *self.t.get_mut()
//     }
// }

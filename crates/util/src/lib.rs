#![feature(map_try_insert)]
#![feature(allocator_api)]
#![feature(unchecked_shifts)]
#![feature(map_entry_replace)]
#![feature(get_many_mut)]

mod allocator;
mod r#async;
mod atomics;
mod broadcast_event;
mod crash_handler;
mod dense_set;
mod global_allocator;
mod log;
mod pinger;
mod registrar;
mod timer;
mod uuid;
mod window;

use std::ops::Deref;
use std::sync::Arc;

pub use allocator::*;
pub use r#async::*;
pub use atomics::*;
pub use broadcast_event::*;
pub use crash_handler::*;
pub use global_allocator::*;
pub use log::*;
pub use pinger::*;
pub use registrar::*;
pub use timer::*;
pub use uuid::*;
pub use window::*;

/// # Safety
///
/// This function unsafely changes the lifetime of the given reference
/// this should only be used when you have extra information that rust cannot
/// understand
pub const unsafe fn extend_lifetime<'a, T>(t: &T) -> &'a T
{
    std::mem::transmute(t)
}

#[cold]
#[inline(never)]
/// # Safety
///
/// Don't use this not for testing
pub unsafe fn asan_test() -> i32
{
    #[cfg(not(debug_assertions))]
    panic!("Don't use this in a release build");

    let xs = [0, 1, 2, 3];
    std::hint::black_box(unsafe {
        *std::hint::black_box(xs.as_ptr()).offset(std::hint::black_box(4))
    })
}

#[derive(Clone, Copy)]
pub struct SendSyncMutPtr<T>(pub *mut T);

unsafe impl<T> Send for SendSyncMutPtr<T> {}
unsafe impl<T> Sync for SendSyncMutPtr<T> {}

impl<T> Ord for SendSyncMutPtr<T>
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering
    {
        self.0.cmp(&other.0)
    }
}

impl<T> PartialOrd for SendSyncMutPtr<T>
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>
    {
        Some(self.cmp(other))
    }
}

impl<T> PartialEq for SendSyncMutPtr<T>
{
    fn eq(&self, other: &Self) -> bool
    {
        self.0 == other.0
    }
}

impl<T> Eq for SendSyncMutPtr<T> {}

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

#[derive(PartialEq, Eq)]
pub enum SuffixType
{
    Short,
    Full
}

pub fn bytes_as_string(bytes: f64, suffix: SuffixType) -> String
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

    // Add suffix
    [
        format!("{:.3}", UNIT.powf(base - base.floor()),).trim_end_matches(".0"),
        match suffix
        {
            SuffixType::Short => SHORT_SUFFIX,
            SuffixType::Full => FULL_SUFFIX
        }[base.floor() as usize]
    ]
    .join(" ")
}

pub fn map_f32(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
    (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}

pub fn map_f64(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64
{
    (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}

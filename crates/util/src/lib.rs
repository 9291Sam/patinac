#![feature(map_try_insert)]
#![feature(allocator_api)]
#![feature(unchecked_shifts)]

mod allocator;
mod r#async;
mod atomics;
mod crash_handler;
mod global_allocator;
mod log;
mod registrar;
mod timer;
mod uuid;
mod window;

use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::Arc;

pub use allocator::*;
pub use r#async::*;
pub use atomics::*;
pub use crash_handler::*;
pub use global_allocator::*;
pub use log::*;
pub use registrar::*;
pub use timer::*;
pub use uuid::*;
pub use window::*;

// pub fn hash_combine<const L: usize>(mut seed: u64, data: &[u8]) -> u64
// {
//     let mut reinterpreted_data: MaybeUninit<[u64; L.div_ceil(8)]> =
// MaybeUninit::uninit();     let ptr: *mut u8 = reinterpreted_data.as_mut_ptr()
// as *mut u8;

//     unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, L) }
//     unsafe { std::ptr::write_bytes(ptr.add(L), 0, data.len() - L) }

//     let u64_data = unsafe { reinterpreted_data.assume_init() };

//     for b in u64_data
//     {
//         seed = b
//             ^ 0x9E37_79B9_E377_9B9Eu64
//             ^ (unsafe { seed.unchecked_shl(12) })
//             ^ (unsafe { seed.unchecked_shr(48) });
//     }

//     seed
// }

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

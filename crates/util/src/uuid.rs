use std::fmt::{Debug, Display};
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uuid
{
    data:       u64,
    time_stamp: u64
}

impl Uuid
{
    pub fn new() -> Self
    {
        let raw_time_stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let raw_id = get_monotonic_id();

        Uuid {
            data:       fxhash::hash64(&raw_id),
            time_stamp: fxhash::hash64(&raw_time_stamp)
        }
    }
}

impl Display for Uuid
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            f,
            "{:08X}~{:08X}~{:08X}~{:08X}",
            (self.data >> 32) as u32,
            self.data as u32,
            (self.time_stamp >> 32) as u32,
            self.time_stamp as u32
        )
    }
}

impl Debug for Uuid
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "{}", *self)
    }
}

fn get_monotonic_id() -> u64
{
    static MONOTONIC_ID: OnceLock<AtomicU64> = OnceLock::new();

    MONOTONIC_ID
        .get_or_init(|| {
            let mut new_id: MaybeUninit<u64> = MaybeUninit::uninit();

            getrandom::getrandom_uninit(new_id.as_bytes_mut()).unwrap();

            AtomicU64::new(unsafe { new_id.assume_init() })
        })
        .fetch_add(1, AcqRel)
}

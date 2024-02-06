use std::fmt::{Debug, Display};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::OnceLock;

use bytemuck::{bytes_of, bytes_of_mut};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uuid
{
    data:       u64,
    time_stamp: u64
}

impl Uuid
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self
    {
        let raw_time_stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let raw_id = get_monotonic_id();

        Uuid {
            data:       seahash::hash(bytes_of(&raw_id)),
            time_stamp: seahash::hash(bytes_of(&raw_time_stamp))
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
            let mut new_id: u64 = 4378234789237894789;

            getrandom::getrandom(bytes_of_mut(&mut new_id)).unwrap();

            AtomicU64::new(new_id)
        })
        .fetch_add(1, AcqRel)
}

use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct AtomicF32F32
{
    data: AtomicU64
}

impl Debug for AtomicF32F32
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("AtomicF32F32")
            .field("data", &self.load(Ordering::SeqCst))
            .finish()
    }
}

impl AtomicF32F32
{
    pub fn new(val: (f32, f32)) -> Self
    {
        AtomicF32F32 {
            data: AtomicU64::new(Self::tuple_as_u64(val))
        }
    }

    pub fn store(&self, val: (f32, f32), ordering: Ordering)
    {
        self.data.store(Self::tuple_as_u64(val), ordering)
    }

    pub fn load(&self, ordering: Ordering) -> (f32, f32)
    {
        Self::u64_as_tuple(self.data.load(ordering))
    }

    fn tuple_as_u64(val: (f32, f32)) -> u64
    {
        let low: u64 = val.0.to_bits() as u64;
        let high: u64 = (val.1.to_bits() as u64) << 32;

        low | high
    }

    fn u64_as_tuple(val: u64) -> (f32, f32)
    {
        let low: u64 = 0x0000_0000_FFFF_FFFF & val;
        let high: u64 = (0xFFFF_FFFF_0000_0000 & val) >> 32;

        (f32::from_bits(low as u32), f32::from_bits(high as u32))
    }
}

pub struct AtomicU32U32
{
    data: AtomicU64
}

impl Debug for AtomicU32U32
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("AtomicU32U32")
            .field("data", &self.load(Ordering::SeqCst))
            .finish()
    }
}

impl AtomicU32U32
{
    pub fn new(val: (u32, u32)) -> Self
    {
        AtomicU32U32 {
            data: AtomicU64::new(Self::tuple_as_u64(val))
        }
    }

    pub fn store(&self, val: (u32, u32), ordering: Ordering)
    {
        self.data.store(Self::tuple_as_u64(val), ordering)
    }

    pub fn load(&self, ordering: Ordering) -> (u32, u32)
    {
        Self::u64_as_tuple(self.data.load(ordering))
    }

    fn tuple_as_u64(val: (u32, u32)) -> u64
    {
        let low: u64 = val.0 as u64;
        let high: u64 = (val.1 as u64) << 32;

        low | high
    }

    fn u64_as_tuple(val: u64) -> (u32, u32)
    {
        let low: u64 = 0x0000_0000_FFFF_FFFF & val;
        let high: u64 = (0xFFFF_FFFF_0000_0000 & val) >> 32;

        (low as u32, high as u32)
    }
}

#[repr(transparent)]
pub struct AtomicF32
{
    data: AtomicU32
}

impl Debug for AtomicF32
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("AtomicF32")
            .field("data", &self.load(Ordering::SeqCst))
            .finish()
    }
}

impl AtomicF32
{
    pub const fn new(val: f32) -> Self
    {
        AtomicF32 {
            data: AtomicU32::new(val.to_bits())
        }
    }

    pub fn store(&self, val: f32, ordering: Ordering)
    {
        self.data.store(val.to_bits(), ordering)
    }

    pub fn load(&self, ordering: Ordering) -> f32
    {
        f32::from_bits(self.data.load(ordering))
    }
}

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_atomic_f32f32()
    {
        let atom = AtomicF32F32::new((0.0, 0.0));

        for i in 0..1_000_000
        {
            let l = i as f32 / 1403.0;
            let r = i as f32 * 45983.432;

            atom.store((l, r), Ordering::Release);
            assert_eq!((l, r), atom.load(Ordering::Acquire));
        }
    }
}

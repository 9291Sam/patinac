use std::fmt::Debug;
use std::mem::MaybeUninit;

use num::traits::{WrappingAdd, WrappingMul, WrappingNeg, WrappingShl, WrappingShr, WrappingSub};
use num::{Float, Integer, Unsigned};

// Kelsie Loquavian??? in my code!!!
/// A single "phase" of the noise, each of these generators produces continuous
/// noise in some vector space
#[derive(Clone)]
struct Signet<const L: usize>
{
    seed: u64
}

impl<const L: usize> Debug for Signet<L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Signet (Kelsie Loquavian)")
    }
}

impl<const L: usize> Signet<L>
{
    #[inline(always)]
    pub fn new(seed: u64, pos: [i64; L]) -> Self
    {
        todo!()
    }

    #[inline(always)]
    pub fn sample<F: Float>(pos: [F; L]) {}

    #[inline(always)]
    pub fn sample_integer<I>(&self, pos: [i64; L]) -> u64
    where
        I: Integer
            + Unsigned
            + WrappingAdd
            + WrappingMul
            + WrappingSub
            + WrappingNeg
            + WrappingShl
            + WrappingShr,
        [i64; L * 2 + 1]: Sized
    {
        let mut arr: MaybeUninit<[u64; L * 2 + 1]> = MaybeUninit::uninit();
        let arr_ptr: *mut [u64; L * 2 + 1] = arr.as_mut_ptr();
        let arr_data: *mut u64 = arr_ptr as *mut _;

        for i in 0..L
        {
            unsafe { arr_data.add(i).write(*pos.get_unchecked(i) as u64) }
        }

        for i in 0..L
        {
            unsafe { arr_data.add(i + L).write(*pos.get_unchecked(i) as u64) }
        }

        unsafe { arr_data.add(L * 2).write(self.seed) }

        deterministic_rand_combine(unsafe { arr.assume_init() })
    }
}

#[inline(always)]
fn deterministic_rand_combine<const L: usize, I>(data: [I; L]) -> I
where
    I: Integer
        + Unsigned
        + WrappingAdd
        + WrappingMul
        + WrappingSub
        + WrappingNeg
        + WrappingShl
        + WrappingShr
{
    let mut combined: I = I::one();

    for i in data
    {
        let prev = combined;

        todo!()
        // combined.
    }

    combined
}

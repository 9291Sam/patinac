use std::fmt::Debug;
use std::hint::unreachable_unchecked;
use std::intrinsics::transmute_unchecked;
use std::mem::{transmute, MaybeUninit};
use std::ops::BitXor;

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
        + Copy
{
    let mut combined: I = I::one();

    // TODO: replace with the one from verdigris C++
    for (idx, d) in data.into_iter().enumerate()
    {
        let p1 = combined.wrapping_shr(6);
        let p2 = combined.wrapping_shr(6);

        combined = combined.wrapping_mul(&p2.add(combined.xor(p1.xor(p2).xor(d)).xor(d)));
    }

    combined
}

trait XorExtension
{
    fn xor(self, other: Self) -> Self;
}

impl<T: Integer> XorExtension for T
{
    fn xor(self, other: Self) -> Self
    {
        unsafe {
            match std::mem::size_of::<Self>()
            {
                1 =>
                {
                    transmute_unchecked(
                        transmute_unchecked::<Self, u8>(self)
                            .bitxor(transmute_unchecked::<Self, u8>(other))
                    )
                }
                2 =>
                {
                    transmute_unchecked(
                        transmute_unchecked::<Self, u16>(self).bitxor(transmute_unchecked::<
                            Self,
                            u16
                        >(
                            other
                        ))
                    )
                }
                4 =>
                {
                    transmute_unchecked(
                        transmute_unchecked::<Self, u32>(self).bitxor(transmute_unchecked::<
                            Self,
                            u32
                        >(
                            other
                        ))
                    )
                }
                8 =>
                {
                    transmute_unchecked(
                        transmute_unchecked::<Self, u64>(self).bitxor(transmute_unchecked::<
                            Self,
                            u64
                        >(
                            other
                        ))
                    )
                }
                _ =>
                {
                    todo!()
                }
            }
        }
    }
}

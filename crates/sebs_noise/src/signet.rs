use std::fmt::Debug;
use std::mem::MaybeUninit;

use bytemuck::NoUninit;
use num::{Float, FromPrimitive};

// Kelsie Loquavian??? in my code!!!
/// A single "phase" of the noise, each of these generators produces continuous
/// noise in some vector space
#[derive(Clone)]
pub struct Signet<const L: usize>
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
    pub fn new(seed: u64) -> Signet<L>
    {
        Self {
            seed
        }
    }

    #[inline(always)]
    pub fn sample<F: Float + FromPrimitive>(&self, pos: [F; L]) -> F
    where
        [(); 2u32.pow(L as u32) as usize]:
    {
        // floor, ceil, diff
        let data: [(i64, i64, F); L] = std::array::from_fn(|i| {
            let f = unsafe { pos.get_unchecked(i) };

            let ceil = f.ceil();
            let floor = f.floor();

            (
                floor.to_i64().expect("NANL"),
                ceil.to_i64().expect("NANR"),
                ceil - floor
            )
        });

        // make ndimesnsional cartesian prpdocut
        // 2 4 8
        let cartesian_prod_array: [F; 2u32.pow(L as u32) as usize] = std::array::from_fn(|i| {
            let coord: [bool; L] = generate_binary_permutation::<L>(i);

            let sample_point: [i64; L] = std::array::from_fn(|i| {
                unsafe {
                    match coord.get_unchecked(i)
                    {
                        true => data.get_unchecked(i).1,
                        false => data.get_unchecked(i).0
                    }
                }
            });

            self.sample_float(sample_point)
        });

        let mut weighted_sum = F::zero();
        let mut total_weight = F::zero();

        for i in 0..2u32.pow(L as u32) as usize
        {
            let weight = unsafe { data.get_unchecked(0).2 };
            weighted_sum = weighted_sum + cartesian_prod_array[i] * weight;
            total_weight = total_weight + weight;
        }

        weighted_sum / total_weight
    }

    #[inline(always)]
    pub fn sample_integer(&self, pos: [i64; L]) -> u64
    {
        util::hash_combine(self.seed, unsafe {
            std::slice::from_raw_parts(pos.as_ptr() as *const u8, std::mem::size_of_val(&pos))
        })
    }

    #[inline(always)]
    pub fn sample_float<F: Float + FromPrimitive>(&self, pos: [i64; L]) -> F
    {
        unsafe {
            map_float(
                FromPrimitive::from_u64(self.sample_integer(pos)).unwrap_unchecked(),
                F::zero(),
                FromPrimitive::from_u64(u64::MAX).unwrap_unchecked(),
                F::zero(),
                F::one()
            )
        }
    }
}

const fn generate_generic_float<F: Float + From<f64>>(float: f64) -> F
{
    float.into()
}

// <2>(0) -> [false, false]
// <2>(1) -> [false, true]
// <2>(2) -> [true, false]
// <2>(3) -> [true, true]
// <3>(0) -> [false, false, false]
// <3>(1) -> [false, false, true]
// <3>(2) -> [false, true, false]
// <3>(3) -> [false, true, true]
// <3>(4) -> [true, false, false]
// <3>(5) -> [true, false, true]
// <3>(6) -> [true, true, false]
// <3>(7) -> [true, true, true]
#[inline(always)]
fn generate_binary_permutation<const L: usize>(idx: usize) -> [bool; L]
{
    unsafe {
        std::array::from_fn(|i| {
            (idx.unchecked_shr(L.unchecked_sub(1).unchecked_sub(i) as u32) & 1) == 1
        })
    }
}

#[inline(always)]
pub fn map_float<F: Float>(x: F, x_min: F, x_max: F, y_min: F, y_max: F) -> F
{
    let range = x_max - x_min;
    y_min + (x - x_min) * (y_max - y_min) / range
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_generate_binary_permutation_2d()
    {
        assert_eq!(generate_binary_permutation::<2>(0), [false, false]);
        assert_eq!(generate_binary_permutation::<2>(1), [false, true]);
        assert_eq!(generate_binary_permutation::<2>(2), [true, false]);
        assert_eq!(generate_binary_permutation::<2>(3), [true, true]);
    }

    #[test]
    fn test_generate_binary_permutation_3d()
    {
        assert_eq!(generate_binary_permutation::<3>(0), [false, false, false]);
        assert_eq!(generate_binary_permutation::<3>(1), [false, false, true]);
        assert_eq!(generate_binary_permutation::<3>(2), [false, true, false]);
        assert_eq!(generate_binary_permutation::<3>(3), [false, true, true]);
        assert_eq!(generate_binary_permutation::<3>(4), [true, false, false]);
        assert_eq!(generate_binary_permutation::<3>(5), [true, false, true]);
        assert_eq!(generate_binary_permutation::<3>(6), [true, true, false]);
        assert_eq!(generate_binary_permutation::<3>(7), [true, true, true]);
    }

    #[test]
    fn test_generate_binary_permutation_4d()
    {
        assert_eq!(
            generate_binary_permutation::<4>(0),
            [false, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(1),
            [false, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(2),
            [false, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(3),
            [false, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(4),
            [false, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(5),
            [false, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(6),
            [false, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(7),
            [false, true, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(8),
            [true, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(9),
            [true, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(10),
            [true, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(11),
            [true, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(12),
            [true, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(13),
            [true, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<4>(14),
            [true, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<4>(15),
            [true, true, true, true]
        );
    }

    #[test]
    fn test_generate_binary_permutation_5d()
    {
        assert_eq!(
            generate_binary_permutation::<5>(0),
            [false, false, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(1),
            [false, false, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(2),
            [false, false, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(3),
            [false, false, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(4),
            [false, false, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(5),
            [false, false, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(6),
            [false, false, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(7),
            [false, false, true, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(8),
            [false, true, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(9),
            [false, true, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(10),
            [false, true, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(11),
            [false, true, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(12),
            [false, true, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(13),
            [false, true, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(14),
            [false, true, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(15),
            [false, true, true, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(16),
            [true, false, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(17),
            [true, false, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(18),
            [true, false, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(19),
            [true, false, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(20),
            [true, false, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(21),
            [true, false, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(22),
            [true, false, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(23),
            [true, false, true, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(24),
            [true, true, false, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(25),
            [true, true, false, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(26),
            [true, true, false, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(27),
            [true, true, false, true, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(28),
            [true, true, true, false, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(29),
            [true, true, true, false, true]
        );
        assert_eq!(
            generate_binary_permutation::<5>(30),
            [true, true, true, true, false]
        );
        assert_eq!(
            generate_binary_permutation::<5>(31),
            [true, true, true, true, true]
        );
    }
}

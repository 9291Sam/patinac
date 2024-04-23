use core::hash;
use std::fmt::Debug;
use std::hash::Hasher;

use num::{Float, FromPrimitive};

// Kelsie Loquavian??? in my code!!!
/// A single "phase" of the noise, each of these generators produces continuous
/// noise in some vector space
/// TODO: make generic over dimensions
#[derive(Clone)]
pub struct Signet2D
{
    seed: u64
}

impl Debug for Signet2D
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Signet (Kelsie Loquavian)")
    }
}

impl Signet2D
{
    const DIM: usize = 2;

    pub fn new(seed: u64) -> Signet2D
    {
        Self {
            seed
        }
    }

    #[inline(always)]
    // Given a point in some L dimensional space, returns a sample of noise that is
    // continuous in that space
    pub fn sample<F: Float + FromPrimitive>(&self, pos: [F; Signet2D::DIM]) -> F
    {
        // Start by sampling the integers around the given value,
        // take the ceil and the floor and then get the difference
        // floor, ceil, diff: (i64, i64, F);
        let data: [(i64, i64, F); Signet2D::DIM] = std::array::from_fn(|i| {
            let f = unsafe { pos.get_unchecked(i) };

            let ceil = f.ceil();
            let floor = f.floor();

            (
                floor.to_i64().expect("NANL"),
                ceil.to_i64().expect("NANR"),
                ceil - floor
            )
        });

        // the weights along each dimension
        let sample: [F; Signet2D::DIM] =
            std::array::from_fn(|i| unsafe { data.get_unchecked(i).2 });

        const PROD: usize = 2usize.pow(Signet2D::DIM as u32);

        // the values of each coordinate around the sample point in binary step
        // i.e 000, 001, 010, ... around the sample point, and then calculates the value
        // at that integer position
        let cartesian_prod_array: [F; PROD] = std::array::from_fn(|i| {
            let coord: [bool; Signet2D::DIM] = generate_binary_permutation::<{ Signet2D::DIM }>(i);

            let sample_point: [i64; Signet2D::DIM] = std::array::from_fn(|i| {
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

        // 00, 01, 10, 11
        smoothstep(
            smoothstep(
                cartesian_prod_array[0b00],
                cartesian_prod_array[0b01],
                data[0].2
            ),
            smoothstep(
                cartesian_prod_array[0b10],
                cartesian_prod_array[0b11],
                data[0].2
            ),
            data[1].2
        )

        // smoothstep(e1, e2, x)

        // // TODO: using those integer positions around the sample points, and
        // then the // weights saying how much each dimension should
        // matter (which also represent // how close / far the sample
        // point is from one of the integer boundaries) // Interpolate
        // the data into a single float //
        // nd_smoothstep(&cartesian_prod_array, &sample)
    }

    #[inline(always)]
    pub fn sample_integer(&self, pos: [i64; Signet2D::DIM]) -> u64
    {
        let mut hasher = std::hash::DefaultHasher::new();
        hasher.write_u64(self.seed);

        for p in pos
        {
            hasher.write_i64(p);
        }

        hasher.finish()
    }

    #[inline(always)]
    pub fn sample_float<F: Float + FromPrimitive>(&self, pos: [i64; Signet2D::DIM]) -> F
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

#[inline(always)]
pub fn smoothstep<F: Float>(e1: F, e2: F, mut x: F) -> F
{
    x = clamp((x - e1) / (e2 - e1), F::zero(), F::one());

    let two: F = F::one() + F::one();
    let three: F = two + F::one();

    return x * x * (three - two * x);
}

fn clamp<F: Float>(f: F, low: F, high: F) -> F
{
    if f < low
    {
        low
    }
    else if f > high
    {
        high
    }
    else
    {
        f
    }
}
// function generalSmoothStep(N, x) {
//     x = clamp(x, 0, 1); // x must be equal to or between 0 and 1
//     var result = 0;
//     for (var n = 0; n <= N; ++n)
//       result += pascalTriangle(-N - 1, n) *
//                 pascalTriangle(2 * N + 1, N - n) *
//                 Math.pow(x, N + n + 1);
//     return result;
//   }

//   // Returns binomial coefficient without explicit use of factorials,
//   // which can't be used with negative integers

// function pascalTriangle(a, b) {
//     var result = 1;
//     for (var i = 0; i < b; ++i)
//       result *= (a - i) / (i + 1);
//     return result;
//   }

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

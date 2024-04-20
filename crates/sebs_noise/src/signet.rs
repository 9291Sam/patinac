use std::fmt::Debug;
use std::mem::MaybeUninit;

use bytemuck::NoUninit;
use num::{Float, FromPrimitive};

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
    pub fn sample<F: Float>(pos: [F; L])
    {
        let data: [(i64, i64); L] = std::array::from_fn(|i| {
            let f = unsafe { pos.get_unchecked(i) };

            (
                f.floor().to_i64().expect("NANL"),
                f.ceil().to_i64().expect("NANR")
            )
        });

        // make ndimesnsional cartesian prpdocut
        // 2 4 8
        let cartesian_prod_array: [F; 2u32.exp(L)];

        // for each in cartesian prod array
        // generate coordinates
        // write res

        // interp...
    }

    #[inline(always)]
    pub fn sample_integer<I>(&self, pos: [i64; L]) -> u64
    {
        util::hash_combine(self.seed, unsafe {
            std::slice::from_raw_parts(pos.as_ptr() as *const u8, std::mem::size_of_val(&pos))
        })
    }

    pub fn sample_f32(&self, pos: [i64; L]) -> f32
    {
        util::map_f32(
            self.sample_integer::<f32>(pos) as f32,
            0.0,
            u64::MAX as f32,
            0.0,
            1.0
        )
    }

    pub fn sample_f64(&self, pos: [i64; L]) -> f64
    {
        util::map_f64(
            self.sample_integer::<f64>(pos) as f64,
            0.0,
            u64::MAX as f64,
            0.0,
            1.0
        )
    }
}

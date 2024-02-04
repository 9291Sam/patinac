#![feature(map_try_insert)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_as_bytes)]

mod allocator;
mod log;
mod registrar;
mod uuid;

use std::mem::MaybeUninit;
use std::ops::{BitAnd, BitXor, Not, Shr};
use std::ptr::addr_of;

pub use allocator::*;
pub use log::*;
pub use registrar::*;
pub use uuid::*;

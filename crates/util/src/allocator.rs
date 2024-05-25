use std::collections::{BTreeSet, HashSet};

use bytemuck::Contiguous;

#[derive(Debug)]
pub struct FreelistAllocator
{
    free_blocks:     Vec<usize>,
    total_blocks:    usize,
    next_free_block: usize,

    #[cfg(debug_assertions)]
    allocated_blocks: BTreeSet<usize>
}

/// Valid Blocks [0, total_blocks]
impl FreelistAllocator
{
    pub fn new(size: usize) -> Self
    {
        FreelistAllocator {
            free_blocks:                               Vec::new(),
            total_blocks:                              size,
            next_free_block:                           0,
            #[cfg(debug_assertions)]
            allocated_blocks:                          BTreeSet::new()
        }
    }

    pub fn has_next(&self) -> bool
    {
        !self.free_blocks.is_empty() || self.next_free_block < self.total_blocks
    }

    // (Free blocks , total blocks)
    pub fn peek(&self) -> (usize, usize)
    {
        let used_blocks =
            self.total_blocks - (self.total_blocks - self.next_free_block + self.free_blocks.len());

        (used_blocks, self.total_blocks)
    }

    pub fn allocate(&mut self) -> Result<usize, OutOfBlocks>
    {
        match self.free_blocks.pop()
        {
            Some(free_block) =>
            {
                let ret_val = free_block;

                #[cfg(debug_assertions)]
                self.allocated_blocks.insert(ret_val);

                Ok(ret_val)
            }
            None =>
            {
                if self.next_free_block >= self.total_blocks
                {
                    Err(OutOfBlocks)
                }
                else
                {
                    let block = self.next_free_block;

                    self.next_free_block += 1;

                    #[cfg(debug_assertions)]
                    debug_assert!(self.allocated_blocks.insert(block));

                    Ok(block)
                }
            }
        }
    }

    /// # Safety
    /// You must only free an integer allocated by this allocator
    pub unsafe fn free(&mut self, block: usize)
    {
        #[cfg(debug_assertions)]
        if !self.allocated_blocks.remove(&block)
        {
            log::warn!("free of untracked value {block}");
            return;
        }

        self.free_blocks.push(block)
    }

    pub fn get_total_blocks(&self) -> usize
    {
        self.total_blocks.into_integer()
    }

    pub fn extend_size(&mut self, new_cap: usize)
    {
        debug_assert!(new_cap > self.total_blocks.into_integer());

        self.total_blocks = new_cap;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OutOfBlocks;

// #[cfg(test)]
// mod tests
// {
//     #[allow(unused_imports)]
//     use super::*;

//     #[test]
//     pub fn alloc()
//     {
//         let mut allocator = FreelistAllocator::new(5);

//         assert_eq!(1, allocator.allocate().unwrap());
//         assert_eq!(2, allocator.allocate().unwrap());
//         assert_eq!(3, allocator.allocate().unwrap());
//         assert_eq!(4, allocator.allocate().unwrap());
//         assert_eq!(5, allocator.allocate().unwrap());
//         assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
//     }

//     #[test]
//     pub fn free()
//     {
//         let mut allocator = FreelistAllocator::new(5);

//         assert_eq!(1, allocator.allocate().unwrap());
//         assert_eq!(2, allocator.allocate().unwrap());
//         assert_eq!(3, allocator.allocate().unwrap());
//         assert_eq!(4, allocator.allocate().unwrap());
//         assert_eq!(5, allocator.allocate().unwrap());
//         assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());

//         unsafe { allocator.free(5) };
//         unsafe { allocator.free(2) };
//         unsafe { allocator.free(1) };

//         for _ in 0..3
//         {
//             let _ = allocator.allocate();
//         }

//         assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
//     }

//     #[test]
//     pub fn drain()
//     {
//         let mut allocator = FreelistAllocator::new(2);

//         assert_eq!(1, allocator.allocate().unwrap());
//         assert_eq!(2, allocator.allocate().unwrap());

//         unsafe { allocator.free(1) };
//         unsafe { allocator.free(2) };

//         let _ = allocator.allocate().unwrap();
//         let _ = allocator.allocate().unwrap();
//     }

//     #[test]
//     pub fn out_of_blocks()
//     {
//         let mut allocator = FreelistAllocator::new(1);

//         assert_eq!(1, allocator.allocate().unwrap());

//         assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
//         assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
//     }

//     #[test]
//     #[should_panic]
//     pub fn free_untracked()
//     {
//         let mut allocator = FreelistAllocator::new(1);

//         assert_eq!(1, allocator.allocate().unwrap());
//         unsafe { allocator.free(2) };
//     }

//     #[test]
//     #[should_panic]
//     pub fn double_free()
//     {
//         let mut allocator = FreelistAllocator::new(1);

//         assert_eq!(1, allocator.allocate().unwrap());
//         unsafe { allocator.free(1) };

//         unsafe { allocator.free(1) };
//     }

//     #[test]
//     pub fn extend()
//     {
//         let mut allocator = FreelistAllocator::new(1);

//         assert_eq!(1, allocator.allocate().unwrap());
//         allocator.extend_size(2);
//         assert_eq!(2, allocator.allocate().unwrap());
//         assert_eq!(Err(OutOfBlocks), allocator.allocate());
//     }
// }

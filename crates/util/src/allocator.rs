use std::collections::HashSet;

use bytemuck::Contiguous;

#[derive(Debug)]
pub struct FreelistAllocator
{
    free_blocks:     Vec<usize>,
    total_blocks:    usize,
    next_free_block: usize
}

/// Valid Blocks [0, total_blocks]
impl FreelistAllocator
{
    pub fn new(size: usize) -> Self
    {
        FreelistAllocator {
            free_blocks:     Vec::new(),
            total_blocks:    size,
            next_free_block: 0
        }
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
            Some(free_block) => Ok(free_block),
            None =>
            {
                if self.next_free_block >= self.total_blocks
                {
                    Err(OutOfBlocks)
                }
                else
                {
                    let block = self.next_free_block;

                    self.next_free_block = self.next_free_block.checked_add(1).unwrap();

                    Ok(block)
                }
            }
        }
    }

    /// # Safety
    /// You must only free an integer allocated by this allocator
    pub unsafe fn free(&mut self, block: usize)
    {
        if block > self.total_blocks
        {
            panic!("FreeListAllocator Free of Untracked Value")
        };

        self.free_blocks.push(block)
    }

    pub fn get_total_blocks(&self) -> usize
    {
        self.total_blocks.into_integer()
    }

    pub fn extend_size(&mut self, new_cap: usize)
    {
        assert!(new_cap > self.total_blocks.into_integer());

        self.total_blocks = new_cap;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OutOfBlocks;

#[cfg(test)]
mod tests
{
    #[allow(unused_imports)]
    use super::*;

    #[test]
    pub fn alloc()
    {
        let mut allocator = FreelistAllocator::new(5);

        assert_eq!(1, allocator.allocate().unwrap());
        assert_eq!(2, allocator.allocate().unwrap());
        assert_eq!(3, allocator.allocate().unwrap());
        assert_eq!(4, allocator.allocate().unwrap());
        assert_eq!(5, allocator.allocate().unwrap());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    pub fn free()
    {
        let mut allocator = FreelistAllocator::new(5);

        assert_eq!(1, allocator.allocate().unwrap());
        assert_eq!(2, allocator.allocate().unwrap());
        assert_eq!(3, allocator.allocate().unwrap());
        assert_eq!(4, allocator.allocate().unwrap());
        assert_eq!(5, allocator.allocate().unwrap());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());

        unsafe { allocator.free(5) };
        unsafe { allocator.free(2) };
        unsafe { allocator.free(1) };

        for _ in 0..3
        {
            let _ = allocator.allocate();
        }

        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    pub fn drain()
    {
        let mut allocator = FreelistAllocator::new(2);

        assert_eq!(1, allocator.allocate().unwrap());
        assert_eq!(2, allocator.allocate().unwrap());

        unsafe { allocator.free(1) };
        unsafe { allocator.free(2) };

        let _ = allocator.allocate().unwrap();
        let _ = allocator.allocate().unwrap();
    }

    #[test]
    pub fn out_of_blocks()
    {
        let mut allocator = FreelistAllocator::new(1);

        assert_eq!(1, allocator.allocate().unwrap());

        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    #[should_panic]
    pub fn free_untracked()
    {
        let mut allocator = FreelistAllocator::new(1);

        assert_eq!(1, allocator.allocate().unwrap());
        unsafe { allocator.free(2) };
    }

    #[test]
    #[should_panic]
    pub fn double_free()
    {
        let mut allocator = FreelistAllocator::new(1);

        assert_eq!(1, allocator.allocate().unwrap());
        unsafe { allocator.free(1) };

        unsafe { allocator.free(1) };
    }

    #[test]
    pub fn extend()
    {
        let mut allocator = FreelistAllocator::new(1);

        assert_eq!(1, allocator.allocate().unwrap());
        allocator.extend_size(2);
        assert_eq!(2, allocator.allocate().unwrap());
        assert_eq!(Err(OutOfBlocks), allocator.allocate());
    }
}

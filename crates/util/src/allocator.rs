use std::collections::HashSet;
use std::num::NonZeroUsize;

use bytemuck::Contiguous;

#[derive(Debug)]
pub struct FreelistAllocator
{
    free_blocks:     HashSet<NonZeroUsize>,
    total_blocks:    NonZeroUsize,
    next_free_block: NonZeroUsize
}

/// Valid Blocks [1, total_blocks]
impl FreelistAllocator
{
    pub fn new(size: NonZeroUsize) -> Self
    {
        FreelistAllocator {
            free_blocks:     HashSet::new(),
            total_blocks:    size,
            next_free_block: NonZeroUsize::new(1).unwrap()
        }
    }

    // (Free blocks , total blocks)
    pub fn peek(&self) -> (NonZeroUsize, NonZeroUsize)
    {
        let used_blocks = self.total_blocks.into_integer()
            - (self.total_blocks.into_integer() - self.next_free_block.into_integer()
                + self.free_blocks.len());

        (NonZeroUsize::new(used_blocks).unwrap(), self.total_blocks)
    }

    pub fn allocate(&mut self) -> Result<NonZeroUsize, OutOfBlocks>
    {
        let maybe_free_block = self.free_blocks.iter().next().cloned();

        match maybe_free_block
        {
            Some(block) =>
            {
                debug_assert!(self.free_blocks.remove(&block));
                Ok(block)
            }
            None =>
            {
                if self.next_free_block > self.total_blocks
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

    pub fn free(&mut self, block: NonZeroUsize)
    {
        if block > self.total_blocks
        {
            panic!("FreeListAllocator Free of Untracked Value")
        };

        match self.free_blocks.insert(block)
        {
            true => (),
            false => panic!("FreeListAllocator Double Free!")
        }

        // TODO: optimize
        for b in (1..=self.next_free_block.into_integer() - 1).rev()
        {
            let block = NonZeroUsize::new(b).unwrap();

            if self.free_blocks.contains(&block)
            {
                self.free_blocks.remove(&block);

                self.next_free_block = block;
            }
            else
            {
                break;
            }
        }
    }

    pub fn get_total_blocks(&self) -> usize
    {
        self.total_blocks.into_integer()
    }

    pub fn extend_size(&mut self, new_cap: usize)
    {
        assert!(new_cap > self.total_blocks.into_integer());

        self.total_blocks = new_cap.try_into().unwrap();
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
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(5).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(2).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(3).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(4).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(5).unwrap(), allocator.allocate().unwrap());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    pub fn free()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(5).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(2).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(3).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(4).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(5).unwrap(), allocator.allocate().unwrap());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());

        allocator.free(NonZeroUsize::new(5).unwrap());
        allocator.free(NonZeroUsize::new(2).unwrap());
        allocator.free(NonZeroUsize::new(1).unwrap());

        for _ in 0..3
        {
            let _ = allocator.allocate();
        }

        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    pub fn drain()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(2).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        assert_eq!(NonZeroUsize::new(2).unwrap(), allocator.allocate().unwrap());

        allocator.free(NonZeroUsize::new(1).unwrap());
        allocator.free(NonZeroUsize::new(2).unwrap());

        let _ = allocator.allocate().unwrap();
        let _ = allocator.allocate().unwrap();
    }

    #[test]
    pub fn out_of_blocks()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(1).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());

        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
        assert_eq!(OutOfBlocks, allocator.allocate().unwrap_err());
    }

    #[test]
    #[should_panic]
    pub fn free_untracked()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(1).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        allocator.free(NonZeroUsize::new(2).unwrap());
    }

    #[test]
    #[should_panic]
    pub fn double_free()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(1).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        allocator.free(NonZeroUsize::new(1).unwrap());

        allocator.free(NonZeroUsize::new(1).unwrap());
    }

    #[test]
    pub fn extend()
    {
        let mut allocator = FreelistAllocator::new(NonZeroUsize::new(1).unwrap());

        assert_eq!(NonZeroUsize::new(1).unwrap(), allocator.allocate().unwrap());
        allocator.extend_size(2);
        assert_eq!(NonZeroUsize::new(2).unwrap(), allocator.allocate().unwrap());
        assert_eq!(Err(OutOfBlocks), allocator.allocate());
    }
}

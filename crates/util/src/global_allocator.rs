use std::alloc::{Allocator, Layout, System};
use std::isize;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL_ALLOCATOR: GlobalAllocator = GlobalAllocator {
    bytes_allocated_current: AtomicUsize::new(0)
};

pub fn get_bytes_of_active_allocations() -> usize
{
    GLOBAL_ALLOCATOR
        .bytes_allocated_current
        .load(Ordering::Relaxed)
}
struct GlobalAllocator
{
    bytes_allocated_current: AtomicUsize
}

unsafe impl std::alloc::GlobalAlloc for GlobalAllocator
{
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8
    {
        self.bytes_allocated_current
            .fetch_add(layout.size(), Ordering::Relaxed);

        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout)
    {
        self.bytes_allocated_current
            .fetch_sub(layout.size(), Ordering::Relaxed);

        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8
    {
        let maybe_allocation = self.alloc(layout);

        if !maybe_allocation.is_null()
        {
            maybe_allocation.write_bytes(0x00, layout.size());
        }

        maybe_allocation
    }

    unsafe fn realloc(
        &self,
        maybe_ptr: *mut u8,
        old_layout: std::alloc::Layout,
        new_size: usize
    ) -> *mut u8
    {
        match NonNull::new(maybe_ptr)
        {
            Some(ptr) =>
            {
                let old_size = old_layout.size();

                let alloc_difference: isize = new_size as isize - old_size as isize;

                match alloc_difference
                {
                    d if d < 0 =>
                    {
                        self.bytes_allocated_current
                            .fetch_sub(alloc_difference.unsigned_abs(), Ordering::Relaxed);
                    }
                    0 =>
                    {}
                    d if d > 0 =>
                    {
                        self.bytes_allocated_current
                            .fetch_add(alloc_difference.unsigned_abs(), Ordering::Relaxed);
                    }
                    _ => unreachable!()
                }

                if new_size < old_layout.size()
                {
                    return ptr.as_ptr();
                }

                match System.grow(
                    ptr,
                    old_layout,
                    Layout::from_size_align_unchecked(new_size, old_layout.align())
                )
                {
                    Ok(ptr) => ptr.as_non_null_ptr().as_ptr(),
                    Err(_) => std::ptr::null_mut()
                }
            }
            None => std::ptr::null_mut()
        }
    }
}

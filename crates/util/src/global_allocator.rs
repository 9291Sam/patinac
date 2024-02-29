use std::alloc::System;
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL_ALLOCATOR: GlobalAllocator = GlobalAllocator {
    bytes_allocated_current: AtomicUsize::new(0),
    bytes_allocated_total:   AtomicUsize::new(0)
};

pub fn get_bytes_of_active_allocations() -> usize
{
    GLOBAL_ALLOCATOR
        .bytes_allocated_current
        .load(Ordering::Relaxed)
}
pub fn get_bytes_allocated_total() -> usize
{
    GLOBAL_ALLOCATOR
        .bytes_allocated_total
        .load(Ordering::Relaxed)
}

struct GlobalAllocator
{
    bytes_allocated_current: AtomicUsize,
    bytes_allocated_total:   AtomicUsize
}

unsafe impl std::alloc::GlobalAlloc for GlobalAllocator
{
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8
    {
        self.bytes_allocated_current
            .fetch_add(layout.size(), Ordering::Relaxed);

        self.bytes_allocated_total
            .fetch_add(layout.size(), Ordering::Relaxed);

        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout)
    {
        self.bytes_allocated_current
            .fetch_sub(layout.size(), Ordering::Relaxed);

        System.dealloc(ptr, layout)
    }
}

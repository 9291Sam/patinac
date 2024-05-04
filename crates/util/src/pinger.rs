use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub fn create_pingers() -> (PingSender, PingReceiver)
{
    let block = Arc::new(ControlBlock {
        current_count: AtomicU64::new(0)
    });

    (
        PingSender {
            block: block.clone()
        },
        PingReceiver {
            block,
            seen: AtomicU64::new(0)
        }
    )
}

#[derive(Debug)]
pub struct PingSender
{
    block: Arc<ControlBlock>
}

impl PingSender
{
    pub fn ping(&self)
    {
        self.block.current_count.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub struct PingReceiver
{
    block: Arc<ControlBlock>,
    seen:  AtomicU64
}

impl Clone for PingReceiver
{
    fn clone(&self) -> Self
    {
        Self {
            seen:  AtomicU64::new(self.seen.load(Ordering::SeqCst)),
            block: self.block.clone()
        }
    }
}

impl PingReceiver
{
    pub fn recv_one(&self) -> bool
    {
        let current_event: u64 = self.block.current_count.load(Ordering::SeqCst);

        if current_event > self.seen.load(Ordering::SeqCst)
        {
            self.seen.fetch_add(1, Ordering::SeqCst);
            true
        }
        else
        {
            false
        }
    }

    pub fn recv_all(&self) -> bool
    {
        let current_event: u64 = self.block.current_count.load(Ordering::SeqCst);

        if current_event > self.seen.load(Ordering::SeqCst)
        {
            self.seen.store(current_event, Ordering::SeqCst);
            true
        }
        else
        {
            false
        }
    }
}

#[derive(Debug)]
struct ControlBlock
{
    current_count: AtomicU64
}

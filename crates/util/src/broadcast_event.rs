use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

use dashmap::DashMap;

pub struct Broadcaster<T: Clone>(Arc<BroadcasterInternal<T>>);
pub struct BroadcasterReceiver<T: Clone>(Arc<BroadcasterReceiverInternal<T>>);

impl<T: Clone> Broadcaster<T>
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Broadcaster<T>
    {
        Broadcaster(Arc::new(BroadcasterInternal {
            receivers: DashMap::new()
        }))
    }

    pub fn broadcast(&self, val: T)
    {
        self.0.receivers.retain(|_, maybe_receiver| {
            if let Some(strong_receiver) = maybe_receiver.upgrade()
            {
                strong_receiver
                    .buffer
                    .lock()
                    .unwrap()
                    .push_front(val.clone());
                true
            }
            else
            {
                false
            }
        });
    }

    pub fn create_receiver(&self) -> BroadcasterReceiver<T>
    {
        let unregistered_receiver = BroadcasterReceiver(Arc::new(BroadcasterReceiverInternal {
            broadcaster: Arc::downgrade(&self.0),
            buffer:      Mutex::new(VecDeque::new())
        }));

        self.0
            .receivers
            .insert(super::Uuid::new(), Arc::downgrade(&unregistered_receiver.0));

        unregistered_receiver
    }
}

pub enum BroadcasterReceiverError
{
    ReceiverDestroyed,
    NoMoreElements
}

impl<T: Clone> BroadcasterReceiver<T>
{
    pub fn try_recv(&self) -> Result<T, BroadcasterReceiverError>
    {
        if let Some(t) = self.0.buffer.lock().unwrap().pop_back()
        {
            Ok(t)
        }
        else if self.0.broadcaster.upgrade().is_some()
        {
            Err(BroadcasterReceiverError::NoMoreElements)
        }
        else
        {
            Err(BroadcasterReceiverError::ReceiverDestroyed)
        }
    }

    pub fn try_clone(&self) -> Option<BroadcasterReceiver<T>>
    {
        Some(Broadcaster(self.0.broadcaster.upgrade()?).create_receiver())
    }
}

struct BroadcasterReceiverInternal<T: Clone>
{
    broadcaster: Weak<BroadcasterInternal<T>>,
    buffer:      Mutex<VecDeque<T>>
}

struct BroadcasterInternal<T: Clone>
{
    receivers: DashMap<super::Uuid, Weak<BroadcasterReceiverInternal<T>>>
}

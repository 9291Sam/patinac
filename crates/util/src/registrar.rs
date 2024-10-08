use std::collections::hash_map::OccupiedError;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

#[derive(Debug)]
pub struct Registrar<K: Hash + Eq, V>
{
    update_sender:    Sender<UpdateType<K, V>>,
    critical_section: Mutex<CriticalSection<K, V>>
}

impl<K, V> Registrar<K, V>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone
{
    pub fn new() -> Self
    {
        let (sender, receiver) = mpsc::channel();

        Self {
            update_sender:    sender,
            critical_section: Mutex::new(CriticalSection {
                update_receiver: receiver,
                storage:         HashMap::new()
            })
        }
    }

    pub fn access(&self) -> Vec<(K, V)>
    {
        let mut guard = self.critical_section.lock().unwrap();
        let CriticalSection {
            update_receiver,
            storage
        } = &mut *guard;

        while let Ok(update) = update_receiver.try_recv()
        {
            match update
            {
                UpdateType::Insertion(k, v) =>
                {
                    if let Err(OccupiedError {
                        entry,
                        value
                    }) = storage.try_insert(k, v)
                    {
                        log::warn!(
                            "Unexpected duplicate insertion of ({:?}, {:?}) in Registrar<{}, {}>",
                            entry.key(),
                            value,
                            std::any::type_name::<K>(),
                            std::any::type_name::<V>()
                        );
                    }
                }
                UpdateType::Deletion(k) =>
                {
                    if storage.remove(&k).is_none()
                    {
                        log::warn!(
                            "Unexpected deletion of {k:?} in Registrar<{}, {}>",
                            std::any::type_name::<K>(),
                            std::any::type_name::<V>()
                        );
                    }
                }
            }
        }

        storage
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn insert(&self, key: K, value: V)
    {
        self.update_sender
            .send(UpdateType::Insertion(key, value))
            .unwrap()
    }

    pub fn delete(&self, key: K)
    {
        self.update_sender.send(UpdateType::Deletion(key)).unwrap()
    }
}

impl<K, V> Default for Registrar<K, V>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone
{
    fn default() -> Self
    {
        Registrar::new()
    }
}

enum UpdateType<K, V>
{
    Insertion(K, V),
    Deletion(K)
}

#[derive(Debug)]
struct CriticalSection<K, V>
{
    update_receiver: Receiver<UpdateType<K, V>>,
    storage:         HashMap<K, V>
}

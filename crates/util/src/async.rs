use std::any::{Any, TypeId};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::RwLock;
use std::thread::JoinHandle;

use crossbeam::channel::{Receiver, Sender};

#[derive(Debug)]
pub struct Future<T: Send>
where
    Self: Send
{
    resolved:          AtomicBool,
    receiver:          oneshot::Receiver<T>,
    #[cfg(debug_assertions)]
    creation_location: std::panic::Location<'static>
}

impl<T: Send> Drop for Future<T>
{
    fn drop(&mut self)
    {
        if !self.resolved.load(SeqCst)
        {
            // The previous call does noy synchronize with this, the future may now be
            // fulfilled
            match self.receiver.try_recv()
            {
                // Ok, the future was fulfilled in the literal microseconds since we last checked
                Ok(_t) => (),
                Err(e) =>
                {
                    match e
                    {
                        oneshot::TryRecvError::Empty =>
                        {
                            // we haven't received a message yet, let's wait for it\
                            #[cfg(not(debug_assertions))]
                            log::warn!("Tried dropping an unresolved future, attaching!");

                            #[cfg(debug_assertions)]
                            log::warn!(
                                "Tried dropping an unresolved future [{}], attaching!",
                                self.creation_location
                            );

                            let _ = self.get_ref();
                        }
                        oneshot::TryRecvError::Disconnected =>
                        {
                            // Either the sender thread panicked, in which case
                            // we forget  or the message has already been
                            // extracted, in which we also forget
                        }
                    }
                }
            }
        }
    }
}

impl<T: Send> Future<T>
{
    fn new(
        receiver: oneshot::Receiver<T>,
        #[cfg(debug_assertions)] creation_location: std::panic::Location<'static>
    ) -> Future<T>
    {
        Future {
            resolved: AtomicBool::new(false),
            receiver,
            #[cfg(debug_assertions)]
            creation_location
        }
    }

    pub fn get(self) -> T
    {
        self.get_ref()
    }

    fn get_ref(&self) -> T
    {
        match self.receiver.recv_ref()
        {
            Ok(t) =>
            {
                self.resolved.store(true, SeqCst);
                t
            }
            Err(_) => unreachable!("No message was sent!")
        }
    }

    pub fn poll(self) -> Result<T, Self>
    {
        match self.poll_ref()
        {
            Some(t) => Ok(t),
            None => Err(self)
        }
    }

    fn poll_ref(&self) -> Option<T>
    {
        match self.receiver.try_recv()
        {
            Ok(t) =>
            {
                self.resolved.store(true, SeqCst);
                Some(t)
            }
            Err(e) =>
            {
                match e
                {
                    oneshot::TryRecvError::Empty => None,
                    oneshot::TryRecvError::Disconnected => unreachable!("No message was sent!")
                }
            }
        }
    }

    pub fn detach(&self)
    {
        self.resolved.store(true, SeqCst);
    }
}

#[derive(Debug)]
pub enum Promise<T: Send>
where
    Self: Send
{
    Resolved(T),
    Pending(Future<T>)
}

impl<T: Send> From<Future<T>> for Promise<T>
{
    fn from(f: Future<T>) -> Self
    {
        Promise::Pending(f)
    }
}

impl<T: Send> Promise<T>
{
    pub fn poll(self) -> Promise<T>
    {
        match self
        {
            Promise::Resolved(t) => Promise::Resolved(t),
            Promise::Pending(future) =>
            {
                match future.poll()
                {
                    Ok(t) => Promise::Resolved(t),
                    Err(unresolved_future) => Promise::Pending(unresolved_future)
                }
            }
        }
    }

    pub fn poll_ref(&mut self)
    {
        *self = match self
        {
            Promise::Resolved(_) => return,
            Promise::Pending(future) =>
            {
                match future.poll_ref()
                {
                    Some(t) => Promise::Resolved(t),
                    None => return
                }
            }
        }
    }
}

#[track_caller]
pub fn run_async<T, F>(func: F) -> Future<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static
{
    access_global_thread_pool()
        .read()
        .expect("Global threadpool was poisoned!")
        .as_ref()
        .expect("No threadpool was available")
        .run_async(func)
}

pub fn access_global_thread_pool() -> &'static RwLock<Option<ThreadPool>>
{
    &THREAD_POOL
}

static THREAD_POOL: RwLock<Option<ThreadPool>> = RwLock::new(None);

pub struct ThreadPool
{
    threads: Vec<JoinHandle<()>>,
    sender:  Sender<Box<dyn FnOnce() + Send>>
}

impl ThreadPool
{
    #[allow(clippy::new_without_default)]
    pub fn new(number_of_workers: usize, name: &str) -> ThreadPool
    {
        let (sender, receiver) = crossbeam::channel::unbounded();

        let threads = (0..number_of_workers)
            .map(|idx| {
                let this_receiver: Receiver<Box<dyn FnOnce() + Send>> = receiver.clone();

                std::thread::Builder::new()
                    .name(format!("{name} Thread #{idx}"))
                    .spawn(move || {
                        while let Ok(func) = this_receiver.recv()
                        {
                            func()
                        }
                    })
                    .unwrap()
            })
            .collect();

        ThreadPool {
            sender,
            threads
        }
    }

    #[track_caller]
    #[must_use]
    pub fn run_async<T, F>(&self, func: F) -> Future<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static
    {
        let (sender, receiver) = oneshot::channel();

        let caller = std::panic::Location::caller();

        #[cfg(debug_assertions)]
        let future = Future::new(receiver, *caller);

        #[cfg(not(debug_assertions))]
        let future = Future::new(receiver);

        self.enqueue_function(move || {
            if let Err(result) = sender.send(func())
            {
                if result.into_inner().type_id() != TypeId::of::<()>()
                {
                    log::warn!(
                        "Sent notification to dropped Future<{}> from [{}]",
                        std::any::type_name::<T>(),
                        caller
                    );
                }
            }
        });

        future
    }

    pub fn enqueue_function(&self, func: impl FnOnce() + Send + 'static)
    {
        if let Err(func) = self.sender.send(Box::new(func))
        {
            log::warn!("Tried to enqueue a function on a threadpool with closed threads!");

            func.0();
        }
    }

    pub fn join_threads(mut self)
    {
        std::mem::drop(self.sender);

        log::info!("Stopping worker threads");

        self.threads.drain(..).for_each(|t| {
            let _ = t.join();
        });
    }
}

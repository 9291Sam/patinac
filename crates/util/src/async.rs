use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Mutex, RwLock};
use std::thread::JoinHandle;

use bytemuck::Contiguous;
use crossbeam::channel::{Receiver, Sender};

#[derive(Debug)]
pub struct Future<T: Send>
where
    Self: Send + Sync
{
    resolved: AtomicBool,
    receiver: Mutex<oneshot::Receiver<T>>
}

impl<T: Send> Drop for Future<T>
{
    fn drop(&mut self)
    {
        if !self.resolved.load(SeqCst)
        {
            panic!("Tried dropping an unresolved future!")
        }
    }
}

impl<T: Send> Future<T>
{
    fn new(receiver: oneshot::Receiver<T>) -> Future<T>
    {
        Future {
            resolved: AtomicBool::new(false),
            receiver: Mutex::new(receiver)
        }
    }

    fn get(mut self) -> T
    {
        loop
        {
            match self.poll()
            {
                Ok(t) => return t,
                Err(me) => self = me
            }
        }
    }

    fn poll(mut self) -> Result<T, Self>
    {
        match self.poll_mut()
        {
            Some(t) => Ok(t),
            None => Err(self)
        }
    }

    fn poll_mut(&mut self) -> Option<T>
    {
        match self.receiver.lock().unwrap().try_recv()
        {
            Ok(t) =>
            {
                (&self).resolved.store(true, SeqCst);
                Some(t)
            }
            Err(e) =>
            {
                match e
                {
                    oneshot::TryRecvError::Empty => None,
                    oneshot::TryRecvError::Disconnected => unreachable!()
                }
            }
        }
    }

    fn detach(self)
    {
        self.resolved.store(true, SeqCst);
        std::mem::drop(self);
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
                match future.poll_mut()
                {
                    Some(t) => Promise::Resolved(t),
                    None => return
                }
            }
        }
    }
}

pub fn run_async<T, F>(func: F) -> Future<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static
{
    let (sender, receiver) = oneshot::channel();

    let future = Future::new(receiver);

    THREAD_POOL
        .read()
        .unwrap()
        .as_ref()
        .unwrap()
        .enqueue_function(|| sender.send(func()).unwrap());

    future
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
    pub fn new() -> ThreadPool
    {
        let (sender, receiver) = crossbeam::channel::unbounded();

        let threads = (0..std::thread::available_parallelism().unwrap().into_integer())
            .map(|_| {
                let this_receiver: Receiver<Box<dyn FnOnce() + Send>> = receiver.clone();

                std::thread::spawn(move || {
                    while let Ok(func) = this_receiver.recv()
                    {
                        func()
                    }
                })
            })
            .collect();

        ThreadPool {
            sender,
            threads
        }
    }

    fn enqueue_function(&self, func: impl FnOnce() + Send + 'static)
    {
        if self.sender.send(Box::new(func)).is_err()
        {
            panic!("Tried to enqueue a function on a threadpool with closed threads!")
        }
    }

    fn join_threads(&mut self)
    {
        self.threads.drain(..).for_each(|t| t.join().unwrap())
    }
}

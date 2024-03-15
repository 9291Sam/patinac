use std::any::Any;
use std::collections::HashMap;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;

use chrono::Utc;

use crate::Uuid;

pub struct CrashHandlerManagedLoopSpawner
{
    id:             Uuid,
    name:           String,
    owning_handler: Arc<CrashHandler>
}

impl Drop for CrashHandlerManagedLoopSpawner
{
    fn drop(&mut self)
    {
        self.owning_handler.acknowledge_termination(self.id);
    }
}

impl CrashHandlerManagedLoopSpawner
{
    pub fn enter_managed_loop(self, func: impl Fn() -> TerminationResult + RefUnwindSafe)
    {
        if let Err(e) = std::panic::catch_unwind(|| {
            while self.owning_handler.should_loops_iterate()
            {
                match func()
                {
                    TerminationResult::RequestIteration =>
                    {}
                    TerminationResult::Terminate => break
                }

                self.owning_handler.acknowledge_iteration(self.id);
            }
        })
        {
            self.owning_handler.acknowledge_crash(self.id, e);
        }
    }

    pub fn enter_managed_thread(
        self,
        func: impl Fn() -> TerminationResult + UnwindSafe + RefUnwindSafe + Send + 'static
    )
    {
        let name = self.name.clone();
        let crash_handler = self.owning_handler.clone();

        crash_handler.register_join_handle(
            self.id,
            std::thread::Builder::new()
                .name(self.name.clone())
                .spawn(move || self.enter_managed_loop(func))
                .unwrap_or_else(|_| panic!("Failed to spawn thread {}", name))
        );
    }
}

pub enum TerminationResult
{
    RequestIteration,
    Terminate
}

type PanicPayload = Box<dyn Any + Send>;

pub struct CrashHandler
{
    should_loops_iterate:   AtomicBool,
    loop_acknowledge_times: Mutex<HashMap<Uuid, chrono::DateTime<Utc>>>,
    loop_names:             Mutex<HashMap<Uuid, String>>,
    thread_join_handles:    Mutex<HashMap<Uuid, JoinHandle<()>>>,
    panic_data:             Mutex<HashMap<Uuid, PanicPayload>>
}

impl CrashHandler
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Arc<CrashHandler>
    {
        Arc::new(CrashHandler {
            should_loops_iterate:   AtomicBool::new(true),
            loop_acknowledge_times: Mutex::new(HashMap::new()),
            loop_names:             Mutex::new(HashMap::new()),
            thread_join_handles:    Mutex::new(HashMap::new()),
            panic_data:             Mutex::new(HashMap::new())
        })
    }

    pub fn should_loops_iterate(&self) -> bool
    {
        self.should_loops_iterate.load(SeqCst)
    }

    pub fn create_handle(self: Arc<Self>, name: String) -> CrashHandlerManagedLoopSpawner
    {
        let id = Uuid::new();
        self.loop_acknowledge_times
            .lock()
            .unwrap()
            .insert(id, chrono::Utc::now());
        self.loop_names.lock().unwrap().insert(id, name.clone());

        CrashHandlerManagedLoopSpawner {
            id,
            name,
            owning_handler: self
        }
    }

    pub fn acknowledge_iteration(&self, id: Uuid)
    {
        self.loop_acknowledge_times
            .lock()
            .unwrap()
            .insert(id, chrono::Utc::now());
    }

    pub fn register_join_handle(&self, id: Uuid, join_handle: JoinHandle<()>)
    {
        assert!(
            self.thread_join_handles
                .lock()
                .unwrap()
                .insert(id, join_handle)
                .is_none()
        )
    }

    pub fn acknowledge_crash(&self, id: Uuid, panic_data: PanicPayload)
    {
        self.should_loops_iterate.store(false, SeqCst);

        assert!(
            self.panic_data
                .lock()
                .unwrap()
                .insert(id, panic_data)
                .is_none()
        )
    }

    pub fn acknowledge_termination(&self, id: Uuid)
    {
        self.loop_acknowledge_times.lock().unwrap().remove(&id);
        self.loop_names.lock().unwrap().remove(&id);
        self.thread_join_handles.lock().unwrap().remove(&id);
        self.panic_data.lock().unwrap().remove(&id);
    }
}

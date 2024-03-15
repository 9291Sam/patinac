use std::any::Any;
use std::collections::HashMap;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::{Arc, Mutex};
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
    pub fn enter_oneshot<R>(self, func: impl FnOnce() -> R + UnwindSafe) -> Option<R>
    {
        match std::panic::catch_unwind(func)
        {
            Ok(r) => Some(r),
            Err(e) =>
            {
                self.owning_handler.acknowledge_crash(self.id, e);
                None
            }
        }
    }

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
pub trait CrashHandlerFunction = Fn(CrashLog) + Send + 'static;

pub struct CrashHandler
{
    handle_crash_function:  Mutex<Option<Box<dyn CrashHandlerFunction>>>,
    should_loops_iterate:   AtomicBool,
    loop_acknowledge_times: Mutex<HashMap<Uuid, chrono::DateTime<Utc>>>,
    loop_names:             Mutex<HashMap<Uuid, String>>,
    thread_join_handles:    Mutex<Option<HashMap<Uuid, JoinHandle<()>>>>,
    panic_data:             Mutex<Option<HashMap<Uuid, (std::time::Instant, PanicPayload)>>>
}

impl CrashHandler
{
    #[allow(clippy::new_without_default)]
    pub fn new(crash_func: impl CrashHandlerFunction) -> Arc<CrashHandler>
    {
        Arc::new(CrashHandler {
            handle_crash_function:  Mutex::new(Some(Box::new(crash_func))),
            should_loops_iterate:   AtomicBool::new(true),
            loop_acknowledge_times: Mutex::new(HashMap::new()),
            loop_names:             Mutex::new(HashMap::new()),
            thread_join_handles:    Mutex::new(Some(HashMap::new())),
            panic_data:             Mutex::new(Some(HashMap::new()))
        })
    }

    pub fn into_guarded_scope(&self, func: FnOnce() -> )
    {

    }

    pub fn handle_crash(&self)
    {
        // no threads have crashed
        if self.panic_data.lock().unwrap().unwrap().is_empty()
        {
            return;
        }

        self.should_loops_iterate.store(false, SeqCst);

        self.thread_join_handles
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .into_iter()
            .for_each(|(_, handle)| {
                let _ = handle.join();
            });

        let mut panicked_threads = self
            .panic_data
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();

        panicked_threads.sort_by(|l, r| l.1.0.cmp(&r.1.0));

        self.crash_log.lock().unwrap() = CrashLog::new(panicked_threads.into_iter().map(|f| ThreadCrashLog { thread_name: todo!(), panic_time: todo!(), payload: todo!() }))
        
        std::panic::panic_any("Crash::handle_crash() Forcing Unwind!");
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
                .as_mut()
                .expect("Crash has already occurred!")
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
                .as_mut()
                .expect("Crash has already occurred!")
                .insert(id, (std::time::Instant::now(), panic_data))
                .is_none()
        )
    }

    pub fn acknowledge_termination(&self, id: Uuid)
    {
        self.loop_acknowledge_times.lock().unwrap().remove(&id);
        self.loop_names.lock().unwrap().remove(&id);
        self.thread_join_handles.lock().unwrap().as_mut().unwrap().remove(&id);
        self.panic_data.lock().unwrap().as_mut().unwrap().remove(&id);
    }
}


struct ThreadCrashLog {
    thread_name: String,
    panic_time: std::time::Instant,
    // TODO: backtrace
    payload: Box<PanicPayload>
}

struct CrashLog {}

impl CrashLog {
    pub fn new(threads: impl IntoIterator<Item = ThreadCrashLog>)
    {
        
    }
}

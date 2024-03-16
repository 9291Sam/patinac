use std::any::Any;
use std::borrow::Cow;
use std::fmt::Display;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Mutex, Once};
use std::thread::{Scope, ScopedJoinHandle};

pub struct CrashHandlerManagedLoopSpawner<'scope, 'env>
{
    handler:         &'scope CrashHandler,
    thread_scope:    &'scope Scope<'scope, 'env>,
    spawned_handles: Mutex<Vec<ScopedJoinHandle<'scope, ()>>>
}

impl UnwindSafe for CrashHandlerManagedLoopSpawner<'_, '_> {}
impl RefUnwindSafe for CrashHandlerManagedLoopSpawner<'_, '_> {}

type ContinueLoopingFunc<'l> = dyn Fn() -> bool + 'l;
type TerminateLoopsFunc<'l> = dyn Fn() + 'l;

impl<'scope, 'env> CrashHandlerManagedLoopSpawner<'scope, 'env>
{
    pub fn enter_constrained<R, F>(&self, name: String, func: F) -> R
    where
        F: FnOnce(&ContinueLoopingFunc<'scope>, &TerminateLoopsFunc<'scope>) -> R + UnwindSafe
    {
        let iter_func = || self.handler.should_loops_iterate();
        let terminate_func = || self.handler.terminate_loops();

        match std::panic::catch_unwind(|| func(&iter_func, &terminate_func))
        {
            Ok(r) => r,
            Err(payload) =>
            {
                std::panic::panic_any(CrashInfo {
                    thread_name: name,
                    panic_time: std::time::Instant::now(),
                    payload
                });
            }
        }
    }

    // help
    pub fn enter_constrained_thread<F>(&self, name: String, func: F)
    where
        F: FnOnce(&ContinueLoopingFunc<'scope>, &TerminateLoopsFunc<'scope>)
            + UnwindSafe
            + Send
            + 'scope
    {
        let iter_func = || self.handler.should_loops_iterate();
        let terminate_func = || self.handler.terminate_loops();

        self.spawned_handles.lock().unwrap().push(
            self.thread_scope
                .spawn(move || func(&iter_func, &terminate_func))
        );
    }
}

pub struct CrashHandler
{
    has_crash_occurred:   AtomicBool,
    crash_receiver:       Mutex<Receiver<CrashInfo>>,
    crash_sender:         Sender<CrashInfo>,
    should_loops_iterate: AtomicBool
}

impl CrashHandler
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> CrashHandler
    {
        static INIT_ONCE: Once = Once::new();

        assert!(!INIT_ONCE.is_completed());
        INIT_ONCE.call_once(|| {});

        // std::panic::set_hook(Box::new(|_| {}));

        let (tx, rx) = std::sync::mpsc::channel();

        CrashHandler {
            has_crash_occurred:   AtomicBool::new(false),
            crash_sender:         tx,
            crash_receiver:       Mutex::new(rx),
            should_loops_iterate: AtomicBool::new(true)
        }
    }

    pub fn into_guarded_scope(
        &self,
        func: impl FnOnce(&CrashHandlerManagedLoopSpawner) + UnwindSafe
    )
    {
        std::thread::scope(|s| {
            let spawner = CrashHandlerManagedLoopSpawner {
                handler:         self,
                thread_scope:    s,
                spawned_handles: Mutex::new(Vec::new())
            };

            if let Err(e) = std::panic::catch_unwind(|| func(&spawner))
            {
                match e.downcast::<CrashInfo>()
                {
                    Ok(crash_info) => self.crash_sender.send(*crash_info).unwrap(),
                    Err(e) => std::panic::panic_any(e)
                }
            }
        })
    }

    pub fn should_loops_iterate(&self) -> bool
    {
        self.should_loops_iterate.load(SeqCst)
    }

    pub fn terminate_loops(&self)
    {
        self.should_loops_iterate.store(false, SeqCst)
    }

    pub fn poll_threads_for_crashes(&self) {}

    pub fn finish(self)
    {
        std::mem::drop(self.crash_sender);

        let frames = self
            .crash_receiver
            .into_inner()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        frames.into_iter().for_each(|f| log::error!("{f}"));
    }
}

pub struct CrashInfo
{
    thread_name: String,
    panic_time:  std::time::Instant,
    // TODO: backtrace
    payload:     Box<dyn Any + Send>
}

impl Display for CrashInfo
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        let message = if let Some(s) = self.payload.downcast_ref::<&'static str>()
        {
            Cow::Borrowed(*s)
        }
        else if let Some(s) = self.payload.downcast_ref::<String>()
        {
            Cow::Owned(s.clone())
        }
        else
        {
            Cow::Owned(format!("{:#?}", Any::type_id(&self.payload)))
        };

        writeln!(
            f,
            "{} has crashed @ {:?} with message |{}|",
            self.thread_name, self.panic_time, message
        )
    }
}

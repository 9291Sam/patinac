use std::any::Any;
use std::borrow::Cow;
use std::fmt::Display;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, Once};
use std::thread::{Scope, ScopedJoinHandle};

pub struct CrashHandlerManagedLoopSpawner<'scope, 'env>
{
    handler:         &'scope CrashHandler,
    thread_scope:    &'scope Scope<'scope, 'env>,
    spawned_handles: Mutex<Vec<ScopedJoinHandle<'scope, Result<(), Box<dyn Any + Send>>>>>
}

impl UnwindSafe for CrashHandlerManagedLoopSpawner<'_, '_> {}
impl RefUnwindSafe for CrashHandlerManagedLoopSpawner<'_, '_> {}

type ContinueLoopingFunc<'l> = dyn Fn() -> bool + 'l;
type TerminateLoopsFunc<'l> = dyn Fn() + 'l;
type CrashPollFunc<'l> = dyn Fn() + 'l;

impl<'scope, 'env> CrashHandlerManagedLoopSpawner<'scope, 'env>
{
    pub fn enter_constrained<R, F>(&self, name: String, func: F) -> R
    where
        F: FnOnce(
                &ContinueLoopingFunc<'scope>,
                &TerminateLoopsFunc<'scope>,
                &CrashPollFunc<'scope>
            ) -> R
            + UnwindSafe
    {
        let iter_func = || self.handler.should_loops_iterate();
        let terminate_func = || self.handler.terminate_loops();
        let crash_poll_func = || self.handler.poll_threads_for_crashes();

        match std::panic::catch_unwind(|| func(&iter_func, &terminate_func, &crash_poll_func))
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
        F: FnOnce(
                &ContinueLoopingFunc<'scope>,
                &TerminateLoopsFunc<'scope>,
                &CrashPollFunc<'scope>
            ) + UnwindSafe
            + Send
            + 'scope
    {
        let iter_func = || self.handler.should_loops_iterate();
        let terminate_func = || self.handler.terminate_loops();
        let crash_poll_func = || self.handler.poll_threads_for_crashes();

        let thread_crash_notifier = self.handler.has_thread_crashed.clone();
        self.spawned_handles
            .lock()
            .unwrap()
            .push(self.thread_scope.spawn(move || {
                match std::panic::catch_unwind(|| {
                    func(&iter_func, &terminate_func, &crash_poll_func)
                })
                {
                    Ok(()) => Ok(()),
                    Err(e) =>
                    {
                        thread_crash_notifier.store(true, SeqCst);

                        Err(e)
                    }
                }
            }));
    }
}

pub struct CrashHandler
{
    crash_receiver:       Mutex<Receiver<CrashInfo>>,
    crash_sender:         Sender<CrashInfo>,
    should_loops_iterate: AtomicBool,

    // if you decide that you hate yourself, you should remove this Arc
    has_thread_crashed: Arc<AtomicBool>
}

impl CrashHandler
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> CrashHandler
    {
        static INIT_ONCE: Once = Once::new();

        assert!(!INIT_ONCE.is_completed());
        INIT_ONCE.call_once(|| {});

        // TODO: capture backtrace
        std::panic::set_hook(Box::new(|info| {
            if info.payload().type_id() == std::any::TypeId::of::<CrashInfo>()
            {
                return;
            }

            let file = info.location().unwrap().file().replace('\\', "/");
            let line = info.location().unwrap().line();

            let prettified_filepath: &str;

            if let Some(idx) = file.find("index.crates.io-")
            {
                let idx_of_next_slash = file[idx..].find('/').unwrap();

                prettified_filepath = &file[idx_of_next_slash + idx + 1..];
            }
            else
            {
                prettified_filepath = &file;
            }

            let file_path: String = format!("[{prettified_filepath}:{line}]");

            log::error!(
                "Thread ??? has crashed @ {file_path} with message |{}|",
                panic_payload_as_cow(info.payload())
            );
        }));

        let (tx, rx) = std::sync::mpsc::channel();

        CrashHandler {
            crash_sender:         tx,
            crash_receiver:       Mutex::new(rx),
            should_loops_iterate: AtomicBool::new(true),
            has_thread_crashed:   Arc::new(AtomicBool::new(false))
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

    pub fn poll_threads_for_crashes(&self)
    {
        if self.has_thread_crashed.load(SeqCst)
        {
            self.terminate_loops();
        }
    }

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
            .collect::<Vec<_>>(); // TODO: change because now this receiver doesnt receive anything

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
        writeln!(
            f,
            "{} has crashed @ {:?} with message |{}|",
            self.thread_name,
            self.panic_time,
            panic_payload_as_cow(&*self.payload)
        )
    }
}

pub fn panic_payload_as_cow(payload: &(dyn Any + Send)) -> Cow<'static, str>
{
    if let Some(s) = payload.downcast_ref::<&'static str>()
    {
        Cow::Borrowed(*s)
    }
    else if let Some(s) = payload.downcast_ref::<String>()
    {
        Cow::Owned(s.clone())
    }
    else if let Some(s) = payload.downcast_ref::<CrashInfo>()
    {
        Cow::Owned(format!("{s}"))
    }
    else if let Some(s) = payload.downcast_ref::<Box<dyn Any + Send>>()
    {
        panic_payload_as_cow(&**s)
    }
    else
    {
        Cow::Owned(format!("{:#?}", Any::type_id(payload)))
    }
}

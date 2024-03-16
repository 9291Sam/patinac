use std::any::Any;
use std::fmt::Display;
use std::panic::UnwindSafe;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;

pub struct CrashHandlerManagedLoopSpawner<'h>
{
    handler: &'h CrashHandler
}

type ContinueLoopingFunc = dyn Fn() -> bool;
type TerminateLoopsFunc = dyn Fn();

impl CrashHandlerManagedLoopSpawner<'_>
{
    pub fn enter_oneshot<R, F>(&self, name: String, func: F) -> R
    where
        F: FnOnce() -> R + UnwindSafe
    {
        match std::panic::catch_unwind(|| func(&|| self.handler.should_loops_iterate()))
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

    // pub fn enter_managed_thread

    // pub fn enter_managed_loop(&self, func: impl Fn() -> TerminationResult +
    // RefUnwindSafe) {
    //     if let Err(e) = std::panic::catch_unwind(|| {
    //         while self.owning_handler.should_loops_iterate()
    //         {
    //             match func()
    //             {
    //                 TerminationResult::RequestIteration =>
    //                 {}
    //                 TerminationResult::Terminate => break
    //             }

    //             self.owning_handler.acknowledge_iteration(self.id);
    //         }
    //     })
    //     {
    //         self.owning_handler.acknowledge_crash(self.id, e);
    //     }
    // }
}

pub enum TerminationResult
{
    RequestIteration,
    Terminate
}

type PanicPayload = Box<dyn Any + Send>;

pub struct CrashHandler
{
    crash_receiver: Receiver<CrashInfo>,
    crash_sender:   Sender<CrashInfo>
}

impl CrashHandler
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> CrashHandler
    {
        let (tx, rx) = std::sync::mpsc::channel();

        CrashHandler {
            crash_sender:   tx,
            crash_receiver: rx
        }
    }

    pub fn into_guarded_scope(
        &self,
        func: impl FnOnce(&CrashHandlerManagedLoopSpawner) + UnwindSafe
    )
    {
        if let Err(e) = std::panic::catch_unwind(|| {
            func(&CrashHandlerManagedLoopSpawner {
                handler: self
            })
        })
        {
            match e.downcast::<CrashInfo>()
            {
                Ok(crash_info) => self.crash_sender.send(*crash_info).unwrap(),
                Err(e) => std::panic::panic_any(e)
            }
        }
    }

    pub fn should_loops_iterate(&self) -> bool
    {
        todo!()
    }

    pub fn

    pub fn finish(self)
    {
        std::mem::drop(self.crash_sender);

        let frames = self
            .crash_receiver
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
    payload:     PanicPayload
}

impl Display for CrashInfo
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        writeln!(
            f,
            "{} has crashed @ {:?} with message {:?}",
            self.thread_name, self.panic_time, self.payload
        )
    }
}

#![allow(clippy::type_complexity)]
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::io::Write;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::thread::{ScopedJoinHandle, ThreadId};

// TODO: sort threads by panic tine

#[derive(Clone)]
pub struct CrashInfo
{
    thread_name:    String,
    panic_location: String,
    panic_time:     chrono::DateTime<chrono::Local>,
    backtrace:      backtrace::Backtrace,
    message:        Cow<'static, str>
}

impl Debug for CrashInfo
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            f,
            "Thread {} has crashed @ [{}] and {} with message: {} \n {:?}",
            self.thread_name,
            super::format_chrono_time(self.panic_time),
            self.panic_location,
            self.message,
            self.backtrace
        )
    }
}

impl Display for CrashInfo
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            f,
            "Thread {} has crashed @ [{}] and {} with message: {}",
            self.thread_name,
            super::format_chrono_time(self.panic_time),
            self.panic_location,
            self.message
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
    else if let Some(s) = payload.downcast_ref::<Box<dyn Any + Send>>()
    {
        panic_payload_as_cow(&**s)
    }
    else
    {
        Cow::Owned(format!("Unknown {:#?}", Any::type_id(payload)))
    }
}

static THREAD_CRASH_INFOS: OnceLock<Mutex<HashMap<ThreadId, CrashInfo>>> = OnceLock::new();
static SHOULD_LOOPS_KEEP_ITERATING: AtomicBool = AtomicBool::new(true);

pub type ShouldLoopsContinue = dyn Fn() -> bool + Send + Sync + RefUnwindSafe;
pub type TerminateLoops = dyn Fn() + Send + Sync;

pub fn handle_crashes(
    // thread spawn func, not labeled for lifetime reasons
    func: impl FnOnce(
        &dyn (Fn(String, Box<dyn FnOnce() + Send + UnwindSafe>)),
        Arc<ShouldLoopsContinue>,
        Arc<TerminateLoops>
    ) + UnwindSafe
)
{
    assert!(
        THREAD_CRASH_INFOS
            .try_insert(Mutex::new(HashMap::new()))
            .is_ok()
    );

    std::panic::set_hook(Box::new(move |panic_info| {
        static PRINT_THREADS_PANIC_MESSAGE_ONCE: Once = Once::new();

        PRINT_THREADS_PANIC_MESSAGE_ONCE.call_once(|| {
            log::error!(
                "Thread {} has panicked",
                std::thread::current().name().unwrap_or("???")
            )
        });

        let thread = std::thread::current();

        // TODO: make function
        let file = panic_info.location().unwrap().file().replace('\\', "/");
        let line = panic_info.location().unwrap().line();

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

        THREAD_CRASH_INFOS.get().unwrap().lock().unwrap().insert(
            thread.id(),
            CrashInfo {
                thread_name:    thread.name().unwrap_or("???").to_string(),
                panic_location: file_path,
                panic_time:     chrono::Local::now(),
                backtrace:      backtrace::Backtrace::new(),
                message:        panic_payload_as_cow(panic_info.payload())
            }
        );

        SHOULD_LOOPS_KEEP_ITERATING.store(false, Ordering::Release);
    }));

    let should_loops_continue = || SHOULD_LOOPS_KEEP_ITERATING.load(Acquire);
    let terminate_loops = || SHOULD_LOOPS_KEEP_ITERATING.store(false, Release);

    std::thread::scope(|thread_scope| {
        let spawned_handles: Mutex<HashMap<ThreadId, ScopedJoinHandle<'_, ()>>> =
            Mutex::new(HashMap::new());

        let new_thread_func = |name: String, execute: Box<dyn FnOnce() + Send + UnwindSafe>| {
            let join_handle = std::thread::Builder::new()
                .name(name)
                .spawn_scoped(thread_scope, || {
                    let _ = std::panic::catch_unwind(execute);
                })
                .unwrap();

            spawned_handles
                .lock()
                .unwrap()
                .insert(join_handle.thread().id(), join_handle);
        };

        // if this thread crashes it still puts its info in THREAD_CRASH_INFOS
        let _ = std::panic::catch_unwind(|| {
            func(
                &new_thread_func,
                Arc::new(should_loops_continue),
                Arc::new(terminate_loops)
            )
        });

        // func has finished, threads are joined.
        terminate_loops();
    });

    let mut local_crash_infos: Vec<CrashInfo> = THREAD_CRASH_INFOS
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .clone()
        .into_values()
        .collect::<Vec<_>>();

    local_crash_infos.sort_by(|l, r| l.panic_time.cmp(&r.panic_time));

    if local_crash_infos.is_empty()
    {
        log::info!("No threads crashed")
    }
    else
    {
        local_crash_infos.iter().for_each(|c| log::error!("{}", c));

        let time = super::format_chrono_time(chrono::Local::now()).replace(['/', ':'], "_");

        let mut crash_file = std::fs::OpenOptions::new()
            .read(false)
            .write(true)
            .append(false)
            .truncate(true)
            .create(true)
            .open(format!("patinac_crash-{}.txt", time))
            .expect("Failed to create crash file!");

        local_crash_infos
            .into_iter()
            .for_each(|c| writeln!(crash_file, "{:?}", c).unwrap());
    }
}

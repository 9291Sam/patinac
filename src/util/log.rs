use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[derive(Debug)]
pub struct AsyncLogger
{
    thread_sender: Sender<String>,
    should_stop:   Arc<AtomicBool>,
    worker_thread: Mutex<Option<JoinHandle<()>>>
}

impl AsyncLogger
{
    pub fn new() -> Self
    {
        let (thread_sender, receiver) = mpsc::channel();
        let should_stop = Arc::new(AtomicBool::new(false));

        let thread_should_stop = should_stop.clone();
        let worker_thread = std::thread::spawn(move || {
            let mut log_file = std::fs::OpenOptions::new()
                .read(false)
                .write(true)
                .append(false)
                .truncate(true)
                .create(true)
                .open("patinac_log.txt")
                .expect("Failed to create log file!");

            let mut write_fn = |message: String| {
                println!("{}", message);
                writeln!(log_file, "{}", message).unwrap()
            };

            loop
            {
                if thread_should_stop.load(Ordering::Acquire)
                {
                    break;
                }

                match receiver.try_recv()
                {
                    Ok(message) => write_fn(message),
                    Err(TryRecvError::Disconnected) => break,
                    Err(_) =>
                    {}
                }
            }

            // cleanup loop
            while let Ok(message) = receiver.try_recv()
            {
                write_fn(message);
            }
        });

        AsyncLogger {
            thread_sender,
            worker_thread: Mutex::new(Option::Some(worker_thread)),
            should_stop
        }
    }

    pub fn stop_worker(&self)
    {
        self.should_stop.store(true, Ordering::Release);

        let stolen_worker: JoinHandle<()> = self.worker_thread.lock().unwrap().take().unwrap();

        stolen_worker.join().unwrap();
    }
}

impl log::Log for AsyncLogger
{
    /// Doing this means that the log crate manages the level itself
    fn enabled(&self, _metadata: &log::Metadata) -> bool
    {
        true
    }

    fn log(&self, record: &log::Record)
    {
        let mut working_time_string = chrono::Local::now()
            .format("%b %m/%d/%Y %T%.6f")
            .to_string();

        working_time_string.replace_range(23..24, ":");
        working_time_string.insert(27, ':');

        if let Err(unsent_string) = self.thread_sender.send(format!(
            "[{}] [{}:{}] [{}] {}",
            working_time_string,
            record.file().unwrap_or_default().replace('\\', "/"),
            record.line().unwrap_or_default(),
            record.level(),
            format_args!("{}", record.args())
        ))
        {
            eprintln!("Send After async shutdown! {}", unsent_string.0);
        }
    }

    fn flush(&self) {}
}

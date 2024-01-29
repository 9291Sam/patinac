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
    fn enabled(&self, metadata: &log::Metadata) -> bool
    {
        true
    }

    fn log(&self, record: &log::Record)
    {
        // Silencing useless messages in 3rd party libs
        // if let Some(true) = record.file().map(|f| f.contains(".cargo"))
        // {
        //     if record.level() >= log::Level::Info
        //     {
        //         return;
        //     }
        // }

        let mut working_time_string = chrono::Local::now()
            .format("%b %m/%d/%Y %T%.6f")
            .to_string();

        working_time_string.replace_range(23..24, ":");
        working_time_string.insert(27, ':');

        let file_path: String;

        if let (Some(file), Some(line), true) = (
            record.file().map(|s| s.replace('\\', "/")),
            record.line(),
            cfg!(debug_assertions)
        )
        {
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

            file_path = format!("[{prettified_filepath}:{line}] ");
        }
        else
        {
            file_path = "".into();
        }

        if let Err(unsent_string) = self.thread_sender.send(format!(
            "[{}] {}[{}] {}",
            working_time_string,
            file_path,
            record.level(),
            format_args!("{}", record.args())
        ))
        {
            eprintln!("Send After async shutdown! {}", unsent_string.0);
        }
    }

    fn flush(&self) {}
}

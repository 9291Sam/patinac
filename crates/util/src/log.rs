use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SendError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

struct AsyncLoggerMessage
{
    message:                     String,
    should_be_printed_to_stdout: bool
}

#[derive(Debug)]
pub struct AsyncLogger
{
    thread_sender: Sender<AsyncLoggerMessage>,
    should_stop:   Arc<AtomicBool>,
    worker_thread: Mutex<Option<JoinHandle<()>>>
}

impl AsyncLogger
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self
    {
        let (thread_sender, receiver) = mpsc::channel();
        let should_stop = Arc::new(AtomicBool::new(false));

        let thread_should_stop = should_stop.clone();
        let worker_thread = std::thread::Builder::new()
            .name("Patinac AsyncLogger Thread".into())
            .spawn(move || {
                let mut log_file = std::fs::OpenOptions::new()
                    .read(false)
                    .write(true)
                    .append(false)
                    .truncate(true)
                    .create(true)
                    .open("patinac_log.txt")
                    .expect("Failed to create log file!");

                let mut write_fn = |message: AsyncLoggerMessage| {
                    let AsyncLoggerMessage {
                        message,
                        should_be_printed_to_stdout
                    } = message;

                    writeln!(log_file, "{}", message).unwrap();

                    if should_be_printed_to_stdout
                    {
                        println!("{}", message);
                    }
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
            })
            .unwrap();

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

        assert!(
            self.thread_sender
                .send(AsyncLoggerMessage {
                    message:                     String::new(),
                    should_be_printed_to_stdout: false
                })
                .is_err()
        )
    }
}

const BLOCK_ON_STR: &str = "BLOCK_ON";

impl log::Log for AsyncLogger
{
    /// Doing this means that the log crate manages the level itself
    fn enabled(&self, _metadata: &log::Metadata) -> bool
    {
        true
    }

    fn log(&self, record: &log::Record)
    {
        // Silencing useless messages in 3rd party libs
        if let Some(true) = record.file().map(|f| {
            f.contains(".cargo") || f.contains("wgpu-msl-out-atomic-compare-exchange-weak")
        })
        {
            if record.level() >= log::Level::Info
            {
                return;
            }
        }

        let working_time_string = format_chrono_time(chrono::Local::now());

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

        let user_message = format!("{}", record.args());

        let message_to_send: AsyncLoggerMessage = if let Some(stripped) =
            user_message.strip_prefix(BLOCK_ON_STR)
        {
            let msg = AsyncLoggerMessage {
                message:                     format!(
                    "[{}] {}[{}] {}",
                    working_time_string,
                    file_path,
                    record.level(),
                    stripped
                ),
                should_be_printed_to_stdout: false
            };

            println!("{}", msg.message);

            msg
        }
        else
        {
            AsyncLoggerMessage {
                message:                     format!(
                    "[{}] {}[{}] {}",
                    working_time_string,
                    file_path,
                    record.level(),
                    user_message
                ),
                should_be_printed_to_stdout: true
            }
        };

        if let Err(unsent_string) = self.thread_sender.send(message_to_send)
        {
            eprintln!("Send After async shutdown! {}", unsent_string.0.message);
        }
    }

    fn flush(&self) {}
}

pub fn format_chrono_time(time: chrono::DateTime<chrono::Local>) -> String
{
    let mut working_time_string = time.format("%b %m/%d/%Y %T%.6f").to_string();

    working_time_string.replace_range(23..24, ":");
    working_time_string.insert(27, ':');

    working_time_string
}

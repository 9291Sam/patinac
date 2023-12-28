use std::sync::mpsc;

struct AsyncLogger
{
    level: log::Level
}

impl log::Log for AsyncLogger
{
    fn enabled(&self, metadata: &log::Metadata) -> bool
    {
        metadata.level() >= self.level
    }

    fn log(&self, record: &log::Record)
    {
        todo!()
    }

    fn flush(&self)
    {
        todo!()
    }
}

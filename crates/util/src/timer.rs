use std::borrow::Cow;

#[derive(Debug)]
pub struct Timer
{
    has_terminated: bool,
    name:           Cow<'static, str>,
    start_time:     std::time::Instant
}

impl Drop for Timer
{
    fn drop(&mut self)
    {
        if !self.has_terminated
        {
            self.print_termination_message();

            // dead write lol? I wonder if the compiler optimizes this
            self.has_terminated = true;
        }
    }
}

impl Timer
{
    pub fn new(name: Cow<'static, str>) -> Timer
    {
        Timer {
            has_terminated: false,
            name,
            start_time: std::time::Instant::now()
        }
    }

    pub fn end(mut self)
    {
        self.has_terminated = true;

        self.print_termination_message();
    }

    fn print_termination_message(&self)
    {
        log::info!(
            "Timer {} completed in {:.3}ms",
            self.name,
            std::time::Instant::now()
                .duration_since(self.start_time)
                .as_secs_f64()
                * 1000.0
        );
    }
}

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct Window<T: Send + Clone>
{
    shared_value: Arc<Mutex<T>>
}

impl<T: Send + Clone> Window<T>
{
    pub fn new(init_val: T) -> (Window<T>, WindowUpdater<T>)
    {
        let shared_value = Arc::new(Mutex::new(init_val));

        (
            Window {
                shared_value: shared_value.clone()
            },
            WindowUpdater {
                shared_value
            }
        )
    }

    pub fn get(&self) -> T
    {
        self.shared_value.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone)]
pub struct WindowUpdater<T: Send + Clone>
{
    shared_value: Arc<Mutex<T>>
}

impl<T: Send + Clone> WindowUpdater<T>
{
    pub fn update(&self, t: T)
    {
        *self.shared_value.lock().unwrap() = t;
    }
}

#[derive(Debug, Clone)]
pub struct JointWindow<T: Send + Clone>
{
    sender:   WindowUpdater<T>,
    receiver: Window<T>
}

impl<T: Send + Clone> JointWindow<T>
{
    pub fn new(init_val: T) -> JointWindow<T>
    {
        let (receiver, sender) = Window::new(init_val);

        JointWindow {
            sender,
            receiver
        }
    }

    pub fn update(&self, t: T)
    {
        self.sender.update(t);
    }

    pub fn get(&self) -> T
    {
        self.receiver.get()
    }
}

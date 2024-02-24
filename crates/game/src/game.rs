use std::ops::Add;
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::{Arc, Weak};

use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::Entity;

pub struct TickTag(());

pub struct Game
{
    renderer:         Arc<gfx::Renderer>,
    entities:         util::Registrar<util::Uuid, Weak<dyn Entity>>,
    float_delta_time: AtomicU32,
    float_time_alive: AtomicU64
}

impl Drop for Game
{
    fn drop(&mut self)
    {
        self.entities
            .access()
            .into_iter()
            .filter_map(|(_, weak)| weak.upgrade())
            .for_each(|strong| log::warn!("Retained Entity! {:?}", &*strong));
    }
}

impl Game
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        Game {
            renderer,
            entities: util::Registrar::new(),
            float_delta_time: AtomicU32::new(0.0f32.to_bits()),
            float_time_alive: AtomicU64::new(0.0f64.to_bits())
        }
    }

    pub fn get_renderer(&self) -> &Arc<gfx::Renderer>
    {
        &self.renderer
    }

    #[deprecated]
    pub fn get_time_alive(&self) -> f64
    {
        f64::from_bits(self.float_time_alive.load(Acquire))
    }

    pub fn get_delta_time(&self) -> f32
    {
        f32::from_bits(self.float_delta_time.load(Acquire))
    }

    pub fn register(&self, entity: Arc<dyn Entity>)
    {
        self.entities
            .insert(entity.get_uuid(), Arc::downgrade(&entity));
    }

    pub fn enter_tick_loop(&self, should_stop: &AtomicBool)
    {
        let mut prev = std::time::Instant::now();
        let mut delta_time: f64;

        while !should_stop.load(Acquire)
        {
            let now = std::time::Instant::now();

            delta_time = (now - prev).as_secs_f64();
            prev = now;
            self.float_delta_time
                .store((delta_time as f32).to_bits(), Release);
            self.float_time_alive.store(
                (f64::from_bits(self.float_time_alive.load(Acquire)) + delta_time).to_bits(),
                Release
            );

            let thread_entities = &self.entities;

            // TODO: deadlock and too long detection
            self.entities
                .access()
                .into_iter()
                .filter_map(|(uuid, weak_renderable)| {
                    match weak_renderable.upgrade()
                    {
                        Some(s) => Some(s),
                        None =>
                        {
                            thread_entities.delete(uuid);
                            None
                        }
                    }
                })
                .par_bridge()
                .for_each(|strong_entity| strong_entity.tick(self, TickTag(())));
        }
    }
}

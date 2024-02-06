use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Weak};

use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::entity::{self, Entity};

pub(crate) struct TickTag(());

pub struct Game<'r>
{
    renderer:         &'r gfx::Renderer,
    entities:         util::Registrar<util::Uuid, Weak<dyn Entity>>,
    float_delta_time: AtomicU32 // event_dispatcher: /* */
}

impl Drop for Game<'_>
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

impl<'r> Game<'r>
{
    pub fn new(renderer: &'r gfx::Renderer) -> Self
    {
        Game {
            renderer,
            entities: util::Registrar::new(),
            float_delta_time: AtomicU32::new(0.0f32.to_bits())
        }
    }

    pub fn get_renderer(&self) -> &gfx::Renderer
    {
        self.renderer
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
        let _scene = entity::test_scene::TestScene::new(self);

        let mut prev = std::time::Instant::now();
        let mut delta_time: f32;

        while !should_stop.load(Acquire)
        {
            let now = std::time::Instant::now();

            delta_time = (now - prev).as_secs_f32();
            prev = now;
            self.float_delta_time.store(delta_time.to_bits(), Release);

            self.entities
                .access()
                .into_iter()
                .filter_map(|(uuid, weak_renderable)| {
                    match weak_renderable.upgrade()
                    {
                        Some(s) => Some(s),
                        None =>
                        {
                            self.entities.delete(uuid);
                            None
                        }
                    }
                })
                .par_bridge()
                .for_each(|strong_entity| strong_entity.tick(self, TickTag(())));
        }
    }
}

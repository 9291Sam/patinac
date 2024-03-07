#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::{Arc, OnceLock};

mod recordables;
mod test_scene;

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));

    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    // Safety: we try our best to drop the Renderer on this thread
    let renderer = Arc::new(unsafe { gfx::Renderer::new() });

    let should_stop = AtomicBool::new(false);

    let run = || {
        let game = game::Game::new(renderer.clone());

        std::thread::scope(|s| {
            let _scene_guard = test_scene::TestScene::new(&game);

            s.spawn(|| game.enter_tick_loop(&should_stop));

            renderer.enter_gfx_loop(&should_stop);
        });
    };

    if std::panic::catch_unwind(run).is_err()
    {
        should_stop.store(true, SeqCst);
    }

    if Arc::into_inner(renderer).is_none()
    {
        log::warn!("Renderer was retained via Arc cycle! Drop is not possible");
    }

    match util::get_bytes_of_active_allocations()
    {
        b if b > 2e6 as usize =>
        {
            log::trace!(
                "{} leaked | {} allocated",
                b,
                human_bytes::human_bytes(util::get_bytes_allocated_total() as f64)
            );
        }
        _ =>
        {}
    }

    logger.stop_worker();
}

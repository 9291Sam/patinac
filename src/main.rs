use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use gfx::glm;
use itertools::iproduct;
use rand::Rng;

fn main()
{
    #[cfg(debug_assertions)]
    std::fs::read_dir(".").unwrap().for_each(|p| {
        let dir_entry = p.unwrap();

        if dir_entry.file_type().unwrap().is_file()
        {
            let name = dir_entry.file_name();
            let name_as_str = name.to_string_lossy();

            if name_as_str.starts_with("patinac") && name_as_str.ends_with(".txt")
            {
                std::fs::remove_file(dir_entry.path()).unwrap();
            }
        }
    });

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new(
        std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .get(),
        "Patinac async threadpool"
    ));

    let held_renderer: Mutex<Option<Arc<gfx::Renderer>>> = Mutex::new(None);
    let held_game: Mutex<Option<Arc<game::Game>>> = Mutex::new(None);

    util::handle_crashes(|new_thread_func, should_loops_continue, terminate_loops| {
        let (renderer, renderer_renderpass_updater) =
            unsafe { gfx::Renderer::new(format!("Patinac {}", env!("CARGO_PKG_VERSION"))) };

        let renderer = Arc::new(renderer);
        *held_renderer.lock().unwrap() = Some(renderer.clone());

        let game = game::Game::new(renderer.clone(), renderer_renderpass_updater);
        *held_game.lock().unwrap() = Some(game.clone());

        {
            let _verdigris = verdigris::DemoScene::new(game.clone());
            let _debug_menu = gui::DebugMenu::new(&renderer, game.clone());

            let game_tick = game.clone();
            let game_continue = should_loops_continue.clone();
            new_thread_func(
                "Game Tick Thread".into(),
                Box::new(move || game_tick.enter_tick_loop(&*game_continue))
            );

            let input_game = game.clone();
            let input_update_func =
                move |input_manager: &gfx::InputManager, camera_delta_time: f32| {
                    input_game.poll_input_updates(input_manager, camera_delta_time)
                };

            renderer.enter_gfx_loop(
                &*should_loops_continue,
                &*terminate_loops,
                &input_update_func
            );
        }

        util::access_global_thread_pool()
            .write()
            .unwrap()
            .take()
            .unwrap()
            .join_threads();
    });

    if let Ok(Some(arc_game)) = held_game.into_inner()
    {
        if let Err(g) = Arc::try_unwrap(arc_game)
        {
            log::warn!("Game was retained with {} cycles", Arc::strong_count(&g))
        }
    }
    else
    {
        log::error!("Game was never created!");
    };

    if let Ok(Some(arc_renderer)) = held_renderer.into_inner()
    {
        if let Err(r) = Arc::try_unwrap(arc_renderer)
        {
            log::warn!(
                "Renderer was retained with {} cycles",
                Arc::strong_count(&r)
            )
        }
    }
    else
    {
        log::error!("Renderer was never created!");
    };

    logger.stop_worker();
}

// my ecs: two stage indirection
// Entity: NonZeroU64 (top 8 bits are the number of components this thing has,
// bottom 24 is its id), generational 24 | id 24
// EntityStorageBuffer[number_of_components][id] // two stage buffer to get a
// dense list of components each thing registers a callback on what it wants to
// do, &callbacks are executed simultaneously &mut are done in a chaining
// semaphore like fashion

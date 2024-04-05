use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

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
        let renderer = Arc::new(unsafe {
            gfx::Renderer::new(format!("Patinac {}", env!("CARGO_PKG_VERSION")))
        });
        *held_renderer.lock().unwrap() = Some(renderer.clone());

        log::trace!("Fov: {}", renderer.get_fov());

        let game = game::Game::new(renderer.clone());
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
        if let Some(game) = Arc::into_inner(arc_game)
        {
            std::mem::drop(game);
        }
        else
        {
            log::error!("Game was retained!");
        }
    }
    else
    {
        log::error!("Game was never created!");
    };

    if let Ok(Some(arc_renderer)) = held_renderer.into_inner()
    {
        if let Some(renderer) = Arc::into_inner(arc_renderer)
        {
            std::mem::drop(renderer);
        }
        else
        {
            log::error!("Renderer was retained!");
        }
    }
    else
    {
        log::error!("Renderer was never created!");
    };

    log::info!("Patinac exited");

    logger.stop_worker();
}

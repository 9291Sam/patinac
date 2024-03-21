use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct DebugMenu
{
    id:             util::Uuid,
    rendering_data: Mutex<DebugMenuCriticalSection>
}

struct DebugMenuCriticalSection
{
    font_system:          glyphon::FontSystem,
    cache:                glyphon::SwashCache,
    atlas:                glyphon::TextAtlas,
    text_renderer:        glyphon::TextRenderer,
    buffer:               glyphon::Buffer,
    previous_update_time: std::time::Instant
}

impl DebugMenu
{
    pub fn new(renderer: &gfx::Renderer) -> Arc<Self>
    {
        let mut db = glyphon::fontdb::Database::new();
        db.load_font_data(include_bytes!("unifont-15.1.05.otf").into());
        db.load_font_data(include_bytes!("OpenMoji-color-sbix.ttf").into());

        let mut font_system = glyphon::FontSystem::new_with_locale_and_db(
            sys_locale::get_locale().unwrap_or_else(|| {
                log::warn!("failed to get system locale, falling back to en-US");
                String::from("en-US")
            }),
            db
        );

        // let mut font_system = glyphon::FontSystem::new_with_fonts([source]);
        // let mut font_system = glyphon::FontSystem::new();
        let cache = glyphon::SwashCache::new();
        let mut atlas = glyphon::TextAtlas::new(
            &renderer.device,
            &renderer.queue,
            gfx::Renderer::SURFACE_TEXTURE_FORMAT
        );
        let text_renderer = glyphon::TextRenderer::new(
            &mut atlas,
            &renderer.device,
            gfx::wgpu::MultisampleState::default(),
            None
        );

        let mut buffer = Self::make_buffer(&mut font_system, renderer);
        buffer.set_text(
            &mut font_system,
            "DEFAULT BUFFER TEXT",
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced
        );

        let this = Arc::new(DebugMenu {
            id:             util::Uuid::new(),
            rendering_data: Mutex::new(DebugMenuCriticalSection {
                font_system,
                cache,
                atlas,
                text_renderer,
                buffer,
                previous_update_time: std::time::Instant::now()
            })
        });

        renderer.register(this.clone());

        this
    }

    fn make_buffer(
        font_system: &mut glyphon::FontSystem,
        renderer: &gfx::Renderer
    ) -> glyphon::Buffer
    {
        let mut buffer = glyphon::Buffer::new_empty(glyphon::Metrics::new(16.0, 16.0));

        let (physical_width, physical_height) = {
            let dim = renderer.get_framebuffer_size();

            (dim.x, dim.y)
        };

        buffer.set_size(font_system, physical_width as f32, physical_height as f32);
        buffer.shape_until_scroll(font_system);

        buffer
    }
}

impl Debug for DebugMenuCriticalSection
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("DebugMenuCriticalSection")
            .field("font_system", &self.font_system)
            .field("cache", &self.cache)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl gfx::Recordable for DebugMenu
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Debug Menu")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::MenuRender
    }

    fn get_pipeline(&self) -> Option<&gfx::GenericPipeline>
    {
        None
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        _: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let DebugMenuCriticalSection {
            ref mut font_system,
            ref mut cache,
            ref mut atlas,
            ref mut text_renderer,
            ref mut buffer,
            ref mut previous_update_time
        } = &mut *self.rendering_data.lock().unwrap();

        let now = std::time::Instant::now();

        const UPDATE_SPACING_TIME: std::time::Duration = std::time::Duration::from_millis(75);

        if now.duration_since(*previous_update_time) > UPDATE_SPACING_TIME
        {
            *previous_update_time = now;
            // ╔═╦═╗
            // ║ ║ ║
            // ╠═╬═╣
            // ║ ║ ║
            // ╚═╩═╝

            #[rustfmt::skip]
        buffer.set_text(
            font_system,
            &format!(
r#"╔═══════════════════╦════════════╗
║ Frames per second ║ {:<10.3} ║
╠═══════════════════╬════════════╣
║  Frame Time (ms)  ║ {:<10.3} ║
╠═══════════════════╬════════════╣
║     Ram Usage     ║ {:<10} ║
╠═══════════════════╬════════════╩══════╗
║  Camera Position  ║ {} ║
╚═══════════════════╩═══════════════════╝"#,
                1.0 / renderer.get_delta_time(),
                renderer.get_delta_time() * 1000.0,
                util::bytes_as_string(
                    util::get_bytes_of_active_allocations() as f64,
                    util::SuffixType::Short
                ),
                camera.get_position(),
            ),
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced
        );
            let (width, height) = {
                let dim = renderer.get_framebuffer_size();

                (dim.x, dim.y)
            };

            text_renderer
                .prepare(
                    &renderer.device,
                    &renderer.queue,
                    font_system,
                    atlas,
                    glyphon::Resolution {
                        width,
                        height
                    },
                    [glyphon::TextArea {
                        buffer,
                        left: 2.0,
                        top: 0.0,
                        scale: 1.0,
                        bounds: glyphon::TextBounds {
                            left:   0,
                            top:    0,
                            right:  600,
                            bottom: 320
                        },
                        default_color: glyphon::Color::rgb(255, 255, 255)
                    }],
                    cache
                )
                .unwrap();
        }

        gfx::RecordInfo {
            should_draw: true,
            transform:   None,
            bind_groups: [None, None, None, None]
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(pass), None) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        let (atlas, text_renderer) = {
            let DebugMenuCriticalSection {
                ref mut atlas,
                ref mut text_renderer,
                ..
            } = &mut *self.rendering_data.lock().unwrap();

            (
                atlas as *const glyphon::TextAtlas,
                text_renderer as *const glyphon::TextRenderer
            )
        };

        // That's right! the square peg goes into the round hole!
        // Hillariously enough, this isn't actually a problem as the menu is dropped
        // before the renderer, and calls to pre_record_update and record may never
        // alias one another
        // TODO: rework the lifetimes on the entire renderer subsystem to fix this,
        // because this 1000% can be statically verified
        unsafe { (*text_renderer).render(&*atlas, pass).unwrap() }
    }
}

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
    font_system:   glyphon::FontSystem,
    cache:         glyphon::SwashCache,
    atlas:         glyphon::TextAtlas,
    text_renderer: glyphon::TextRenderer,
    buffer:        glyphon::Buffer
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
        let mut buffer = glyphon::Buffer::new(&mut font_system, glyphon::Metrics::new(16.0, 20.0));

        let physical_width = 640.0;
        let physical_height = 480.0;

        buffer.set_size(&mut font_system, physical_width, physical_height);
        buffer.set_text(
            &mut font_system,
            "Hello world! 👋\nThis is rendered with 🦅 glyphon 🦁\nThe text below should be \
             partially clipped.\na b c d e f g h i j k l m n o p q r s t u v w x y z",
            glyphon::Attrs::new().family(glyphon::Family::Monospace),
            glyphon::Shaping::Advanced
        );
        buffer.shape_until_scroll(&mut font_system);

        let this = Arc::new(DebugMenu {
            id:             util::Uuid::new(),
            rendering_data: Mutex::new(DebugMenuCriticalSection {
                font_system,
                cache,
                atlas,
                text_renderer,
                buffer
            })
        });

        renderer.register(this.clone());

        this
    }
}

impl Debug for DebugMenuCriticalSection
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        f.debug_struct("DebugMenuCriticalSection")
            .field("font_system", &self.font_system)
            .field("cache", &self.cache)
            // .field("atlas", &self.atlas)
            // .field("text_renderer", &self.text_renderer)
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
        _: &gfx::Camera,
        _: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let DebugMenuCriticalSection {
            ref mut font_system,
            ref mut cache,
            ref mut atlas,
            ref mut text_renderer,
            ref mut buffer
        } = &mut *self.rendering_data.lock().unwrap();

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
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: glyphon::TextBounds {
                        left:   0,
                        top:    0,
                        right:  600,
                        bottom: 160
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255)
                }],
                cache
            )
            .unwrap();

        gfx::RecordInfo {
            should_draw: true,
            transform:   None,
            bind_groups: [None, None, None, None]
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
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
            unsafe { (*text_renderer).render(&*atlas, pass).unwrap() }
        }
    }
}

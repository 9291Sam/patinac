use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use util::AtomicU32U32;

pub struct ScreenSizedTexture
{
    renderer:         Arc<super::Renderer>,
    descriptor:       ScreenSizedTextureDescriptor,
    current_size:     AtomicU32U32,
    critical_section: Mutex<ScreenSizedTextureCriticalSection>
}

impl Debug for ScreenSizedTexture
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        let (x, y) = self.current_size.load(Ordering::SeqCst);

        write!(
            f,
            "Screen Sized Texture {} with size {x}x{y}",
            self.descriptor.label
        )
    }
}

impl ScreenSizedTexture
{
    // TODO: should this be on the renderer?
    pub fn new(
        renderer: Arc<super::Renderer>,
        descriptor: ScreenSizedTextureDescriptor
    ) -> Arc<ScreenSizedTexture>
    {
        let c_renderer = renderer.clone();

        let (texture, view, size) = descriptor.create_texture(&renderer);

        let this = Arc::new(ScreenSizedTexture {
            renderer,
            descriptor,
            current_size: AtomicU32U32::new(size),
            critical_section: Mutex::new(ScreenSizedTextureCriticalSection {
                texture,
                view
            })
        });

        c_renderer.register_screen_sized_image(this.clone());

        this
    }

    pub(crate) fn resize_to_screen_size(&self)
    {
        let ScreenSizedTextureCriticalSection {
            texture,
            view
        } = &mut *self.critical_section.lock().unwrap();

        let (t, v, size) = self.descriptor.create_texture(&self.renderer);
        (*texture, *view) = (t, v);

        self.current_size.store(size, Ordering::Release);
    }
}

pub struct ScreenSizedTextureDescriptor
{
    pub label:           Cow<'static, str>,
    pub mip_level_count: u32,
    pub sample_count:    u32,
    pub format:          wgpu::TextureFormat,
    pub usage:           wgpu::TextureUsages,
    pub view_format:     wgpu::TextureFormat
}

impl ScreenSizedTextureDescriptor
{
    fn create_texture(
        &self,
        renderer: &super::Renderer
    ) -> (wgpu::Texture, wgpu::TextureView, (u32, u32))
    {
        let size = renderer.get_framebuffer_size();

        let texture = renderer.create_texture(&wgpu::TextureDescriptor {
            label:           Some(&self.label),
            size:            wgpu::Extent3d {
                width:                 size.x,
                height:                size.y,
                depth_or_array_layers: 1
            },
            mip_level_count: self.mip_level_count,
            sample_count:    self.sample_count,
            dimension:       wgpu::TextureDimension::D2,
            format:          self.format,
            usage:           self.usage,
            view_formats:    &[self.view_format]
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&self.label),
            ..Default::default()
        });

        (texture, view, (size.x, size.y))
    }
}

struct ScreenSizedTextureCriticalSection
{
    texture: wgpu::Texture,
    view:    wgpu::TextureView
}
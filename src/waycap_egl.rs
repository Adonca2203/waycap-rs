use std::{
    cell::Cell,
    ffi::{c_void, CStr},
};

use khronos_egl::{self as egl, ClientBuffer, Dynamic, Instance};

use crate::types::{error::Result, video_frame::DmaBufPlane};

type PFNGLEGLIMAGETARGETTEXTURE2DOESPROC =
    unsafe extern "C" fn(target: gl::types::GLenum, image: *const c_void);

unsafe impl Sync for EglContext {}
unsafe impl Send for EglContext {}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    INTEL,
    UNKNOWN,
}

impl From<&CStr> for GpuVendor {
    fn from(value: &CStr) -> Self {
        match value.to_str() {
            Ok(s) => {
                let s_lower = s.to_lowercase();
                if s_lower.contains("nvidia") {
                    Self::NVIDIA
                } else if s_lower.contains("ati")
                    || s_lower.contains("amd")
                    || s_lower.contains("advanced micro devices")
                {
                    Self::AMD
                } else if s_lower.contains("intel") {
                    Self::INTEL
                } else {
                    log::error!("The GPU vendor {s:?} is not supported.");
                    Self::UNKNOWN
                }
            }
            _ => Self::UNKNOWN,
        }
    }
}

pub struct EglContext {
    egl_instance: Instance<Dynamic<libloading::Library, egl::EGL1_5>>,
    display: egl::Display,
    context: egl::Context,
    surface: Option<egl::Surface>, // Optional for surfaceless context
    _config: egl::Config,
    dmabuf_supported: bool,
    dmabuf_modifiers_supported: bool,
    persistent_texture_id: Cell<Option<u32>>,
    gpu_vendor: GpuVendor,
    width: i32,
    height: i32,

    // Keep Wayland display alive
    _wayland_display: wayland_client::Display,
}

impl EglContext {
    pub fn new(width: i32, height: i32) -> Result<Self> {
        let lib =
            unsafe { libloading::Library::new("libEGL.so.1") }.expect("unable to find libEGL.so.1");
        let egl_instance = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib) }
            .expect("unable to load libEGL.so.1");

        egl_instance.bind_api(egl::OPENGL_ES_API)?;

        let wayland_display = wayland_client::Display::connect_to_env().unwrap();
        let display =
            unsafe { egl_instance.get_display(wayland_display.c_ptr() as *mut std::ffi::c_void) }
                .unwrap();

        egl_instance.initialize(display)?;

        let attributes = [
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES_BIT,
            egl::NONE,
        ];

        let config = egl_instance
            .choose_first_config(display, &attributes)?
            .or_else(|| {
                log::warn!("pbuffer config not found, trying window config");
                let fallback_attributes = [
                    egl::SURFACE_TYPE,
                    egl::WINDOW_BIT,
                    egl::RENDERABLE_TYPE,
                    egl::OPENGL_ES_BIT,
                    egl::NONE,
                ];
                egl_instance
                    .choose_first_config(display, &fallback_attributes)
                    .ok()
                    .flatten()
            })
            .expect("unable to find an appropriate EGL configuration");

        let context_attributes = [egl::CONTEXT_CLIENT_VERSION, 2, egl::NONE];

        let context = egl_instance.create_context(display, config, None, &context_attributes)?;

        let extensions = egl_instance.query_string(Some(display), egl::EXTENSIONS)?;
        let ext_str = extensions.to_string_lossy();

        // Check supported surface types for this config
        let surface_type = egl_instance.get_config_attrib(display, config, egl::SURFACE_TYPE)?;
        let supports_pbuffer = (surface_type & egl::PBUFFER_BIT) != 0;

        let surface = if supports_pbuffer {
            log::debug!("Using pbuffer surface");
            let surface_attributes = [egl::WIDTH, width, egl::HEIGHT, height, egl::NONE];
            let surface =
                egl_instance.create_pbuffer_surface(display, config, &surface_attributes)?;
            egl_instance.make_current(display, Some(surface), Some(surface), Some(context))?;
            Some(surface)
        } else if ext_str.contains("EGL_KHR_surfaceless_context") {
            log::debug!("Using surfaceless context");
            egl_instance.make_current(display, None, None, Some(context))?;
            None
        } else {
            return Err("No suitable surface type available".into());
        };

        gl::load_with(|symbol| egl_instance.get_proc_address(symbol).unwrap() as *const _);

        let (dmabuf_supported, dmabuf_modifiers_supported) =
            Self::check_dmabuf_support(&egl_instance, display).unwrap();

        let gpu_vendor = get_gpu_vendor();

        Ok(Self {
            egl_instance,
            display,
            _config: config,
            context,
            surface,
            dmabuf_supported,
            dmabuf_modifiers_supported,
            persistent_texture_id: Cell::new(None),
            gpu_vendor,
            width,
            height,

            _wayland_display: wayland_display,
        })
    }

    pub fn update_texture_from_image(&self, egl_image: egl::Image) -> Result<()> {
        assert!(self.persistent_texture_id.get().is_some());

        unsafe {
            // Create a temporary texture from the EGL image
            let mut temp_texture = 0;
            gl::GenTextures(1, &mut temp_texture);
            gl::BindTexture(gl::TEXTURE_2D, temp_texture);

            // Bind EGL image to temporary texture
            let egl_texture_2d = {
                let proc_name = "glEGLImageTargetTexture2DOES";
                let proc_addr = self.egl_instance.get_proc_address(proc_name);

                if proc_addr.is_none() {
                    gl::DeleteTextures(1, &temp_texture);
                    return Err("glEGLImageTargetTexture2DOES not available".into());
                } else {
                    std::mem::transmute::<
                        Option<extern "system" fn()>,
                        PFNGLEGLIMAGETARGETTEXTURE2DOESPROC,
                    >(proc_addr)
                }
            };

            egl_texture_2d(gl::TEXTURE_2D, egl_image.as_ptr());

            let gl_error = gl::GetError();
            if gl_error != gl::NO_ERROR {
                gl::DeleteTextures(1, &temp_texture);
                return Err(
                    format!("Failed to bind EGL image to temp texture: 0x{gl_error:x}").into(),
                );
            }

            // Get dimensions from the EGL image texture
            let mut width = 0;
            let mut height = 0;
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_WIDTH, &mut width);
            gl::GetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_HEIGHT, &mut height);

            // Create framebuffer for copying
            let mut fbo = 0;
            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);

            // Attach temporary EGL texture as source
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                temp_texture,
                0,
            );

            // Check framebuffer status
            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            if status != gl::FRAMEBUFFER_COMPLETE {
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &fbo);
                gl::DeleteTextures(1, &temp_texture);
                return Err(format!("Framebuffer not complete: 0x{status:x}").into());
            }

            // Bind persistent texture as destination
            gl::BindTexture(gl::TEXTURE_2D, self.persistent_texture_id.get().unwrap());

            // Use CopyTexSubImage2D instead of CopyTexImage2D
            // This updates existing texture data rather than reallocating
            gl::CopyTexSubImage2D(
                gl::TEXTURE_2D,
                0, // mipmap level
                0,
                0, // destination x, y offset in texture
                0,
                0,      // source x, y offset in framebuffer
                width,  // width to copy
                height, // height to copy
            );

            let gl_error = gl::GetError();
            if gl_error != gl::NO_ERROR {
                gl::BindTexture(gl::TEXTURE_2D, 0);
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &fbo);
                gl::DeleteTextures(1, &temp_texture);
                return Err(format!("Failed to copy texture data: 0x{gl_error:x}").into());
            }

            // Cleanup
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::DeleteFramebuffers(1, &fbo);
            gl::DeleteTextures(1, &temp_texture);

            Ok(())
        }
    }

    pub fn create_persistent_texture(&self) -> Result<()> {
        unsafe {
            let mut texture_id = 0;
            gl::GenTextures(1, &mut texture_id);
            gl::BindTexture(gl::TEXTURE_2D, texture_id);

            // Allocate texture storage with CUDA-compatible RGBA8 format
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as i32, // CUDA-compatible format
                self.width,
                self.height,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                std::ptr::null(), // No initial data
            );

            // Set texture parameters for better performance
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);

            gl::BindTexture(gl::TEXTURE_2D, 0);

            let gl_error = gl::GetError();
            if gl_error != gl::NO_ERROR {
                gl::DeleteTextures(1, &texture_id);
                return Err(format!("Failed to create persistent texture: 0x{gl_error:x}").into());
            }

            log::trace!(
                "✓ Created persistent texture: ID {texture_id} ({}x{})",
                self.width,
                self.height
            );
            self.persistent_texture_id.set(Some(texture_id));
            Ok(())
        }
    }

    fn check_dmabuf_support(
        egl_instance: &Instance<Dynamic<libloading::Library, egl::EGL1_5>>,
        display: egl::Display,
    ) -> Result<(bool, bool)> {
        let extensions = egl_instance.query_string(Some(display), egl::EXTENSIONS)?;
        let ext_str = extensions.to_string_lossy();

        let dmabuf_import = ext_str.contains("EGL_EXT_image_dma_buf_import");
        let dmabuf_modifiers = ext_str.contains("EGL_EXT_image_dma_buf_import_modifiers");

        if !dmabuf_import {
            return Err("EGL_EXT_image_dma_buf_import not supported".into());
        }

        Ok((dmabuf_import, dmabuf_modifiers))
    }

    pub fn create_image_from_dmabuf(
        &self,
        planes: &[DmaBufPlane],
        format: u32,
        width: u32,
        height: u32,
        modifier: u64,
    ) -> Result<egl::Image> {
        if !self.dmabuf_supported {
            return Err("DMA-BUF import not supported".into());
        }

        let mut attributes = vec![
            // EGL_LINUX_DRM_FOURCC_EXT
            0x3271,
            format as usize,
            egl::WIDTH as usize,
            width as usize,
            egl::HEIGHT as usize,
            height as usize,
        ];

        for (i, plane) in planes.iter().enumerate() {
            let plane_attrs = match i {
                0 => vec![
                    // EGL_DMA_BUF_PLANE0_FD_EXT
                    0x3272,
                    plane.fd as usize,
                    // EGL_DMA_BUF_PLANE0_OFFSET_EXT
                    0x3273,
                    plane.offset as usize,
                    // EGL_DMA_BUF_PLANE0_PITCH_EXT
                    0x3274,
                    plane.stride as usize,
                ],
                1 => vec![
                    // EGL_DMA_BUF_PLANE1_FD_EXT
                    0x3275,
                    plane.fd as usize,
                    // EGL_DMA_BUF_PLANE1_OFFSET_EXT
                    0x3276,
                    plane.offset as usize,
                    // EGL_DMA_BUF_PLANE1_PITCH_EXT
                    0x3277,
                    plane.stride as usize,
                ],
                2 => vec![
                    // EGL_DMA_BUF_PLANE2_FD_EXT
                    0x3278,
                    plane.fd as usize,
                    // EGL_DMA_BUF_PLANE2_OFFSET_EXT
                    0x3279,
                    plane.offset as usize,
                    // EGL_DMA_BUF_PLANE2_PITCH_EXT
                    0x327A,
                    plane.stride as usize,
                ],
                _ => break,
            };

            attributes.extend(plane_attrs);

            // Add modifiers if supported
            if self.dmabuf_modifiers_supported {
                let modifier_attrs = match i {
                    0 => vec![
                        // EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT
                        0x3443,
                        (modifier & 0xFFFFFFFF) as usize,
                        // EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT
                        0x3444,
                        (modifier >> 32) as usize,
                    ],
                    1 => vec![
                        // EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT
                        0x3445,
                        (modifier & 0xFFFFFFFF) as usize,
                        // EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT
                        0x3446,
                        (modifier >> 32) as usize,
                    ],
                    2 => vec![
                        // EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT
                        0x3447,
                        (modifier & 0xFFFFFFFF) as usize,
                        // EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT
                        0x3448,
                        (modifier >> 32) as usize,
                    ],
                    _ => break,
                };
                attributes.extend(modifier_attrs);
            }
        }

        attributes.push(egl::NONE as usize);

        // Create EGL image
        let image = self
            .egl_instance
            .create_image(
                self.display,
                unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
                // EGL_LINUX_DMA_BUF_EXT
                0x3270,
                unsafe { ClientBuffer::from_ptr(std::ptr::null_mut()) },
                &attributes,
            )
            .map_err(|e| format!("Failed to create EGL image from DMA-BUF: {e:?}"))?;

        Ok(image)
    }

    pub fn destroy_image(&self, image: egl::Image) -> Result<()> {
        self.egl_instance
            .destroy_image(self.display, image)
            .map_err(|e| format!("Failed to destroy EGL image: {e:?}").into())
    }

    pub fn delete_texture(&self, texture_id: u32) {
        unsafe {
            gl::DeleteTextures(1, &texture_id);
        }
    }

    pub fn make_current(&self) -> Result<()> {
        self.egl_instance.make_current(
            self.display,
            self.surface,
            self.surface,
            Some(self.context),
        )?;
        Ok(())
    }

    pub fn release_current(&self) -> Result<()> {
        self.egl_instance
            .make_current(self.display, None, None, None)?;
        Ok(())
    }

    pub fn get_texture_id(&self) -> Option<u32> {
        self.persistent_texture_id.get()
    }

    pub fn get_gpu_vendor(&self) -> GpuVendor {
        self.gpu_vendor
    }
}

impl Drop for EglContext {
    fn drop(&mut self) {
        let _ = self
            .egl_instance
            .make_current(self.display, None, None, None);

        if let Some(surface) = self.surface {
            let _ = self.egl_instance.destroy_surface(self.display, surface);
        }

        let _ = self
            .egl_instance
            .destroy_context(self.display, self.context);
        let _ = self.egl_instance.terminate(self.display);
        if let Some(texture) = self.persistent_texture_id.get() {
            self.delete_texture(texture);
        }
    }
}

fn get_gpu_vendor() -> GpuVendor {
    unsafe {
        let vendor_ptr = gl::GetString(gl::VENDOR);
        if vendor_ptr.is_null() {
            GpuVendor::UNKNOWN
        } else {
            let vendor = CStr::from_ptr(vendor_ptr as *const std::ffi::c_char);
            GpuVendor::from(vendor)
        }
    }
}

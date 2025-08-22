use std::ptr::null_mut;

use crossbeam::channel::{bounded, Receiver, Sender};
use cust::{
    prelude::Context,
    sys::{
        cuCtxSetCurrent, cuGraphicsMapResources, cuGraphicsResourceSetMapFlags_v2,
        cuGraphicsSubResourceGetMappedArray, cuGraphicsUnmapResources,
        cuGraphicsUnregisterResource, cuMemcpy2D_v2, CUDA_MEMCPY2D_v2, CUarray, CUdeviceptr,
        CUgraphicsResource, CUmemorytype, CUresult,
    },
};
use ffmpeg_next::{
    self as ffmpeg,
    ffi::{
        av_buffer_ref, av_buffer_unref, av_hwdevice_ctx_alloc, av_hwdevice_ctx_init,
        av_hwframe_ctx_init, av_hwframe_get_buffer, AVHWDeviceContext, AVHWFramesContext,
        AVPixelFormat,
    },
    Rational,
};
use pipewire as pw;

use crate::{
    encoders::video::{PipewireSPA, ProcessingThread, VideoEncoder},
    types::{
        config::QualityPreset,
        error::{Result, WaycapError},
        video_frame::{EncodedVideoFrame, RawVideoFrame},
    },
    utils::{extract_dmabuf_planes, TIME_UNIT_NS},
    waycap_egl::EglContext,
};
use khronos_egl::Image;

use super::{
    cuda::{cuGraphicsGLRegisterImage, AVCUDADeviceContext},
    video::{create_hw_frame_ctx, GOP_SIZE},
};

// Literally stole these by looking at what OBS uses
// just magic numbers to me no clue what these are
// but they enable DMA Buf so it is what it is
const NVIDIA_MODIFIERS: &[i64] = &[
    216172782120099856,
    216172782120099857,
    216172782120099858,
    216172782120099859,
    216172782120099860,
    216172782120099861,
    216172782128496656,
    216172782128496657,
    216172782128496658,
    216172782128496659,
    216172782128496660,
    216172782128496661,
    72057594037927935,
];

/// Encoder which provides frames encoded using Nvenc
///
/// Only available for Nvidia GPUs
pub struct NvencEncoder {
    encoder: Option<ffmpeg::codec::encoder::Video>,
    width: u32,
    height: u32,
    encoder_name: String,
    quality: QualityPreset,
    encoded_frame_recv: Option<Receiver<EncodedVideoFrame>>,
    encoded_frame_sender: Sender<EncodedVideoFrame>,

    cuda_ctx: Context,
    graphics_resource: CUgraphicsResource,
    egl_context: Option<Box<EglContext>>, // boxed egl context because its huge
    egl_texture: u32,
}

unsafe impl Send for NvencEncoder {}
unsafe impl Sync for NvencEncoder {}

impl VideoEncoder for NvencEncoder {
    type Output = EncodedVideoFrame;
    fn reset(&mut self) -> Result<()> {
        self.drop_processor();
        let new_encoder = Self::create_encoder(
            self.width,
            self.height,
            &self.encoder_name,
            &self.quality,
            &self.cuda_ctx,
        )?;

        self.encoder = Some(new_encoder);
        Ok(())
    }

    fn drop_processor(&mut self) {
        self.encoder.take();
    }

    fn output(&mut self) -> Option<Receiver<EncodedVideoFrame>> {
        self.encoded_frame_recv.clone()
    }

    fn drain(&mut self) -> Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            // Drain encoder
            encoder.send_eof()?;
            let mut packet = ffmpeg::codec::packet::Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {} // Discard these frames
        }
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video> {
        &self.encoder
    }
}
impl ProcessingThread for NvencEncoder {
    fn thread_setup(&mut self) -> Result<()> {
        self.egl_context = Some(Box::new(EglContext::new(
            self.width as i32,
            self.height as i32,
        )?));
        self.make_current()?;
        self.init_gl(None)?;
        Ok(())
    }

    fn thread_teardown(&mut self) -> Result<()> {
        self.egl_context.as_mut().unwrap().release_current()
    }

    fn process(&mut self, frame: RawVideoFrame) -> Result<()> {
        match egl_img_from_dmabuf(self.egl_context.as_ref().unwrap(), &frame) {
            Ok(img) => {
                if let Some(ref mut encoder) = self.encoder {
                    let mut cuda_frame = ffmpeg::util::frame::Video::new(
                        ffmpeg_next::format::Pixel::CUDA,
                        encoder.width(),
                        encoder.height(),
                    );

                    unsafe {
                        let ret = av_hwframe_get_buffer(
                            (*encoder.as_ptr()).hw_frames_ctx,
                            cuda_frame.as_mut_ptr(),
                            0,
                        );
                        if ret < 0 {
                            return Err(WaycapError::Encoding(format!(
                                "Failed to allocate CUDA frame buffer: {ret}",
                            )));
                        }

                        let result =
                            cuGraphicsMapResources(1, &mut self.graphics_resource, null_mut());
                        if result != CUresult::CUDA_SUCCESS {
                            gl::BindTexture(gl::TEXTURE_2D, 0);
                            return Err(WaycapError::Encoding(format!(
                                "Error mapping GL image to CUDA: {result:?}",
                            )));
                        }

                        let mut cuda_array: CUarray = null_mut();

                        let result = cuGraphicsSubResourceGetMappedArray(
                            &mut cuda_array,
                            self.graphics_resource,
                            0,
                            0,
                        );
                        if result != CUresult::CUDA_SUCCESS {
                            cuGraphicsUnmapResources(1, &mut self.graphics_resource, null_mut());
                            gl::BindTexture(gl::TEXTURE_2D, 0);
                            return Err(WaycapError::Encoding(format!(
                                "Error getting CUDA Array: {result:?}",
                            )));
                        }

                        let copy_params = CUDA_MEMCPY2D_v2 {
                            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                            srcArray: cuda_array,
                            srcXInBytes: 0,
                            srcY: 0,
                            srcHost: std::ptr::null(),
                            srcDevice: 0,
                            srcPitch: 0,

                            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                            dstDevice: (*cuda_frame.as_ptr()).data[0] as CUdeviceptr,
                            dstPitch: (*cuda_frame.as_ptr()).linesize[0] as usize,
                            dstXInBytes: 0,
                            dstY: 0,
                            dstHost: std::ptr::null_mut(),
                            dstArray: std::ptr::null_mut(),

                            // RGBA is 4 bytes per pixel
                            WidthInBytes: (encoder.width() * 4) as usize,
                            Height: encoder.height() as usize,
                        };

                        let result = cuMemcpy2D_v2(&copy_params);
                        if result != CUresult::CUDA_SUCCESS {
                            cuGraphicsUnmapResources(1, &mut self.graphics_resource, null_mut());
                            gl::BindTexture(gl::TEXTURE_2D, 0);
                            return Err(WaycapError::Encoding(format!(
                                "Error mapping cuda frame: {result:?}",
                            )));
                        }

                        // Cleanup
                        let result =
                            cuGraphicsUnmapResources(1, &mut self.graphics_resource, null_mut());
                        if result != CUresult::CUDA_SUCCESS {
                            return Err(WaycapError::Encoding(format!(
                                "Could not unmap resource: {result:?}",
                            )));
                        }

                        gl::BindTexture(gl::TEXTURE_2D, 0);
                    }

                    cuda_frame.set_pts(Some(frame.timestamp));
                    encoder.send_frame(&cuda_frame)?;

                    let mut packet = ffmpeg::codec::packet::Packet::empty();
                    if encoder.receive_packet(&mut packet).is_ok() {
                        if let Some(data) = packet.data() {
                            match self.encoded_frame_sender.try_send(EncodedVideoFrame {
                                data: data.to_vec(),
                                is_keyframe: packet.is_key(),
                                pts: packet.pts().unwrap_or(0),
                                dts: packet.dts().unwrap_or(0),
                            }) {
                                Ok(_) => {}
                                Err(crossbeam::channel::TrySendError::Full(_)) => {
                                    log::error!(
                                        "Could not send encoded video frame. Receiver is full"
                                    );
                                }
                                Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                                    log::error!(
                                        "Could not send encoded video frame. Receiver disconnected"
                                    );
                                }
                            }
                        };
                    }
                }
                self.egl_context.as_ref().unwrap().destroy_image(img)?;
            }
            Err(e) => log::error!("Could not process dma buf frame: {e:?}"),
        }
        Ok(())
    }
}

impl PipewireSPA for NvencEncoder {
    fn get_spa_definition() -> Result<pw::spa::pod::Object> {
        let nvidia_mod_property = pw::spa::pod::Property {
            key: pw::spa::param::format::FormatProperties::VideoModifier.as_raw(),
            flags: pw::spa::pod::PropertyFlags::empty(),
            value: pw::spa::pod::Value::Choice(pw::spa::pod::ChoiceValue::Long(
                pw::spa::utils::Choice::<i64>(
                    pw::spa::utils::ChoiceFlags::empty(),
                    pw::spa::utils::ChoiceEnum::<i64>::Enum {
                        default: NVIDIA_MODIFIERS[0],
                        alternatives: NVIDIA_MODIFIERS.to_vec(),
                    },
                ),
            )),
        };

        Ok(pw::spa::pod::object!(
            pw::spa::utils::SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaType,
                Id,
                pw::spa::param::format::MediaType::Video
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaSubtype,
                Id,
                pw::spa::param::format::MediaSubtype::Raw
            ),
            nvidia_mod_property,
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoFormat,
                Choice,
                Enum,
                Id,
                pw::spa::param::video::VideoFormat::NV12,
                pw::spa::param::video::VideoFormat::I420,
                pw::spa::param::video::VideoFormat::BGRA
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoSize,
                Choice,
                Range,
                Rectangle,
                pw::spa::utils::Rectangle {
                    width: 2560,
                    height: 1440
                }, // Default
                pw::spa::utils::Rectangle {
                    width: 1,
                    height: 1
                }, // Min
                pw::spa::utils::Rectangle {
                    width: 4096,
                    height: 4096
                } // Max
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoFramerate,
                Choice,
                Range,
                Fraction,
                pw::spa::utils::Fraction { num: 240, denom: 1 }, // Default
                pw::spa::utils::Fraction { num: 0, denom: 1 },   // Min
                pw::spa::utils::Fraction { num: 244, denom: 1 }  // Max
            ),
        ))
    }
}

fn egl_img_from_dmabuf(egl_ctx: &EglContext, raw_frame: &RawVideoFrame) -> Result<Image> {
    let dma_buf_planes = extract_dmabuf_planes(raw_frame)?;

    let format = drm_fourcc::DrmFourcc::Argb8888 as u32;
    let modifier = raw_frame.modifier;

    let egl_image = egl_ctx.create_image_from_dmabuf(
        &dma_buf_planes,
        format,
        raw_frame.dimensions.width,
        raw_frame.dimensions.height,
        modifier,
    )?;

    egl_ctx.update_texture_from_image(egl_image)?;

    Ok(egl_image)
}

impl NvencEncoder {
    pub(crate) fn new(width: u32, height: u32, quality: QualityPreset) -> Result<Self> {
        let encoder_name = "h264_nvenc";

        let (frame_tx, frame_rx): (Sender<EncodedVideoFrame>, Receiver<EncodedVideoFrame>) =
            bounded(10);
        let cuda_ctx = cust::quick_init().unwrap();

        let encoder = Self::create_encoder(width, height, encoder_name, &quality, &cuda_ctx)?;

        Ok(Self {
            encoder: Some(encoder),
            width,
            height,
            encoder_name: encoder_name.to_string(),
            quality,
            encoded_frame_recv: Some(frame_rx),
            encoded_frame_sender: frame_tx,
            cuda_ctx,
            graphics_resource: null_mut(),
            egl_context: None,
            egl_texture: 0,
        })
    }

    fn create_encoder(
        width: u32,
        height: u32,
        encoder: &str,
        quality: &QualityPreset,
        cuda_ctx: &Context,
    ) -> Result<ffmpeg::codec::encoder::Video> {
        let encoder_codec =
            ffmpeg::codec::encoder::find_by_name(encoder).ok_or(ffmpeg::Error::EncoderNotFound)?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .video()?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(ffmpeg::format::Pixel::CUDA);
        encoder_ctx.set_bit_rate(16_000_000);

        unsafe {
            // Set up the cuda context
            let nvenc_device =
                av_hwdevice_ctx_alloc(ffmpeg_next::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA);

            if nvenc_device.is_null() {
                return Err(WaycapError::Init(
                    "Could not initialize nvenc device".into(),
                ));
            }

            let hw_device_ctx = (*nvenc_device).data as *mut AVHWDeviceContext;
            let cuda_device_ctx = (*hw_device_ctx).hwctx as *mut AVCUDADeviceContext;
            (*cuda_device_ctx).cuda_ctx = cuda_ctx.as_raw();

            let err = av_hwdevice_ctx_init(nvenc_device);

            if err < 0 {
                return Err(WaycapError::Init(format!(
                    "Error trying to initialize hw device context: {err:?}",
                )));
            }

            let hw_device_ctx = (*nvenc_device).data as *mut AVHWDeviceContext;
            let cuda_device_ctx = (*hw_device_ctx).hwctx as *mut AVCUDADeviceContext;
            (*cuda_device_ctx).cuda_ctx = cuda_ctx.as_raw();

            let mut frame_ctx = create_hw_frame_ctx(nvenc_device)?;

            if frame_ctx.is_null() {
                return Err(WaycapError::Init(
                    "Could not initialize hw frame context".into(),
                ));
            }

            let hw_frame_context = &mut *((*frame_ctx).data as *mut AVHWFramesContext);

            hw_frame_context.width = width as i32;
            hw_frame_context.height = height as i32;
            hw_frame_context.sw_format = AVPixelFormat::AV_PIX_FMT_RGBA;
            hw_frame_context.format = encoder_ctx.format().into();
            hw_frame_context.device_ctx = hw_device_ctx;
            // Decides buffer size if we do not pop frame from the encoder we cannot
            // keep pushing. Smaller better as we reserve less GPU memory
            hw_frame_context.initial_pool_size = 2;

            let err = av_hwframe_ctx_init(frame_ctx);
            if err < 0 {
                return Err(WaycapError::Init(format!(
                    "Error trying to initialize hw frame context: {err:?}",
                )));
            }

            (*encoder_ctx.as_mut_ptr()).hw_device_ctx = av_buffer_ref(nvenc_device);
            (*encoder_ctx.as_mut_ptr()).hw_frames_ctx = av_buffer_ref(frame_ctx);

            av_buffer_unref(&mut frame_ctx);
        }

        encoder_ctx.set_time_base(Rational::new(1, TIME_UNIT_NS as i32));
        encoder_ctx.set_gop(GOP_SIZE);

        let encoder_params = ffmpeg::codec::Parameters::new();

        let opts = Self::get_encoder_params(quality);

        encoder_ctx.set_parameters(encoder_params)?;
        let encoder = encoder_ctx.open_with(opts)?;

        Ok(encoder)
    }

    fn get_encoder_params(quality: &QualityPreset) -> ffmpeg::Dictionary<'_> {
        let mut opts = ffmpeg::Dictionary::new();
        opts.set("vsync", "vfr");
        opts.set("rc", "vbr");
        opts.set("tune", "hq");
        match quality {
            QualityPreset::Low => {
                opts.set("preset", "p2");
                opts.set("cq", "30");
                opts.set("b:v", "20M");
            }
            QualityPreset::Medium => {
                opts.set("preset", "p4");
                opts.set("cq", "25");
                opts.set("b:v", "40M");
            }
            QualityPreset::High => {
                opts.set("preset", "p7");
                opts.set("cq", "20");
                opts.set("b:v", "80M");
            }
            QualityPreset::Ultra => {
                opts.set("preset", "p7");
                opts.set("cq", "15");
                opts.set("b:v", "120M");
            }
        }
        opts
    }

    fn init_gl(&mut self, texture_id: Option<u32>) -> Result<()> {
        self.egl_texture = match texture_id {
            Some(texture_id) => texture_id,
            None => {
                self.egl_context
                    .as_ref()
                    .unwrap()
                    .create_persistent_texture()?;
                self.egl_context.as_ref().unwrap().get_texture_id().unwrap()
            }
        };

        unsafe {
            // Try to register GL texture with CUDA
            let result = cuGraphicsGLRegisterImage(
                &mut self.graphics_resource,
                self.egl_texture,
                gl::TEXTURE_2D, // GL_TEXTURE_2D
                0x00,           // CU_GRAPHICS_REGISTER_FLAGS_READ_NONE
            );

            if result != CUresult::CUDA_SUCCESS {
                return Err(WaycapError::Init(format!(
                    "Error registering GL texture to CUDA: {result:?}",
                )));
            }

            let result = cuGraphicsResourceSetMapFlags_v2(self.graphics_resource, 0);

            if result != CUresult::CUDA_SUCCESS {
                cuGraphicsUnregisterResource(self.graphics_resource);
                gl::BindTexture(gl::TEXTURE_2D, 0);
                return Err(WaycapError::Init(format!(
                    "Failed to set graphics resource map flags: {result:?}",
                )));
            }
        }

        Ok(())
    }

    /// Set cuda  context to current thread
    fn make_current(&self) -> Result<()> {
        unsafe { cuCtxSetCurrent(self.cuda_ctx.as_raw()) };
        Ok(())
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        if let Err(e) = self.drain() {
            if let WaycapError::FFmpeg(ffmpeg::Error::Other { errno: 541478725 }) = e {
                // This seems normal when a stream is empty,
                // like when its been drained before (in Capture::finish for example)
                // see: https://trac.ffmpeg.org/ticket/7290
            } else {
                log::error!("Error while draining nvenc encoder during drop: {e:?}");
            }
        }
        self.drop_processor();

        self.egl_context.as_ref().unwrap().make_current().unwrap();
        if let Err(e) = self.make_current() {
            log::error!("Could not make context current during drop: {e:?}");
        }

        let result = unsafe { cuGraphicsUnregisterResource(self.graphics_resource) };
        if result != CUresult::CUDA_SUCCESS {
            log::error!("Error cleaning up graphics resource: {result:?}");
        }
    }
}

use std::{any::Any, ptr::null_mut};

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

use crate::types::{
    config::QualityPreset,
    error::{Result, WaycapError},
    video_frame::{EncodedVideoFrame, RawVideoFrame},
};

use super::{
    cuda::{cuGraphicsGLRegisterImage, AVCUDADeviceContext},
    video::{create_hw_frame_ctx, VideoEncoder, GOP_SIZE},
};

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
    egl_texture: u32,
}

unsafe impl Send for NvencEncoder {}
unsafe impl Sync for NvencEncoder {}

impl VideoEncoder for NvencEncoder {
    fn new(width: u32, height: u32, quality: QualityPreset) -> Result<Self>
    where
        Self: Sized,
    {
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
            egl_texture: 0,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn process(&mut self, frame: &RawVideoFrame) -> Result<()> {
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

                let result = cuGraphicsMapResources(1, &mut self.graphics_resource, null_mut());
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
                let result = cuGraphicsUnmapResources(1, &mut self.graphics_resource, null_mut());
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
                            log::error!("Could not send encoded video frame. Receiver is full");
                        }
                        Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                            log::error!(
                                "Cound not send encoded video frame. Receiver disconnected"
                            );
                        }
                    }
                };
            }
        }
        Ok(())
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

    fn reset(&mut self) -> Result<()> {
        self.drop_encoder();
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

    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video> {
        &self.encoder
    }

    fn drop_encoder(&mut self) {
        self.encoder.take();
    }

    fn take_encoded_recv(&mut self) -> Option<Receiver<EncodedVideoFrame>> {
        self.encoded_frame_recv.take()
    }
}

impl NvencEncoder {
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

        encoder_ctx.set_time_base(Rational::new(1, 1_000_000));
        encoder_ctx.set_gop(GOP_SIZE);

        let encoder_params = ffmpeg::codec::Parameters::new();

        let opts = Self::get_encoder_params(quality);

        encoder_ctx.set_parameters(encoder_params)?;
        let encoder = encoder_ctx.open_with(opts)?;

        Ok(encoder)
    }

    fn get_encoder_params(quality: &QualityPreset) -> ffmpeg::Dictionary {
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

    pub fn init_gl(&mut self, texture_id: u32) -> Result<()> {
        unsafe {
            self.egl_texture = texture_id;
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

    pub fn make_current(&self) -> Result<()> {
        unsafe { cuCtxSetCurrent(self.cuda_ctx.as_raw()) };
        Ok(())
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        if let Err(e) = self.drain() {
            log::error!("Error while draining nvenc encoder during drop: {e:?}");
        }
        self.drop_encoder();

        if let Err(e) = self.make_current() {
            log::error!("Could not make context current during drop: {e:?}");
        }

        let result = unsafe { cuGraphicsUnregisterResource(self.graphics_resource) };
        if result != CUresult::CUDA_SUCCESS {
            log::error!("Error cleaning up graphics resource: {result:?}");
        }
    }
}

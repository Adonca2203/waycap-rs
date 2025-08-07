use std::{any::Any, ptr::null_mut};

use crossbeam::channel::{bounded, Receiver, Sender};
use drm_fourcc::DrmFourcc;
use ffmpeg_next::{
    self as ffmpeg,
    ffi::{
        av_buffer_create, av_buffer_default_free, av_buffer_ref, av_buffer_unref,
        av_hwframe_ctx_init, AVDRMFrameDescriptor, AVHWDeviceContext, AVHWFramesContext,
        AVPixelFormat,
    },
    Rational,
};

use crate::{
    types::{
        config::QualityPreset,
        error::{Result, WaycapError},
        video_frame::{EncodedVideoFrame, RawVideoFrame},
    },
    utils::TIME_UNIT_NS,
};

use super::video::{create_hw_device, create_hw_frame_ctx, VideoEncoder, GOP_SIZE};

pub struct VaapiEncoder {
    encoder: Option<ffmpeg::codec::encoder::Video>,
    width: u32,
    height: u32,
    encoder_name: String,
    quality: QualityPreset,
    encoded_frame_recv: Option<Receiver<EncodedVideoFrame>>,
    encoded_frame_sender: Sender<EncodedVideoFrame>,
    filter_graph: Option<ffmpeg::filter::Graph>,
}

impl VideoEncoder for VaapiEncoder {
    fn new(width: u32, height: u32, quality: QualityPreset) -> Result<Self>
    where
        Self: Sized,
    {
        let encoder_name = "h264_vaapi";
        let encoder = Self::create_encoder(width, height, encoder_name, &quality)?;

        let (frame_tx, frame_rx): (Sender<EncodedVideoFrame>, Receiver<EncodedVideoFrame>) =
            bounded(10);
        let filter_graph = Some(Self::create_filter_graph(&encoder, width, height)?);

        Ok(Self {
            encoder: Some(encoder),
            width,
            height,
            encoder_name: encoder_name.to_string(),
            quality,
            encoded_frame_recv: Some(frame_rx),
            encoded_frame_sender: frame_tx,
            filter_graph,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn process(&mut self, frame: &RawVideoFrame) -> Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            if let Some(fd) = frame.dmabuf_fd {
                let mut drm_frame = ffmpeg::util::frame::Video::new(
                    ffmpeg_next::format::Pixel::DRM_PRIME,
                    encoder.width(),
                    encoder.height(),
                );
                unsafe {
                    // Create DRM descriptor that points to the DMA buffer
                    let drm_desc =
                        Box::into_raw(Box::new(std::mem::zeroed::<AVDRMFrameDescriptor>()));

                    (*drm_desc).nb_objects = 1;
                    (*drm_desc).objects[0].fd = fd;
                    (*drm_desc).objects[0].size = 0;
                    (*drm_desc).objects[0].format_modifier = 0;

                    (*drm_desc).nb_layers = 1;
                    (*drm_desc).layers[0].format = DrmFourcc::Argb8888 as u32;
                    (*drm_desc).layers[0].nb_planes = 1;
                    (*drm_desc).layers[0].planes[0].object_index = 0;
                    (*drm_desc).layers[0].planes[0].offset = frame.offset as isize;
                    (*drm_desc).layers[0].planes[0].pitch = frame.stride as isize;

                    // Attach descriptor to frame
                    (*drm_frame.as_mut_ptr()).data[0] = drm_desc as *mut u8;
                    (*drm_frame.as_mut_ptr()).buf[0] = av_buffer_create(
                        drm_desc as *mut u8,
                        std::mem::size_of::<AVDRMFrameDescriptor>(),
                        Some(av_buffer_default_free),
                        null_mut(),
                        0,
                    );

                    (*drm_frame.as_mut_ptr()).hw_frames_ctx =
                        av_buffer_ref((*encoder.as_ptr()).hw_frames_ctx);
                }

                drm_frame.set_pts(Some(frame.timestamp));
                self.filter_graph
                    .as_mut()
                    .unwrap()
                    .get("in")
                    .unwrap()
                    .source()
                    .add(&drm_frame)
                    .unwrap();

                let mut filtered = ffmpeg::util::frame::Video::empty();
                if self
                    .filter_graph
                    .as_mut()
                    .unwrap()
                    .get("out")
                    .unwrap()
                    .sink()
                    .frame(&mut filtered)
                    .is_ok()
                {
                    encoder.send_frame(&filtered)?;
                }
            }

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

    /// Drain the filter graph and encoder of any remaining frames it is processing
    fn drain(&mut self) -> Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            // Drain the filter graph
            let mut filtered = ffmpeg::util::frame::Video::empty();
            while self
                .filter_graph
                .as_mut()
                .unwrap()
                .get("out")
                .unwrap()
                .sink()
                .frame(&mut filtered)
                .is_ok()
            {
                encoder.send_frame(&filtered)?;
            }

            // Drain encoder
            encoder.send_eof()?;
            let mut packet = ffmpeg::codec::packet::Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {} // Discard these frames
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.drop_encoder();
        let new_encoder =
            Self::create_encoder(self.width, self.height, &self.encoder_name, &self.quality)?;

        let new_filter_graph = Self::create_filter_graph(&new_encoder, self.width, self.height)?;

        self.encoder = Some(new_encoder);
        self.filter_graph = Some(new_filter_graph);
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video> {
        &self.encoder
    }

    fn drop_encoder(&mut self) {
        self.encoder.take();
        self.filter_graph.take();
    }

    fn get_encoded_recv(&mut self) -> Option<Receiver<EncodedVideoFrame>> {
        self.encoded_frame_recv.clone()
    }
}

impl VaapiEncoder {
    fn create_encoder(
        width: u32,
        height: u32,
        encoder: &str,
        quality: &QualityPreset,
    ) -> Result<ffmpeg::codec::encoder::Video> {
        let encoder_codec =
            ffmpeg::codec::encoder::find_by_name(encoder).ok_or(ffmpeg::Error::EncoderNotFound)?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .video()?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(ffmpeg::format::Pixel::VAAPI);
        // Configuration inspiration from
        // https://git.dec05eba.com/gpu-screen-recorder/tree/src/capture/xcomposite_drm.c?id=8cbdb596ebf79587a432ed40583630b6cd39ed88
        let mut vaapi_device =
            create_hw_device(ffmpeg_next::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI)?;
        let mut frame_ctx = create_hw_frame_ctx(vaapi_device)?;

        unsafe {
            let hw_frame_context = &mut *((*frame_ctx).data as *mut AVHWFramesContext);
            hw_frame_context.width = width as i32;
            hw_frame_context.height = height as i32;
            hw_frame_context.sw_format = AVPixelFormat::AV_PIX_FMT_NV12;
            hw_frame_context.format = encoder_ctx.format().into();
            hw_frame_context.device_ref = av_buffer_ref(vaapi_device);
            hw_frame_context.device_ctx = (*vaapi_device).data as *mut AVHWDeviceContext;
            // Decides buffer size if we do not pop frame from the encoder we cannot
            // keep pushing. Smaller better as we reserve less GPU memory
            hw_frame_context.initial_pool_size = 2;

            let err = av_hwframe_ctx_init(frame_ctx);
            if err < 0 {
                return Err(WaycapError::Init(format!(
                    "Error trying to initialize hw frame context: {err:?}",
                )));
            }

            (*encoder_ctx.as_mut_ptr()).hw_device_ctx = av_buffer_ref(vaapi_device);
            (*encoder_ctx.as_mut_ptr()).hw_frames_ctx = av_buffer_ref(frame_ctx);

            av_buffer_unref(&mut vaapi_device);
            av_buffer_unref(&mut frame_ctx);
        }

        // These should be part of a config file
        encoder_ctx.set_time_base(Rational::new(1, TIME_UNIT_NS as i32));

        // Needed to insert I-Frames more frequently so we don't lose full seconds
        // when popping frames from the front
        encoder_ctx.set_gop(GOP_SIZE);

        let encoder_params = ffmpeg::codec::Parameters::new();

        let opts = Self::get_encoder_params(quality);

        encoder_ctx.set_parameters(encoder_params)?;
        let encoder = encoder_ctx.open_with(opts)?;
        Ok(encoder)
    }

    fn get_encoder_params(quality: &QualityPreset) -> ffmpeg::Dictionary<'_>{
        let mut opts = ffmpeg::Dictionary::new();
        opts.set("vsync", "vfr");
        opts.set("rc", "VBR");
        match quality {
            QualityPreset::Low => {
                opts.set("qp", "30");
            }
            QualityPreset::Medium => {
                opts.set("qp", "25");
            }
            QualityPreset::High => {
                opts.set("qp", "20");
            }
            QualityPreset::Ultra => {
                opts.set("qp", "15");
            }
        }
        opts
    }

    fn create_filter_graph(
        encoder: &ffmpeg::codec::encoder::Video,
        width: u32,
        height: u32,
    ) -> Result<ffmpeg::filter::Graph> {
        let mut graph = ffmpeg::filter::Graph::new();

        let args = format!("video_size={width}x{height}:pix_fmt=bgra:time_base=1/1000000",);

        let mut input = graph.add(&ffmpeg::filter::find("buffer").unwrap(), "in", &args)?;

        let mut hwmap = graph.add(
            &ffmpeg::filter::find("hwmap").unwrap(),
            "hwmap",
            "mode=read+write:derive_device=vaapi",
        )?;

        let scale_args = format!("w={width}:h={height}:format=nv12:out_range=tv");
        let mut scale = graph.add(
            &ffmpeg::filter::find("scale_vaapi").unwrap(),
            "scale",
            &scale_args,
        )?;

        let mut out = graph.add(&ffmpeg::filter::find("buffersink").unwrap(), "out", "")?;
        unsafe {
            let dev = (*encoder.as_ptr()).hw_device_ctx;

            (*hwmap.as_mut_ptr()).hw_device_ctx = av_buffer_ref(dev);
        }

        input.link(0, &mut hwmap, 0);
        hwmap.link(0, &mut scale, 0);
        scale.link(0, &mut out, 0);

        graph.validate()?;
        log::trace!("VAAPI Graph\n{}", graph.dump());

        Ok(graph)
    }
}

impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        if let Err(e) = self.drain() {
            log::error!("Error while draining vaapi encoder during drop: {e:?}");
        }
        self.drop_encoder();
    }
}

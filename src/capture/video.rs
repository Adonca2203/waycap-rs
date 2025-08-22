use std::{
    os::fd::{FromRawFd, OwnedFd, RawFd},
    sync::{
        mpsc::{self},
        Arc,
    },
};

use crossbeam::channel::Sender;
use pipewire::{
    self as pw,
    context::Context,
    core::{Core, Listener},
    main_loop::MainLoop,
    spa::{
        buffer::{Data, DataType},
        utils::Direction,
    },
    stream::{Stream, StreamFlags, StreamListener, StreamState},
    sys::pw_stream_get_nsec,
};
use pw::{properties::properties, spa};

use spa::pod::Pod;

use crate::{
    types::{
        error::{Result, WaycapError},
        video_frame::RawVideoFrame,
    }, CaptureControls, ReadyState, Resolution
};

use super::Terminate;



pub struct VideoCapture {
    termination_recv: Option<pw::channel::Receiver<Terminate>>,
    pipewire_state: PipewireState,
}

// Need to keep all of these alive even if never referenced
struct PipewireState {
    pw_loop: MainLoop,
    _pw_context: Context,
    _core: Core,
    _core_listener: Listener,
    _stream: Stream,
    _stream_listener: StreamListener<UserData>,
}

#[derive(Clone, Copy, Default)]
struct UserData {
    video_format: spa::param::video::VideoInfoRaw,
}

impl VideoCapture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipewire_fd: RawFd,
        stream_node: u32,
        ready_state: Arc<ReadyState>,
        controls: Arc<CaptureControls>,
        resolution_sender: mpsc::Sender<Resolution>,
        frame_tx: Sender<RawVideoFrame>,
        termination_recv: pw::channel::Receiver<Terminate>,
        pw_obj: spa::pod::Object,
    ) -> Result<Self> {
        let pw_loop = MainLoop::new(None)?;
        let context = Context::new(&pw_loop)?;
        let mut core = context.connect_fd(unsafe { OwnedFd::from_raw_fd(pipewire_fd) }, None)?;
        let core_listener = Self::setup_core_listener(&mut core)?;
        let mut stream = Self::create_stream(&core)?;
        let stream_listener = Self::setup_stream_listener(
            &mut stream,
            UserData::default(),
            ready_state,
            &controls,
            resolution_sender.clone(),
            frame_tx.clone(),
        )?;
        Self::connect_stream(&mut stream, stream_node, pw_obj)?;

        Ok(Self {
            termination_recv: Some(termination_recv),
            pipewire_state: PipewireState {
                pw_loop,
                _pw_context: context,
                _core: core,
                _core_listener: core_listener,
                _stream: stream,
                _stream_listener: stream_listener,
            },
        })
    }

    fn create_stream(core: &Core) -> Result<Stream> {
        match Stream::new(
            core,
            "waycap-video",
            properties! {
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Screen",
            },
        ) {
            Ok(stream) => Ok(stream),
            Err(e) => Err(WaycapError::from(e)),
        }
    }

    fn setup_core_listener(core: &mut Core) -> Result<Listener> {
        Ok(core
            .add_listener_local()
            .info(|i| log::debug!("VIDEO CORE:\n{i:#?}"))
            .error(|e, f, g, h| log::error!("{e},{f},{g},{h}"))
            .done(|d, _| log::debug!("DONE: {d}"))
            .register())
    }

    #[allow(clippy::too_many_arguments)]
    fn setup_stream_listener(
        stream: &mut Stream,
        data: UserData,
        ready_state: Arc<ReadyState>,
        controls: &Arc<CaptureControls>,
        resolution_sender: mpsc::Sender<Resolution>,
        frame_tx: Sender<RawVideoFrame>,
    ) -> Result<StreamListener<UserData>> {
        let ready_state_clone = Arc::clone(&ready_state);
        let controls_clone = Arc::clone(controls);

        let stream_listener = stream
            .add_local_listener_with_user_data(data)
            .state_changed(move |_, _, old, new| {
                log::info!("Video Stream State Changed: {old:?} -> {new:?}");
                ready_state.video.store(
                    new == StreamState::Streaming,
                    std::sync::atomic::Ordering::Release,
                );
            })
            .param_changed(move |_, user_data, id, param| {
                let Some(param) = param else {
                    return;
                };

                if id != pw::spa::param::ParamType::Format.as_raw() {
                    return;
                }

                let (media_type, media_subtype) =
                    match pw::spa::param::format_utils::parse_format(param) {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                if media_type != pw::spa::param::format::MediaType::Video
                    || media_subtype != pw::spa::param::format::MediaSubtype::Raw
                {   
                    return;
                }

                user_data
                    .video_format
                    .parse(param)
                    .expect("Failed to parse param");

                log::debug!(
                    "  format: {} ({:?})",
                    user_data.video_format.format().as_raw(),
                    user_data.video_format.format()
                );

                let (width, height) = (

                    user_data.video_format.size().width,
                    user_data.video_format.size().height,
                    );
                match resolution_sender.send(Resolution { width, height }) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Tried to send resolution update {width}x{height} but ran into an error on the channel: {e}");
                    }
                };

                log::debug!(
                    "  size: {}x{}",
                    user_data.video_format.size().width,
                    user_data.video_format.size().height
                );
                log::debug!(
                    "  framerate: {}/{}",
                    user_data.video_format.framerate().num,
                    user_data.video_format.framerate().denom
                );
            })
            .process(move |stream, udata| {
                match stream.dequeue_buffer() {
                    None => log::debug!("out of buffers"),
                    Some(mut buffer) => {
                        // Wait until audio is streaming before we try to process
                        if !ready_state_clone.audio_ready() || controls_clone.skip_processing() {
                            return;
                        }

                        let datas = buffer.datas_mut();
                        if datas.is_empty() {
                            return;
                        }

                        let data = &mut datas[0];

                        let fd = Self::get_dmabuf_fd(data);

                        match frame_tx.try_send(RawVideoFrame {
                            data: data.data().unwrap_or_default().to_vec(),
                            timestamp: unsafe { pw_stream_get_nsec(stream.as_raw_ptr())} as i64,
                            dmabuf_fd: fd,
                            stride: data.chunk().stride(),
                            offset: data.chunk().offset(),
                            size: data.chunk().size(),
                            modifier: udata.video_format.modifier(),
                            format: udata.video_format.format(),
                            dimensions: udata.video_format.size()
                        }) {
                            Ok(_) => {}
                            Err(crossbeam::channel::TrySendError::Full(frame)) => {
                                log::error!(
                                    "Could not send video frame at: {}. Channel full.",
                                    frame.timestamp
                                );
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(frame)) => {
                                // TODO: If we disconnected, terminate the session instead of
                                // throwing an error it means the receiver was dropped.
                                log::error!(
                                    "Could not send video frame at: {}. Connection closed.",
                                    frame.timestamp
                                );
                            }
                        }
                    }
                }
            })
            .register()?;

        Ok(stream_listener)
    }

    fn connect_stream(
        stream: &mut Stream,
        stream_node: u32,
        pw_obj: spa::pod::Object,
    ) -> Result<()> {
        let video_spa_values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(pw_obj),
        )
        .unwrap()
        .0
        .into_inner();

        let mut video_params = [Pod::from_bytes(&video_spa_values).unwrap()];
        stream.connect(
            Direction::Input,
            Some(stream_node),
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut video_params,
        )?;

        Ok(())
    }

    /// Finalizes the pipewire run loop with a terminate receiver and runs it
    /// Blocks the current thread so this must be called in a separate thread
    pub fn run(&mut self) -> Result<()> {
        let terminate_loop = self.pipewire_state.pw_loop.clone();
        let terminate_recv = self.termination_recv.take().unwrap();
        let _recv = terminate_recv.attach(self.pipewire_state.pw_loop.loop_(), move |_| {
            log::debug!("Terminating video capture loop");
            terminate_loop.quit();
        });

        self.pipewire_state.pw_loop.run();

        Ok(())
    }

    fn get_dmabuf_fd(data: &Data) -> Option<RawFd> {
        let raw_data = data.as_raw();

        if data.type_() == DataType::DmaBuf {
            let fd = raw_data.fd;

            if fd > 0 {
                return Some(fd as i32);
            }
        }

        None
    }
}

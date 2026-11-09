use std::sync::Arc;

use crate::{
    types::{
        audio_frame::RawAudioFrame,
        config::AudioSource,
        error::Result as WaycapResult,
    },
    CaptureControls, ReadyState,
};
use crossbeam::channel::Sender;
use pipewire::{
    self as pw,
    context::ContextRc,
    core::CoreRc,
    main_loop::MainLoopRc,
    properties::properties,
    spa::{
        self,
        param::format::{MediaSubtype, MediaType},
        pod::Pod,
        utils::Direction,
    },
    stream::{StreamFlags, StreamListener, StreamRc, StreamState},
    sys::pw_stream_get_nsec,
};

use super::{audio_device, Terminate};

#[derive(Clone, Copy, Default)]
struct UserData {
    audio_format: spa::param::audio::AudioInfoRaw,
}

struct PipewireState {
    pw_loop: MainLoopRc,
    _pw_context: ContextRc,
    _core: CoreRc,
    _core_listener: pw::core::Listener,
    _stream: StreamRc,
    _stream_listener: StreamListener<UserData>,
}

pub struct AudioCapture {
    termination_recv: Option<pw::channel::Receiver<Terminate>>,
    pipewire_state: PipewireState,
}

impl AudioCapture {
    pub fn new(
        ready_state: Arc<ReadyState>,
        audio_sender: Sender<RawAudioFrame>,
        termination_recv: pw::channel::Receiver<Terminate>,
        controls: Arc<CaptureControls>,
        audio_source: AudioSource,
    ) -> WaycapResult<Self> {
        let pw_loop = MainLoopRc::new(None)?;
        let pw_context = ContextRc::new(&pw_loop, None)?;
        let core = pw_context.connect_rc(None)?;

        let target_node_id = audio_device::resolve_node_id(&core, &pw_loop, &audio_source)?;
        log::debug!("Resolved audio target node id: {target_node_id} for {audio_source:?}");

        let mut core_mut = core.clone();
        let core_listener = Self::setup_core_listener(&mut core_mut);

        let mut stream = Self::create_stream(core.clone())?;
        let stream_listener =
            Self::setup_stream_listener(&mut stream, ready_state, controls, audio_sender)?;
        Self::connect_stream(&mut stream, target_node_id)?;

        Ok(Self {
            termination_recv: Some(termination_recv),
            pipewire_state: PipewireState {
                pw_loop,
                _pw_context: pw_context,
                _core: core,
                _core_listener: core_listener,
                _stream: stream,
                _stream_listener: stream_listener,
            },
        })
    }

    fn create_stream(core: CoreRc) -> Result<StreamRc, pw::Error> {
        StreamRc::new(
            core,
            "waycap-audio",
            properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Music",
                *pw::keys::NODE_LATENCY => "1024/48000",
            },
        )
    }

    fn setup_core_listener(core: &mut CoreRc) -> pw::core::Listener {
        core.add_listener_local()
            .info(|i| log::debug!("AUDIO CORE:\n{i:#?}"))
            .error(|e, f, g, h| log::error!("{e},{f},{g},{h}"))
            .done(|d, _| log::debug!("DONE: {d}"))
            .register()
    }

    fn setup_stream_listener(
        stream: &mut StreamRc,
        ready_state: Arc<ReadyState>,
        controls: Arc<CaptureControls>,
        audio_sender: Sender<RawAudioFrame>,
    ) -> Result<StreamListener<UserData>, pw::Error> {
        let ready_state_clone = Arc::clone(&ready_state);

        let stream_listener = stream
            .add_local_listener_with_user_data(UserData::default())
            .state_changed(move |_, _, old, new| {
                log::info!("Audio Stream State Changed: {old:?} -> {new:?}");
                ready_state.note_audio_stream_transition(
                    old == StreamState::Streaming,
                    new == StreamState::Streaming,
                );
            })
            .param_changed(|_, udata, id, param| {
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

                if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                    return;
                }

                udata
                    .audio_format
                    .parse(param)
                    .expect("Failed to parse audio params");

                log::debug!(
                    "Capturing Rate:{} channels:{}, format: {}",
                    udata.audio_format.rate(),
                    udata.audio_format.channels(),
                    udata.audio_format.format().as_raw()
                );
            })
            .process(move |stream, _| match stream.dequeue_buffer() {
                None => log::debug!("Out of audio buffers"),
                Some(mut buffer) => {
                    if !ready_state_clone.video_ready() || controls.skip_processing() {
                        return;
                    }

                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }

                    let data = &mut datas[0];
                    let n_samples = data.chunk().size() / (std::mem::size_of::<f32>()) as u32;

                    if let Some(samples) = data.data() {
                        let samples_f32: &[f32] = bytemuck::cast_slice(samples);
                        let audio_samples = &samples_f32[..n_samples as usize];
                        match audio_sender.try_send(RawAudioFrame {
                            samples: audio_samples.to_vec(),
                            timestamp: unsafe { pw_stream_get_nsec(stream.as_raw_ptr()) } as i64,
                        }) {
                            Ok(_) => {}
                            Err(crossbeam::channel::TrySendError::Full(frame)) => {
                                log::error!(
                                    "channel is full when trying to send frame at: {}.",
                                    frame.timestamp
                                );
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(frame)) => {
                                log::error!(
                                    "channel is disconnected when trying to send frame at: {}.",
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

    fn connect_stream(stream: &mut StreamRc, target_node_id: u32) -> WaycapResult<()> {
        // Fixed rate/channels so mic + system audio samples can be safely summed when mixing.
        let mut audio_info = spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
        audio_info.set_rate(crate::utils::TARGET_AUDIO_SAMPLE_RATE);
        audio_info.set_channels(crate::utils::TARGET_AUDIO_CHANNELS);

        let audio_spa_obj = pw::spa::pod::Object {
            type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: pw::spa::param::ParamType::EnumFormat.as_raw(),
            properties: audio_info.into(),
        };

        let audio_spa_values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(audio_spa_obj),
        )
        .unwrap()
        .0
        .into_inner();

        let mut audio_params = [Pod::from_bytes(&audio_spa_values).unwrap()];

        stream
            .connect(
                Direction::Input,
                Some(target_node_id),
                StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
                &mut audio_params,
            )
            .map_err(Into::into)
    }

    pub fn run(&mut self) {
        let terminate_loop = self.pipewire_state.pw_loop.clone();
        let terminate_recv = self.termination_recv.take().unwrap();
        let _recv = terminate_recv.attach(self.pipewire_state.pw_loop.loop_(), move |_| {
            log::debug!("Terminating audio capture loop");
            terminate_loop.quit();
        });

        log::debug!("Audio Stream: {:?}", self.pipewire_state._stream);
        self.pipewire_state.pw_loop.run();
    }
}

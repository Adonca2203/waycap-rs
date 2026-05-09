use std::{process::Command, sync::Arc};

use crate::{types::audio_frame::RawAudioFrame, CaptureControls, ReadyState};
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

use super::Terminate;

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
    ) -> Result<Self, pw::Error> {
        let pw_loop = MainLoopRc::new(None)?;
        let pw_context = ContextRc::new(&pw_loop, None)?;
        let core = pw_context.connect_rc(None)?;

        let mut core_mut = core.clone();
        let core_listener = Self::setup_core_listener(&mut core_mut);

        let mut stream = Self::create_stream(core.clone())?;
        let stream_listener =
            Self::setup_stream_listener(&mut stream, ready_state, controls, audio_sender)?;
        Self::connect_stream(&mut stream)?;

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
                ready_state.audio.store(
                    new == StreamState::Streaming,
                    std::sync::atomic::Ordering::Release,
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

    fn connect_stream(stream: &mut StreamRc) -> Result<(), pw::Error> {
        let audio_spa_obj = pw::spa::pod::object! {
            pw::spa::utils::SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaType,
                Id,
                pw::spa::param::format::MediaType::Audio
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaSubtype,
                Id,
                pw::spa::param::format::MediaSubtype::Raw
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::AudioFormat,
                Id,
                pw::spa::param::audio::AudioFormat::F32LE
            )
        };

        let audio_spa_values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(audio_spa_obj),
        )
        .unwrap()
        .0
        .into_inner();

        let mut audio_params = [Pod::from_bytes(&audio_spa_values).unwrap()];

        let sink_id_to_use = get_default_sink_node_id();
        log::debug!("Default sink id: {sink_id_to_use:?}");

        stream.connect(
            Direction::Input,
            sink_id_to_use,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut audio_params,
        )
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

// Theres gotta be a less goofy way to do this
fn get_default_sink_node_id() -> Option<u32> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(r#"pactl list sinks | awk -v sink="$(pactl info | grep 'Default Sink' | cut -d' ' -f3)" '$0 ~ "Name: " sink { found=1 } found && /object.id/ { print $NF; exit }'"#)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let cleaned = stdout.replace('"', "");
    cleaned.trim().parse::<u32>().ok()
}

use std::{
    process::Command,
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

use crate::types::audio_frame::RawAudioFrame;
use crossbeam::channel::Sender;
use pipewire::{
    self as pw,
    context::Context,
    main_loop::MainLoop,
    properties::properties,
    spa::{
        self,
        param::format::{MediaSubtype, MediaType},
        pod::Pod,
        utils::Direction,
    },
    stream::{StreamFlags, StreamState},
};

use super::Terminate;

#[derive(Clone, Copy, Default)]
struct UserData {
    audio_format: spa::param::audio::AudioInfoRaw,
}

pub struct AudioCapture {
    video_ready: Arc<AtomicBool>,
    audio_ready: Arc<AtomicBool>,
}

// TODO: Similar approach to video capture in how the struct should look
impl AudioCapture {
    pub fn new(video_ready: Arc<AtomicBool>, audio_ready: Arc<AtomicBool>) -> Self {
        Self {
            video_ready,
            audio_ready,
        }
    }

    pub fn run(
        &self,
        audio_sender: Sender<RawAudioFrame>,
        start_time: Instant,
        termination_recv: pw::channel::Receiver<Terminate>,
        saving: Arc<AtomicBool>,
    ) -> Result<(), pw::Error> {
        let pw_loop = MainLoop::new(None)?;
        let terminate_loop = pw_loop.clone();

        let _recv = termination_recv.attach(pw_loop.loop_(), move |_| {
            log::debug!("Terminating audio capture loop");
            terminate_loop.quit();
        });

        let pw_context = Context::new(&pw_loop)?;
        let audio_core = pw_context.connect(None)?;

        let _audio_core_listener = audio_core
            .add_listener_local()
            .info(|i| log::debug!("AUDIO CORE:\n{i:#?}"))
            .error(|e, f, g, h| log::error!("{e},{f},{g},{h}"))
            .done(|d, _| log::debug!("DONE: {d}"))
            .register();

        let data = UserData::default();

        // Audio Stream
        let audio_stream = pw::stream::Stream::new(
            &audio_core,
            "waycap-audio",
            properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::NODE_LATENCY => "1024/48000",
            },
        )?;

        let video_ready_clone = Arc::clone(&self.video_ready);
        let audio_ready_clone = Arc::clone(&self.audio_ready);
        let _audio_stream_shared_data_listener = audio_stream
            .add_local_listener_with_user_data(data)
            .state_changed(move |_, _, old, new| {
                log::info!("Audio Stream State Changed: {old:?} -> {new:?}");
                audio_ready_clone.store(
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

                // only accept raw audio
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
                    // Wait until video is streaming before we try to process
                    if !video_ready_clone.load(std::sync::atomic::Ordering::Acquire)
                        || saving.load(std::sync::atomic::Ordering::Acquire)
                    {
                        return;
                    }

                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }

                    let time_us = start_time.elapsed().as_micros() as i64;

                    let data = &mut datas[0];
                    let n_samples = data.chunk().size() / (std::mem::size_of::<f32>()) as u32;

                    if let Some(samples) = data.data() {
                        let samples_f32: &[f32] = bytemuck::cast_slice(samples);
                        let audio_samples = &samples_f32[..n_samples as usize];
                        match audio_sender.try_send(RawAudioFrame {
                            samples: audio_samples.to_vec(),
                            timestamp: time_us,
                        }) {
                            Ok(_) => {}
                            Err(crossbeam::channel::TrySendError::Full(frame)) => {
                                log::error!(
                                    "channel is full when trying to send frame at: {}.",
                                    frame.timestamp
                                );
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(frame)) => {
                                // TODO: If we disconnected, terminate the session instead of
                                // throwing an error it means the receiver was dropped.
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
        audio_stream.connect(
            Direction::Input,
            sink_id_to_use,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut audio_params,
        )?;

        log::debug!("Audio Stream: {audio_stream:?}");

        pw_loop.run();
        Ok(())
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

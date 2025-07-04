use std::{
    os::fd::{FromRawFd, OwnedFd, RawFd},
    sync::{
        atomic::AtomicBool,
        mpsc::{self},
        Arc,
    },
    time::Instant,
};

use crossbeam::channel::Sender;
use pipewire::{
    self as pw,
    context::Context,
    main_loop::MainLoop,
    spa::{
        buffer::{Data, DataType},
        pod::{Property, PropertyFlags},
        utils::{Choice, ChoiceEnum, ChoiceFlags, Direction},
    },
    stream::{Stream, StreamFlags, StreamState},
};
use pw::{properties::properties, spa};

use spa::pod::Pod;

use crate::types::video_frame::RawVideoFrame;

use super::Terminate;

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

pub struct VideoCapture {
    video_ready: Arc<AtomicBool>,
    audio_ready: Arc<AtomicBool>,
    use_nvidia_modifiers: bool,
}

#[derive(Clone, Copy, Default)]
struct UserData {
    video_format: spa::param::video::VideoInfoRaw,
}

impl VideoCapture {
    pub fn new(
        video_ready: Arc<AtomicBool>,
        audio_ready: Arc<AtomicBool>,
        use_nvidia_modifiers: bool,
    ) -> Self {
        Self {
            video_ready,
            audio_ready,
            use_nvidia_modifiers,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        pipewire_fd: RawFd,
        stream_node: u32,
        frame_tx: Sender<RawVideoFrame>,
        termination_recv: pw::channel::Receiver<Terminate>,
        saving: Arc<AtomicBool>,
        start_time: Instant,
        resolution_negotiation_channel: mpsc::Sender<(u32, u32)>,
    ) -> Result<(), pipewire::Error> {
        let pw_loop = MainLoop::new(None)?;
        let terminate_loop = pw_loop.clone();

        let _recv = termination_recv.attach(pw_loop.loop_(), move |_| {
            log::debug!("Terminating video capture loop");
            terminate_loop.quit();
        });

        let pw_context = Context::new(&pw_loop)?;
        let core = pw_context.connect_fd(unsafe { OwnedFd::from_raw_fd(pipewire_fd) }, None)?;

        let data = UserData::default();

        let _listener = core
            .add_listener_local()
            .info(|i| log::info!("VIDEO CORE:\n{0:#?}", i))
            .error(|e, f, g, h| log::error!("{0},{1},{2},{3}", e, f, g, h))
            .done(|d, _| log::info!("DONE: {0}", d))
            .register();

        // Set up video stream
        let video_stream = Stream::new(
            &core,
            "waycap-video",
            properties! {
                *pw::keys::MEDIA_TYPE => "Video",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Screen",
            },
        )?;

        let ready_clone = Arc::clone(&self.video_ready);
        let audio_ready_clone = Arc::clone(&self.audio_ready);
        let _video_stream = video_stream
            .add_local_listener_with_user_data(data)
            .state_changed(move |_, _, old, new| {
                log::info!("Video Stream State Changed: {0:?} -> {1:?}", old, new);
                ready_clone.store(
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
                match resolution_negotiation_channel.send((width, height)) {
                    Ok(_) => {}
                    Err(_) => {
                        log::error!("Tried to send resolution update {}x{} but ran into an error on the channel.", width, height);
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
                        if !audio_ready_clone.load(std::sync::atomic::Ordering::Acquire)
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

                        let fd = Self::get_dmabuf_fd(data);

                        match frame_tx.try_send(RawVideoFrame {
                            data: data.data().unwrap_or_default().to_vec(),
                            timestamp: time_us,
                            dmabuf_fd: fd,
                            stride: data.chunk().stride(),
                            offset: data.chunk().offset(),
                            size: data.chunk().size(),
                            modifier: udata.video_format.modifier(),
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

        // TODO: Use features? Probably should not have runtime conditionals like this
        let pw_obj = if self.use_nvidia_modifiers {
            let nvidia_mod_property = Property {
                key: pw::spa::param::format::FormatProperties::VideoModifier.as_raw(),
                flags: PropertyFlags::empty(),
                value: spa::pod::Value::Choice(spa::pod::ChoiceValue::Long(Choice::<i64>(
                    ChoiceFlags::empty(),
                    ChoiceEnum::<i64>::Enum {
                        default: NVIDIA_MODIFIERS[0],
                        alternatives: NVIDIA_MODIFIERS.to_vec(),
                    },
                ))),
            };

            Some(pw::spa::pod::object!(
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
                    pw::spa::param::video::VideoFormat::BGRA,
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
        } else {
            Some(pw::spa::pod::object!(
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
                pw::spa::pod::property!(
                    pw::spa::param::format::FormatProperties::VideoModifier,
                    Long,
                    0
                ),
                pw::spa::pod::property!(
                    pw::spa::param::format::FormatProperties::VideoFormat,
                    Choice,
                    Enum,
                    Id,
                    pw::spa::param::video::VideoFormat::NV12,
                    pw::spa::param::video::VideoFormat::I420,
                    pw::spa::param::video::VideoFormat::BGRA,
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
        };

        let video_spa_values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(pw_obj.unwrap()),
        )
        .unwrap()
        .0
        .into_inner();
        let mut video_params = [Pod::from_bytes(&video_spa_values).unwrap()];
        video_stream.connect(
            Direction::Input,
            Some(stream_node),
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            &mut video_params,
        )?;

        log::debug!("Video Stream: {0:?}", video_stream);

        pw_loop.run();
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

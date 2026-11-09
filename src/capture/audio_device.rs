//! Native PipeWire registry/metadata lookup for audio devices, replacing a `pactl` shell-out.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::{Duration, Instant},
};

use pipewire::{
    self as pw, context::ContextRc, core::CoreRc, loop_::Timeout, main_loop::MainLoopRc,
    metadata::Metadata, types::ObjectType,
};

use crate::types::{
    config::{AudioDeviceInfo, AudioDeviceKind, AudioSource},
    error::{Result, WaycapError},
};

const SINK_CLASS: &str = "Audio/Sink";
const SOURCE_CLASS: &str = "Audio/Source";

struct DiscoveredNode {
    id: u32,
    name: String,
    description: String,
    media_class: String,
}

struct RegistrySnapshot {
    nodes: Vec<DiscoveredNode>,
    default_sink_name: Option<String>,
    default_source_name: Option<String>,
}

/// Enumerate available audio playback/capture devices; feed [`AudioDeviceInfo::name`] into [`AudioSource::Named`].
pub fn list_audio_devices() -> Result<Vec<AudioDeviceInfo>> {
    let pw_loop = MainLoopRc::new(None)?;
    let pw_context = ContextRc::new(&pw_loop, None)?;
    let core = pw_context.connect_rc(None)?;

    let snapshot = take_snapshot(&core, &pw_loop)?;

    Ok(snapshot
        .nodes
        .into_iter()
        .map(|node| AudioDeviceInfo {
            name: node.name,
            description: node.description,
            kind: if node.media_class == SOURCE_CLASS {
                AudioDeviceKind::Input
            } else {
                AudioDeviceKind::Output
            },
        })
        .collect())
}

/// Resolve an [`AudioSource`] to a concrete PipeWire node id to connect the capture stream to.
pub(crate) fn resolve_node_id(
    core: &CoreRc,
    pw_loop: &MainLoopRc,
    source: &AudioSource,
) -> Result<u32> {
    let snapshot = take_snapshot(core, pw_loop)?;

    match source {
        AudioSource::DefaultOutput => resolve_default(
            &snapshot,
            SINK_CLASS,
            snapshot.default_sink_name.as_deref(),
            "default playback device (system audio)",
        ),
        AudioSource::DefaultInput => resolve_default(
            &snapshot,
            SOURCE_CLASS,
            snapshot.default_source_name.as_deref(),
            "default microphone",
        ),
        AudioSource::Named(needle) => resolve_named(&snapshot, needle),
    }
}

fn resolve_default(
    snapshot: &RegistrySnapshot,
    class: &str,
    default_name: Option<&str>,
    label: &str,
) -> Result<u32> {
    if let Some(name) = default_name {
        if let Some(node) = snapshot
            .nodes
            .iter()
            .find(|n| n.media_class == class && n.name == name)
        {
            return Ok(node.id);
        }
    }

    // No "default" metadata (or it's stale); fall back only if the choice is unambiguous.
    let candidates: Vec<&DiscoveredNode> = snapshot
        .nodes
        .iter()
        .filter(|n| n.media_class == class)
        .collect();

    match candidates.as_slice() {
        [] => Err(WaycapError::Device(format!("No {label} found"))),
        [only] => {
            log::warn!(
                "Could not determine {label} via PipeWire metadata; falling back to the only available device: {}",
                only.name
            );
            Ok(only.id)
        }
        _ => Err(WaycapError::Device(format!(
            "Could not determine the {label} and multiple candidate devices exist; pass one explicitly via AudioSource::Named (see waycap_rs::list_audio_devices)"
        ))),
    }
}

fn resolve_named(snapshot: &RegistrySnapshot, needle: &str) -> Result<u32> {
    let needle_lower = needle.to_lowercase();
    snapshot
        .nodes
        .iter()
        .find(|n| {
            n.name.to_lowercase().contains(&needle_lower)
                || n.description.to_lowercase().contains(&needle_lower)
        })
        .map(|n| n.id)
        .ok_or_else(|| WaycapError::Device(format!("No audio device matching '{needle}' was found")))
}

fn take_snapshot(core: &CoreRc, pw_loop: &MainLoopRc) -> Result<RegistrySnapshot> {
    let registry = core.get_registry_rc()?;

    let nodes = Rc::new(RefCell::new(Vec::new()));
    let default_sink = Rc::new(RefCell::new(None));
    let default_source = Rc::new(RefCell::new(None));
    // Metadata proxies + their listeners must stay alive to receive property events.
    let metadata_handles: Rc<RefCell<Vec<(Metadata, pw::metadata::MetadataListener)>>> =
        Rc::new(RefCell::new(Vec::new()));

    let nodes_clone = Rc::clone(&nodes);
    let default_sink_clone = Rc::clone(&default_sink);
    let default_source_clone = Rc::clone(&default_source);
    let metadata_handles_clone = Rc::clone(&metadata_handles);
    let registry_for_bind = registry.clone();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| match global.type_ {
            ObjectType::Node => {
                let Some(props) = global.props else {
                    return;
                };
                let media_class = props.get("media.class").unwrap_or_default();
                if media_class != SINK_CLASS && media_class != SOURCE_CLASS {
                    return;
                }

                nodes_clone.borrow_mut().push(DiscoveredNode {
                    id: global.id,
                    name: props.get("node.name").unwrap_or_default().to_string(),
                    description: props
                        .get("node.description")
                        .or_else(|| props.get("node.nick"))
                        .unwrap_or_default()
                        .to_string(),
                    media_class: media_class.to_string(),
                });
            }
            ObjectType::Metadata => {
                let is_default_metadata = global
                    .props
                    .and_then(|p| p.get("metadata.name"))
                    .map(|name| name == "default")
                    .unwrap_or(false);

                if !is_default_metadata {
                    return;
                }

                let Ok(metadata) = registry_for_bind.bind::<Metadata, _>(global) else {
                    return;
                };

                let default_sink_inner = Rc::clone(&default_sink_clone);
                let default_source_inner = Rc::clone(&default_source_clone);
                let listener = metadata
                    .add_listener_local()
                    .property(move |_subject, key, _type_, value| {
                        if let (Some(key), Some(value)) = (key, value) {
                            match key {
                                "default.audio.sink" => {
                                    *default_sink_inner.borrow_mut() = extract_json_name(value);
                                }
                                "default.audio.source" => {
                                    *default_source_inner.borrow_mut() = extract_json_name(value);
                                }
                                _ => {}
                            }
                        }
                        0
                    })
                    .register();

                metadata_handles_clone
                    .borrow_mut()
                    .push((metadata, listener));
            }
            _ => {}
        })
        .register();

    roundtrip(core, pw_loop)?;
    // Second roundtrip ensures metadata binds issued during the first have delivered their properties.
    roundtrip(core, pw_loop)?;

    Ok(RegistrySnapshot {
        nodes: nodes.take(),
        default_sink_name: default_sink.take(),
        default_source_name: default_source.take(),
    })
}

fn roundtrip(core: &CoreRc, pw_loop: &MainLoopRc) -> Result<()> {
    let pending = core.sync(0)?;
    let done = Rc::new(Cell::new(false));
    let done_clone = Rc::clone(&done);

    let _listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == pending {
                done_clone.set(true);
            }
        })
        .register();

    let deadline = Instant::now() + Duration::from_secs(2);
    while !done.get() && Instant::now() < deadline {
        pw_loop
            .loop_()
            .iterate(Timeout::Finite(Duration::from_millis(50)));
    }

    if !done.get() {
        return Err(WaycapError::PipeWire(
            "Timed out waiting for PipeWire registry roundtrip".into(),
        ));
    }

    Ok(())
}

/// Pulls `name` out of the small JSON object PipeWire reports for `default.audio.sink`/`source`.
fn extract_json_name(value: &str) -> Option<String> {
    let idx = value.find("\"name\"")?;
    let rest = &value[idx + "\"name\"".len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

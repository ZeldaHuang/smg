//! Read-only observation of engine control calls that pass through the
//! proxy: what a 2xx answer to a route tells SMG about the engine. The
//! forwarded bytes are never modified.

use serde::Deserialize;
use serde_json::Value;

use crate::{table::ControlState, version::Version};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observation {
    Version(Version),
    Control(ControlState),
}

/// Only the version fields are read; everything else in a refit body is
/// skipped without allocation.
#[derive(Deserialize)]
struct VersionProbe {
    #[serde(default)]
    weight_version: Option<Value>,
    #[serde(default)]
    new_version: Option<Value>,
}

fn version_field(value: Option<Value>) -> Option<Version> {
    match value? {
        Value::String(s) if !s.trim().is_empty() => Some(Version::parse(&s)),
        Value::Number(n) => Some(Version::parse(&n.to_string())),
        _ => None,
    }
}

/// What a successful call to `path` (the proxied engine route, no leading
/// slash) with request `body` says about the engine, if anything.
pub fn observe(path: &str, body: &[u8]) -> Option<Observation> {
    match path {
        "update_weights_from_disk"
        | "update_weights_from_tensor"
        | "update_weights_from_distributed" => {
            let probe: VersionProbe = serde_json::from_slice(body).ok()?;
            version_field(probe.weight_version).map(Observation::Version)
        }
        "update_weight_version" => {
            let probe: VersionProbe = serde_json::from_slice(body).ok()?;
            version_field(probe.new_version).map(Observation::Version)
        }
        "pause_generation" | "pause" => Some(Observation::Control(ControlState::Paused)),
        "continue_generation" | "resume" | "resume_memory_occupation" | "wake_up" => {
            Some(Observation::Control(ControlState::Active))
        }
        "release_memory_occupation" | "sleep" => Some(Observation::Control(ControlState::Asleep)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refit_routes_report_the_version_field_when_present() {
        let v = observe(
            "update_weights_from_disk",
            br#"{"model_path": "/ckpt", "weight_version": "42"}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("42"))));
        let v = observe(
            "update_weights_from_tensor",
            br#"{"serialized_named_tensors": ["..."], "weight_version": 7}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("7"))));
        let v = observe(
            "update_weights_from_distributed",
            br#"{"names": [], "weight_version": "s3"}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("s3"))));
        assert_eq!(
            observe(
                "update_weight_version",
                br#"{"new_version": "43", "abort_all_requests": false}"#
            ),
            Some(Observation::Version(Version::parse("43")))
        );
    }

    #[test]
    fn refit_routes_without_a_version_observe_nothing() {
        assert_eq!(
            observe("update_weights_from_disk", br#"{"model_path": "/ckpt"}"#),
            None
        );
        assert_eq!(
            observe("update_weights_from_disk", br#"{"weight_version": ""}"#),
            None
        );
        assert_eq!(
            observe("update_weights_from_disk", br#"{"weight_version": null}"#),
            None
        );
        assert_eq!(observe("update_weights_from_disk", b"not json"), None);
        assert_eq!(
            observe("update_weight_version", br#"{"weight_version": "1"}"#),
            None,
            "wrong field for this route"
        );
    }

    #[test]
    fn control_routes_map_to_states_regardless_of_body() {
        for (path, state) in [
            ("pause_generation", ControlState::Paused),
            ("pause", ControlState::Paused),
            ("continue_generation", ControlState::Active),
            ("resume", ControlState::Active),
            ("release_memory_occupation", ControlState::Asleep),
            ("sleep", ControlState::Asleep),
            ("resume_memory_occupation", ControlState::Active),
            ("wake_up", ControlState::Active),
        ] {
            assert_eq!(
                observe(path, b"{}"),
                Some(Observation::Control(state)),
                "{path}"
            );
            assert_eq!(
                observe(path, b""),
                Some(Observation::Control(state)),
                "{path}"
            );
        }
        assert_eq!(observe("flush_cache", b"{}"), None);
        assert_eq!(observe("server_info", b""), None);
    }
}

//! Weight versions as engines report them, with the ordering the pre-filter
//! and the per-model fleet maximum use.

use std::{
    cmp::Ordering,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
};

use tracing::warn;

/// The label an engine reports before any refit; seeds as "no version".
pub const UNVERSIONED_LABEL: &str = "default";

/// A weight version: numeric when the text parses as `u64`, opaque text
/// otherwise. Numeric versions compare by value, text versions lexically,
/// and a numeric/text pair lexically with one warning per process.
#[derive(Clone, Debug)]
pub struct Version {
    raw: Arc<str>,
    numeric: Option<u64>,
}

static MIXED_ORDERING_WARNED: AtomicBool = AtomicBool::new(false);

impl Version {
    pub fn parse(raw: &str) -> Self {
        let raw = raw.trim();
        Self {
            raw: Arc::from(raw),
            numeric: raw.parse::<u64>().ok(),
        }
    }

    /// The version a registration label carries: `None` for an absent or
    /// empty label and for [`UNVERSIONED_LABEL`].
    pub fn from_label(label: Option<&str>) -> Option<Self> {
        let label = label?.trim();
        if label.is_empty() || label == UNVERSIONED_LABEL {
            return None;
        }
        Some(Self::parse(label))
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn numeric(&self) -> Option<u64> {
        self.numeric
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Version {}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.numeric, other.numeric) {
            (Some(a), Some(b)) => a.cmp(&b),
            (None, None) => self.raw.cmp(&other.raw),
            _ => {
                if !MIXED_ORDERING_WARNED.swap(true, AtomicOrdering::Relaxed) {
                    warn!(
                        target: "smg_rl",
                        a = %self.raw, b = %other.raw,
                        "comparing a numeric weight version with a text one; using lexical order"
                    );
                }
                self.raw.cmp(&other.raw)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_versions_compare_by_value() {
        assert!(Version::parse("10") > Version::parse("9"));
        assert_eq!(Version::parse("042"), Version::parse("42"));
        assert_eq!(Version::parse(" 7 ").as_str(), "7");
        assert_eq!(Version::parse("7").numeric(), Some(7));
    }

    #[test]
    fn text_versions_compare_lexically() {
        assert!(Version::parse("step-9") > Version::parse("step-10"));
        assert_eq!(Version::parse("abc").numeric(), None);
        assert_ne!(Version::parse("abc"), Version::parse("acd"));
    }

    #[test]
    fn mixed_versions_fall_back_to_lexical_order() {
        // '1' (0x31) sorts before 'd' (0x64).
        assert!(Version::parse("1") < Version::parse("default"));
        assert!(Version::parse("1") < Version::parse("1a"));
    }

    #[test]
    fn labels_seed_versions_except_the_placeholder() {
        assert_eq!(Version::from_label(None), None);
        assert_eq!(Version::from_label(Some("")), None);
        assert_eq!(Version::from_label(Some("default")), None);
        assert_eq!(
            Version::from_label(Some("42")).map(|v| v.numeric()),
            Some(Some(42))
        );
        assert_eq!(format!("{}", Version::parse("v3")), "v3");
    }
}

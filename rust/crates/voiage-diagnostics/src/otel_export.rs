//! Bounded, application-owned trace export qualification.
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use opentelemetry::trace::SpanContext;

/// OpenTelemetry API line qualified by this optional feature.
pub const OPENTELEMETRY_API_VERSION: &str = "0.32";

/// Maximum queued records, even when a larger capacity is requested.
pub const MAX_QUEUED_EVENTS: usize = 1024;
/// Maximum retained attributes on one record.
pub const MAX_EVENT_ATTRIBUTES: usize = 32;
/// Maximum total allocated string bytes retained in one record.
pub const MAX_EVENT_STRING_BYTES: usize = 64 * 1024;

/// A bounded trace record suitable for an application-owned exporter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    /// Stable operation name.
    pub name: String,
    /// W3C trace identifier.
    pub trace_id: String,
    /// Optional parent span identifier.
    pub parent_id: Option<String>,
    /// Bounded semantic attributes.
    pub attributes: BTreeMap<String, String>,
}

/// Explicit opt-in and resource limits for trace export.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceExportConfig {
    enabled: bool,
    max_retries: u8,
    sample_rate: f64,
    max_attributes: usize,
    capacity: usize,
}

impl TraceExportConfig {
    /// Creates a bounded configuration with conservative defaults.
    ///
    /// # Errors
    ///
    /// Returns an error when retries exceed three attempts.
    pub fn new(enabled: bool, max_retries: u8) -> Result<Self, &'static str> {
        if max_retries > 3 {
            return Err("max_retries must be <= 3");
        }
        Ok(Self {
            enabled,
            max_retries,
            sample_rate: 1.0,
            max_attributes: MAX_EVENT_ATTRIBUTES,
            capacity: MAX_QUEUED_EVENTS,
        })
    }

    /// Sets a sampling probability in the inclusive range `0..=1`.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the maximum attributes retained per event, capped at the global limit.
    #[must_use]
    pub fn with_max_attributes(mut self, max_attributes: usize) -> Self {
        self.max_attributes = max_attributes.min(MAX_EVENT_ATTRIBUTES);
        self
    }

    /// Sets the queue capacity, capped at the global limit.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.min(MAX_QUEUED_EVENTS);
        self
    }

    /// Returns whether export is explicitly enabled.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the bounded retry budget.
    #[must_use]
    pub const fn max_retries(&self) -> u8 {
        self.max_retries
    }
}

/// A non-blocking, bounded collector for local qualification and adapters.
#[derive(Clone, Debug)]
pub struct TraceCollector {
    config: TraceExportConfig,
    events: Arc<Mutex<VecDeque<TraceEvent>>>,
}

impl TraceCollector {
    /// Creates a collector without initializing global logging state.
    #[must_use]
    pub fn new(config: TraceExportConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Records an event when opt-in, sampling and retained-size limits allow it.
    #[must_use]
    pub fn record(&self, mut event: TraceEvent) -> bool {
        if !self.config.enabled || !sampled(&event.trace_id, self.config.sample_rate) {
            return false;
        }
        while event.attributes.len() > self.config.max_attributes {
            if let Some(key) = event.attributes.keys().next().cloned() {
                event.attributes.remove(&key);
            } else {
                break;
            }
        }
        let mut bytes = event
            .name
            .capacity()
            .saturating_add(event.trace_id.capacity());
        if let Some(parent) = &event.parent_id {
            bytes = bytes.saturating_add(parent.capacity());
        }
        for (key, value) in &event.attributes {
            bytes = bytes
                .saturating_add(key.capacity())
                .saturating_add(value.capacity());
        }
        if bytes > MAX_EVENT_STRING_BYTES {
            return false;
        }
        let Ok(mut events) = self.events.try_lock() else {
            return false;
        };
        if self.config.capacity == 0 {
            return false;
        }
        if events.len() >= self.config.capacity {
            events.pop_front();
        }
        events.push_back(event);
        true
    }

    /// Records a span context supplied by an application-owned OpenTelemetry tracer.
    #[must_use]
    pub fn record_span_context(
        &self,
        name: impl Into<String>,
        context: &SpanContext,
        attributes: BTreeMap<String, String>,
    ) -> bool {
        self.record(TraceEvent {
            name: name.into(),
            trace_id: context.trace_id().to_string(),
            parent_id: context.is_valid().then(|| context.span_id().to_string()),
            attributes,
        })
    }

    /// Drains records for an application-owned exporter or shutdown hook.
    #[must_use]
    pub fn drain(&self) -> Vec<TraceEvent> {
        let Ok(mut events) = self.events.try_lock() else {
            return Vec::new();
        };
        std::mem::take(&mut *events).into_iter().collect()
    }
}

fn sampled(trace_id: &str, sample_rate: f64) -> bool {
    let hash = trace_id.bytes().fold(0x811c_9dc5_u32, |hash, byte| {
        hash.wrapping_mul(16_777_619) ^ u32::from(byte)
    });
    let bucket = hash % 1000;
    f64::from(bucket) / 1000.0 < sample_rate
}

#[cfg(test)]
mod tests {
    use super::{
        TraceCollector, TraceEvent, TraceExportConfig, MAX_EVENT_ATTRIBUTES,
        MAX_EVENT_STRING_BYTES, MAX_QUEUED_EVENTS,
    };
    use std::collections::BTreeMap;

    fn event(id: &str) -> TraceEvent {
        TraceEvent {
            name: "voi.run".into(),
            trace_id: id.into(),
            parent_id: Some("parent".into()),
            attributes: BTreeMap::from([(String::from("run_id"), String::from("redacted"))]),
        }
    }

    #[test]
    fn opt_in_and_parent_context_are_preserved() {
        let collector = TraceCollector::new(TraceExportConfig::new(true, 3).expect("valid"));
        assert!(collector.record(event("trace-1")));
        let records = collector.drain();
        assert!(!records.is_empty(), "expected at least one record");
        assert_eq!(records[0].parent_id.as_deref(), Some("parent"));
    }

    #[test]
    fn disabled_sampling_and_capacity_do_not_block() {
        let config = TraceExportConfig::new(false, 0)
            .expect("valid")
            .with_capacity(1);
        let collector = TraceCollector::new(config);
        assert!(!collector.record(event("trace-2")));
        assert_eq!(collector.drain(), [] as [TraceEvent; 0]);
    }

    #[test]
    fn attributes_are_bounded_and_queue_evicts_oldest() {
        let config = TraceExportConfig::new(true, 0)
            .expect("valid")
            .with_max_attributes(1)
            .with_capacity(1);
        let collector = TraceCollector::new(config);
        assert!(collector.record(event("trace-3")));
        assert!(collector.record(event("trace-4")));
        let records = collector.drain();
        assert!(!records.is_empty(), "expected at least one record");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].attributes.len(), 1);
    }

    #[test]
    fn oversized_event_is_not_retained() {
        let collector = TraceCollector::new(TraceExportConfig::new(true, 0).expect("valid"));
        let mut oversized = event("trace-large");
        oversized.name = "x".repeat(1024 * 1024);
        assert!(!collector.record(oversized));
        assert!(collector.drain().is_empty());
    }

    #[test]
    fn allocated_string_capacity_is_bounded() {
        let collector = TraceCollector::new(TraceExportConfig::new(true, 0).expect("valid"));
        let mut oversized = event("trace-capacity");
        oversized.attributes.insert(
            "key".into(),
            String::with_capacity(MAX_EVENT_STRING_BYTES + 1),
        );
        assert!(!collector.record(oversized));
        assert!(collector.drain().is_empty());
        assert!(collector.record(event("trace-normal")));
        assert_eq!(collector.drain().len(), 1);
    }

    #[test]
    fn requested_limits_cannot_exceed_global_caps() {
        let config = TraceExportConfig::new(true, 0)
            .expect("valid")
            .with_capacity(usize::MAX)
            .with_max_attributes(usize::MAX);
        assert_eq!(config.capacity, MAX_QUEUED_EVENTS);
        assert_eq!(config.max_attributes, MAX_EVENT_ATTRIBUTES);
        let collector = TraceCollector::new(config);
        for _ in 0..=MAX_QUEUED_EVENTS {
            assert!(collector.record(event("trace-normal")));
        }
        assert_eq!(collector.drain().len(), MAX_QUEUED_EVENTS);
    }
}

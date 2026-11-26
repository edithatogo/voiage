# Existing telemetry baseline review

Reviewed at base `19454f7f7825b9142c8cc84eab52f3fd336d3193`. This is a
source inspection, not execution or acceptance evidence.

- **AC1:** The retained pilot acceptance records demonstrate functional and
  replay behavior. They do not identify a measured telemetry need or budget.
- **AC2:** The diagnostics crate declares optional OpenTelemetry API and SDK
  dependencies. A source declaration and two Python string checks do not
  qualify a bridge/exporter, semantic conventions, or an MSRV matrix.
- **AC3:** The existing collector stores local `TraceEvent` records and drains
  them to a caller. It contains no network exporter or local collector test;
  its hand-built parent string test does not demonstrate propagated span
  parentage through an actual export path.
- **AC4:** The queue bounds the number of records, and `try_lock` avoids waiting
  on its mutex. Attribute and string sizes, configured capacity, outage handling,
  retries and shutdown are not qualified by the existing three Rust tests. The
  retry count is stored but no retrying export operation is present. Record-count
  limits alone do not prove a byte-memory bound.
- **AC5:** The source is gated by an optional feature, but the retained tests
  inspected here do not prove the required feature matrix, incompatible-toolchain
  behavior, or numerical equivalence.

The packet's initial absent-file assumptions are historical. Refresh the phase
packet against these existing files and obtain behavioral witnesses before
implementation. Do not mark the track complete or infer a need from the mere
presence of the collector.

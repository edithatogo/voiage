# voiage-diagnostics

Structured diagnostics and error contracts for the `voiage` Value of
Information library. Diagnostics are designed to remain stable across the
Rust core and its Python, R, and Julia bindings.

## Local trace export

Enable the optional `otel-http` Cargo feature to send OTLP/HTTP JSON spans
to a collector listening on the same host. Construct the provider in your
application, keep it alive for the workload, and shut it down during teardown.
For example, with a receiver listening on port 4318:

```rust
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry_sdk::trace::Sampler;
use voiage_diagnostics::otlp_http::loopback_trace_provider;

let provider = loopback_trace_provider(4318, Sampler::TraceIdRatioBased(0.1))?;
let tracer = provider.tracer("analysis");
let value = tracer.in_span("voi.evaluate", |_| evaluate());
provider.shutdown()?;
```

Provider initialization works in synchronous applications and inside Tokio.
Call flush and shutdown outside latency-sensitive code because they wait for
the background exporter.

The application chooses the traced operation and its attributes. Use fixed
operation names and omit raw input data. The exporter uses the IPv4 loopback
interface, bypasses proxies and redirects, and sets a two-second request
timeout. It queues up to 1,024 completed spans and exports up to 64 per batch.
The SDK may drop spans when the queue fills. Standard OpenTelemetry resource
and header environment settings apply. The default build has no HTTP exporter.

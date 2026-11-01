//! Opt-in OTLP/HTTP JSON trace export to an application-owned local collector.
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider,
};
use std::time::Duration;

/// Creates a provider sending completed spans to the IPv4 loopback collector.
/// Obtain a tracer from this provider and call `shutdown` after the workload.
/// No global provider is installed. A background worker exports batches from
/// a bounded queue; the application may call `force_flush` at a checkpoint.
///
/// # Errors
/// Returns an error if the HTTP client or exporter cannot be initialized.
pub fn loopback_trace_provider(
    port: u16,
    sampler: Sampler,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    std::thread::Builder::new()
        .name("voiage-otlp-init".into())
        .spawn(move || build_provider(port, sampler))?
        .join()
        .map_err(|_| std::io::Error::other("OTLP initialization thread panicked"))?
}

fn build_provider(
    port: u16,
    sampler: Sampler,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    let timeout = Duration::from_secs(2);
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(timeout)
        .build()?;
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_http_client(client)
        .with_protocol(Protocol::HttpJson)
        .with_endpoint(format!("http://127.0.0.1:{port}/v1/traces"))
        .with_timeout(timeout)
        .build()?;
    let batch = BatchConfigBuilder::default()
        .with_max_queue_size(1024)
        .with_max_export_batch_size(64)
        .build();
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(batch)
        .build();
    Ok(SdkTracerProvider::builder()
        .with_span_processor(processor)
        .with_sampler(Sampler::ParentBased(Box::new(sampler)))
        .with_max_attributes_per_span(32)
        .build())
}

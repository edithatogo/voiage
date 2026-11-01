//! Exercise the opt-in OTLP transport across a real loopback HTTP connection.
#![cfg(feature = "otel-http")]

use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider};
use opentelemetry::KeyValue;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::Duration;
use voiage_diagnostics::otlp_http::loopback_trace_provider;

#[test]
fn collector_receives_parent_child_and_operation_attribute() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let receiver = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(8);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "collector received no request"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.starts_with("POST /v1/traces "));
        let mut length = None;
        loop {
            line.clear();
            assert!(reader.read_line(&mut line).unwrap() > 0);
            if line == "\r\n" {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                if key.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
        }
        let mut body = vec![0; length.unwrap()];
        reader.read_exact(&mut body).unwrap();
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}").unwrap();
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()
    });
    let provider =
        loopback_trace_provider(port, opentelemetry_sdk::trace::Sampler::AlwaysOn).unwrap();
    let tracer = provider.tracer("qualification");
    tracer.in_span("analysis", |_| {
        tracer.in_span("voi.evaluate", |cx| {
            cx.span().set_attribute(KeyValue::new("voi.method", "evpi"));
        });
    });
    provider.force_flush().unwrap();
    provider.shutdown().unwrap();
    let packet = receiver.join().unwrap();
    let spans = packet["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap();
    let parent = spans.iter().find(|s| s["name"] == "analysis").unwrap();
    let child = spans.iter().find(|s| s["name"] == "voi.evaluate").unwrap();
    assert_eq!(child["traceId"], parent["traceId"]);
    assert_eq!(child["parentSpanId"], parent["spanId"]);
    assert!(child["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["key"] == "voi.method" && a["value"]["stringValue"] == "evpi"));
}

#[test]
fn unsampled_spans_make_no_connection() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let provider = loopback_trace_provider(
        listener.local_addr().unwrap().port(),
        opentelemetry_sdk::trace::Sampler::AlwaysOff,
    )
    .unwrap();
    provider.tracer("qualification").in_span("analysis", |_| ());
    provider.force_flush().unwrap();
    provider.shutdown().unwrap();
    listener.set_nonblocking(true).unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn stalled_collector_does_not_block_the_operation() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let provider = loopback_trace_provider(
        listener.local_addr().unwrap().port(),
        opentelemetry_sdk::trace::Sampler::AlwaysOn,
    )
    .unwrap();
    let started = std::time::Instant::now();
    let result = provider.tracer("qualification").in_span("analysis", |_| 42);
    assert_eq!(result, 42);
    assert!(started.elapsed() < Duration::from_secs(1));
    // Leave the socket listening without servicing the request.
    let _flush = provider.force_flush();
    let _shutdown = provider.shutdown();
    assert!(started.elapsed() < Duration::from_secs(7));
    listener.set_nonblocking(true).unwrap();
    listener.accept().expect("export attempted");
}

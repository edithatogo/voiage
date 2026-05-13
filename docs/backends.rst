Backends
========

`voiage` supports a NumPy-first execution path with optional JAX acceleration
for workflows that benefit from compilation or hardware acceleration.

JAX is an implementation detail of the acceleration path rather than part of
the stable user-facing contract. The canonical public data structures remain
xarray-backed regardless of which backend is selected.

Summary:

- NumPy is the default baseline.
- JAX is used when backend-accelerated execution is requested and the runtime
  environment supports it.
- Apple Metal is available as an optional internal backend on macOS Apple
  Silicon when PyTorch provides an MPS device.
- Public APIs should continue to work when JAX is unavailable; accelerated
  paths are optional, not mandatory.
- The stable contract requires backend-dependent results to disclose that
  status explicitly.

Packaging and release guidance:

- Keep the Apple Metal backend optional and runtime-detected.
- Preserve the CPU fallback path as the authoritative reference path.
- Treat the benchmark handoff packet as the release evidence for device-backed
  comparison claims.
- Do not introduce a public API change just to expose the backend.

See also:

- `docs/developer_guide/architecture.rst`
- `specs/core-api/method-metadata.md`

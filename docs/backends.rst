Backends
========

`voiage` supports a NumPy-first execution path with optional JAX acceleration
for workflows that benefit from compilation or hardware acceleration.

Summary:

- NumPy is the default baseline.
- JAX is used when backend-accelerated execution is requested and the runtime
  environment supports it.
- The stable contract requires backend-dependent results to disclose that
  status explicitly.

See also:

- `docs/developer_guide/architecture.rst`
- `specs/core-api/method-metadata.md`


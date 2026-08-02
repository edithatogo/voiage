# Binding dispositions

The v1.0.0 COSS and EVSI/EVPI numerical policy is implemented by Rust and
exposed through the experimental Python façade. The retained R and Julia
bindings do not export these methods, and Mojo remains an external boundary.
Those three surfaces must report `unsupported_capability`; they must not route
to a Python fallback or imply cross-language parity.

`bindings.json` is the machine-readable authority. The normative fixture at
`specs/frontier/study-design-efficiency/v1/fixtures/normative/coss-efficiency.json`
is executed through Python against the Rust-owned kernels. This establishes
Rust/Python conformance only. Stable polyglot parity remains false until each
additional language has an installed-runtime result for the same fixture and
passes the separate scientific-promotion gate.

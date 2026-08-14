# Research-software review

Score: **826/1000**
Disposition: **not ready for submission**
Fail-closed cap: **applies**

## Repository assessment

The repository provides a Rust numerical core, a packaged Python surface,
direct native EVPI bindings for R and Julia, extensive tests, continuous
integration, signed-release evidence, and a Software Heritage snapshot. Python
is the broadest interface. R retains optional Python-backed EVPPI and EVSI;
Julia exposes EVPI only.

## Manuscript blockers

1. The original build-versus-contribute rationale did not explain why an
   environment-specific upstream package could not own the shared boundary.
2. The original Julia CI claim was stale: the current workflow has a
   Linux/macOS/Windows matrix.

## Packaging boundaries

- Python wheels bundle the native runtime and are the primary reviewer path.
- R native EVPI currently requires a separately supplied `voiage-ffi` library.
- Julia source testing also requires the native library; General/JLL
  publication is a separate registry gate.

These limitations should remain explicit and must not be converted into claims
of full cross-language feature parity.

## External gate

No manuscript wording can substitute for completed research use or independent
human engagement.

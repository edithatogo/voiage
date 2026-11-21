# Implementation Plan

ML01–ML03 and ML00 are complete. The [Mac benchmark and receipt](./evidence/ml00/README.md)
record a no-go on 21 November 2026. Both Metal routes exceeded the
preregistered common-offset p95 limit. MPS also exceeded the RSS budget.
ML04–ML06 are inapplicable to this outcome.

## Phase 1 — Feasibility

- [x] **ML01 — Qualify candidate runtimes.** Check MLX and its Metal wheels
  on a named Apple Silicon device with the repository's qualified Python
  3.15 interpreter. Record the exact artifacts, license closure and device
  availability. Compare with the existing PyTorch MPS path. Document the
  JAX Metal alternative and its suitability if it is qualified.
- [x] **ML02 — Freeze the comparison.** Bind the current Rust path, G05
  and independent analytic cases. Register representative workload shapes,
  fixture identities and the benchmark acceptance criteria before timing.
  Include the minimum useful end-to-end gain and minimum eligible share,
  plus fallback overhead, peak memory and repeated-run variation.
- [x] **ML03 — Prototype eligibility and accuracy.** Use NumPy/host float64
  processing and maintained GPU reductions. Establish a conservative error
  bound for admitted inputs and exercise the shifted, near-zero and extreme
  cases. Retain the admission cost and rejected-input behavior, including
  the evidence if no candidate clears the numerical gate.
- [x] **ML00 — Record the go/no-go benchmark.** On Apple Silicon, compare
  the direct Rust kernel, supported Python API and accuracy-qualified GPU
  candidate as specified. Record kernel and workflow timings, cold and warm
  runs, synchronization, memory, fallback and crossover points. Retain the
  raw results and choose go, no-go or blocked against the preregistered
  criteria. Failed numerical qualification can establish no-go before
  performance timing. On no-go, record ML04–ML06 as inapplicable and retain
  the evidence. Leave the gate open when required evidence is blocked.

## Phase 2 — Inapplicable after ML00

- **ML04 — Integrate one opt-in route (inapplicable).** With a go, reuse or adapt the
  winning Metal route at the validated per-decision EVPI boundary. Reuse
  the Rust fallback, shared scaling and appropriate capability metadata.
  Qualify any platform-gated optional extra through the normal dependency
  policy and record actual dispatch and fallback in diagnostics.
- **ML05 — Verify the public workflow and fallback (inapplicable).** Test the selected
  entry point with the G05 and analytic cases, invalid inputs, scaling,
  missing packages or devices, and rejected inputs. Confirm the base
  install and stable CPU behavior, including error semantics and caching.
  Run actual device checks on Apple Silicon and leave the CPU suite portable.
- **ML06 — Verify the final candidate (inapplicable).** Re-run the ML00 benchmark on
  the integrated path and validate the device evidence. Run the focused
  tests and the repository's final local and hosted gates. Document the
  measured scope and update the changelog only for delivered behavior.

## Feasibility follow-up

- [x] Profile MPS process memory on the same Mac. Compare Rust in the base
  and PyTorch-installed environments with the admitted MPS route. Retain
  the measurements separately from the preregistered ML00 gate.
  [Follow-up memory profile](./evidence/ml00/memory-profile.md)

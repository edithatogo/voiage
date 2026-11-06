# MLX/Metal EVPI experiment scope — 5 November 2026

## Candidate and boundary

Evaluate the existing EVPI kernel in `voiage/methods/basic.py` on an Apple
Silicon Metal device. The experiment is optional and does not change the public
API, dependency policy, CPU implementation, or supported numerical tolerances.
Observed host: arm64 Apple M1 Max, Metal 4; selected project Python: 3.15.1.
An isolated Python 3.15.1 environment imported MLX 0.32.4 and mlx-metal
0.32.4; MLX reported Metal available and GPU device 0 as the default.
A single three-row float32 GPU EVPI calculation returned 0.6666669845581055
against the analytic 2/3 (absolute difference about 3.18e-7). The attempted
float64 reduction reported that float64 is unsupported on the GPU. This is a
small capability probe, not parity, fallback or speedup qualification.

A further four-case probe compared the published installed voiage 3.1.0 Rust
CPU path with the naive MLX float32 calculation on the same M1 Max, using the
isolated Python 3.15.1 consumer environment and MLX 0.32.4. The unshifted
example differed by 3.18e-7; tie and single-strategy cases both returned zero.
Adding 1e9 to the example produced CPU 0.6666666269302368 and GPU 0.0. This
observed loss of the decision signal disqualifies the naive float32 route for
that input; it does not prove that a different algorithm can or cannot meet the
contract. The probe used a separately composed experimental environment, not a
packaged extension, and did not verify fallback or measure performance.
Centering the shifted input in float64 before conversion to GPU float32
returned 0.6666667461395264, an absolute CPU difference of 1.19e-7. This
recovers that example's signal but does not establish parity or safe behavior
for other input ranges, and its host preprocessing cost is unmeasured. Both
observed differences exceed the G05 numerical case register's relative
`1e-10` and absolute `1e-12` tolerances, so neither probe qualifies for
promotion under that witness.

## Qualification sequence

1. In an isolated Python 3.15.1 environment, resolve an authenticated MLX and
   Metal backend release; record versions, archive hashes, licences and actual
   import/device availability. A release announcement of Python support is not
   evidence that a compatible wheel exists for this exact interpreter and device.
2. Freeze the existing CPU EVPI function and representative deterministic inputs:
   empty inputs, single and multiple strategies, ties, invalid shapes, nonfinite
   values, large/small magnitudes and population-scaling options. Include exact
   analytic examples and the current numerical test tolerances. The selected
   public EVPI API does not take sample weights; weighted contracts require a
   separately scoped API and must not be inferred from this experiment.
3. Prototype the same EVPI calculation in MLX using an explicitly selected GPU.
   Compare every result and error case with the current CPU behavior, including
   dtype and reduction-order effects. The current stable path uses the Rust
   runtime and clamps the per-decision result at zero. If the device cannot
   satisfy the current precision contract, retain that CPU path for the input;
   do not loosen the oracle.
4. Verify the application remains functional when MLX or the GPU is unavailable,
   and that fallback selects the existing CPU path without silent precision loss.
5. Measure warm and cold execution on the named device at several input sizes;
   include transfer and synchronization costs, repetitions and CPU baseline.
   MLX evaluation is lazy, so force evaluation inside timed regions. Record
   hardware, OS, interpreter, dependency versions, numerical differences and
   allocation limits. Report regressions and crossover points, not only peak speed.
6. Promote an optional extension only after review of exact dependency and
   licence closure, numerical equivalence, fallback and real-device measurements.

MLX's [installation requirements](https://github.com/ml-explore/mlx/blob/main/docs/src/install.rst)
and [lazy-evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
are upstream references; live package and device qualification remain necessary.

## Exploratory device timings

A Python 3.15.1 experiment on the M1 Max compared the installed voiage
3.1.0 CPU API with the unqualified MLX 0.32.4 float32 GPU calculation, using
seeded normal inputs with four strategies. For 1,024, 65,536 and 262,144
rows, the five-run warm medians were respectively 0.349/0.312 ms,
21.661/0.943 ms and 92.829/1.716 ms for CPU/GPU. GPU timings include dtype
conversion, transfer and forced evaluation; CPU timings include the public
API and its data conversion. First calls per size are recorded, but only the
first size is a fresh-process startup. The [receipt](./mlx-metal-evpi-timings-20261105.json)
retains individual timings and numerical differences. This is not an
apples-to-apples kernel benchmark or a qualified speedup: the GPU path fails
the numerical witness, the workloads are small and system contention was not
controlled. The initial attempt from the source checkout failed because it
shadowed the installed wheel; the corrected run used a neutral working directory.

# Installed adapter qualification on 1 November 2026

The Voiage consumer uses Python 3.15.1 and an installed non-editable wheel.
The retained sample used the Voiage wheel built from commit
`89ab5f5ac466149c308496bb57b451bfb8fd8a38`, SHA-256
`8f46b93023c35962e7848c5e49215156395de9338aa7b0168119248e1eeaf231`.
The application chooses the synthetic payoff and joins draws by ID.
Consumer data and source checkouts live outside the producer environment.
See the compact JSON receipts for expected numerical values and artifact hashes.

## Inputs

In a private replay directory, place this packet's files alongside an
`inputs` directory. Check out these exact commits with a clean worktree:

| Checkout | Repository | Revision |
| --- | --- | --- |
| `inputs/innovate-src` | `edithatogo/innovate` | `83d556ba4995fd5f5238f4953e9a8af34976f464` |
| `inputs/mars` | `edithatogo/mars` | `d68b54a1c9a5eee40e1b7a724e002d5083ff6d8f` |
| `inputs/vop` | `edithatogo/vop_poc_nz` | `2c46db2fe5f907d894bb07f1127c008f10ee462e` |

Set `REPO` to the checked-out Voiage repository, `WHEEL` to the independently
built candidate wheel and `WHEEL_SHA` to its recorded expected digest. Set
`PY` to the Python 3.15.1 interpreter of the private candidate environment.
Build and install the pinned MARS wheel there. Run verification under
`scripts/guard_verification.py` with a distinct work directory for each test.

```sh
"$PY" -V
uv build --wheel --python "$PY" inputs/mars --out-dir wheels
uv pip install --python "$PY" "$WHEEL" wheels/mars_earth-*.whl
fixture=inputs/innovate-src/specs/ecosystem/voiage/uncertainty/diffusion_v1
"$PY" -I probe.py innovate "$fixture" --voiage-sha "$WHEEL_SHA"
"$PY" -I probe.py mars "$fixture" --voiage-sha "$WHEEL_SHA"
```

The first command using the fixture reads its documented CSVs. It joins the
two scenarios at the final time point and assigns a synthetic 1/20 penalty
to accelerated adoption. The second adapter passes the installed public
`pymars.EarthRegressor` to Voiage's optional custom-estimator EVPPI API using
a typed parameter set. Two affine payoffs switch sign at zero, giving the
independent finite reference 17/33.

## Fresh Innovate native handoff

Generate six keyed Bass inputs from the declared parameter fixture, then run
Innovate's pinned native prediction API using Rust 1.99.1. Its Kairos
dependencies use commit `fae901558f07b7b717a676adbafbe2cdc78dea1c`.
The locked consumer binary rejects responses without the Rust-native runtime
marker. Independent 50-digit decimal calculations check all 24 new trajectory
points and expected value before the installed Voiage call. The 35-unit
penalty is an illustrative application assumption.

```sh
"$PY" -I prepare_native.py "$fixture/parameter_draws.csv" native-inputs.json
rustup run 1.99.1 cargo run -j2 --locked --manifest-path native/Cargo.toml \
  <native-inputs.json >native-output.json
"$PY" -I native_evpi.py native-inputs.json native-output.json
```

The native qualification is independent of Innovate's Python wheel install.
Its Python 0.5.0 wheel built successfully, while binary-only installation on
macOS arm64 / Python 3.15.1 lacked a matching Arrow wheel within its declared
`>=23.0.1,<26` range. No claim is made for that Python installation.

## VOP

Use a separate Python 3.15.1 environment for the pinned historical source
export. The minimal exporter environment used numpy, scipy, PyYAML, the
dataframe, plotting and statistics dependencies listed in the source
project, and its schema validation dependencies. Install only the packages
needed to import the historical exporter. Run from a clean source checkout.

```sh
"$EXPORT_PY" -I "$REPO/scripts/run_vop_research_handoff.py" export \
  --vop-root inputs/vop --output vop-export
"$PY" -I candidate_vop.py vop-export "$REPO/scripts/run_vop_research_handoff.py" \
  "$WHEEL" "$WHEEL_SHA"
```

The reference seed is 20260727 and the receipt binds 500 model draws. The
candidate verifier checks every installed wheel member except the installer-
rewritten RECORD. The historical VOP package declares Python >=3.12,<3.15,
including at the separately observed current revision
`ad84d3090a2d97f2805a86e94a7b9614a3d1ab14`. This establishes source
export and installed Voiage evaluation only. The existing public-wheel
verifier is unchanged. Installing its reviewed macOS wheel with ordinary
dependency resolution on Python 3.15.1 stopped during the Arrow 25.0.1 source
build. The candidate check does not promote that public-wheel environment.

For the separate MARS compiled-prediction witness run
`"$PY" -I native_mars.py`. This fits through the public estimator and
explicitly sends its JSON model artifact to the compiled predictor. The
receipt includes the extension digest and the maximum error over 33 points.
It does not establish native training parity.

The VOP historical reference in `paper/research-use/v2.2.0/export.json`
independently binds the expected CSV SHA, VOP revision, draw count and seed.
The Innovate manifest and both CSV files have independently retained SHA-256
values. The negative controls alter the Innovate manifest and substitute an
all-zero VOP CSV with a self-consistently rehashed receipt. Both changes were
accepted by the first draft and rejected after review. To replay:

```sh
"$PY" -I challenge_inputs.py . "$fixture" vop-export \
  "$REPO/scripts/run_vop_research_handoff.py" "$WHEEL" "$WHEEL_SHA"
```


The fixture and estimator probes check Python and distribution versions, the
selected wheel SHA-256, installed member bytes and the imported module path.
The retained MARS result binds the exact wheel in `package-identities.json`.
A different MARS build needs its own identity receipt. The negative suite
also substitutes an outside import origin and requires its specific rejection.

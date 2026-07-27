# Voiage.jl

Julia package scaffold for the voiage core API contract.

## Setup

From `bindings/julia/`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## First workflow

```julia
using Voiage

net_benefits = [10.0 1.0; 2.0 8.0]
evpi_value = evpi(net_benefits)

@show evpi_value
```

This example returns `3.0` for the simple two-strategy matrix above.

Decision Problem and statistical-assurance documents can also be validated
without Python. Both functions return the Rust-normalized contract:

```julia
problem = normalize_decision_problem(Dict(
    "decision_problem_id" => "example",
    "title" => "Example decision",
    "analysis_type" => "net-benefit-first",
    "currency" => "AUD",
    "willingness_to_pay" => 50_000,
    "interventions" => [
        Dict("intervention_id" => "usual-care", "name" => "Usual care"),
        Dict("intervention_id" => "new-care", "name" => "New care"),
    ],
))

table = read_voiage_arrow("decision-problems.arrow")
```

`read_voiage_arrow` reads the pinned Arrow IPC contract and requires the
lossless `payload_json` column. Set `VOIAGE_FFI_LIBRARY` to a built
`libvoiage_ffi` library; distribution through a Julia JLL remains a separate
registry-owned release step.

## Release and caveats

Julia package publication is handled through the General registry. The release
workflow verifies that `Project.toml` matches the release tag, runs `Pkg.test()`,
and attaches a source archive to the GitHub release. Registry updates should
use the Julia Registrator flow, and the scheduled TagBot workflow keeps GitHub
tags and releases aligned after registry merges. This adapter keeps Rust as the
semantic authority. General-registry and JLL publication are external release
gates and are not implied by passing source tests.

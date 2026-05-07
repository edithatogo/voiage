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

## Release and caveats

Julia package publication is handled through the General registry. The release
workflow verifies that `Project.toml` matches the release tag, runs `Pkg.test()`,
and attaches a source archive to the GitHub release. Registry updates should
use the Julia Registrator flow, and the scheduled TagBot workflow keeps GitHub
tags and releases aligned after registry merges. The walkthrough here is a
thin adapter story: it is deliberately small because the Rust core remains the
semantic authority.

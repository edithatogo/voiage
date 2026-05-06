# Track Specification: HPC Distribution And Acceleration Strategy

## Overview
This track decides how `voiage` should present itself in HPC-adjacent
distribution channels and what accelerator strategy is realistic for the
current and future Rust-core architecture. The aim is to separate "can be
installed on HPC systems" from "is HPC-native" and from "should ever be pushed
onto GPU/TPU/custom-circuit hardware."

The relevant ecosystem targets are:

- HPSF
- E4S
- Spack
- EasyBuild

## Functional Requirements
1. The track must define the HPC fit of the library in plain terms:
   - HPC-deployable
   - HPC-friendly
   - HPC-native
   - not worth HPC registry treatment
2. The track must explain what "listing" means for each target ecosystem.
   There is no single universal HPC registry, so the track must distinguish
   between:
   - curated ecosystem inclusion
   - package recipe availability
   - module or environment compatibility
   - adoption by an HPC stack such as E4S
3. The track must produce a distribution matrix for Spack and EasyBuild,
   including the source artifacts, dependency declarations, compiler/runtime
   expectations, and validation gates needed for each.
4. The track must assess accelerator options in a rank-ordered way:
   - CPU parallelism
   - SIMD / vectorization
   - GPU acceleration
   - TPU or graph-compiler acceleration
   - custom-circuit / ASIC-style acceleration
5. The track must document where the library should stop short of accelerator
   work if the control flow or economics do not justify it.
6. The track must state the benchmark evidence required before promoting any
   accelerator idea into an implementation track.

## Non-Functional Requirements
1. The track must prefer portability and reproducibility over speculative
   hardware work.
2. The track must treat HPSF and E4S as ecosystem/curation targets rather than
   a substitute for package-manager distribution.
3. Accelerator recommendations must be based on workload shape, not on a
   generic desire to "use the fastest hardware."

## Acceptance Criteria
1. A distribution path exists for each relevant HPC ecosystem target.
2. The repo can state whether it is HPC-deployable, HPC-friendly, or HPC-native.
3. Accelerator options are ranked with clear escalation criteria.
4. The track explicitly calls out where GPU, TPU, and custom-circuit work is
   not worth pursuing.

## Out of Scope
1. Building or maintaining a real HPC cluster deployment.
2. Implementing new accelerator kernels.
3. Claiming HPC-native status without evidence from benchmarks and deployment
   recipes.

# Track Specification: HPC Distribution Contract

## Overview

This track turns the archived HPC strategy outcome into an explicit contract
for what the HPC-facing version of ``voiage`` should look like. The contract
is intentionally conservative: the library should be HPC-deployable and
HPC-friendly, but not HPC-native.

The relevant targets are:

1. Spack
2. EasyBuild
3. HPSF
4. E4S

## Functional Requirements

1. Define the HPC fit of the library in plain terms so the repo can state the
   difference between deployable, friendly, native, and out-of-scope.
2. Define what "listing" means for each HPC-facing target.
3. Record the source, dependency, and reproducibility rules needed for Spack
   and EasyBuild recipes.
4. Record the curation and visibility expectations for HPSF and E4S.
5. Rank CPU parallelism, SIMD, GPU, TPU, and custom-circuit ideas by plausibility.
6. State the benchmark evidence required before any accelerator idea becomes a
   real implementation track.

## Non-Functional Requirements

1. Prefer portability and reproducibility over speculative hardware work.
2. Treat HPSF and E4S as ecosystem targets rather than substitutes for package
   recipes.
3. Avoid claiming HPC-native status without evidence from benchmarks and site
   deployment.

## Acceptance Criteria

1. The repo can state whether ``voiage`` is HPC-deployable, HPC-friendly, or
   HPC-native.
2. A distribution path exists for Spack and EasyBuild.
3. The registry / curation meaning of HPSF and E4S is explicit.
4. Accelerator options are ranked with clear escalation criteria.

## Out of Scope

1. Building or maintaining a real HPC cluster deployment.
2. Implementing new accelerator kernels.
3. Claiming HPC-native status without benchmark-backed evidence.

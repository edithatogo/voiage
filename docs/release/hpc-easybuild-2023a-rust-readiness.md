# Stable Rust validation providers for foss 2023a

The candidate in `packaging/easybuild-2023a-rust-overlay/` extends the 2023a
foundation with source-bound stable Rust, Maturin, Pydantic and JSON Schema
recipes. Pydantic 2.13.4 uses its exact pydantic-core 2.46.4 requirement.
Build-only backend providers avoid changing the existing foundation recipes.

Actual dependency resolution, verified source archives, offline Cargo metadata,
Python backend source builds and a scoped module load/unload probe are retained.
These checks do not establish native EasyBuild compiler or extension builds.
Rust's prebuilt stage-zero compiler remains an explicit bootstrap exception;
its checksums were read from verified source, not from executed binary archives.

Use the overlay README for the dry-run command, source and compiler boundaries,
and remaining full-stack requirements. Both foss generations still require
complete Voiage graphs, native build evidence and upstream review.

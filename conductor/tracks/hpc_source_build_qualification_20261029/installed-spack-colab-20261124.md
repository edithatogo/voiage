# Installed Linux Spack qualification — 24 November 2026

Spack 1.2.2 installed Voiage 3.1.0 from the qualified source on a two-vCPU x86-64
Ubuntu 24.04.5 Colab runtime with glibc 2.39. The installed Python reported
3.15.2 and GCC 13.3.0. The signed recipe and qualification revision is
`ce94ed0ac22f7c7608a7bc4bcdc29a113bab7599` with tree
`dfb0473e03eb1d74c926cf981d00068799243deb`.

The Rust extension reports immutable Voiage source
`0a961dafd2e68a480c668b30600430d9ac71be38`, tree
`6561ac4d928fa2bf75ff15ca19981c6e1b416018`, and clean source state.
The selected stable Rust 1.99.1 and dated nightly 1.100.0 toolchains both
passed compiled fixtures after their source builds. The final network-denied
stage replayed the Voiage Cargo lock against 161 verified vendored packages.
The final solve retained 136 identical specs from the previous lock and
rebuilt the Voiage root with the selected Spack C compiler and runtime.

| Selected package | Version | Spack DAG hash |
| --- | --- | --- |
| Python | 3.15.2 | `5po3ujjtuloeuzhv72y2is4wzc6px3iy` |
| Arrow | 26.0.0 | `dye6s27syj4t4izpkdquhdgta7a3o5yg` |
| PyArrow | 26.0.0 | `dre4tlwbh47quxj4psnv6u376v7ovjn3` |
| Polars runtime | 2.1.0 | `dudzqqxaagcugghs7uxhk6ikymkbtjt2` |
| GCC runtime | 13.3.0 | `xkwmvkqhsrpuoh5ftqhp7n52f7ex5ppr` |
| Voiage | 3.1.0 | `hdmjlksd3ntv2uvbgcb3jp6i4ygbelxt` |

Spack's install-time imports of `voiage` and `voiage._core` passed. The
consumer ran outside the source checkout and returned EVPI of 1.0.
Arrow IPC and Parquet round trips preserved the schema and nulls. Polars
applied the expected null filter and returned the matching Arrow table.
Load, CLI help and unload isolation passed. The four native extensions
resolved every library to the selected Spack DAG or the allowed system glibc
libraries. The Voiage extension selected Spack's `libgcc_s.so.1`. The Arrow
and Parquet SDK had no dependency on the host's shared URI parser.

The final install stage and aggregate guard exited zero with complete process
cleanup. The final continuation took 193.669 seconds and preserved at least
215,821,119,488 bytes of free space. The preceding run built and installed
the reused Polars and PyArrow packages. It stopped on the Voiage import test
after reaching 11,046,520 KiB peak aggregate RSS.

Earlier failed attempts and their corrections remain in the recovery records.
Arrow built its checksum-pinned simdjson and URI parser sources. The PyArrow
consumer used the selected LZ4 prefix, and Polars used the graph's GNU Make
4.4.1 for Cargo's FIFO jobserver. Voiage's final test used Spack's normal load
environment and the compiler wrapper selected through its declared C build
dependency. [Arrow dependencies](https://github.com/edithatogo/voiage/pull/1400)
· [URI parser and import-path repair](https://github.com/edithatogo/voiage/pull/1401)
· [Polars and PyArrow build tools](https://github.com/edithatogo/voiage/pull/1402)

## Retained evidence

| Artifact | SHA-256 |
| --- | --- |
| Spack lock | `6a78216c6bd8ae1a4352670039781315a8a231c013ed11b58ff7630d6d36dbf1` |
| `spack-ce94ed0ac-source-replay-receipt.tar.gz` | `f9108d30e6a10335e77d9053758d311188b83047f74346c18be1adfdb6a8b5a0` |
| `spack-ce94ed0ac-installed-acceptance.tar.gz` | `7ae4f8314d149aec5b5ab3aacab6fc9e7d778c61744c80d5b6f77cabd2b39e53` |
| Installed consumer probe JSON | `1b6bf33c8c15907109b013bff6ed7f0dfa25c79b06b78c6c6323283c04c1c821` |

Both archives are digest-verified in private recovery storage. The installed
archive includes the terminal, import, CLI, linkage, load/unload and resource
receipts. The earlier branch tips for PRs #1400–#1402 were retired after an
independent Git restore, exact tree comparisons and strict Git fsck.

EasyBuild foss/2023a and foss/2024a remain deferred. Their former Colab
sessions were unavailable at the final status read, and their retained
hosted attempts failed. Matrix acceptance and final archival await successful
installed evidence for both generations. [Hosted attempt](https://github.com/edithatogo/voiage/actions/runs/45034305228)

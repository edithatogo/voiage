# Native EasyBuild qualification

The native driver builds one maintained generation at a time and refuses to
run without an exact clean source commit, the pinned EasyBuild catalogue, a
fresh VM-local workspace, Linux, Environment Modules, eight CPUs and 250 GiB
free. It does not use a repository, virtual-environment or Spack installation
as native evidence.

Run `foss-2023a` to a terminal receipt before starting `foss-2024a`. The two
generations may share downloaded source archives, but use separate build,
install and module roots. Invoke the driver only inside the prepared Linux VM:

```sh
VOIAGE_VM_LOCAL_STORAGE=confirmed python3 scripts/native_easybuild_qualification.py \
  2023a --root /vm/voiage-9307d9ec --catalogue /vm/easyconfigs-58e8b5a \
  --workspace /vm/easybuild --source-cache /vm/prepared-sources \
  --run-id native-20260904 --execute
```

Repeat with `2024a` only after reviewing the first receipt, using the same
`--run-id` and adding
`--predecessor-receipt /vm/easybuild/voiage-easybuild-2023a/receipt.json`.
The prepared cache must contain the complete release and transitive source set;
the driver copies it into a read-only generation-local tree and disables downloads.
A successful build
is followed in a fresh shell by module, CLI, Rust engine, exact EVPI, Arrow,
Polars, linkage, threaded-core and unload-isolation probes. The aggregate
validator accepts readiness only when exact terminal receipts for both
generations pass. Upstream submission and production-cluster qualification
remain separate gates.

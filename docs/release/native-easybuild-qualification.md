# Native EasyBuild qualification

The native driver builds one maintained generation at a time and refuses to
run without an exact clean source commit, the pinned EasyBuild catalogue, a
fresh VM-local workspace, Linux, Environment Modules, eight CPUs and at least
2 GiB free. This floor is a safety reserve, not a sufficient build-space estimate;
monitor free space during execution and stop before exhausting the filesystem.
It does not use a repository, virtual-environment or Spack installation as
native evidence.

Use two distinct clean checkouts: the candidate checkout remains pinned to
`9307d9ec`, while the tooling checkout contains this driver, its receipt schema,
validator and probe. The receipt binds both Git identities and the exact driver
and probe bytes. Run the command from the tooling checkout; `--root` always
names the candidate being qualified and `--tooling-root` names the reviewed
tooling checkout.

Run `foss-2023a` to a terminal receipt before starting `foss-2024a`. The two
generations may share downloaded source archives, but use separate build,
install and module roots. Invoke the driver only inside the prepared Linux VM,
under the resource guard. The guard monitors the workspace and its temporary
filesystem; also monitor the host backing storage separately. Set appropriate
memory and time limits for the qualified builder before starting:

```sh
cd /vm/voiage-native-easybuild-tooling
VOIAGE_VM_LOCAL_STORAGE=confirmed .venv/bin/python scripts/guard_verification.py \
  --work-dir /vm/easybuild/guard-2023a --min-free-mib 2048 \
  -- .venv/bin/python scripts/native_easybuild_qualification.py \
  2023a --root /vm/voiage-9307d9ec --tooling-root "$PWD" \
  --catalogue /vm/easyconfigs-58e8b5a \
  --workspace /vm/easybuild --source-cache /vm/prepared-sources \
  --run-id native-20260904 --execute
```

Repeat with `2024a` only after reviewing the first receipt, using a distinct
`--work-dir /vm/easybuild/guard-2024a`, the same `--run-id` and adding
`--predecessor-receipt /vm/easybuild/voiage-easybuild-2023a/receipt.json`.
The prepared cache must contain the complete release and transitive source set;
the driver copies it into a read-only generation-local tree and disables downloads.
A successful build
is followed in a fresh shell by module, CLI, Rust engine, exact EVPI, Arrow,
Polars, linkage, threaded-core and unload-isolation probes. The aggregate
validator accepts readiness only when exact terminal receipts for both
generations pass. Upstream submission and production-cluster qualification
remain separate gates.

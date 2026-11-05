#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 CELL OUTPUT" >&2; exit 2; }
cell=$1
out=$2
case "$cell" in spack|easybuild-2023a|easybuild-2024a) ;; *) exit 2 ;; esac
mkdir -p "$out"
trap 'status=$?; printf "{\"installed_cell_accepted\":false,\"exit_code\":%d}\n" "$status" > "$out/result.json"; exit "$status"' ERR
root=$PWD
work="${RUNNER_TEMP:?}/voiage-sources-$cell"
bash scripts/qualify_linux_sources.sh "$cell" "$out/source"
export VOIAGE_COMPILER_RECEIPTS="$out/compiler-configurations"
mkdir -p "$VOIAGE_COMPILER_RECEIPTS" "$work/consumer"
preserved=$("$PWD/.venv/bin/python" -c 'from scripts.enter_native_namespace import preserved_environment_names; print(preserved_environment_names())')
build=(sudo --non-interactive "--preserve-env=$preserved" env "PATH=$PATH" "$root/.venv/bin/python" -m scripts.observe_native_stage
  --evidence "$out/install-stage" --workspace "$work" --cwd "$root" --timeout 14400 --uid "$(id -u)" --gid "$(id -g)" --)
export CARGO_NET_OFFLINE=true CARGO_NET_RETRY=0
if [[ "$cell" == spack ]]; then
  export SPACK_USER_CONFIG_PATH="$out/source/graph/config"
  export SPACK_USER_CACHE_PATH="$RUNNER_TEMP/voiage-spack-cache"
  spack config --scope user add 'config:build_jobs:4'
  spack -e "$work/environment" config get > "$out/effective-config.yaml"
  "${build[@]}" "$(command -v spack)" -e "$work/environment" install --only-concrete --no-cache --fail-fast --jobs 4 --concurrent-packages 1 --test root
  spack -e "$work/environment" find --json > "$out/installed-specs.json"
  spack -e "$work/environment" load --sh py-voiage > "$out/load.sh"
  export VOIAGE_SPACK_BIN
  VOIAGE_SPACK_BIN=$(command -v spack)
  export VOIAGE_SPACK_ENV="$work/environment"
  prefix="$work/install"
else
  export FPATH="${FPATH:-}" # Newer Lmod initialization expects this under nounset.
  source /etc/profile.d/lmod.sh
  generation=${cell#easybuild-}
  export VOIAGE_QUALIFIED_SOURCE_ARCHIVE="$out/source/graph/source-dist/voiage-2.2.0.tar.gz"
  export VOIAGE_QUALIFIED_SOURCE_SHA256
  VOIAGE_QUALIFIED_SOURCE_SHA256=$(sha256sum "$VOIAGE_QUALIFIED_SOURCE_ARCHIVE" | cut -d ' ' -f1)
  export VOIAGE_SOURCE_ROOTS="$work/sources:$root/packaging/current/easybuild:$root/.graph-tools/easyconfigs/easybuild/easyconfigs:$out/source/graph/source-dist"
  export VOIAGE_SOURCE_RECEIPTS="$out/resolved-install"
  robot="$root/packaging/current/easybuild/$generation:$root/packaging/current/easybuild:$root/.graph-tools/easyconfigs/easybuild/easyconfigs"
  "${build[@]}" "$root/.venv/bin/python" -m easybuild.main "$out/source/graph/root.eb" --robot --minimal-toolchains "--robot-paths=$robot" --enforce-checksums --trace \
    "--sourcepath=$work/sources" "--buildpath=$work/build" "--prefix=$work/prefix" "--hooks=$root/scripts/easybuild_source_hooks.py" --parallel=4 --max-parallel=4
  prefix="$work/prefix"
  printf 'export FPATH="${FPATH:-}"\nsource /etc/profile.d/lmod.sh\nmodule purge\nmodule use %q\nmodule load %q\n' "$prefix/modules/all" "voiage/2.2.0-foss-$generation-candidate-Python-3.15.1" > "$out/load.sh"
  printf 'module purge\n' > "$out/unload.sh"
fi
[[ $(find "$VOIAGE_COMPILER_RECEIPTS" -name '*.toml' -type f | wc -l) -ge 2 ]]
export VOIAGE_INSTALL_CELL="$cell" VOIAGE_INSTALL_OUTPUT="$out" VOIAGE_INSTALL_PREFIX="$prefix" VOIAGE_PROBE_SCRIPT="$root/scripts/probe_current_hpc_install.py" VOIAGE_FRONTEND_PYTHON="$root/.venv/bin/python"
cd "$work/consumer"
bash -euo pipefail <<'SH'
source "$VOIAGE_INSTALL_OUTPUT/load.sh"
voiage --help > "$VOIAGE_INSTALL_OUTPUT/cli-help.txt"
python "$VOIAGE_PROBE_SCRIPT" --prefix "$VOIAGE_INSTALL_PREFIX" --cell "$VOIAGE_INSTALL_CELL" --output "$VOIAGE_INSTALL_OUTPUT/installed-probe.json"
if [[ "$VOIAGE_INSTALL_CELL" == spack ]]; then
  "$VOIAGE_SPACK_BIN" -e "$VOIAGE_SPACK_ENV" unload --sh --all > "$VOIAGE_INSTALL_OUTPUT/unload.sh"
fi
source "$VOIAGE_INSTALL_OUTPUT/unload.sh"
"$VOIAGE_FRONTEND_PYTHON" -I -c 'import importlib.util,os; from pathlib import Path; prefix=Path(os.environ["VOIAGE_INSTALL_PREFIX"]); assert importlib.util.find_spec("voiage") is None; assert all(not Path(p).is_relative_to(prefix) for k in ("PATH","PYTHONPATH","LD_LIBRARY_PATH") for p in os.environ.get(k,"").split(os.pathsep) if p)'
SH
printf '{"installed_cell_accepted":true,"whole_matrix_accepted":false}\n' > "$out/result.json"

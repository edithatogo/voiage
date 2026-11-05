#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 CELL OUTPUT" >&2; exit 2; }
cell=$1
out=$2
case "$cell" in spack|easybuild-2023a|easybuild-2024a) ;; *) exit 2 ;; esac
work="${RUNNER_TEMP:?Linux runner private temporary directory required}/voiage-sources-$cell"
mkdir "$work" # A prior run is evidence, never an implicit mutable cache.
mkdir -p "$out"
trap 'status=$?; printf "{\"source_replay_passed\":false,\"installed_acceptance\":false,\"reason\":\"source_preflight_failed\",\"exit_code\":%d}\n" "$status" > "$out/result.json"; exit "$status"' ERR
export VOIAGE_SOURCE_WORK="$work"
bash scripts/qualify_linux_graph.sh "$cell" "$out/graph"

# Verify this kernel can deny networking before relying on offline replay.
# Drop back to the ordinary runner user inside the private network namespace.
preserved=$("$PWD/.venv/bin/python" -c 'from scripts.enter_native_namespace import preserved_environment_names; print(preserved_environment_names())')
offline=(sudo --non-interactive "--preserve-env=$preserved" unshare --net -- sudo --non-interactive "--preserve-env=$preserved" -u "$(id -un)")
"${offline[@]}" "$PWD/.venv/bin/python" - "$out/network-denial.json" <<'PY'
import errno
import json
from pathlib import Path
import socket
import sys
try:
    with socket.create_connection(("192.0.2.1", 443), timeout=2):
        raise RuntimeError("External network unexpectedly available")
except OSError as error:
    if error.errno != errno.ENETUNREACH:
        raise RuntimeError("Network denial was not established") from error
    Path(sys.argv[1]).write_text(json.dumps({"network_denied": True, "errno": error.errno}) + "\n")
PY

if [[ "$cell" == spack ]]; then
  export SPACK_USER_CONFIG_PATH="$out/graph/config"
  export SPACK_USER_CACHE_PATH="${RUNNER_TEMP}/voiage-spack-cache"
  spack config --scope user add "config:build_stage:[$work/stage]"
  spack config --scope user add "config:source_cache:$work/cache"
  spack config --scope user add "config:install_tree:root:$work/install"
  spack env create --dir "$work/environment" --without-view
  spack -e "$work/environment" add py-voiage@develop
  spack -e "$work/environment" concretize --fresh > "$out/concretize.log" 2>&1
  cp "$work/environment/spack.lock" "$out/spack.lock"
  spack -e "$work/environment" mirror create --all --directory "$work/mirror" --jobs 2 > "$out/fetch.log" 2>&1
  # Spack cannot checksum mirrored VCS archives itself. Freeze both its private
  # cache and mirror immediately after online fetch, and check before replay.
  mkdir -p "$work/cache"
  for cache in cache mirror; do
    .venv/bin/python scripts/source_cache_manifest.py "$work/$cache" "$out/$cache-manifest.json"
  done
  spack mirror add --scope user qualified "file://$work/mirror"
  for cache in cache mirror; do
    .venv/bin/python scripts/source_cache_manifest.py "$work/$cache" "$out/$cache-manifest.json" --check
  done
  # With no spec argument, supported Spack stage traverses the concrete DAG.
  # --path aliases resources; --dependencies is not a supported stage option.
  "${offline[@]}" "$(command -v spack)" -e "$work/environment" stage > "$out/offline-stage.log" 2>&1
  .venv/bin/python scripts/source_cache_manifest.py "$work/mirror" "$out/mirror-manifest.json" --check
  .venv/bin/python scripts/source_cache_manifest.py "$work/cache" "$out/cache-manifest.json" --check --mirror-manifest "$out/mirror-manifest.json" --after-manifest "$out/cache-after-manifest.json"
  PYTHONPATH="$PWD" spack -e "$work/environment" python scripts/spack_source_witness.py "$work" "$out"
  spack -e "$work/environment" config get > "$out/effective-config.yaml"
else
  export FPATH="${FPATH:-}" # Newer Lmod initialization expects this under nounset.
  source /etc/profile.d/lmod.sh
  generation=${cell#easybuild-}
  robot="$PWD/packaging/current/easybuild/$generation:$PWD/packaging/current/easybuild:$PWD/.graph-tools/easyconfigs/easybuild/easyconfigs"
  export VOIAGE_QUALIFIED_SOURCE_ARCHIVE="$out/graph/source-dist/voiage-2.2.0.tar.gz"
  export VOIAGE_QUALIFIED_SOURCE_SHA256
  VOIAGE_QUALIFIED_SOURCE_SHA256=$(sha256sum "$VOIAGE_QUALIFIED_SOURCE_ARCHIVE" | cut -d ' ' -f1)
  export VOIAGE_SOURCE_ROOTS="$work/sources:$PWD/packaging/current/easybuild:$PWD/.graph-tools/easyconfigs/easybuild/easyconfigs:$out/graph/source-dist"
  mkdir "$work/sources"
  # Seed EasyBuild's exact auxiliary helper before its optional GNU download.
  # The fixed mirror revision must match EasyBuild's own pinned SHA-256.
  .venv/bin/python - "$work/sources" "$out/config-guess-source.json" <<'PY'
import json
from pathlib import Path
import sys
from easybuild.easyblocks.generic.configuremake import CONFIG_GUESS_COMMIT_ID, CONFIG_GUESS_SHA256
from easybuild.easyblocks.generic.configuremake import EASYBLOCKS_VERSION
from scripts.qualify_source_vendor import fetch, digest
entry = {
    "filename": "config.guess",
    "url": f"https://raw.githubusercontent.com/chipp/gnu-config/{CONFIG_GUESS_COMMIT_ID}/config.guess",
    "sha256": CONFIG_GUESS_SHA256,
}
cache = Path(sys.argv[1]) / "generic" / f"eb_v{EASYBLOCKS_VERSION}" / "ConfigureMake"
path = fetch(entry, cache)
path.chmod(0o755)
Path(sys.argv[2]).write_text(json.dumps(dict(entry, resolved_path=str(path), observed_sha256=digest(path)), indent=2) + "\n")
PY
  args=("$out/graph/root.eb" --robot --minimal-toolchains "--robot-paths=$robot" --fetch --enforce-checksums --trace
    "--sourcepath=$work/sources" "--buildpath=$work/build" "--prefix=$work/prefix"
    "--hooks=$PWD/scripts/easybuild_source_hooks.py" --parallel=4 --max-parallel=4)
  export VOIAGE_SOURCE_RECEIPTS="$out/resolved-online"
  .venv/bin/python -m easybuild.main "${args[@]}" > "$out/fetch.log" 2>&1
  export VOIAGE_SOURCE_RECEIPTS="$out/resolved-offline"
  "${offline[@]}" "$PWD/.venv/bin/python" -m easybuild.main "${args[@]}" > "$out/offline-fetch.log" 2>&1
  diff --recursive "$out/resolved-online" "$out/resolved-offline" > "$out/replay-diff.txt"
fi
printf '{"source_replay_passed":true,"installed_acceptance":false}\n' > "$out/result.json"

#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 spack|easybuild-2023a|easybuild-2024a OUTPUT" >&2; exit 2; }
case "$1" in spack|easybuild-2023a|easybuild-2024a) ;; *) exit 2 ;; esac
cell=$1
out=$2
mkdir -p "$out"
git rev-parse HEAD > "$out/source-commit.txt"
git rev-parse 'HEAD^{tree}' > "$out/source-tree.txt"
uname -a > "$out/host.txt"
python --version > "$out/python.txt" 2>&1
if [[ "$cell" == spack ]]; then
  export SPACK_USER_CONFIG_PATH="$out/config"
  export SPACK_USER_CACHE_PATH="${RUNNER_TEMP:-/tmp}/voiage-spack-cache"
  mkdir -p "$SPACK_USER_CONFIG_PATH"
  cp packaging/current/spack/{concretizer,packages}.yaml "$SPACK_USER_CONFIG_PATH/"
  spack repo add --scope user "$PWD/packaging/current/spack"
  spack config --scope user add 'repos:builtin:commit:fa170d11f60ea44045d9a02370002ed153e35f1a'
  spack compiler find
  spack --version > "$out/manager-version.txt"
  spack spec --json py-voiage@develop > "$out/graph.json" 2> "$out/solver.stderr"
  spack spec -Il py-voiage@develop > "$out/graph.txt"
else
  generation=${cell#easybuild-}
  source /etc/profile.d/lmod.sh
  eb --version > "$out/manager-version.txt"
  # Graph evaluation only. No source is fetched or built by the dry-run command.
  export VOIAGE_QUALIFIED_SOURCE_ARCHIVE="$out/graph-only-source-not-qualified.tar.gz"
  export VOIAGE_QUALIFIED_SOURCE_SHA256=0000000000000000000000000000000000000000000000000000000000000000
  python - "$generation" "$out" <<'PY'
from pathlib import Path
import sys
generation, output = sys.argv[1:]
source = Path("packaging/current/easybuild/voiage-candidate-Python-3.15.1.eb")
text = source.read_text().replace(
    'toolchain = {"name": "system", "version": ""}',
    f'toolchain = {{"name": "foss", "version": "{generation}"}}',
)
# These three import surfaces are supplied by the checksum-pinned support bundle.
for dependency in (
    '    ("click", "8.5.0"),\n',
    '    ("typing_extensions", "4.16.0"),\n',
    '    ("typer", "0.27.2"),\n',
):
    if dependency not in text:
        raise ValueError("Current support-bundle mapping drift")
    text = text.replace(dependency, "")
text = text.replace('dependencies = [\n', 'dependencies = [\n    ("Voiage-Python-support", "2.2.0"),\n', 1)
Path(output, "root.eb").write_text(text)
PY
  eb "$out/root.eb" --dry-run --robot --minimal-toolchains --robot-paths="$PWD/packaging/current/easybuild/$generation:$PWD/packaging/current/easybuild:$PWD/.graph-tools/easyconfigs/easybuild/easyconfigs" --allow-use-as-root-and-accept-consequences > "$out/graph.txt" 2>&1
  python - "$out/graph.txt" <<'PYTHON'
from pathlib import Path
import re
import sys
modules = re.findall(r"module: (Python/[^)]+)", Path(sys.argv[1]).read_text())
if not modules or any(not module.startswith("Python/3.15.1-") for module in modules):
    raise ValueError(f"Unqualified Python modules in resolved graph: {modules}")
PYTHON
fi

#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

spack_repo="$work_dir/spack-repo"
mkdir -p "$spack_repo/packages/py-voiage"
printf 'repo:\n  namespace: voiage_hpc\n' > "$spack_repo/repo.yaml"
cp "$repo_root/packaging/spack/package.py" "$spack_repo/packages/py-voiage/package.py"

SPACK_USER_CONFIG_PATH="$work_dir/spack-config" \
SPACK_USER_CACHE_PATH="$work_dir/spack-cache" \
  spack repo add --scope user "$spack_repo"
SPACK_USER_CONFIG_PATH="$work_dir/spack-config" \
SPACK_USER_CACHE_PATH="$work_dir/spack-cache" \
  spack spec py-voiage@2.0.0

# Lmod's shell initialisation makes its module command available to EasyBuild.
# Prefer it because current EasyBuild releases do not recognise the version
# interface emitted by the Homebrew Environment Modules formula.
if [ -r /opt/homebrew/opt/lmod/init/zsh ]; then
  # shellcheck disable=SC1091
  source /opt/homebrew/opt/lmod/init/zsh
elif command -v modulecmd >/dev/null 2>&1; then
  :
else
  echo "EasyBuild syntax requires Lmod or Environment Modules." >&2
  exit 2
fi

uv tool run --from easybuild-framework --with easybuild-easyblocks --with pycodestyle \
  eb --modules-tool=Lmod --module-syntax=Lua --check-style \
  "$repo_root/packaging/easybuild/voiage-2.0.0-foss-2023a.eb"

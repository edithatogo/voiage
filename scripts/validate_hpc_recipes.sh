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

if command -v modulecmd >/dev/null 2>&1; then
  eb --check-style "$repo_root/packaging/easybuild/voiage-2.0.0-foss-2023a.eb"
else
  echo "EasyBuild syntax requires a modules tool; install Environment Modules or Lmod." >&2
  exit 2
fi

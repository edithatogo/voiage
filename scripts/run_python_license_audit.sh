#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
repo_python="$repo_root/.venv/bin/python"
"$repo_python" -c 'import sys; assert sys.version_info[:3] == (3, 15, 2), sys.version'
output_path=${1:-$repo_root/.conductor/local/python-runtime-licenses.json}
mkdir -p "$(dirname "$output_path")"
audit_dir=$(mktemp -d)
trap 'rm -rf "$audit_dir"' EXIT

uv export --quiet --frozen --python "$repo_python" --no-dev --no-emit-project --format requirements-txt \
  --output-file "$audit_dir/runtime-requirements.txt"
uv venv --quiet --python "$repo_python" "$audit_dir/venv"
"$audit_dir/venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 15, 2), sys.version'
uv pip install --quiet --python "$audit_dir/venv" \
  -r "$audit_dir/runtime-requirements.txt"
uv run --no-project --python "$audit_dir/venv/bin/python" \
  --with pip-licenses==5.5.0 \
  pip-licenses --format=json --with-urls \
  --ignore-packages pip-licenses prettytable wcwidth > "$output_path"
"$repo_python" "$repo_root/scripts/check_python_licenses.py" "$output_path"

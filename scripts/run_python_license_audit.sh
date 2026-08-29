#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
output_path=${1:-$repo_root/.conductor/local/python-runtime-licenses.json}
audit_dir=$(mktemp -d)
trap 'rm -rf "$audit_dir"' EXIT

uv export --quiet --frozen --no-dev --no-emit-project --format requirements-txt \
  --output-file "$audit_dir/runtime-requirements.txt"
uv venv --quiet --python 3.12 "$audit_dir/venv"
uv pip install --quiet --python "$audit_dir/venv" \
  -r "$audit_dir/runtime-requirements.txt"
uv run --no-project --python "$audit_dir/venv/bin/python" \
  --with pip-licenses==5.5.0 \
  pip-licenses --format=json --with-urls \
  --ignore-packages pip-licenses prettytable wcwidth > "$output_path"
uv run --frozen --no-sync python scripts/check_python_licenses.py "$output_path"

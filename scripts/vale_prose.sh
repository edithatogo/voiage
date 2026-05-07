#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_bin="$(mktemp -d)"

cleanup() {
    rm -rf "$tmp_bin"
}

trap cleanup EXIT

if command -v rst2html >/dev/null 2>&1; then
    :
elif command -v rst2html.py >/dev/null 2>&1; then
    ln -s "$(command -v rst2html.py)" "$tmp_bin/rst2html"
    export PATH="$tmp_bin:$PATH"
else
    echo "rst2html or rst2html.py is required for Vale RST linting." >&2
    exit 1
fi

cd "$repo_root"
find README.md *.md docs conductor specs -type f \( -name '*.md' -o -name '*.rst' \) -print0 \
    | xargs -0 vale --minAlertLevel=warning "$@"

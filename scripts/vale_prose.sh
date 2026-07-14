#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
find README.md *.md docs conductor specs -type f \( -name '*.md' -o -name '*.mdx' \) -print0 \
    | xargs -0 vale --minAlertLevel=warning "$@"

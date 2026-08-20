#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

R_LIBS_USER="$work_dir/r-library" \
PKGCHECK_CACHE_DIR="$work_dir/pkgcheck-cache" \
  Rscript -e '
    options(repos = c(
      ropenscireviewtools = "https://ropensci-review-tools.r-universe.dev",
      CRAN = "https://cloud.r-project.org"
    ))
    dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
    install.packages("pkgcheck", lib = Sys.getenv("R_LIBS_USER"))
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) < 1L) stop("Usage: script requires path to R package directory")
    checks <- pkgcheck::pkgcheck(args[[1]], use_cache = FALSE)
    print(summary(checks))
  ' "$repo_root/r-package/voiageR"

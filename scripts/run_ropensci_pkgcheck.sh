#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
pkgcheck_revision="af25295aed5fbb0229c20a8f68e91ed9e53e8d19"
srr_revision="d186fe6f93657805ed86177f03333c478e136709"
pkgstats_r43_revision="2679f1e899a9e3777eaa9a9ac5566a2aeafc11d9"
pkgstats_r43_sha256="e0679a1bf759d637dc779108818265c681bdbec77542433534ece0e3a1fec977"
# The dollar expressions in the following single-quoted program belong to R.
# shellcheck disable=SC2016
r_version=$(Rscript -e 'cat(paste(R.version$major, R.version$minor, sep = "."))')
tool_library="${PKGCHECK_TOOL_LIBRARY:-$repo_root/.cache/ropensci-tools/r-$r_version-$pkgcheck_revision}"

# pkgcheck queries public GitHub Actions results only when a token is present.
# Reuse an authenticated gh session when callers have not already supplied a
# token, without printing or persisting the credential.
if [[ -z "${GITHUB_TOKEN:-}" && -z "${GH_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
  GITHUB_TOKEN=$(gh auth token 2>/dev/null || true)
  export GITHUB_TOKEN
fi

(
  cd "$repo_root"
  git diff --quiet -- r-package/voiageR
  git diff --cached --quiet -- r-package/voiageR
)

source_archive="$work_dir/voiageR_2.2.0.tar.gz"
pkgcheck_repo="$work_dir/voiageR-repository"
(
  cd "$work_dir"
  R CMD build "$repo_root/r-package/voiageR"
)
test -f "$source_archive"
shasum -a 256 "$source_archive"

# Present the R subpackage as a clean repository root so repository-level
# contribution and CI evidence remains visible to pkgcheck without allowing
# ignored local build products to contaminate the package scan.
mkdir -p "$pkgcheck_repo/.github/workflows"
git -C "$repo_root" archive HEAD:r-package/voiageR | tar -x -C "$pkgcheck_repo"
cp "$repo_root/CONTRIBUTING.md" "$repo_root/CODE_OF_CONDUCT.md" "$pkgcheck_repo/"
cp "$repo_root/.github/workflows/bindings-ci.yml" \
  "$pkgcheck_repo/.github/workflows/"

mkdir -p "$tool_library" "$work_dir/pkgcheck-cache"
tools_are_complete=false
if test -f "$tool_library/.voiage-pkgcheck-tools-complete" && \
  R_LIBS_USER="$tool_library" Rscript -e \
    'stopifnot(requireNamespace("pkgcheck", quietly = TRUE), requireNamespace("pkgstats", quietly = TRUE), requireNamespace("srr", quietly = TRUE))'; then
  tools_are_complete=true
fi
if ! "$tools_are_complete"; then
  R_LIBS_USER="$tool_library" \
  VOIAGE_TOOL_LIBRARY="$tool_library" \
    Rscript -e '
      options(repos = c(
        ropenscireviewtools = "https://ropensci-review-tools.r-universe.dev",
        CRAN = "https://cloud.r-project.org"
      ))
      library_path <- Sys.getenv("VOIAGE_TOOL_LIBRARY")
      install.packages("pkgcheck", lib = library_path)
    '

  if Rscript -e 'quit(status = as.integer(getRversion() >= "4.5.0"))'; then
    pkgstats_archive="$work_dir/pkgstats-$pkgstats_r43_revision.tar.gz"
    pkgstats_source="$work_dir/pkgstats-$pkgstats_r43_revision"
    curl -fsSL \
      "https://github.com/ropensci-review-tools/pkgstats/archive/$pkgstats_r43_revision.tar.gz" \
      -o "$pkgstats_archive"
    printf '%s  %s\n' "$pkgstats_r43_sha256" "$pkgstats_archive" | shasum -a 256 -c -
    tar -xzf "$pkgstats_archive" -C "$work_dir"
    # The dollar expression below is literal R source.
    # shellcheck disable=SC2016
    grep -F 'grepv ("^R\\-", basename (s$translations))' \
      "$pkgstats_source/R/pkgstats-summary.R"
    perl -pi -e \
      'if (/tr <- grepv/) { $_ = "    tr <- grep (\"^R\\\\-\", basename (s\$translations), invert = TRUE, value = TRUE)\n"; }' \
      "$pkgstats_source/R/pkgstats-summary.R"
    # The dollar expression below is literal R source.
    # shellcheck disable=SC2016
    grep -F 'grep ("^R\\-", basename (s$translations), invert = TRUE, value = TRUE)' \
      "$pkgstats_source/R/pkgstats-summary.R"
    R_LIBS_USER="$tool_library" R CMD INSTALL --library="$tool_library" \
      "$pkgstats_source"
  fi
  R_LIBS_USER="$tool_library" \
  VOIAGE_TOOL_LIBRARY="$tool_library" \
  VOIAGE_SRR_REVISION="$srr_revision" \
    Rscript -e '
      remotes::install_github(
        paste0("ropensci-review-tools/srr@", Sys.getenv("VOIAGE_SRR_REVISION")),
        lib = Sys.getenv("VOIAGE_TOOL_LIBRARY"),
        dependencies = FALSE,
        upgrade = "never"
      )
    '
  R_LIBS_USER="$tool_library" \
  VOIAGE_TOOL_LIBRARY="$tool_library" \
  VOIAGE_PKGCHECK_REVISION="$pkgcheck_revision" \
    Rscript -e '
      remotes::install_github(
        paste0(
          "ropensci-review-tools/pkgcheck@",
          Sys.getenv("VOIAGE_PKGCHECK_REVISION")
        ),
        lib = Sys.getenv("VOIAGE_TOOL_LIBRARY"),
        dependencies = FALSE,
        upgrade = "never"
      )
    '
  R_LIBS_USER="$tool_library" Rscript -e \
    'stopifnot(requireNamespace("pkgcheck"), requireNamespace("pkgstats"), requireNamespace("srr"))'
  touch "$tool_library/.voiage-pkgcheck-tools-complete"
fi

R_LIBS_USER="$tool_library" \
PKGCHECK_CACHE_DIR="$work_dir/pkgcheck-cache" \
  Rscript -e '
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) < 2L) {
      stop("Usage: script requires package directory and exact source archive")
    }
    if (!file.exists(args[[2]])) stop("Exact source archive does not exist")
    checks <- pkgcheck::pkgcheck(args[[1]], use_cache = FALSE)
    summary_lines <- pkgcheck:::summarise_all_checks(checks)
    print(summary(checks))
    if (!isTRUE(attr(summary_lines, "checks_okay"))) quit(status = 1L)
  ' "$pkgcheck_repo" "$source_archive"

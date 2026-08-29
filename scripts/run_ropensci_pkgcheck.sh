#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
pkgcheck_revision="af25295"
srr_revision="d186fe6f93657805ed86177f03333c478e136709"
pkgstats_r43_revision="f25191d8a15dacb42daeab5ebc89afc92efcfdbf"
# shellcheck disable=SC2016 -- the dollar expressions belong to R.
r_version=$(Rscript -e 'cat(paste(R.version$major, R.version$minor, sep = "."))')
tool_library="${PKGCHECK_TOOL_LIBRARY:-$repo_root/.cache/ropensci-tools/r-$r_version-$pkgcheck_revision}"

(
  cd "$repo_root"
  git diff --quiet -- r-package/voiageR
  git diff --cached --quiet -- r-package/voiageR
)

source_archive="$work_dir/voiageR_2.1.0.tar.gz"
(
  cd "$work_dir"
  R CMD build "$repo_root/r-package/voiageR"
)
test -f "$source_archive"
shasum -a 256 "$source_archive"

mkdir -p "$tool_library" "$work_dir/pkgcheck-cache"
if ! test -f "$tool_library/.voiage-pkgcheck-tools-complete"; then
  R_LIBS_USER="$tool_library" \
  VOIAGE_PKGCHECK_REVISION="$pkgcheck_revision" \
  VOIAGE_SRR_REVISION="$srr_revision" \
  VOIAGE_PKGSTATS_R43_REVISION="$pkgstats_r43_revision" \
    Rscript -e '
      options(repos = c(
        ropenscireviewtools = "https://ropensci-review-tools.r-universe.dev",
        CRAN = "https://cloud.r-project.org"
      ))
      library_path <- Sys.getenv("R_LIBS_USER")
      install.packages("pkgcheck", lib = library_path)
      if (getRversion() < "4.5.0") {
        remotes::install_github(
          paste0(
            "ropensci-review-tools/pkgstats@",
            Sys.getenv("VOIAGE_PKGSTATS_R43_REVISION")
          ),
          lib = library_path,
          dependencies = FALSE,
          upgrade = "never",
          quiet = TRUE
        )
      }
      remotes::install_github(
        paste0("ropensci-review-tools/srr@", Sys.getenv("VOIAGE_SRR_REVISION")),
        lib = library_path,
        dependencies = FALSE,
        upgrade = "never",
        quiet = TRUE
      )
      remotes::install_github(
        paste0(
          "ropensci-review-tools/pkgcheck@",
          Sys.getenv("VOIAGE_PKGCHECK_REVISION")
        ),
        lib = library_path,
        dependencies = FALSE,
        upgrade = "never",
        quiet = TRUE
      )
    '
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
  ' "$repo_root/r-package/voiageR" "$source_archive"

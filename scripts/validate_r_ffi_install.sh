#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

cargo build --manifest-path "$repo_root/rust/Cargo.toml" --release --locked --package voiage-ffi
mkdir -p "$work_dir/r-library"
R CMD INSTALL --library="$work_dir/r-library" "$repo_root/r-package/voiageR"
case "$(uname -s)" in
  Darwin) ffi_library="$repo_root/rust/target/release/libvoiage_ffi.dylib" ;;
  Linux) ffi_library="$repo_root/rust/target/release/libvoiage_ffi.so" ;;
  *) echo "Unsupported native-library platform: $(uname -s)" >&2; exit 2 ;;
esac
VOIAGE_FFI_LIBRARY="$ffi_library" \
R_LIBS_USER="$work_dir/r-library" \
  Rscript -e '
    value <- voiageR::evpi(matrix(c(0, 1, 1, 0), ncol = 2, byrow = TRUE))
    stopifnot(is.finite(value), identical(value, 0.5))
  '

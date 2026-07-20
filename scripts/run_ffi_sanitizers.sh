#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: the ABI sanitizer gate requires Linux" >&2
  exit 2
fi

for command in cargo clang nm rustup; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "error: required sanitizer tool is unavailable: ${command}" >&2
    exit 2
  fi
done

toolchain="${VOIAGE_SANITIZER_TOOLCHAIN:-nightly}"
target="${VOIAGE_SANITIZER_TARGET:-x86_64-unknown-linux-gnu}"
if ! rustup run "${toolchain}" rustc --version >/dev/null 2>&1; then
  echo "error: Rust toolchain '${toolchain}' is not installed" >&2
  exit 2
fi
if ! rustup component list --toolchain "${toolchain}" --installed | grep -qx 'rust-src'; then
  echo "error: rust-src is required for instrumented standard-library builds" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rust_root="${root}/rust"
target_dir="${RUNNER_TEMP:-${rust_root}/target}/voiage-ffi-sanitizers"
library_dir="${target_dir}/${target}/debug"
library="${library_dir}/libvoiage_ffi.so"
consumer="${target_dir}/voiage_v1_sanitizer_smoke"

export CARGO_TARGET_DIR="${target_dir}"
export RUSTFLAGS="-Zsanitizer=address -Cforce-frame-pointers=yes"

cargo "+${toolchain}" build \
  --manifest-path "${rust_root}/Cargo.toml" \
  --locked \
  --package voiage-ffi \
  --target "${target}" \
  -Zbuild-std

if [[ ! -f "${library}" ]]; then
  echo "error: instrumented ABI library was not produced: ${library}" >&2
  exit 1
fi
if ! nm -D "${library}" | grep -q '__asan_'; then
  echo "error: Rust ABI library does not contain ASan instrumentation" >&2
  exit 1
fi

clang \
  -std=c11 \
  -Wall -Wextra -Werror -pedantic \
  -fno-omit-frame-pointer \
  -fsanitize=address,undefined \
  -I"${rust_root}/crates/voiage-ffi/include" \
  "${root}/tests/ffi/voiage_v1_sanitizer_smoke.c" \
  -L"${library_dir}" -lvoiage_ffi \
  -Wl,-rpath,"${library_dir}" \
  -o "${consumer}"

ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1:strict_string_checks=1" \
LSAN_OPTIONS="exitcode=23:report_objects=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
  "${consumer}"

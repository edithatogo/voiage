## R CMD check results

The submission was checked from a fresh source bundle with:

```text
R CMD build --compact-vignettes=gs+qpdf voiageR
R CMD check --as-cran --run-donttest --timings voiageR_2.0.0.tar.gz
```

The result was 0 errors | 0 warnings | 2 notes.

* This is a New submission.
* The check reported `unable to verify current time`. This is an environmental
  clock-verification note from the local runner.

The package also builds and checks successfully on r-universe for R release and
R-devel on Linux, R old-release and release on macOS, R old-release, release and
devel on Windows, and R release for WebAssembly.

## External software

The direct EVPI interface uses the separately distributed Apache-2.0
`voiage-ffi` shared library documented in `SystemRequirements`. EVPPI and EVSI
use the optional `reticulate` bridge to the published Python `voiage` package.
Package installation, loading, documentation, examples and tests remain
available when those optional runtimes are absent.

## Reverse dependencies

There are no known reverse dependencies.

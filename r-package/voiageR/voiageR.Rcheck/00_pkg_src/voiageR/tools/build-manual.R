#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
pkg_dir <- normalizePath(
  if (length(args) >= 1 && nzchar(args[[1]])) args[[1]] else ".",
  winslash = "/",
  mustWork = TRUE
)
output_path <- if (length(args) >= 2 && nzchar(args[[2]])) {
  args[[2]]
} else {
  file.path(pkg_dir, "voiageR-manual.pdf")
}

output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

status <- system2(
  "R",
  c(
    "CMD",
    "Rd2pdf",
    "--batch",
    "--no-preview",
    paste0("--output=", output_path),
    pkg_dir
  )
)

if (status != 0) {
  quit(status = status)
}

message(output_path)

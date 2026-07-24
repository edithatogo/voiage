test_that("voiageR help and package index are available", {
  rd_db <- source_rd_db()
  description <- package_description_record()

  expect_true("voiageR.Rd" %in% names(rd_db))
  expect_identical(unname(description[["Package"]]), "voiageR")
})

test_that("the Rd surface matches the exported package API", {
  rd_db <- source_rd_db()
  rd_topics <- sort(sub("\\.Rd$", "", names(rd_db)))

  expect_setequal(
    rd_topics,
    c(
      "evpi",
      "evppi",
      "evsi",
      "init_voiage",
      "is_voiage_available",
      "set_voiage_env",
      "voiageR"
    )
  )
})

test_that("exported Rd examples can be extracted consistently", {
  rd_db <- source_rd_db()
  topics <- setdiff(names(rd_db), "voiageR.Rd")

  for (topic in topics) {
    out <- tempfile(fileext = ".R")
    tools::Rd2ex(rd_db[[topic]], out = out)
    lines <- readLines(out, warn = FALSE)

    expect_gt(length(lines), 0)
    expect_true(any(nzchar(lines)))
  }
})

test_that("the manual build helper assembles the Rd2pdf invocation", {
  script_path <- testthat::test_path("..", "..", "tools", "build-manual.R")
  skip_if_not(file.exists(script_path))

  pkg_dir <- normalizePath(testthat::test_path("..", ".."), winslash = "/", mustWork = TRUE)
  output_path <- file.path(tempdir(), "voiageR-manual-smoke", "voiageR-manual.pdf")
  output_dir <- dirname(output_path)

  if (dir.exists(output_dir)) {
    unlink(output_dir, recursive = TRUE)
  }

  manual_env <- new.env(parent = baseenv())
  manual_env$commandArgs <- function(trailingOnly = TRUE) {
    c(pkg_dir, output_path)
  }
  manual_env$system2 <- function(cmd, args, ...) {
    manual_env$call <- list(cmd = cmd, args = args)
    0L
  }
  manual_env$message <- function(...) {
    manual_env$message_value <- paste(..., collapse = " ")
  }

  source(script_path, local = manual_env)

  expect_true(dir.exists(output_dir))
  expect_identical(manual_env$call$cmd, "R")
  expect_identical(
    manual_env$call$args,
    c(
      "CMD",
      "Rd2pdf",
      "--batch",
      "--no-preview",
      paste0("--output=", output_path),
      pkg_dir
    )
  )
  expect_identical(manual_env$message_value, output_path)
})

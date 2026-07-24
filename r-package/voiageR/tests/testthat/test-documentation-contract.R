test_that("the R package documentation surface includes the package help topic", {
  rd_db <- source_rd_db()
  expected_rd_files <- c(
    "voiageR.Rd",
    "evpi.Rd",
    "evppi.Rd",
    "evsi.Rd",
    "init_voiage.Rd",
    "is_voiage_available.Rd",
    "set_voiage_env.Rd"
  )

  expect_true(all(expected_rd_files %in% names(rd_db)))
})

test_that("the R package documentation policy is explicit in metadata", {
  description <- package_description_record()
  suggests <- unlist(strsplit(description[["Suggests"]], ",\\s*"))

  expect_identical(description[["VignetteBuilder"]], "knitr")
  expect_true(any(grepl("^knitr", suggests)))
  expect_true(any(grepl("^rmarkdown", suggests)))
})

test_that("EVSI documentation states the corrected R facade boundary", {
  package_root <- source_package_root()
  skip_if(is.null(package_root), "source documentation is unavailable")
  documentation_paths <- c(
    file.path(package_root, "README.md"),
    file.path(package_root, "man", "evsi.Rd"),
    file.path(package_root, "vignettes", "voiageR-getting-started.Rmd"),
    file.path(package_root, "vignettes", "voiageR-intro.Rmd")
  )
  documentation <- vapply(
    documentation_paths,
    function(path) paste(readLines(path, warn = FALSE), collapse = "\n"),
    character(1)
  )

  expect_true(all(grepl("mean_<normalised arm>", documentation, fixed = TRUE)))
  expect_true(all(grepl("sd_outcome", documentation, fixed = TRUE)))
  expect_true(all(grepl("Python-only", documentation, fixed = TRUE)))
  expect_false(any(grepl("n_simulations", documentation, fixed = TRUE)))
})

test_that("documented two-loop examples use the built-in study contract", {
  package_root <- source_package_root()
  skip_if(is.null(package_root), "source documentation is unavailable")
  paths <- c(
    file.path(package_root, "README.md"),
    file.path(package_root, "man", "evsi.Rd"),
    file.path(package_root, "vignettes", "voiageR-getting-started.Rmd"),
    file.path(package_root, "vignettes", "voiageR-intro.Rmd")
  )
  documentation <- paste(
    unlist(lapply(paths, readLines, warn = FALSE)),
    collapse = "\n"
  )

  expect_match(documentation, "mean_treatment")
  expect_match(documentation, "mean_control")
  expect_match(documentation, "sd_outcome = rep\\(1\\.0, draws\\)")
  expect_match(documentation, 'method = "two_loop"')
  expect_match(documentation, "seed = 20260724L")
})

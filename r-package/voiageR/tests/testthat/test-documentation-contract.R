test_that("the R package documentation surface includes the package help topic", {
  rd_db <- tools::Rd_db("voiageR")
  expected_rd_files <- c(
    "voiageR.Rd",
    "evpi.Rd",
    "evppi.Rd",
    "evsi.Rd",
    "init_voiage.Rd",
    "is_voiage_available.Rd",
    "normalize_decision_problem.Rd",
    "normalize_statistical_assurance.Rd",
    "read_voiage_arrow.Rd",
    "set_voiage_env.Rd"
  )

  expect_true(all(expected_rd_files %in% names(rd_db)))
})

test_that("the R package documentation policy is explicit in metadata", {
  description <- packageDescription("voiageR")
  suggests <- unlist(strsplit(description[["Suggests"]], ",\\s*"))

  expect_identical(description[["VignetteBuilder"]], "knitr")
  expect_true(any(grepl("^knitr", suggests)))
  expect_true(any(grepl("^rmarkdown", suggests)))
})

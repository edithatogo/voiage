test_that("the package-level documentation topic is present", {
  rd_db <- tools::Rd_db("voiageR")
  rd_files <- names(rd_db)
  expect_true("voiageR.Rd" %in% rd_files)
  expect_true("evpi.Rd" %in% rd_files)
  expect_true("evppi.Rd" %in% rd_files)
  expect_true("evsi.Rd" %in% rd_files)

  exported <- getNamespaceExports("voiageR")
  expect_setequal(
    exported,
    c(
      "evpi",
      "evppi",
      "evsi",
      "init_voiage",
      "is_voiage_available",
      "normalize_decision_problem",
      "normalize_statistical_assurance",
      "read_voiage_arrow",
      "set_voiage_env"
    )
  )
})

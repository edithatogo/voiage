test_that("the R release workflow is tied to r-v tags and source artifacts", {
  workflow_path <- testthat::test_path("..", "..", "..", "..", ".github", "workflows", "bindings-release.yml")
  workflow_text <- readLines(workflow_path, warn = FALSE)

  expect_true(any(grepl("r-v\\*", workflow_text)))
  expect_true(any(grepl("Verify DESCRIPTION version matches release tag", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("R CMD build", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("tools/build-manual.R", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("softprops/action-gh-release@v2", workflow_text, fixed = TRUE)))
})

test_that("the R submission checklist records the bridge role and external registries", {
  checklist_path <- testthat::test_path("..", "..", "..", "..", "docs", "release", "binding-submission-checklist.md")
  checklist_text <- readLines(checklist_path, warn = FALSE)

  expect_true(any(grepl("The R package remains the thin reticulate bridge to the Python façade", checklist_text, fixed = TRUE)))
  expect_true(any(grepl("CRAN submission remains external/manual", checklist_text, fixed = TRUE)))
  expect_true(any(grepl("r-universe indexing remains external/manual", checklist_text, fixed = TRUE)))
})

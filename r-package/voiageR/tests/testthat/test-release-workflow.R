repo_file <- function(...) {
  candidates <- c(
    Sys.getenv("VOIAGE_REPO_ROOT", unset = NA_character_),
    Sys.getenv("GITHUB_WORKSPACE", unset = NA_character_),
    testthat::test_path("..", "..", "..", "..")
  )
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  paths <- file.path(candidates, ...)
  existing <- paths[file.exists(paths)]
  if (length(existing) == 0) {
    testthat::skip(paste("Repository file not available:", file.path(...)))
  }
  existing[[1]]
}

test_that("the R release workflow is tied to r-v tags and source artifacts", {
  workflow_path <- repo_file(".github", "workflows", "bindings-release.yml")
  workflow_text <- readLines(workflow_path, warn = FALSE)

  expect_true(any(grepl("r-v\\*", workflow_text)))
  expect_true(any(grepl("Verify DESCRIPTION version matches release tag", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("R CMD build", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("tools/build-manual.R", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("r-lib/actions/setup-tinytex@v2", workflow_text, fixed = TRUE)))
  expect_true(any(grepl("softprops/action-gh-release@v2", workflow_text, fixed = TRUE)))
})

test_that("the R submission checklist records the bridge role and external registries", {
  checklist_path <- repo_file("docs", "release", "binding-submission-checklist.md")
  checklist_text <- readLines(checklist_path, warn = FALSE)

  expect_true(any(grepl("The R package remains the thin reticulate bridge", checklist_text, fixed = TRUE)))
  expect_true(any(grepl("shared contract", checklist_text, fixed = TRUE)))
  expect_true(any(grepl("CRAN submission remains external/manual", checklist_text, fixed = TRUE)))
  expect_true(any(grepl("r-universe indexing remains external/manual", checklist_text, fixed = TRUE)))
})

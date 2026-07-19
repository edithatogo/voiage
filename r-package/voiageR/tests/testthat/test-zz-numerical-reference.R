find_numerical_reference <- function() {
  directory <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  repeat {
    candidate <- file.path(
      directory,
      "specs",
      "numerical-reference",
      "v1",
      "evpi-cases.json"
    )
    if (file.exists(candidate)) {
      return(candidate)
    }
    parent <- dirname(directory)
    if (identical(parent, directory)) {
      stop("Could not locate the shared EVPI reference fixture.")
    }
    directory <- parent
  }
}

test_that("R binding consumes the shared EVPI numerical reference", {
  json <- reticulate::import("json", convert = TRUE)
  reference <- json$loads(
    paste(readLines(find_numerical_reference(), warn = FALSE), collapse = "\n")
  )
  expect_identical(reference$schema_version, "1.0.0")
  expect_identical(reference$method, "evpi")

  cache <- get(".voiage_cache", envir = asNamespace("voiageR"))
  old_module <- cache$module
  on.exit(cache$module <- old_module, add = TRUE)
  cache$module <- list(
    evpi = function(
      net_benefits,
      population = NULL,
      time_horizon = NULL,
      discount_rate = NULL
    ) {
      matrix <- as.matrix(net_benefits)
      mean(apply(matrix, 1, max)) - max(colMeans(matrix))
    }
  )

  for (fixture_case in reference$cases) {
    matrix <- do.call(rbind, fixture_case$net_benefits)
    expect_equal(
      evpi(matrix),
      fixture_case$expected$value,
      tolerance = fixture_case$expected$atol,
      info = fixture_case$id
    )
  }
})

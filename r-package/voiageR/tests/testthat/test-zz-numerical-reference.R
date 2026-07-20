find_numerical_reference <- function() {
  configured <- Sys.getenv("VOIAGE_NUMERICAL_REFERENCE", unset = "")
  if (nzchar(configured)) {
    return(normalizePath(configured, winslash = "/", mustWork = TRUE))
  }
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
    rows <- lapply(
      fixture_case$net_benefits,
      function(row) as.numeric(unlist(row, use.names = FALSE))
    )
    matrix <- do.call(rbind, rows)
    actual <- as.numeric(evpi(matrix))
    expected <- as.numeric(unlist(
      fixture_case$expected$value,
      use.names = FALSE
    ))
    absolute_tolerance <- as.numeric(unlist(
      fixture_case$expected$atol,
      use.names = FALSE
    ))
    relative_tolerance <- if (is.null(fixture_case$expected$rtol)) {
      0
    } else {
      as.numeric(unlist(fixture_case$expected$rtol, use.names = FALSE))
    }
    expect_length(actual, 1)
    expect_length(expected, 1)
    expect_length(absolute_tolerance, 1)
    expect_length(relative_tolerance, 1)
    expect_true(
      all(is.finite(c(actual, expected, absolute_tolerance, relative_tolerance))),
      info = fixture_case$id
    )
    expect_true(absolute_tolerance >= 0, info = fixture_case$id)
    expect_true(relative_tolerance >= 0, info = fixture_case$id)
    permitted_error <- absolute_tolerance + relative_tolerance * abs(expected)
    expect_true(
      abs(actual - expected) <= permitted_error,
      info = sprintf(
        "%s: actual %.17g, expected %.17g, permitted error %.17g",
        fixture_case$id,
        actual,
        expected,
        permitted_error
      )
    )
  }
})

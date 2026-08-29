test_that("installed package calls its packaged Rust EVPI symbol", {
  cache <- get(".voiage_cache", envir = asNamespace("voiageR"))
  old_module <- cache$module
  on.exit(cache$module <- old_module, add = TRUE)
  cache$module <- NULL

  net_benefits <- matrix(
    c(
      0, 1,
      2, 0
    ),
    nrow = 2,
    byrow = TRUE
  )

  expect_equal(evpi(net_benefits), 0.5, tolerance = 1e-12)
})

test_that("installed package calls its packaged Rust ENBS symbol", {
  cache <- get(".voiage_cache", envir = asNamespace("voiageR"))
  old_module <- cache$module
  on.exit(cache$module <- old_module, add = TRUE)
  cache$module <- NULL

  expect_equal(enbs(12.5, 3.0), 9.5, tolerance = 1e-12)
  expect_equal(enbs(2.0, 3.0), -1.0, tolerance = 1e-12)
  expect_error(enbs(-1.0, -1.0))
})

test_that("voiageR package loads", {
  expect_true(TRUE)
})

test_that("is_voiage_available returns logical", {
  result <- is_voiage_available()
  expect_type(result, "logical")
})

test_that("init_voiage works when package is available", {
  skip_if_not(is_voiage_available(), "voiage Python package not available")
  
  expect_silent(init_voiage())
  expect_false(is.null(.voiage))
})

test_that("set_voiage_env works without error", {
  # We can't actually test this without a real environment, but we can check it doesn't error
  expect_silent(set_voiage_env("test_env", type = "virtualenv"))
})
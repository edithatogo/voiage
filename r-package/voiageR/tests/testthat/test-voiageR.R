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

test_that("set_voiage_env validates environment type", {
  expect_error(set_voiage_env("test_env", type = "invalid"))
})

test_that("installed package calls the separately loaded Rust EVPI symbol", {
  library_path <- Sys.getenv("VOIAGE_FFI_LIBRARY", unset = "")
  skip_if(
    !nzchar(library_path),
    "VOIAGE_FFI_LIBRARY is required for the native installed-package test"
  )
  skip_if_not(file.exists(library_path), "configured voiage-ffi library does not exist")

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

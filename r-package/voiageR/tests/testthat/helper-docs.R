source_package_root <- function() {
  candidate <- normalizePath(
    testthat::test_path("..", ".."),
    winslash = "/",
    mustWork = FALSE
  )
  if (file.exists(file.path(candidate, "DESCRIPTION"))) {
    return(candidate)
  }
  NULL
}

source_rd_db <- function() {
  package_root <- source_package_root()
  if (is.null(package_root)) {
    return(tools::Rd_db("voiageR"))
  }
  paths <- list.files(
    file.path(package_root, "man"),
    pattern = "\\.Rd$",
    full.names = TRUE
  )
  stats::setNames(lapply(paths, tools::parse_Rd), basename(paths))
}

package_description_record <- function() {
  package_root <- source_package_root()
  if (is.null(package_root)) {
    return(packageDescription("voiageR"))
  }
  read.dcf(file.path(package_root, "DESCRIPTION"))[1, ]
}

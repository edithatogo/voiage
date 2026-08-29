#' rOpenSci statistical standards for tabular and missing-value preprocessing
#'
#' @srrstats {G2.7} Net benefits accept matrices and data frames.
#' @srrstats {G2.8} Data frames, matrices, and lists are normalised before internal calls.
#' @srrstats {G2.9} Arm-name normalisation rejects collisions rather than losing identity.
#' @srrstats {G2.10} Tabular inputs are converted before column extraction.
#' @srrstats {G2.11} Non-numeric tabular columns fail with an informative validation error.
#' @srrstats {G2.12} List columns fail validation rather than reaching native routines.
#' @srrstats {G2.13} Missing data are rejected during preprocessing.
#' @srrstats {G2.15} Missing data never reach numerical base or native routines.
#' @noRd
NULL

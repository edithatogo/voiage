#' rOpenSci statistical standards implemented by the package and documentation
#'
#' @srrstats {G1.0} The paper and package documentation cite the primary VOI methods.
#' @srrstats {G1.1} The README and paper distinguish prior methods from this implementation.
#' @srrstats {G1.2} CONTRIBUTING.md states the supported lifecycle and maintenance policy.
#' @srrstats {G1.3} Function help and vignettes define VOI terminology and estimands.
#' @srrstats {G1.4} Every exported function has roxygen2-generated help.
#' @srrstats {G1.4a} Every internal helper in voiageR.R has an @noRd documentation block.
#' @srrstats {G2.0} Scalar and vector lengths are checked before computation.
#' @srrstats {G2.0a} Function help documents scalar, matrix, list, and vector lengths.
#' @srrstats {G2.1} Public functions validate input types before native or Python dispatch.
#' @srrstats {G2.1a} Function help documents the accepted data types.
#' @srrstats {G2.2} Scalar controls reject multivalued input.
#' @srrstats {G2.3} Character controls are validated before dispatch.
#' @srrstats {G2.3a} Environment and method choices use match.arg or explicit sets.
#' @srrstats {G2.3b} Case-sensitive controls and arm-name normalisation are documented.
#' @srrstats {G2.4} Inputs are normalised at the public boundary.
#' @srrstats {G2.4a} Positive integral controls are converted with as.integer.
#' @srrstats {G2.4b} Native numerical inputs are converted with as.double.
#' @srrstats {G2.6} One-dimensional sample inputs are normalised independent of class.
#' @srrstats {PD1.0} The documentation cites the empirical VOI expectation definitions.
#' @srrstats {PD3.0} Numerical empirical integration is justified by the VOI definitions.
#' @srrstats {PD3.4} Finite-input and positive-dimension checks define stable integration.
#' @srrstats {PD3.5} EVPI uses the documented empirical Monte Carlo expectation sum.
#' @srrstats {PD3.5a} Finite bounded samples make every empirical expectation finite.
#' @noRd
NULL

#' NA_standards
#'
#' @srrstatsNA {G1.5} The R package makes no package-specific performance claim in a publication.
#' @srrstatsNA {G1.6} No comparative performance claim against another R package is made.
#' @srrstatsNA {G2.4c} The bounded numerical API accepts no character data requiring conversion.
#' @srrstatsNA {G2.4d} The bounded numerical API accepts no factor data requiring conversion.
#' @srrstatsNA {G2.4e} The bounded numerical API accepts no factor data requiring conversion.
#' @srrstatsNA {G2.5} No public parameter accepts ordered or unordered factors.
#' @srrstatsNA {G2.14} Complete joint draws define the estimands, so missing-value policies are invalid.
#' @srrstatsNA {G2.14a} Missing data always produce an error under the documented complete-draw contract.
#' @srrstatsNA {G2.14b} Ignoring missing draws would silently change the joint empirical distribution.
#' @srrstatsNA {G2.14c} Imputation is an upstream modelling decision and is never performed implicitly.
#' @srrstatsNA {G2.16} Undefined values violate the estimand and are rejected rather than removed.
#' @srrstatsNA {G3.1} The R facade does not calculate covariance matrices.
#' @srrstatsNA {G3.1a} No covariance algorithm is exposed by the R facade.
#' @srrstatsNA {G4.0} No exported function writes statistical results to a local file.
#' @srrstatsNA {G5.4c} No correctness fixture is transcribed from a publication output.
#' @srrstatsNA {G5.6} VOI summaries are not parameter estimators, so parameter recovery is undefined.
#' @srrstatsNA {G5.6a} The package has no parameter-estimation recovery tolerance to define.
#' @srrstatsNA {G5.6b} The package has no stochastic parameter-recovery experiment.
#' @srrstatsNA {G5.7} The R package makes no asymptotic algorithm-performance claim.
#' @srrstatsNA {G5.10} All bounded R tests run routinely; there is no separate extended suite.
#' @srrstatsNA {G5.11} The R tests require no downloaded large data assets.
#' @srrstatsNA {G5.11a} No extended-test download exists whose failure needs skip behavior.
#' @srrstatsNA {G5.12} No separate extended-test platform or artifact requirements exist.
#' @srrstatsNA {PD2.0} The API consumes empirical draws and does not represent distribution objects.
#' @srrstatsNA {PD3.1} The non-parametric empirical algorithm has no named parametric distribution.
#' @srrstatsNA {PD3.2} No optimisation routine estimates distributional parameters.
#' @srrstatsNA {PD3.3} No optimisation result object or convergence state is produced.
#' @srrstatsNA {PD4.2} No function selects among named parametric distributions.
#' @srrstatsNA {PD4.3} The empirical expectation has no optimisation or integration controls.
#' @noRd
NULL

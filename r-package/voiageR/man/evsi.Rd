\name{evsi}
\alias{evsi}
\title{Calculate Expected Value of Sample Information}
\usage{
evsi(model_func, prior_samples, trial_design, population = NULL, time_horizon = NULL,
  discount_rate = NULL, method = c("two_loop", "regression", "efficient",
  "moment_based"), n_outer_loops = 100, n_inner_loops = 1000,
  metamodel = "linear", seed = NULL)
}
\arguments{
\item{model_func}{A function that takes parameter samples and returns net benefits.}
\item{prior_samples}{A named list, data frame, or matrix with prior parameter
samples.}
\item{trial_design}{A list containing an \code{arms} list. Each arm must have a
unique \code{name} and a positive integer \code{sample_size}.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
\item{method}{EVSI estimator. \code{two_loop} uses the corrected built-in
joint-normal study model; the other values are compatibility estimators.}
\item{n_outer_loops}{Number of simulated trial data sets for
\code{two_loop}.}
\item{n_inner_loops}{Number of joint posterior draws per simulated trial.}
\item{metamodel}{Metamodel used by compatible approximation methods.}
\item{seed}{Optional non-negative integer seed for reproducible
\code{two_loop} simulation.}
}
\value{The calculated EVSI value.}
\description{
Calculates Expected Value of Sample Information using the voiage Python
library.
}
\details{
The built-in \code{two_loop} method uses a fitted joint
multivariate-normal prior and a known-variance arm-mean likelihood.
\code{prior_samples} must contain one parameter named
\code{mean_<normalised arm>} for each trial arm. Normalisation lowercases the
arm name and replaces spaces with underscores. It must also contain a finite,
strictly positive \code{sd_outcome} whose value is fixed across all prior
draws.

Custom \code{trial_simulator} and \code{posterior_sampler} callbacks are
Python-only. The R facade does not convert or execute those callbacks and does
not claim parity for custom two-loop models.
}
\examples{
\dontrun{
model_func <- function(params) {
  cbind(
    Treatment = 50000 * params$mean_treatment - 3000,
    Control = 50000 * params$mean_control
  )
}
set.seed(17)
draws <- 1000L
prior_samples <- list(
  mean_treatment = rnorm(draws, mean = 0.06, sd = 0.03),
  mean_control = rnorm(draws, mean = 0.00, sd = 0.02),
  sd_outcome = rep(1.0, draws)
)
trial_design <- list(
  arms = list(
    list(name = "Treatment", sample_size = 100L),
    list(name = "Control", sample_size = 100L)
  )
)
evsi(
  model_func,
  prior_samples,
  trial_design,
  method = "two_loop",
  n_outer_loops = 100L,
  n_inner_loops = 1000L,
  seed = 20260724L
)
}
}

\name{evsi}
\alias{evsi}
\title{Calculate Expected Value of Sample Information}
\usage{
evsi(model_func, prior_samples, trial_design, population = NULL, time_horizon = NULL,
  discount_rate = NULL, method = c("two_loop", "regression", "efficient",
  "moment_based"), n_outer_loops = 100, n_inner_loops = 1000, metamodel = "linear")
}
\arguments{
\item{model_func}{A function that takes parameter samples and returns net benefits.}
\item{prior_samples}{A named list or data frame with prior parameter samples.}
\item{trial_design}{A list of trial arm specifications.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
\item{method}{EVSI approximation method.}
\item{n_outer_loops}{Number of outer Monte Carlo loops.}
\item{n_inner_loops}{Number of inner Monte Carlo loops.}
\item{metamodel}{Metamodel used for the efficient approximation.}
}
\value{The calculated EVSI value.}
\description{
Calculates Expected Value of Sample Information using the voiage Python
library.
}
\examples{
\dontrun{
model_func <- function(params) {
  cbind(params$param1, params$param2)
}
prior_samples <- list(param1 = rnorm(1000), param2 = rnorm(1000))
trial_design <- list(
  arms = list(
    list(name = "Treatment", sample_size = 50),
    list(name = "Control", sample_size = 50)
  )
)
evsi(model_func, prior_samples, trial_design, method = "efficient")
}
}

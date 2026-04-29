\name{evsi}
\alias{evsi}
\title{Calculate Expected Value of Sample Information}
\usage{
evsi(model_func, prior_samples, trial_design, population = NULL,
  time_horizon = NULL, discount_rate = NULL, n_simulations = 1000, ...)
}
\arguments{
\item{model_func}{A function that takes parameter samples and returns net benefits.}
\item{prior_samples}{A named list or data frame with prior parameter samples.}
\item{trial_design}{A list describing the trial design.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
\item{n_simulations}{Number of simulations to run.}
\item{...}{Additional arguments passed to the Python voiage implementation.}
}
\value{The calculated EVSI value.}
\description{
Calculates Expected Value of Sample Information using the voiage Python library.
}

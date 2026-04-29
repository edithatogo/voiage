\name{evppi}
\alias{evppi}
\title{Calculate Expected Value of Partial Perfect Information}
\usage{
evppi(net_benefits, parameter_samples, population = NULL, time_horizon = NULL,
  discount_rate = NULL, ...)
}
\arguments{
\item{net_benefits}{A matrix or data frame with PSA samples in rows and strategies in columns.}
\item{parameter_samples}{A named list or data frame with parameter samples.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
\item{...}{Additional arguments passed to the Python voiage implementation.}
}
\value{The calculated EVPPI value.}
\description{
Calculates Expected Value of Partial Perfect Information using the voiage Python library.
}

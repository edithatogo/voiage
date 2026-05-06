\name{evppi}
\alias{evppi}
\title{Calculate Expected Value of Partial Perfect Information}
\usage{
evppi(net_benefits, parameter_samples,
  parameters_of_interest = NULL, population = NULL, time_horizon = NULL,
  discount_rate = NULL)
}
\arguments{
\item{net_benefits}{A matrix or data frame with PSA samples in rows and strategies in columns.}
\item{parameter_samples}{A named list or data frame with parameter samples.}
\item{parameters_of_interest}{Optional character vector of parameter names to retain. Defaults to all named parameters in \code{parameter_samples}.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
}
\value{The calculated EVPPI value.}
\description{
Calculates Expected Value of Partial Perfect Information using the voiage
Python library.
}
\examples{
\dontrun{
net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)
param_samples <- list(effect = rnorm(1000), cost = rnorm(1000))
evppi(net_benefits, param_samples)
}
}

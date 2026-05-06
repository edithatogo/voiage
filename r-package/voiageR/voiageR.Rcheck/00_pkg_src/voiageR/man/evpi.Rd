\name{evpi}
\alias{evpi}
\title{Calculate Expected Value of Perfect Information}
\usage{
evpi(net_benefits, population = NULL, time_horizon = NULL,
  discount_rate = NULL)
}
\arguments{
\item{net_benefits}{A matrix or data frame with PSA samples in rows and strategies in columns.}
\item{population}{Optional population size for scaling.}
\item{time_horizon}{Optional time horizon for scaling.}
\item{discount_rate}{Optional discount rate for scaling.}
}
\value{The calculated EVPI value.}
\description{
Calculates Expected Value of Perfect Information using the voiage Python
library.
}
\examples{
\dontrun{
net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)
evpi(net_benefits)
}
}

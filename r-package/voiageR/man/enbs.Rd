\name{enbs}
\alias{enbs}
\title{Calculate Expected Net Benefit of Sampling}
\usage{
enbs(evsi_result, research_cost, population = NULL, time_horizon = NULL,
  discount_rate = NULL)
}
\arguments{
\item{evsi_result}{A finite numeric value representing the expected value of sample information.}
\item{research_cost}{A non-negative numeric value representing the total cost of the proposed research study.}
\item{population}{Optional population size for scaling the EVSI result.}
\item{time_horizon}{Optional time horizon for scaling the EVSI result.}
\item{discount_rate}{Optional discount rate for scaling the EVSI result.}
}
\value{The calculated signed ENBS value.}
\description{
Calculates Expected Net Benefit of Sampling as \code{EVSI - research_cost},
optionally scaling EVSI to a population before subtracting the research cost.
}
\examples{
\dontrun{
enbs(12.5, 3.0)
}
}

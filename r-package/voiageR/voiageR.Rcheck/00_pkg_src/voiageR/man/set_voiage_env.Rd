\name{set_voiage_env}
\alias{set_voiage_env}
\title{Set the Python Environment for voiage}
\usage{
set_voiage_env(env, type = c("virtualenv", "conda"))
}
\arguments{
\item{env}{The virtualenv or conda environment name or path.}
\item{type}{The environment type, either \code{"virtualenv"} or \code{"conda"}.}
}
\value{Invisibly returns \code{NULL}.}
\description{
Configures the Python environment reticulate should use before importing voiage.
}
\examples{
\dontrun{
set_voiage_env("myenv", type = "virtualenv")
set_voiage_env("myenv", type = "conda")
}
}

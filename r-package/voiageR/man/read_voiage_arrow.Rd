\name{read_voiage_arrow}
\alias{read_voiage_arrow}
\title{Read a canonical voiage Arrow table}
\usage{read_voiage_arrow(path)}
\arguments{\item{path}{Path to an Arrow IPC, Feather, or Parquet file.}}
\value{A data frame retaining the canonical \code{payload_json} column.}
\description{Reads a canonical voiage table with the optional R Arrow package.}
\examples{\dontrun{
read_voiage_arrow("decision-problems.arrow")
}}

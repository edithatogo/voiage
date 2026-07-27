\name{normalize_statistical_assurance}
\alias{normalize_statistical_assurance}
\title{Validate and normalize statistical assurance with the Rust core}
\usage{normalize_statistical_assurance(assurance)}
\arguments{\item{assurance}{An R list or scalar JSON string matching the canonical v1 contract.}}
\value{A normalized R list produced by the Rust-authoritative serializer.}
\description{Validates statistical assurance directly through the versioned voiage C ABI.}
\examples{\dontrun{
normalize_statistical_assurance('{"reporting_class":"analytic"}')
}}

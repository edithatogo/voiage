\name{normalize_decision_problem}
\alias{normalize_decision_problem}
\title{Validate and normalize a Decision Problem with the Rust core}
\usage{normalize_decision_problem(problem)}
\arguments{\item{problem}{An R list or scalar JSON string matching the canonical v1 contract.}}
\value{A normalized R list produced by the Rust-authoritative serializer.}
\description{Validates a Decision Problem directly through the versioned voiage C ABI.}
\examples{\dontrun{
normalize_decision_problem(list(
  decision_problem_id = "example",
  title = "Example decision",
  analysis_type = "net-benefit-first",
  currency = "AUD",
  willingness_to_pay = 50000,
  interventions = list(
    list(intervention_id = "usual-care", name = "Usual care"),
    list(intervention_id = "new-care", name = "New care")
  )
))
}}

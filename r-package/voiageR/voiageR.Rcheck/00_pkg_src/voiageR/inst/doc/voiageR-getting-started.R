## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  eval = FALSE,
  echo = TRUE,
  collapse = TRUE,
  comment = "#>"
)

## ----setup-env----------------------------------------------------------------
# library(voiageR)
# 
# set_voiage_env("voiage", type = "virtualenv")
# init_voiage()

## ----setup-conda--------------------------------------------------------------
# set_voiage_env("voiage", type = "conda")

## ----evpi---------------------------------------------------------------------
# net_benefits <- matrix(
#   c(10.0, 12.0, 11.0, 9.0, 13.0, 14.0),
#   nrow = 3,
#   byrow = TRUE,
#   dimnames = list(NULL, c("Strategy A", "Strategy B"))
# )
# 
# evpi(net_benefits)

## ----evppi--------------------------------------------------------------------
# parameter_samples <- list(
#   effect = c(0.10, 0.12, 0.11),
#   cost = c(1000, 980, 1025)
# )
# 
# evppi(net_benefits, parameter_samples, parameters_of_interest = "effect")

## ----evsi---------------------------------------------------------------------
# model_func <- function(params) {
#   cbind(
#     Strategy_A = params$effect * 100 - params$cost,
#     Strategy_B = params$effect * 110 - params$cost
#   )
# }
# 
# prior_samples <- list(
#   effect = c(0.10, 0.12, 0.11),
#   cost = c(1000, 980, 1025)
# )
# 
# trial_design <- list(
#   arms = list(
#     list(name = "Treatment", sample_size = 50),
#     list(name = "Control", sample_size = 50)
#   )
# )
# 
# evsi(
#   model_func,
#   prior_samples,
#   trial_design,
#   method = "efficient",
#   n_outer_loops = 100,
#   n_inner_loops = 1000
# )


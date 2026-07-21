#' voiageR: An R Interface to the voiage Python Library
#'
#' This package provides an R interface to the voiage Python library for
#' Value of Information analysis, with helper functions for Python
#' environment management and availability checks.
#'
#' @docType package
#' @name voiageR
NULL

# Global cache
.voiage_cache <- new.env(parent = emptyenv())
.voiage_cache$module <- NULL

.set_voiage_module <- function(value) {
  .voiage_cache$module <- value
  invisible(NULL)
}

.get_voiage_module <- function() {
  if (is.null(.voiage_cache$module)) {
    init_voiage()
  }
  .voiage_cache$module
}

# Keep reticulate calls behind package-local seams so environment behavior can
# be tested without activating a real Python installation.
.py_module_available <- function(module) reticulate::py_module_available(module)
.py_import <- function(module) reticulate::import(module)
.use_virtualenv <- function(env) reticulate::use_virtualenv(env)
.use_condaenv <- function(env) reticulate::use_condaenv(env)

.evpi_native <- function(net_benefits) {
  library_path <- Sys.getenv("VOIAGE_FFI_LIBRARY", unset = "libvoiage_ffi")
  loaded <- tryCatch(
    dyn.load(library_path),
    error = function(error) {
      stop("The voiage Rust C ABI library is unavailable: ", error$message, call. = FALSE)
    }
  )
  on.exit(dyn.unload(loaded[["path"]]), add = TRUE)

  values <- as.double(t(net_benefits))
  result <- .C(
    "voiage_v1_evpi_i32_r",
    values = values,
    rows = as.integer(nrow(net_benefits)),
    columns = as.integer(ncol(net_benefits)),
    out_value = double(1),
    out_status = integer(1),
    PACKAGE = "voiageR"
  )
  if (!identical(as.integer(result$out_status), 0L)) {
    stop("voiage Rust EVPI ABI failed", call. = FALSE)
  }
  as.numeric(result$out_value)
}

.scale_evpi <- function(value, population, time_horizon, discount_rate) {
  if (is.null(population) && is.null(time_horizon) && is.null(discount_rate)) {
    return(value)
  }
  if (is.null(population) || is.null(time_horizon)) {
    stop("population and time_horizon must be provided together", call. = FALSE)
  }
  if (!is.numeric(population) || length(population) != 1L || !is.finite(population) || population <= 0) {
    stop("population must be a positive finite number", call. = FALSE)
  }
  if (!is.numeric(time_horizon) || length(time_horizon) != 1L || !is.finite(time_horizon) || time_horizon <= 0) {
    stop("time_horizon must be a positive finite number", call. = FALSE)
  }
  rate <- if (is.null(discount_rate)) 0 else discount_rate
  if (!is.numeric(rate) || length(rate) != 1L || !is.finite(rate)) {
    stop("discount_rate must be a finite number", call. = FALSE)
  }
  annuity <- if (rate == 0) time_horizon else (1 - (1 + rate)^(-time_horizon)) / rate
  as.numeric(value * population * annuity)
}

#' Initialize the voiage Python module
#'
#' This function initializes the voiage Python module using reticulate.
#'
#' @return None (invisible)
#' @export
#'
#' @examples
#' \dontrun{
#' init_voiage()
#' }
init_voiage <- function() {
  if (!.py_module_available("voiage")) {
    stop("The voiage Python package is not available. Please install it with: pip install voiage")
  }

  if (!is.null(.voiage_cache$module)) {
    return(invisible(NULL))
  }

  .set_voiage_module(.py_import("voiage"))

  invisible(NULL)
}

#' Calculate Expected Value of Perfect Information (EVPI)
#'
#' This function calculates the Expected Value of Perfect Information using the voiage library.
#'
#' @param net_benefits A matrix or data frame with rows representing PSA samples and columns representing strategies.
#' @param population Optional population size for scaling the result.
#' @param time_horizon Optional time horizon for scaling the result.
#' @param discount_rate Optional discount rate for scaling the result.
#'
#' @return The calculated EVPI value.
#' @export
#'
#' @examples
#' \dontrun{
#' # Create sample net benefit data
#' net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)
#'
#' # Calculate EVPI
#' evpi_value <- evpi(net_benefits)
#' print(evpi_value)
#' }
evpi <- function(net_benefits, population = NULL, time_horizon = NULL, discount_rate = NULL) {
  if (is.data.frame(net_benefits)) {
    net_benefits <- as.matrix(net_benefits)
  }
  if (!is.matrix(net_benefits)) {
    stop("net_benefits must be a matrix or data frame", call. = FALSE)
  }

  # Preserve injectable test/compatibility modules, while the normal R path
  # calls the Rust core directly and does not require Python or reticulate.
  cache <- get(".voiage_cache", envir = asNamespace("voiageR"))
  if (!is.null(cache$module) && !is.null(cache$module$evpi)) {
    return(cache$module$evpi(
      net_benefits,
      population = population,
      time_horizon = time_horizon,
      discount_rate = discount_rate
    ))
  }
  .scale_evpi(.evpi_native(net_benefits), population, time_horizon, discount_rate)
}

#' Calculate Expected Value of Partial Perfect Information (EVPPI)
#'
#' This function calculates the Expected Value of Partial Perfect Information using the voiage library.
#'
#' @param net_benefits A matrix or data frame with rows representing PSA samples and columns representing strategies.
#' @param parameter_samples A named list or data frame with parameter samples.
#' @param parameters_of_interest Optional character vector of parameter names to retain.
#' @param population Optional population size for scaling the result.
#' @param time_horizon Optional time horizon for scaling the result.
#' @param discount_rate Optional discount rate for scaling the result.
#'
#' @return The calculated EVPPI value.
#' @export
#'
#' @examples
#' \dontrun{
#' # Create sample net benefit data
#' net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)
#'
#' # Create parameter samples
#' param_samples <- list(
#'   param1 = rnorm(1000),
#'   param2 = rnorm(1000)
#' )
#'
#' # Calculate EVPPI
#' evppi_value <- evppi(net_benefits, param_samples)
#' print(evppi_value)
#' }
evppi <- function(
    net_benefits,
    parameter_samples,
    parameters_of_interest = NULL,
    population = NULL,
    time_horizon = NULL,
    discount_rate = NULL
) {
  .voiage <- .get_voiage_module()

  if (is.data.frame(net_benefits)) {
    net_benefits <- as.matrix(net_benefits)
  }

  if (is.data.frame(parameter_samples)) {
    parameter_samples <- as.list(parameter_samples)
  } else if (is.matrix(parameter_samples)) {
    parameter_samples <- as.list(as.data.frame(parameter_samples))
  }

  if (is.null(parameters_of_interest)) {
    parameters_of_interest <- names(parameter_samples)
  }
  if (is.null(parameters_of_interest) || length(parameters_of_interest) == 0) {
    stop("`parameters_of_interest` must be provided or inferable from `parameter_samples`.")
  }

  .voiage$evppi(
    net_benefits,
    parameter_samples,
    parameters_of_interest = as.list(parameters_of_interest),
    population = population,
    time_horizon = time_horizon,
    discount_rate = discount_rate
  )
}

#' Calculate Expected Value of Sample Information (EVSI)
#'
#' This function calculates the Expected Value of Sample Information using the voiage library.
#'
#' @param model_func A function that takes parameter samples and returns net benefits.
#' @param prior_samples A named list or data frame with prior parameter samples.
#' @param trial_design A list of trial arm specifications.
#' @param population Optional population size for scaling the result.
#' @param time_horizon Optional time horizon for scaling the result.
#' @param discount_rate Optional discount rate for scaling the result.
#' @param n_simulations Number of simulations to run.
#' @param ... Additional arguments passed to the underlying Python function.
#'
#' @return The calculated EVSI value.
#' @export
#'
#' @examples
#' \dontrun{
#' # Define a simple model function
#' model_func <- function(params) {
#'   # Simple example - in practice, this would be a more complex economic model
#'   nb_strategy1 <- params$param1
#'   nb_strategy2 <- params$param2
#'   return(cbind(nb_strategy1, nb_strategy2))
#' }
#'
#' # Create prior samples
#' prior_samples <- list(
#'   param1 = rnorm(1000),
#'   param2 = rnorm(1000)
#' )
#'
#' # Define trial design
#' trial_design <- list(
#'   treatment = list(name = "Treatment", sample_size = 50),
#'   control = list(name = "Control", sample_size = 50)
#' )
#'
#' # Calculate EVSI
#' evsi_value <- evsi(model_func, prior_samples, trial_design)
#' print(evsi_value)
#' }
evsi <- function(
    model_func,
    prior_samples,
    trial_design,
    population = NULL,
    time_horizon = NULL,
    discount_rate = NULL,
    method = c("two_loop", "regression", "efficient", "moment_based"),
    n_outer_loops = 100,
    n_inner_loops = 1000,
    metamodel = "linear"
) {
  .voiage <- .get_voiage_module()
  method <- match.arg(method)

  if (is.data.frame(trial_design)) {
    trial_design <- as.list(trial_design)
  }

  if (is.data.frame(prior_samples)) {
    prior_samples <- as.list(prior_samples)
  } else if (is.matrix(prior_samples)) {
    prior_samples <- as.list(as.data.frame(prior_samples))
  }

  py_trial_design <- .voiage$TrialDesign$from_dict(trial_design)
  py_prior_samples <- .voiage$ParameterSet$from_numpy_or_dict(prior_samples)

  py_model_func <- function(params) {
    r_params <- reticulate::py_to_r(params$parameters)
    result <- model_func(r_params)

    if (is.data.frame(result)) {
      result <- as.matrix(result)
    }

    .voiage$ValueArray$from_numpy(result)
  }

  .voiage$evsi(
    model_func = py_model_func,
    psa_prior = py_prior_samples,
    trial_design = py_trial_design,
    population = population,
    time_horizon = time_horizon,
    discount_rate = discount_rate,
    method = method,
    n_outer_loops = n_outer_loops,
    n_inner_loops = n_inner_loops,
    metamodel = metamodel
  )
}

#' Check if voiage Python package is available
#'
#' This function checks if the voiage Python package is available and can be imported.
#'
#' @return Logical indicating whether the voiage package is available.
#' @export
#'
#' @examples
#' \dontrun{
#' is_available <- is_voiage_available()
#' print(is_available)
#' }
is_voiage_available <- function() {
  .py_module_available("voiage")
}

#' Set Python environment for voiage
#'
#' This function sets the Python environment to use for the voiage package.
#'
#' @param env The Python environment to use (virtualenv or conda environment name).
#' @param type The type of environment ("virtualenv" or "conda").
#'
#' @return None (invisible)
#' @export
#'
#' @examples
#' \dontrun{
#' # Use a virtual environment
#' set_voiage_env("myenv", type = "virtualenv")
#'
#' # Use a conda environment
#' set_voiage_env("myenv", type = "conda")
#' }
set_voiage_env <- function(env, type = c("virtualenv", "conda")) {
  type <- match.arg(type)

  if (type == "virtualenv") {
    .use_virtualenv(env)
  } else {
    .use_condaenv(env)
  }

  .set_voiage_module(NULL)

  invisible(NULL)
}

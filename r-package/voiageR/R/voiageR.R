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

.normalise_trial_design <- function(trial_design) {
  if (
    !is.list(trial_design) ||
      is.null(trial_design$arms) ||
      !is.list(trial_design$arms) ||
      length(trial_design$arms) == 0L
  ) {
    stop("`trial_design` must contain a non-empty `arms` list.", call. = FALSE)
  }

  trial_design$arms <- lapply(trial_design$arms, function(arm) {
    if (
      !is.list(arm) ||
        !is.character(arm$name) ||
        length(arm$name) != 1L ||
        is.na(arm$name) ||
        !nzchar(arm$name)
    ) {
      stop(
        "Each trial arm must have one non-empty character `name`.",
        call. = FALSE
      )
    }
    if (
      !is.numeric(arm$sample_size) ||
        length(arm$sample_size) != 1L ||
        is.na(arm$sample_size) ||
        !is.finite(arm$sample_size) ||
        arm$sample_size <= 0 ||
        arm$sample_size != floor(arm$sample_size) ||
        arm$sample_size > .Machine$integer.max
    ) {
      stop(
        "Each trial arm must have one positive integer `sample_size`.",
        call. = FALSE
      )
    }
    arm$sample_size <- as.integer(arm$sample_size)
    arm
  })

  arm_names <- vapply(trial_design$arms, function(arm) arm$name, character(1))
  if (anyDuplicated(arm_names)) {
    stop("Trial arm names must be unique.", call. = FALSE)
  }
  trial_design
}

.arm_parameter_name <- function(arm_name) {
  paste0("mean_", tolower(gsub(" ", "_", arm_name, fixed = TRUE)))
}

.positive_integer_control <- function(value, name) {
  if (
    !is.numeric(value) ||
      length(value) != 1L ||
      is.na(value) ||
      !is.finite(value) ||
      value <= 0 ||
      value != floor(value) ||
      value > .Machine$integer.max
  ) {
    stop(paste0("`", name, "` must be one positive integer."), call. = FALSE)
  }
  as.integer(value)
}

.normalise_seed <- function(seed) {
  if (is.null(seed)) {
    return(NULL)
  }
  if (
    !is.numeric(seed) ||
      length(seed) != 1L ||
      is.na(seed) ||
      !is.finite(seed) ||
      seed < 0 ||
      seed != floor(seed) ||
      seed > .Machine$integer.max
  ) {
    stop("`seed` must be one non-negative integer.", call. = FALSE)
  }
  as.integer(seed)
}

.validate_builtin_two_loop_contract <- function(prior_samples, trial_design) {
  if (!is.list(prior_samples) || is.null(names(prior_samples)) || any(names(prior_samples) == "")) {
    stop("`prior_samples` must be a named list, data frame, or matrix.", call. = FALSE)
  }

  parameter_names <- vapply(
    trial_design$arms,
    function(arm) .arm_parameter_name(arm$name),
    character(1)
  )
  if (anyDuplicated(parameter_names)) {
    stop(
      "Trial arm names must map to unique `mean_<normalised arm>` parameters.",
      call. = FALSE
    )
  }

  missing_parameters <- setdiff(parameter_names, names(prior_samples))
  if (length(missing_parameters) > 0L) {
    stop(
      paste0(
        "Built-in two-loop EVSI requires one `mean_<normalised arm>` ",
        "parameter per trial arm; missing: ",
        paste(missing_parameters, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }

  outcome_sd <- prior_samples$sd_outcome
  if (
    !is.numeric(outcome_sd) ||
      length(outcome_sd) == 0L ||
      anyNA(outcome_sd) ||
      any(!is.finite(outcome_sd)) ||
      any(outcome_sd <= 0) ||
      any(outcome_sd != outcome_sd[[1L]])
  ) {
    stop(
      "Built-in two-loop EVSI requires a finite, strictly positive, fixed `sd_outcome`.",
      call. = FALSE
    )
  }
  invisible(NULL)
}

.evpi_native <- function(net_benefits) {
  library_path <- Sys.getenv("VOIAGE_FFI_LIBRARY", unset = "libvoiage_ffi")
  loaded <- tryCatch(
    dyn.load(library_path),
    error = function(error) {
      stop("The voiage Rust C ABI library is unavailable: ", error$message, call. = FALSE)
    }
  )
  on.exit(dyn.unload(loaded[["path"]]), add = TRUE)

  symbol <- tryCatch(
    getNativeSymbolInfo(
      "voiage_v1_evpi_i32_r",
      PACKAGE = loaded
    )[["address"]],
    error = function(error) {
      stop(
        "The voiage Rust C ABI library does not export the EVPI symbol: ",
        error$message,
        call. = FALSE
      )
    }
  )
  values <- as.double(t(net_benefits))
  result <- .C(
    symbol,
    values = values,
    rows = as.integer(nrow(net_benefits)),
    columns = as.integer(ncol(net_benefits)),
    out_value = double(1),
    out_status = integer(1)
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
#' This function calculates Expected Value of Sample Information through the
#' Python voiage bridge.
#'
#' @param model_func A function that takes parameter samples and returns net benefits.
#' @param prior_samples A named list, data frame, or matrix with prior parameter
#'   samples.
#' @param trial_design A list containing an `arms` list. Each arm must provide
#'   a unique `name` and positive integer `sample_size`.
#' @param population Optional population size for scaling the result.
#' @param time_horizon Optional time horizon for scaling the result.
#' @param discount_rate Optional discount rate for scaling the result.
#' @param method EVSI estimator. `two_loop` is the corrected built-in
#'   joint-normal study model; the other values are compatibility estimators.
#' @param n_outer_loops Number of simulated trial data sets for `two_loop`.
#' @param n_inner_loops Number of joint posterior draws per simulated trial.
#' @param metamodel Metamodel used by compatible approximation methods.
#' @param seed Optional non-negative integer seed for reproducible `two_loop`
#'   simulation.
#'
#' @return The calculated EVSI value.
#' @export
#'
#' @examples
#' \dontrun{
#' model_func <- function(params) {
#'   cbind(
#'     Treatment = 50000 * params$mean_treatment - 3000,
#'     Control = 50000 * params$mean_control
#'   )
#' }
#'
#' set.seed(17)
#' draws <- 1000L
#' prior_samples <- list(
#'   mean_treatment = rnorm(draws, mean = 0.06, sd = 0.03),
#'   mean_control = rnorm(draws, mean = 0.00, sd = 0.02),
#'   sd_outcome = rep(1.0, draws)
#' )
#'
#' trial_design <- list(
#'   arms = list(
#'     list(name = "Treatment", sample_size = 100L),
#'     list(name = "Control", sample_size = 100L)
#'   )
#' )
#'
#' evsi_value <- evsi(
#'   model_func,
#'   prior_samples,
#'   trial_design,
#'   method = "two_loop",
#'   n_outer_loops = 100L,
#'   n_inner_loops = 1000L,
#'   seed = 20260724L
#' )
#' print(evsi_value)
#' }
#'
#' @details
#' The built-in `two_loop` method uses a fitted joint multivariate-normal prior
#' and a known-variance arm-mean likelihood. `prior_samples` must contain one
#' parameter named `mean_<normalised arm>` for each trial arm, where
#' normalisation lowercases the arm name and replaces spaces with underscores.
#' It must also contain a finite, strictly positive `sd_outcome` whose value is
#' fixed across all prior draws.
#'
#' Custom `trial_simulator` and `posterior_sampler` callbacks are available
#' only from the Python API. The R facade does not convert or execute those
#' callbacks and therefore does not claim parity for custom two-loop models.
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
    metamodel = "linear",
    seed = NULL
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

  trial_design <- .normalise_trial_design(trial_design)
  n_outer_loops <- .positive_integer_control(n_outer_loops, "n_outer_loops")
  n_inner_loops <- .positive_integer_control(n_inner_loops, "n_inner_loops")
  seed <- .normalise_seed(seed)
  if (method == "two_loop") {
    .validate_builtin_two_loop_contract(prior_samples, trial_design)
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
    metamodel = metamodel,
    seed = seed
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

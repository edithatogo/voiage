#' voiageR: An R Interface to the voiage Python Library
#'
#' This package provides an R interface to the voiage Python library for 
#' Value of Information analysis.
#'
#' @docType package
#' @name voiageR
NULL

#' @import reticulate
#' @import methods
NULL

# Global variables
.voiage <- NULL

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
  # Check if the voiage Python package is available
  if (!py_module_available("voiage")) {
    stop("The voiage Python package is not available. Please install it with: pip install voiage")
  }
  
  # Import the voiage module
  .voiage <<- import("voiage")
  
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
#' @param ... Additional arguments passed to the underlying Python function.
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
evpi <- function(net_benefits, population = NULL, time_horizon = NULL, discount_rate = NULL, ...) {
  # Initialize if not already done
  if (is.null(.voiage)) {
    init_voiage()
  }
  
  # Convert R data structures to Python
  if (is.data.frame(net_benefits)) {
    net_benefits <- as.matrix(net_benefits)
  }
  
  # Create DecisionAnalysis object
  analysis <- .voiage$analysis$DecisionAnalysis(nb_array = net_benefits)
  
  # Calculate EVPI
  result <- analysis$evpi(
    population = population,
    time_horizon = time_horizon,
    discount_rate = discount_rate,
    ...
  )
  
  return(result)
}

#' Calculate Expected Value of Partial Perfect Information (EVPPI)
#'
#' This function calculates the Expected Value of Partial Perfect Information using the voiage library.
#'
#' @param net_benefits A matrix or data frame with rows representing PSA samples and columns representing strategies.
#' @param parameter_samples A named list or data frame with parameter samples.
#' @param population Optional population size for scaling the result.
#' @param time_horizon Optional time horizon for scaling the result.
#' @param discount_rate Optional discount rate for scaling the result.
#' @param ... Additional arguments passed to the underlying Python function.
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
evppi <- function(net_benefits, parameter_samples, population = NULL, time_horizon = NULL, 
                  discount_rate = NULL, ...) {
  # Initialize if not already done
  if (is.null(.voiage)) {
    init_voiage()
  }
  
  # Convert R data structures to Python
  if (is.data.frame(net_benefits)) {
    net_benefits <- as.matrix(net_benefits)
  }
  
  # Create DecisionAnalysis object
  analysis <- .voiage$analysis$DecisionAnalysis(
    nb_array = net_benefits,
    parameter_samples = parameter_samples
  )
  
  # Calculate EVPPI
  result <- analysis$evppi(
    population = population,
    time_horizon = time_horizon,
    discount_rate = discount_rate,
    ...
  )
  
  return(result)
}

#' Calculate Expected Value of Sample Information (EVSI)
#'
#' This function calculates the Expected Value of Sample Information using the voiage library.
#'
#' @param model_func A function that takes parameter samples and returns net benefits.
#' @param prior_samples A named list or data frame with prior parameter samples.
#' @param trial_design A list describing the trial design.
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
evsi <- function(model_func, prior_samples, trial_design, population = NULL, time_horizon = NULL, 
                 discount_rate = NULL, n_simulations = 1000, ...) {
  # Initialize if not already done
  if (is.null(.voiage)) {
    init_voiage()
  }
  
  # Create a Python wrapper for the R model function
  py_model_func <- function(params) {
    # Convert Python ParameterSet to R list
    r_params <- list()
    for (name in names(params$parameters)) {
      r_params[[name]] <- params$parameters[[name]]
    }
    
    # Call R function
    result <- model_func(r_params)
    
    # Convert result to Python ValueArray
    if (is.data.frame(result)) {
      result <- as.matrix(result)
    }
    
    return(.voiage$schema$ValueArray$from_numpy(result))
  }
  
  # Convert R data structures to Python
  py_prior_samples <- .voiage$schema$ParameterSet$from_numpy_or_dict(prior_samples)
  py_trial_design <- .voiage$schema$TrialDesign(arms = trial_design)
  
  # Calculate EVSI using parallel processing
  result <- .voiage$parallel$monte_carlo$parallel_evsi_calculation(
    model_func = py_model_func,
    psa_prior = py_prior_samples,
    trial_design = py_trial_design,
    population = population,
    time_horizon = time_horizon,
    discount_rate = discount_rate,
    n_simulations = n_simulations,
    ...
  )
  
  return(result)
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
  return(py_module_available("voiage"))
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
    use_virtualenv(env)
  } else {
    use_condaenv(env)
  }
  
  # Reinitialize voiage after setting environment
  .voiage <<- NULL
  
  invisible(NULL)
}
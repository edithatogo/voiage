test_that("voiageR package loads and exports the expected surface", {
  expect_true("package:voiageR" %in% search())
  expect_true("voiageR" %in% loadedNamespaces())

  exported <- getNamespaceExports("voiageR")
  expect_setequal(
    exported,
    c(
      "evpi",
      "evppi",
      "evsi",
      "init_voiage",
      "is_voiage_available",
      "set_voiage_env"
    )
  )
})

voiage_namespace <- function() {
  asNamespace("voiageR")
}

voiage_cache <- function() {
  get(".voiage_cache", envir = voiage_namespace())
}

voiage_module <- function() {
  voiage_cache()$module
}

set_voiage_module <- function(value) {
  cache <- voiage_cache()
  cache$module <- value
  invisible(NULL)
}

with_voiage_stub <- function(fake_voiage, code) {
  old_voiage <- voiage_module()
  on.exit(set_voiage_module(old_voiage), add = TRUE)
  set_voiage_module(fake_voiage)
  eval.parent(substitute(code))
}

test_that("is_voiage_available returns the reticulate availability check", {
  with_mocked_bindings(
    py_module_available = function(module) {
      expect_identical(module, "voiage")
      TRUE
    },
    {
      expect_identical(is_voiage_available(), TRUE)
    }
  )

  with_mocked_bindings(
    py_module_available = function(module) {
      expect_identical(module, "voiage")
      FALSE
    },
    {
      expect_identical(is_voiage_available(), FALSE)
    }
  )
})

test_that("init_voiage errors when voiage is unavailable", {
  with_mocked_bindings(
    py_module_available = function(module) {
      expect_identical(module, "voiage")
      FALSE
    },
    {
      expect_error(
        init_voiage(),
        "The voiage Python package is not available"
      )
    }
  )
})

test_that("init_voiage imports voiage when available", {
  imported <- FALSE
  with_mocked_bindings(
    py_module_available = function(module) {
      expect_identical(module, "voiage")
      TRUE
    },
    import = function(module) {
      expect_identical(module, "voiage")
      imported <<- TRUE
      list(module = module)
    },
    {
      expect_silent(init_voiage())
    }
  )

  expect_true(imported)
  expect_equal(voiage_module(), list(module = "voiage"))
})

test_that("set_voiage_env validates the environment type", {
  expect_error(
    set_voiage_env("test_env", type = "invalid"),
    "one of"
  )
})

test_that("set_voiage_env updates the requested environment backend", {
  virtualenv_called <- NULL
  condaenv_called <- NULL

  old_voiage <- voiage_module()
  on.exit(set_voiage_module(old_voiage), add = TRUE)
  set_voiage_module(list(module = "voiage"))
  with_mocked_bindings(
    use_virtualenv = function(env) {
      virtualenv_called <<- env
    },
    use_condaenv = function(env) {
      condaenv_called <<- env
    },
    {
      set_voiage_env("py310", type = "virtualenv")
      expect_identical(virtualenv_called, "py310")
      expect_true(is.null(voiage_module()))

      set_voiage_module(list(module = "voiage"))
      set_voiage_env("analysis", type = "conda")
      expect_identical(condaenv_called, "analysis")
      expect_true(is.null(voiage_module()))
    }
  )
})

test_that("evpi coerces data frames and dispatches through the top-level Python helper", {
  seen <- new.env(parent = emptyenv())
  old_voiage <- voiage_module()
  on.exit(set_voiage_module(old_voiage), add = TRUE)
  set_voiage_module(
    list(
      evpi = function(
        net_benefits,
        population = NULL,
        time_horizon = NULL,
        discount_rate = NULL
      ) {
        seen$net_benefits <- net_benefits
        seen$args <- list(
          population = population,
          time_horizon = time_horizon,
          discount_rate = discount_rate
        )
        12.5
      }
    )
  )

  result <- evpi(
    data.frame(strategy_a = c(1, 2), strategy_b = c(3, 4)),
    population = 100,
    time_horizon = 10,
    discount_rate = 0.03
  )

  expect_identical(result, 12.5)
  expect_true(is.matrix(seen$net_benefits))
  expect_equal(unname(seen$net_benefits), matrix(c(1, 2, 3, 4), ncol = 2))
  expect_identical(seen$args$population, 100)
  expect_identical(seen$args$time_horizon, 10)
  expect_identical(seen$args$discount_rate, 0.03)
})

test_that("evppi forwards parameter samples and preserves matrix conversion", {
  seen <- new.env(parent = emptyenv())
  old_voiage <- voiage_module()
  on.exit(set_voiage_module(old_voiage), add = TRUE)
  set_voiage_module(
    list(
      evppi = function(
        net_benefits,
        parameter_samples,
        parameters_of_interest = NULL,
        population = NULL,
        time_horizon = NULL,
        discount_rate = NULL
      ) {
        seen$net_benefits <- net_benefits
        seen$parameter_samples <- parameter_samples
        seen$parameters_of_interest <- parameters_of_interest
        seen$args <- list(
          population = population,
          time_horizon = time_horizon,
          discount_rate = discount_rate
        )
        8.75
      }
    )
  )

  result <- evppi(
    data.frame(strategy_a = c(1, 2), strategy_b = c(3, 4)),
    data.frame(param1 = c(0.1, 0.2), param2 = c(0.3, 0.4)),
    parameters_of_interest = c("param1"),
    population = 50,
    time_horizon = 5,
    discount_rate = 0.01
  )

  expect_identical(result, 8.75)
  expect_true(is.matrix(seen$net_benefits))
  expect_equal(
    seen$parameter_samples,
    list(param1 = c(0.1, 0.2), param2 = c(0.3, 0.4))
  )
  expect_identical(seen$parameters_of_interest, list("param1"))
  expect_identical(seen$args$population, 50)
  expect_identical(seen$args$time_horizon, 5)
  expect_identical(seen$args$discount_rate, 0.01)
})

test_that("evsi wraps the model function and dispatches through reticulate", {
  seen <- new.env(parent = emptyenv())
  old_voiage <- voiage_module()
  on.exit(set_voiage_module(old_voiage), add = TRUE)
  set_voiage_module(
    list(
      TrialDesign = list(
        from_dict = function(trial_design) {
          seen$trial_design <- trial_design
          list(tag = "trial")
        }
      ),
      ParameterSet = list(
        from_numpy_or_dict = function(prior_samples) {
          seen$prior_samples <- prior_samples
          list(tag = "psa")
        }
      ),
      ValueArray = list(
        from_numpy = function(result) {
          seen$model_output <- result
          list(tag = "value_array")
        }
      ),
      evsi = function(
        model_func,
        psa_prior,
        trial_design,
        population = NULL,
        time_horizon = NULL,
        discount_rate = NULL,
        method = "two_loop",
        n_outer_loops = 100,
        n_inner_loops = 1000,
        metamodel = "linear"
      ) {
        seen$args <- list(
          psa_prior = psa_prior,
          trial_design = trial_design,
          population = population,
          time_horizon = time_horizon,
          discount_rate = discount_rate,
          method = method,
          n_outer_loops = n_outer_loops,
          n_inner_loops = n_inner_loops,
          metamodel = metamodel
        )
        model_result <- model_func(list(parameters = list(alpha = 1, beta = 2)))
        seen$model_func_result <- model_result
        4.25
      }
    )
  )

  result <- evsi(
    model_func = function(params) {
      expect_equal(params, list(alpha = 1, beta = 2))
      data.frame(strategy_a = c(1, 2), strategy_b = c(3, 4))
    },
    prior_samples = data.frame(alpha = c(0.1, 0.2), beta = c(0.3, 0.4)),
    trial_design = list(arms = list(list(name = "Arm A", sample_size = 10))),
    population = 1000,
    time_horizon = 12,
    discount_rate = 0.04,
    method = "efficient",
    n_outer_loops = 99,
    n_inner_loops = 17,
    metamodel = "linear"
  )

  expect_identical(result, 4.25)
  expect_equal(
    seen$prior_samples,
    list(alpha = c(0.1, 0.2), beta = c(0.3, 0.4))
  )
  expect_equal(seen$trial_design, list(arms = list(list(name = "Arm A", sample_size = 10))))
  expect_identical(seen$args$population, 1000)
  expect_identical(seen$args$time_horizon, 12)
  expect_identical(seen$args$discount_rate, 0.04)
  expect_identical(seen$args$method, "efficient")
  expect_identical(seen$args$n_outer_loops, 99)
  expect_identical(seen$args$n_inner_loops, 17)
  expect_identical(seen$args$metamodel, "linear")
  expect_equal(
    seen$model_output,
    matrix(
      c(1, 2, 3, 4),
      ncol = 2,
      dimnames = list(NULL, c("strategy_a", "strategy_b"))
    )
  )
  expect_equal(seen$model_func_result, list(tag = "value_array"))
})

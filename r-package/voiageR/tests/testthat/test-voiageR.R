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

test_that("the R EVSI facade exposes supported controls without custom callbacks", {
  evsi_arguments <- names(formals(evsi))

  expect_true(all(c("n_outer_loops", "n_inner_loops", "metamodel", "seed") %in% evsi_arguments))
  expect_false(any(c("trial_simulator", "posterior_sampler") %in% evsi_arguments))
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
    .py_module_available = function(module) {
      expect_identical(module, "voiage")
      TRUE
    },
    {
      expect_identical(is_voiage_available(), TRUE)
    }
  )

  with_mocked_bindings(
    .py_module_available = function(module) {
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
    .py_module_available = function(module) {
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
    .py_module_available = function(module) {
      expect_identical(module, "voiage")
      TRUE
    },
    .py_import = function(module) {
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
    .use_virtualenv = function(env) {
      virtualenv_called <<- env
    },
    .use_condaenv = function(env) {
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
        metamodel = "linear",
        seed = NULL
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
          metamodel = metamodel,
          seed = seed
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
    trial_design = list(arms = list(list(name = "Arm A", sample_size = 10L))),
    population = 1000,
    time_horizon = 12,
    discount_rate = 0.04,
    method = "efficient",
    n_outer_loops = 99,
    n_inner_loops = 17,
    metamodel = "linear",
    seed = NULL
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
  expect_identical(seen$args$n_outer_loops, 99L)
  expect_identical(seen$args$n_inner_loops, 17L)
  expect_identical(seen$args$metamodel, "linear")
  expect_null(seen$args$seed)
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

test_that("evsi forwards the corrected built-in two-loop contract", {
  seen <- new.env(parent = emptyenv())
  fake_voiage <- list(
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
    ValueArray = list(from_numpy = function(result) result),
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
      metamodel = "linear",
      seed = NULL
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
        metamodel = metamodel,
        seed = seed
      )
      6.5
    }
  )

  result <- with_voiage_stub(
    fake_voiage,
    evsi(
      model_func = function(params) {
        cbind(Treatment = params$mean_treatment, Control = params$mean_control)
      },
      prior_samples = list(
        mean_treatment = c(0.04, 0.06, 0.08),
        mean_control = c(-0.01, 0.00, 0.01),
        sd_outcome = rep(1.0, 3L)
      ),
      trial_design = list(
        arms = list(
          list(name = "Treatment", sample_size = 75),
          list(name = "Control", sample_size = 80L)
        )
      ),
      population = 1300,
      time_horizon = 10,
      discount_rate = 0.03,
      method = "two_loop",
      n_outer_loops = 200L,
      n_inner_loops = 500L,
      metamodel = "linear",
      seed = 42L
    )
  )

  expect_identical(result, 6.5)
  expect_identical(
    vapply(seen$trial_design$arms, function(arm) arm$sample_size, integer(1)),
    c(75L, 80L)
  )
  expect_named(
    seen$prior_samples,
    c("mean_treatment", "mean_control", "sd_outcome")
  )
  expect_identical(seen$args$method, "two_loop")
  expect_identical(seen$args$n_outer_loops, 200L)
  expect_identical(seen$args$n_inner_loops, 500L)
  expect_identical(seen$args$seed, 42L)
})

test_that("evsi rejects incomplete built-in two-loop inputs before Python dispatch", {
  fake_voiage <- list(
    TrialDesign = list(
      from_dict = function(...) stop("Python dispatch should not occur.")
    ),
    ParameterSet = list(
      from_numpy_or_dict = function(...) stop("Python dispatch should not occur.")
    )
  )
  design <- list(
    arms = list(
      list(name = "Treatment", sample_size = 50L),
      list(name = "Control", sample_size = 50L)
    )
  )
  model <- function(params) cbind(params$mean_treatment, params$mean_control)

  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(
        model,
        list(mean_treatment = c(0.1, 0.2), sd_outcome = c(1, 1)),
        design
      ),
      "missing: mean_control"
    )
  )
  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(
        model,
        list(
          mean_treatment = c(0.1, 0.2),
          mean_control = c(0.0, 0.1),
          sd_outcome = c(1, 2)
        ),
        design
      ),
      "strictly positive, fixed"
    )
  )
})

test_that("evsi rejects ambiguous arm normalisation and invalid designs", {
  fake_voiage <- list()
  prior <- list(mean_a_b = c(0.1, 0.2), sd_outcome = c(1, 1))
  model <- function(params) cbind(params$mean_a_b, params$mean_a_b)

  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(
        model,
        prior,
        list(
          arms = list(
            list(name = "A B", sample_size = 10L),
            list(name = "A_B", sample_size = 10L)
          )
        )
      ),
      "unique `mean_<normalised arm>`"
    )
  )
  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(
        model,
        prior,
        list(arms = list(list(name = "A B", sample_size = 1.5)))
      ),
      "positive integer"
    )
  )
})

test_that("evsi validates integer simulation controls before Python dispatch", {
  fake_voiage <- list()
  prior <- list(mean_treatment = c(0.1, 0.2), sd_outcome = c(1, 1))
  design <- list(arms = list(list(name = "Treatment", sample_size = 10L)))
  model <- function(params) cbind(params$mean_treatment, params$mean_treatment)

  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(model, prior, design, n_outer_loops = 1.5),
      "`n_outer_loops` must be one positive integer"
    )
  )
  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(model, prior, design, n_inner_loops = 0),
      "`n_inner_loops` must be one positive integer"
    )
  )
  with_voiage_stub(
    fake_voiage,
    expect_error(
      evsi(model, prior, design, seed = -1),
      "`seed` must be one non-negative integer"
    )
  )
})

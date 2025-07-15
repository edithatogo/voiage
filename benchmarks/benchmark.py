import timeit

import numpy as np

from voiage import evpi, evppi, set_backend
from voiage.methods.basic import joint_evppi, conditional_evppi, sequential_evppi
from voiage.config import get_default_dtype

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def run_evpi_benchmark(n_samples, n_strategies):
    print(f"--- Running EVPI benchmark (samples={n_samples}, strategies={n_strategies}) ---")

    # NumPy backend
    set_backend("numpy")
    nb_array_np = np.random.rand(n_samples, n_strategies).astype(get_default_dtype())
    numpy_time = timeit.timeit(lambda: evpi(nb_array_np), number=10)
    print(f"NumPy backend: {numpy_time:.4f} seconds")

    # JAX backend
    if JAX_AVAILABLE:
        set_backend("jax")
        nb_array_jax = jnp.array(nb_array_np)
        # JIT compilation time
        jax_time_initial = timeit.timeit(lambda: evpi(nb_array_jax), number=1)
        # Rerunning to get a more accurate time after compilation
        jax_time = timeit.timeit(lambda: evpi(nb_array_jax), number=10)
        print(f"JAX backend (initial run): {jax_time_initial:.4f} seconds")
        print(f"JAX backend (subsequent runs): {jax_time:.4f} seconds")
    else:
        jax_time_initial = 0
        jax_time = 0

    set_backend("numpy")
    return numpy_time, jax_time_initial, jax_time


def run_evppi_benchmark(n_samples, n_strategies, n_params):
    print(f"\n--- Running EVPPI benchmark (samples={n_samples}, strategies={n_strategies}, params={n_params}) ---")

    # NumPy backend
    set_backend("numpy")
    nb_array_np = np.random.rand(n_samples, n_strategies).astype(get_default_dtype())
    param_samples_np = {f"p{i}": np.random.rand(n_samples).astype(get_default_dtype()) for i in range(n_params)}
    numpy_time = timeit.timeit(lambda: evppi(nb_array_np, param_samples_np), number=10)
    print(f"NumPy backend: {numpy_time:.4f} seconds")

    # JAX backend
    if JAX_AVAILABLE:
        set_backend("jax")
        nb_array_jax = jnp.array(nb_array_np)
        param_samples_jax = jnp.stack([param_samples_np[f"p{i}"] for i in range(n_params)], axis=1)
        # JIT compilation time
        jax_time_initial = timeit.timeit(lambda: evppi(nb_array_jax, param_samples_jax), number=1)
        # Rerunning to get a more accurate time after compilation
        jax_time = timeit.timeit(lambda: evppi(nb_array_jax, param_samples_jax), number=10)
        print(f"JAX backend (initial run): {jax_time_initial:.4f} seconds")
        print(f"JAX backend (subsequent runs): {jax_time:.4f} seconds")
    else:
        jax_time_initial = 0
        jax_time = 0

    set_backend("numpy")
    return numpy_time, jax_time_initial, jax_time


def run_joint_evppi_benchmark(n_samples, n_strategies, n_params):
    print(f"\n--- Running Joint EVPPI benchmark (samples={n_samples}, strategies={n_strategies}, params={n_params}) ---")
    set_backend("numpy")
    nb_array_np = np.random.rand(n_samples, n_strategies).astype(get_default_dtype())
    param_samples_np = {f"p{i}": np.random.rand(n_samples).astype(get_default_dtype()) for i in range(n_params)}
    numpy_time = timeit.timeit(lambda: joint_evppi(nb_array_np, param_samples_np), number=10)
    print(f"NumPy backend: {numpy_time:.4f} seconds")


def run_conditional_evppi_benchmark(n_samples, n_strategies, n_params):
    print(f"\n--- Running Conditional EVPPI benchmark (samples={n_samples}, strategies={n_strategies}, params={n_params}) ---")
    set_backend("numpy")
    nb_array_np = np.random.rand(n_samples, n_strategies).astype(get_default_dtype())
    param_samples_of_interest_np = {f"p{i}": np.random.rand(n_samples).astype(get_default_dtype()) for i in range(n_params)}
    param_samples_given_np = {f"p{i}": np.random.rand(n_samples).astype(get_default_dtype()) for i in range(n_params)}
    numpy_time = timeit.timeit(lambda: conditional_evppi(nb_array_np, param_samples_of_interest_np, param_samples_given_np), number=10)
    print(f"NumPy backend: {numpy_time:.4f} seconds")


def run_sequential_evppi_benchmark(n_samples, n_strategies, n_params):
    print(f"\n--- Running Sequential EVPPI benchmark (samples={n_samples}, strategies={n_strategies}, params={n_params}) ---")
    set_backend("numpy")
    nb_array_np = np.random.rand(n_samples, n_strategies).astype(get_default_dtype())
    param_samples_np = {f"p{i}": np.random.rand(n_samples).astype(get_default_dtype()) for i in range(n_params)}
    numpy_time = timeit.timeit(lambda: sequential_evppi(nb_array_np, param_samples_np), number=10)
    print(f"NumPy backend: {numpy_time:.4f} seconds")


if __name__ == "__main__":
    run_evpi_benchmark(1000, 5)
    run_evpi_benchmark(10000, 20)

    run_evppi_benchmark(1000, 5, 2)
    run_evppi_benchmark(10000, 20, 10)

    run_joint_evppi_benchmark(1000, 5, 2)
    run_joint_evppi_benchmark(10000, 20, 10)

    run_conditional_evppi_benchmark(1000, 5, 2)
    run_conditional_evppi_benchmark(10000, 20, 10)

    run_sequential_evppi_benchmark(1000, 5, 2)
    run_sequential_evppi_benchmark(10000, 20, 10)

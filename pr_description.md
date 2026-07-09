<<<<<<< HEAD
💡 **What:**
Optimized `markov_cohort_model` in `voiage/healthcare/utilities.py` to use a pre-allocated array (`np.empty`) and `out=` parameter for dot products in the loop. This avoids the cost of repeatedly allocating new arrays for each cycle and doing a copy of the initial state.

🎯 **Why:**
The simulation loop historically calculated `current_state = np.dot(current_state, transition_matrix)`, which meant that Python was constantly allocating memory for new state arrays. We pre-allocate `state_trajectories` anyways, and we can directly write the result of the `np.dot` operation sequentially into it.

I initially also considered compiling the operation using JAX's `lax.scan`. However, benchmarking revealed that for smaller problems (e.g. 5 states, 100 cycles), Python dispatching and JAX's overhead makes it quite slow compared to simple NumPy array ops. Given that NumPy's pre-allocation strategy was consistently the fastest on small arrays and comparably fast on large arrays, I went with the zero-dependency pure-NumPy optimization.

📊 **Measured Improvement:**
Tests are implemented via `pytest-benchmark`.
* Small sizes (e.g. 5 states, 100 cycles):
  * **Original:** ~180 μs
  * **Optimized NumPy:** ~150 μs (~16% improvement)
  * **JAX:** ~250 μs (much slower due to dispatch overhead)
* Large sizes (e.g. 100 states, 1000 cycles):
  * **Original:** ~3.5 ms
  * **Optimized NumPy:** ~3.0 ms (~15% improvement)
  * **JAX:** ~2.9 ms
=======
<<<<<<< HEAD
💡 **What:**
Optimized the `pareto_strategy_indices` function in `voiage/methods/_frontier_profiles.py` by replacing a nested Python `for` loop with vectorized NumPy operations.

🎯 **Why:**
The original implementation had an O(N^2) time complexity for comparing strategies. As the number of expected net benefit samples (profiles and strategies) increased, the nested loop created a massive performance bottleneck. Vectorizing this using NumPy broadcasting drastically improves execution speeds.

📊 **Measured Improvement:**
The optimization delivers massive speedups, especially for larger arrays.

Baseline results:
- Small (10 profiles, 10 strategies): 0.5208 ms
- Medium (50 profiles, 100 strategies): 58.2311 ms
- Large (100 profiles, 1000 strategies): 5997.1644 ms

Optimized results:
- Small (10 profiles, 10 strategies): 0.0330 ms (~15x faster)
- Medium (50 profiles, 100 strategies): 1.1813 ms (~49x faster)
- Large (100 profiles, 1000 strategies): 304.9709 ms (~19x faster)
=======
⚡ Performance Optimization Task: Vectorized NumPy Operations

💡 **What:** The loops updating net benefit values per sample have been rewritten into explicit vectorized operations across the full NumPy array inside `voiage/methods/adaptive.py`. The `bayesian_adaptive_trial_simulator` has been mostly rewritten using clean matrix math over iterables. Additionally, the outer loop test mocker in `adaptive_evsi` was updated to accurately simulate the new vectorized execution flow inside `sophisticated_adaptive_trial_simulator`.

🎯 **Why:** To simulate Bayesian adaptive trials, the software iterates through thousands of patient sample variations. The original nested python loops over `n_samples` were slow, causing unnecessary iterations, context switching, and slow overall simulator output. Vectorizing shifts the execution natively into C via NumPy arrays, allowing calculations across rows or columns instantly in parallel rather than element-by-element iteratively.

📊 **Measured Improvement:** Utilizing `cProfile`, the time necessary for executing `bayesian_adaptive_trial_simulator` acting on 100,000 independent trial samples dropped significantly.
- Baseline Run Time: ~2.27 seconds (or 2.857 in earlier cProfile runs)
- Improved Run Time: ~0.044 seconds
- **Improvement:** ~50x speed increase over baseline. The code produces identical net benefit logic while removing 100k individual loops.
>>>>>>> origin/main
>>>>>>> origin/main

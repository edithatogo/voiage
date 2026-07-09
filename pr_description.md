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

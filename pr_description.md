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

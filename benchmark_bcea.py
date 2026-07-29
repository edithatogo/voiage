import time
import json
import os
import numpy as np
from voiage.health_economics import HealthEconomicsAnalysis, Treatment
from voiage.ecosystem_integration import RPackageConnector

def run_benchmark():
    analysis = HealthEconomicsAnalysis(willingness_to_pay=50000, currency="USD")
    # Add multiple treatments
    for i in range(10):
        analysis.add_treatment(Treatment(
            name=f"Treatment {i}",
            description="desc",
            effectiveness=0.5 + i*0.05,
            cost_per_cycle=100 + i*10,
            cycles_required=10
        ))

    connector = RPackageConnector()
    output_path = "bench_out.json"

    start_time = time.perf_counter()
    connector.export_for_bcea(analysis, output_path, num_simulations=100000)
    end_time = time.perf_counter()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == '__main__':
    run_benchmark()

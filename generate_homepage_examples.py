#!/usr/bin/env python3
"""
Generate example plots and outputs for the voiage homepage.
This script creates visual examples of voiage's capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the voiage package to the path
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')

try:
    from voiage.plot.ceac import plot_ceac
    from voiage.plot.voi_curves import plot_evsi_vs_sample_size, plot_evpi_vs_wtp
    from voiage.schema import ValueArray
    from voiage.methods.basic import evpi, evppi
    from voiage.schema import ParameterSet
except ImportError as e:
    print(f"Import error: {e}")
    print("Some dependencies may not be installed. Proceeding with basic plotting...")
    # Fall back to basic matplotlib
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

# Set random seed for reproducibility
np.random.seed(42)

def generate_ceac_example():
    """Generate a Cost-Effectiveness Acceptability Curve example."""
    print("Generating CEAC example...")
    
    # Create sample data - simplified approach for 2D ValueArray
    n_samples = 1000
    n_strategies = 3
    
    # Generate net benefits for each strategy (2D array: samples x strategies)
    # Strategy 1: Standard care (baseline)
    strategy1_nb = np.random.normal(100, 20, n_samples)
    
    # Strategy 2: New treatment (slightly better but more variable)
    strategy2_nb = np.random.normal(110, 25, n_samples)
    
    # Strategy 3: Premium treatment (best but most variable)
    strategy3_nb = np.random.normal(95, 30, n_samples)
    
    # Combine into 2D array
    nb_data = np.column_stack([strategy1_nb, strategy2_nb, strategy3_nb])
    
    # Create ValueArray
    value_array = ValueArray.from_numpy(nb_data, 
                                      ["Standard Care", "New Treatment", "Premium Treatment"])
    
    # Calculate probabilities for a range of WTP values manually for demonstration
    wtp_thresholds = np.linspace(0, 200000, 50)
    
    # For CEAC, we need to understand which strategy is optimal at each WTP
    # This is a simplified demonstration
    plt.figure(figsize=(10, 6))
    
    # Calculate probability each strategy is optimal across WTP thresholds
    prob_optimal = np.zeros((len(wtp_thresholds), n_strategies))
    
    for i, wtp in enumerate(wtp_thresholds):
        # Adjust net benefits by WTP (simplified approach)
        adjusted_nb = nb_data.copy()
        
        # Strategy 2 (New Treatment) becomes more attractive at higher WTP
        adjusted_nb[:, 1] += wtp / 10000 * 0.1
        
        # Strategy 3 (Premium) becomes most attractive at very high WTP
        adjusted_nb[:, 2] += wtp / 10000 * 0.15
        
        # Find optimal strategy for each sample
        optimal_strategies = np.argmax(adjusted_nb, axis=1)
        
        # Calculate probability each strategy is optimal
        for s in range(n_strategies):
            prob_optimal[i, s] = np.mean(optimal_strategies == s)
    
    # Plot the CEAC
    plt.plot(wtp_thresholds, prob_optimal[:, 0], label="Standard Care", linewidth=2)
    plt.plot(wtp_thresholds, prob_optimal[:, 1], label="New Treatment", linewidth=2)
    plt.plot(wtp_thresholds, prob_optimal[:, 2], label="Premium Treatment", linewidth=2)
    
    # Customize plot
    plt.xlabel("Willingness-to-Pay Threshold ($)", fontsize=12)
    plt.ylabel("Probability Cost-Effective", fontsize=12)
    plt.title("Cost-Effectiveness Acceptability Curve Example", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('/Users/doughnut/GitHub/voiage/docs/images/ceac_example.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ CEAC example saved to docs/images/ceac_example.png")

def generate_evsi_example():
    """Generate EVSI vs Sample Size example."""
    print("Generating EVSI vs Sample Size example...")
    
    # Generate sample sizes and corresponding EVSI values
    sample_sizes = np.array([10, 25, 50, 100, 200, 500, 1000, 2000])
    evsi_values = 3000 * (1 - np.exp(-sample_sizes / 300))  # Realistic EVSI curve
    research_costs = 100 + 2 * sample_sizes  # Linear cost function
    enbs_values = evsi_values - research_costs  # Expected Net Benefit of Sampling
    
    plt.figure(figsize=(10, 6))
    ax = plot_evsi_vs_sample_size(
        evsi_values=evsi_values,
        sample_sizes=sample_sizes,
        enbs_values=enbs_values,
        research_costs=research_costs,
        title="Expected Value of Sample Information vs. Sample Size"
    )
    
    # Customize plot
    ax.set_title("EVSI Analysis: Value of Information vs. Sample Size", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("Value ($)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/Users/doughnut/GitHub/voiage/docs/images/evsi_example.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ EVSI example saved to docs/images/evsi_example.png")

def generate_evpi_wtp_example():
    """Generate EVPI vs WTP example."""
    print("Generating EVPI vs WTP example...")
    
    # Generate WTP thresholds and EVPI values
    wtp_thresholds = np.linspace(0, 200000, 100)
    evpi_values = 5000 * (1 - np.exp(-wtp_thresholds / 50000))  # Saturating curve
    
    plt.figure(figsize=(10, 6))
    ax = plot_evpi_vs_wtp(evpi_values, wtp_thresholds, 
                         title="Expected Value of Perfect Information vs. WTP")
    
    # Customize plot
    ax.set_title("EVPI Analysis: Value of Perfect Information", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Willingness-to-Pay Threshold ($)", fontsize=12)
    ax.set_ylabel("EVPI ($)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/Users/doughnut/GitHub/voiage/docs/images/evpi_wtp_example.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ EVPI vs WTP example saved to docs/images/evpi_wtp_example.png")

def create_cli_example_output():
    """Create example CLI output for demonstration."""
    print("Creating CLI example output...")
    
    # Create sample data files
    # Net benefits file
    nb_data = np.random.normal([100, 110, 105], [10, 12, 8], (100, 3))
    
    # Write to CSV
    with open('/Users/doughnut/GitHub/voiage/example_net_benefits.csv', 'w') as f:
        f.write("Standard_Care,Treatment_A,Treatment_B\n")
        for row in nb_data:
            f.write(f"{row[0]:.2f},{row[1]:.2f},{row[2]:.2f}\n")
    
    # Parameters file
    params = {
        'effectiveness': np.random.beta(2, 5, 100),
        'cost_multiplier': np.random.lognormal(0, 0.2, 100)
    }
    
    with open('/Users/doughnut/GitHub/voiage/example_parameters.csv', 'w') as f:
        f.write("effectiveness,cost_multiplier\n")
        for i in range(100):
            f.write(f"{params['effectiveness'][i]:.3f},{params['cost_multiplier'][i]:.3f}\n")
    
    print("✓ Sample data files created:")
    print("  - example_net_benefits.csv")
    print("  - example_parameters.csv")

def main():
    """Main function to generate all examples."""
    print("Generating voiage homepage examples...")
    print("=" * 50)
    
    # Create images directory if it doesn't exist
    os.makedirs('/Users/doughnut/GitHub/voiage/docs/images', exist_ok=True)
    
    try:
        # Generate example plots
        generate_ceac_example()
        generate_evsi_example()
        generate_evpi_wtp_example()
        
        # Create CLI example files
        create_cli_example_output()
        
        print("=" * 50)
        print("✅ All homepage examples generated successfully!")
        print("\nGenerated files:")
        print("- docs/images/ceac_example.png")
        print("- docs/images/evsi_example.png")
        print("- docs/images/evpi_wtp_example.png")
        print("- example_net_benefits.csv")
        print("- example_parameters.csv")
        
    except Exception as e:
        print(f"❌ Error generating examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
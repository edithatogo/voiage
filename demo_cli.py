#!/usr/bin/env python3
"""
Quick demo of voiage CLI capabilities.
Run this script to see voiage in action with generated example data.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print the output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/Users/doughnut/GitHub/voiage')
        if result.stdout:
            print("Output:", result.stdout.strip())
        if result.stderr:
            print("Stderr:", result.stderr.strip())
        print("-" * 50)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("=" * 60)
    print("🎯 voiage CLI Demo")
    print("=" * 60)
    print()
    
    print("This demo shows voiage's command-line capabilities using")
    print("the generated example data files.")
    print()
    
    # Check if example files exist
    if not os.path.exists('/Users/doughnut/GitHub/voiage/example_net_benefits.csv'):
        print("❌ Example data files not found. Please run generate_homepage_examples.py first.")
        return 1
    
    print("📊 Example data files found:")
    print("- example_net_benefits.csv (net benefits for 3 strategies)")
    print("- example_parameters.csv (parameter samples for EVPPI)")
    print()
    
    # Demo 1: Basic EVPI calculation
    print("1️⃣ Basic EVPI Calculation")
    run_command("python -m voiage.cli calculate-evpi example_net_benefits.csv")
    
    # Demo 2: EVPI with population scaling
    print("2️⃣ EVPI with Population Scaling")
    run_command("python -m voiage.cli calculate-evpi example_net_benefits.csv --population 100000 --time-horizon 10 --discount-rate 0.03")
    
    # Demo 3: Save to file
    print("3️⃣ EVPI with File Output")
    run_command("python -m voiage.cli calculate-evpi example_net_benefits.csv --output demo_evpi.txt")
    if os.path.exists('/Users/doughnut/GitHub/voiage/demo_evpi.txt'):
        print("📁 Output saved to demo_evpi.txt:")
        with open('/Users/doughnut/GitHub/voiage/demo_evpi.txt', 'r') as f:
            print(f.read().strip())
        print("-" * 50)
    
    # Demo 4: EVPPI calculation
    print("4️⃣ EVPPI Calculation")
    run_command("python -m voiage.cli calculate-evppi example_net_benefits.csv example_parameters.csv")
    
    # Demo 5: Full EVPPI with all options
    print("5️⃣ Full EVPPI Analysis")
    run_command("python -m voiage.cli calculate-evppi example_net_benefits.csv example_parameters.csv --population 100000 --time-horizon 15 --discount-rate 0.035 --output demo_evppi.txt")
    
    print()
    print("✨ Demo complete!")
    print()
    print("💡 To use voiage with your own data:")
    print("1. Prepare CSV files with your net benefits and parameters")
    print("2. Run: python -m voiage.cli calculate-evpi your_data.csv")
    print("3. Or for EVPPI: python -m voiage.cli calculate-evppi benefits.csv params.csv")
    print()
    print("📖 For more examples, see the documentation at:")
    print("   https://edithatogo.github.io/voiage")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
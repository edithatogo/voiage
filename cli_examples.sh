#!/bin/bash
# voiage CLI Examples
# This script demonstrates the voiage command-line interface using the generated example data

echo "=== voiage CLI Examples ==="
echo

echo "1. Calculate EVPI for the example dataset:"
echo "voiage calculate-evpi example_net_benefits.csv --population 100000 --time_horizon 10 --discount-rate 0.03"
echo

echo "2. Calculate EVPI and save to file:"
echo "voiage calculate-evpi example_net_benefits.csv --output evpi_result.txt"
echo

echo "3. Calculate EVPPI for specific parameters:"
echo "voiage calculate-evppi example_net_benefits.csv example_parameters.csv --population 100000"
echo

echo "4. Full EVPPI analysis with all options:"
echo "voiage calculate-evppi example_net_benefits.csv example_parameters.csv --population 100000 --time_horizon 15 --discount-rate 0.035 --output full_evppi_result.txt"
echo

echo "=== Sample Data Structure ==="
echo
echo "example_net_benefits.csv (first 5 rows):"
head -5 example_net_benefits.csv
echo
echo "example_parameters.csv (first 5 rows):"
head -5 example_parameters.csv
echo

echo "=== Expected Output Format ==="
echo "EVPI: 12345.67"
echo "EVPPI: 8901.23"
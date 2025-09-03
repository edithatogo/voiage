# Command-Line Interface (CLI)

This guide covers the command-line interface for voiage, which allows you to perform VOI analyses without writing Python code.

## Overview

The voiage CLI provides access to core VOI methods through simple command-line commands. This is particularly useful for users who want to quickly analyze their data without writing Python scripts.

## Installation

The CLI is installed automatically with voiage. You can access it by running:

```bash
voiage --help
```

## Basic Commands

### EVPI Calculation

Calculate Expected Value of Perfect Information from net benefit data:

```bash
voiage calculate-evpi net_benefits.csv
```

With population scaling:

```bash
voiage calculate-evpi net_benefits.csv --population 100000 --time-horizon 10 --discount-rate 0.03
```

### EVPPI Calculation

Calculate Expected Value of Partial Perfect Information from net benefit and parameter data:

```bash
voiage calculate-evppi net_benefits.csv parameters.csv
```

With population scaling:

```bash
voiage calculate-evppi net_benefits.csv parameters.csv --population 100000 --time-horizon 10 --discount-rate 0.03
```

## File Formats

### Net Benefits CSV

The net benefits CSV file should have one column per strategy, with rows representing PSA samples:

```csv
Standard Care,New Treatment
100,110
95,115
105,108
...
```

### Parameters CSV

The parameters CSV file should have one column per parameter, with rows representing PSA samples:

```csv
effectiveness,cost
0.7,50
0.8,45
0.6,55
...
```

## Output Options

### Save to File

Save results to a file instead of printing to console:

```bash
voiage calculate-evpi net_benefits.csv --output results.txt
```

### Verbose Output

Get more detailed output:

```bash
voiage calculate-evpi net_benefits.csv --verbose
```

## Advanced Options

### Custom Regression Model

Specify a custom regression model for EVPPI calculations:

```bash
voiage calculate-evppi net_benefits.csv parameters.csv --regression-model RandomForest
```

### Sample Size Control

Control the number of samples used for regression fitting:

```bash
voiage calculate-evppi net_benefits.csv parameters.csv --n-regression-samples 500
```

## Examples

### Simple EVPI Analysis

```bash
# Calculate EVPI
voiage calculate-evpi net_benefits.csv

# Output: EVPI: 15.23
```

### Population-Scaled EVPPI Analysis

```bash
# Calculate population-scaled EVPPI
voiage calculate-evppi net_benefits.csv parameters.csv --population 100000 --time-horizon 10 --discount-rate 0.03

# Output: Population EVPPI: 1523000.00
```

## Best Practices

1. **Data Format**: Ensure your CSV files are properly formatted with headers
2. **Sample Size**: Use sufficient PSA samples for reliable results
3. **Parameter Selection**: Include only relevant parameters in EVPPI analysis
4. **Validation**: Always validate results with domain experts
# Command-Line Interface (CLI) Usage

The voiage library provides a command-line interface for performing Value of Information analyses without writing Python code.

## Installation

After installing voiage, the CLI is available as the `voiage` command:

```bash
pip install voiage
```

## Available Commands

### calculate-evpi

Calculate Expected Value of Perfect Information (EVPI) from input data.

```bash
voiage calculate-evpi [OPTIONS] NET_BENEFIT_FILE
```

**Arguments:**
- `NET_BENEFIT_FILE`: Path to CSV containing net benefits (samples x strategies)

**Options:**
- `--population FLOAT`: Population size for population-adjusted EVPI
- `--discount-rate FLOAT`: Annual discount rate (e.g., 0.03)
- `--time-horizon FLOAT`: Time horizon in years
- `--output, -o PATH`: File to save EVPI result
- `--help`: Show help message

**Example:**
```bash
voiage calculate-evpi net_benefits.csv --population 100000 --time-horizon 10 --discount-rate 0.03
```

### calculate-evppi

Calculate Expected Value of Partial Perfect Information (EVPPI).

```bash
voiage calculate-evppi [OPTIONS] NET_BENEFIT_FILE PARAMETER_FILE
```

**Arguments:**
- `NET_BENEFIT_FILE`: Path to CSV containing net benefits (samples x strategies)
- `PARAMETER_FILE`: Path to CSV for parameters of interest (samples x params)

**Options:**
- `--population FLOAT`: Population size for population-adjusted EVPPI
- `--discount-rate FLOAT`: Annual discount rate (e.g., 0.03)
- `--time-horizon FLOAT`: Time horizon in years
- `--output, -o PATH`: File to save EVPPI result
- `--help`: Show help message

**Example:**
```bash
voiage calculate-evppi net_benefits.csv parameters.csv --population 100000 --time-horizon 10 --discount-rate 0.03
```

## Input File Formats

### Net Benefit File

A CSV file with:
- First row: Strategy names (headers)
- Subsequent rows: Net benefit values for each strategy

Example:
```csv
Strategy A,Strategy B,Strategy C
1000,1200,1100
950,1250,1050
1050,1150,1150
```

### Parameter File

A CSV file with:
- First row: Parameter names (headers)
- Subsequent rows: Parameter values

Example:
```csv
param1,param2
0.5,0.3
0.6,0.4
0.4,0.2
```

## Output

Results are printed to the console by default. Use the `--output` option to save results to a file.

Example output:
```
EVPI: 8530202.836776
```
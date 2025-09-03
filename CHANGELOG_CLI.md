# CLI Implementation Summary

## Overview

This document summarizes the implementation of the command-line interface (CLI) for the voiage library. The CLI allows users to perform Value of Information analyses without writing Python code.

## Features Implemented

### 1. Core CLI Commands

- **calculate-evpi**: Calculate Expected Value of Perfect Information
- **calculate-evppi**: Calculate Expected Value of Partial Perfect Information

### 2. File I/O Functionality

- Reading net benefit data from CSV files
- Reading parameter data from CSV files
- Writing results to output files

### 3. Command-Line Argument Parsing

- Using Typer for robust argument parsing
- Support for positional arguments and optional flags
- Built-in help system

### 4. Population Scaling

- Support for population-adjusted VOI calculations
- Time horizon and discount rate parameters
- Proper annuity factor calculations

## Usage Examples

### Basic EVPI Calculation

```bash
voiage calculate-evpi net_benefits.csv
```

### Population-Adjusted EVPI

```bash
voiage calculate-evpi net_benefits.csv --population 100000 --time-horizon 10 --discount-rate 0.03
```

### EVPPI Calculation

```bash
voiage calculate-evppi net_benefits.csv parameters.csv
```

### Saving Results to File

```bash
voiage calculate-evpi net_benefits.csv --output result.txt
```

## File Formats

### Net Benefit CSV Format

- First row: Strategy names (headers)
- Subsequent rows: Net benefit values for each strategy

Example:
```csv
Strategy A,Strategy B,Strategy C
1000,1200,1100
950,1250,1050
```

### Parameter CSV Format

- First row: Parameter names (headers)
- Subsequent rows: Parameter values

Example:
```csv
effectiveness,cost
0.5,0.3
0.6,0.4
```

## Implementation Details

### Dependencies

- Typer for CLI argument parsing
- Standard voiage modules for calculations
- Built-in Python CSV module for file I/O

### Error Handling

- File not found errors
- Invalid data format errors
- Parameter validation
- Graceful error messages

### Testing

- Unit tests for CLI functions
- Integration tests with sample data
- Error condition testing
- Help system verification

## Files Modified/Added

1. `voiage/cli.py` - Main CLI implementation
2. `pyproject.toml` - Added CLI entry point
3. `tests/test_cli.py` - Basic CLI tests
4. `tests/test_cli_comprehensive.py` - Comprehensive CLI tests
5. `docs/cli_usage.md` - CLI documentation
6. `examples/cli_example.py` - CLI usage examples
7. `voiage/core/io.py` - Updated to match ValueArray requirements

## Future Enhancements

- Add EVSI command
- Add portfolio optimization commands
- Add plotting capabilities
- Support for more file formats (JSON, Excel)
- Configuration file support
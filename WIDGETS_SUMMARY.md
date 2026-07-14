# voiage Jupyter Widgets Summary

## Overview

The voiage Jupyter widgets provide an interactive interface for performing Value of Information (VOI) analysis directly within Jupyter notebooks. This makes it easy for researchers and analysts to explore VOI concepts and perform analyses without writing extensive code.

## Features Implemented

1. **Interactive Data Input**: Upload or manually enter net benefit and parameter data
2. **Analysis Type Selection**: Choose between EVPI and EVPPI calculations
3. **Configuration Controls**: Adjust parameters like population size, time horizon, discount rate, and chunk size
4. **Real-time Calculation**: Perform VOI calculations with a single button click
5. **Visual Feedback**: Status indicators and result display areas
6. **Data Management**: Clear data and start fresh with the click of a button

## Widget Components

### Data Input Section
- **File Upload**: Upload net benefit and parameter data from CSV, text, or numpy files
- **Manual Input**: Text areas for entering data directly
- **Data Validation**: Automatic parsing and validation of input data

### Configuration Section
- **Analysis Type**: Radio buttons to select EVPI or EVPPI
- **Population Input**: Float text field for population size
- **Time Horizon Input**: Float text field for time horizon
- **Discount Rate Slider**: Interactive slider for discount rate (0-20%)
- **Chunk Size Input**: Integer text field for chunk size in incremental computation
- **JIT Compilation**: Checkbox to enable/disable Just-In-Time compilation

### Action Buttons
- **Calculate Button**: Trigger VOI calculations
- **Clear Button**: Clear all input data and results

### Output Areas
- **Status Label**: Real-time status updates
- **Output Area**: Detailed output and error messages
- **Result Area**: Display of calculation results

## Usage Example

```python
# In a Jupyter notebook cell
from voiage.widgets.voi_widgets import create_voi_widget

# Create and display the widget
widget = create_voi_widget()
```

## Data Formats

### Net Benefits
- **File Upload**: CSV or text files with samples as rows and strategies as columns
- **Manual Input**: Comma-separated values with one sample per line

### Parameters
- **File Upload**: CSV or text files with samples as rows and parameters as columns
- **Manual Input**: Key:value pairs with one sample per line

## Technical Implementation

The widgets are built using the `ipywidgets` library and follow these design principles:

1. **Modular Design**: Separate creation, layout, and callback setup
2. **Error Handling**: Comprehensive error handling with user-friendly messages
3. **Data Validation**: Automatic validation of input data formats
4. **State Management**: Proper management of data and analysis state
5. **Responsive UI**: Real-time feedback and status updates

## Dependencies

- `ipywidgets`: Core widget library
- `IPython`: Display functionality
- `traitlets`: Widget trait management
- `voiage`: Core VOI analysis library
- `numpy`: Numerical computing
- `pandas`: Data parsing (optional)

## Future Enhancements

- Add support for EVSI calculations
- Implement data visualization within the widget
- Add export functionality for results
- Include sample datasets for demonstration
- Add advanced configuration options
- Implement real-time plotting of results
- Add support for streaming data analysis
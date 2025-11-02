"""Jupyter widgets for interactive VOI analysis."""

import numpy as np

try:
    from IPython.display import display
    import ipywidgets as widgets
    from traitlets import Bool, Float, Int, Unicode
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from voiage.analysis import DecisionAnalysis
from voiage.config_objects import VOIAnalysisConfig
from voiage.schema import ParameterSet, ValueArray


class VOIAnalysisWidget:
    """Interactive widget for VOI analysis."""

    def __init__(self):
        """Initialize the VOI analysis widget."""
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets is required for interactive widgets. Install with: pip install ipywidgets")

        # Data storage
        self.net_benefits = None
        self.parameters = None
        self.analysis = None

        # Create UI components
        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Data input widgets
        self.upload_net_benefits_button = widgets.FileUpload(
            accept='.csv,.txt,.npy',  # Accept CSV, text, and numpy files
            multiple=False,
            description='Upload Net Benefits'
        )

        self.upload_parameters_button = widgets.FileUpload(
            accept='.csv,.txt,.npy',
            multiple=False,
            description='Upload Parameters'
        )

        # Manual input widgets
        self.net_benefits_text = widgets.Textarea(
            value='',
            placeholder='Enter net benefits as comma-separated values (one sample per line)',
            description='Net Benefits:',
            layout=widgets.Layout(width='400px', height='100px')
        )

        self.parameters_text = widgets.Textarea(
            value='',
            placeholder='Enter parameters as key:value pairs (one sample per line)',
            description='Parameters:',
            layout=widgets.Layout(width='400px', height='100px')
        )

        # Analysis type selection
        self.analysis_type = widgets.RadioButtons(
            options=['EVPI', 'EVPPI'],
            value='EVPI',
            description='Analysis Type:',
            disabled=False
        )

        # Configuration widgets
        self.population_input = widgets.FloatText(
            value=100000,
            description='Population:',
            disabled=False
        )

        self.time_horizon_input = widgets.FloatText(
            value=10,
            description='Time Horizon:',
            disabled=False
        )

        self.discount_rate_input = widgets.FloatSlider(
            value=0.03,
            min=0,
            max=0.2,
            step=0.001,
            description='Discount Rate:',
            disabled=False,
            readout_format='.3f'
        )

        self.chunk_size_input = widgets.IntText(
            value=1000,
            description='Chunk Size:',
            disabled=False
        )

        self.use_jit_checkbox = widgets.Checkbox(
            value=True,
            description='Use JIT Compilation',
            disabled=False
        )

        # Action buttons
        self.calculate_button = widgets.Button(
            description='Calculate VOI',
            disabled=False,
            button_style='success',
            tooltip='Calculate Value of Information'
        )

        self.clear_button = widgets.Button(
            description='Clear Data',
            disabled=False,
            button_style='warning',
            tooltip='Clear all input data'
        )

        # Output areas
        self.output_area = widgets.Output()
        self.result_area = widgets.Output()

        # Status indicator
        self.status_label = widgets.Label(
            value='Ready',
            layout=widgets.Layout(width='200px')
        )

    def _setup_layout(self):
        """Setup the widget layout."""
        # Data input section
        data_input_box = widgets.VBox([
            widgets.HTML("<h3>Data Input</h3>"),
            widgets.HBox([self.upload_net_benefits_button, self.upload_parameters_button]),
            widgets.HTML("<p>Or enter data manually:</p>"),
            self.net_benefits_text,
            self.parameters_text
        ])

        # Configuration section
        config_box = widgets.VBox([
            widgets.HTML("<h3>Configuration</h3>"),
            self.analysis_type,
            self.population_input,
            self.time_horizon_input,
            self.discount_rate_input,
            self.chunk_size_input,
            self.use_jit_checkbox
        ])

        # Action buttons
        buttons_box = widgets.HBox([
            self.calculate_button,
            self.clear_button,
            self.status_label
        ])

        # Main layout
        self.widget_layout = widgets.VBox([
            data_input_box,
            config_box,
            buttons_box,
            self.output_area,
            self.result_area
        ])

    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.calculate_button.on_click(self._on_calculate_clicked)
        self.clear_button.on_click(self._on_clear_clicked)
        self.upload_net_benefits_button.observe(self._on_net_benefits_upload, names='value')
        self.upload_parameters_button.observe(self._on_parameters_upload, names='value')

    def _on_calculate_clicked(self, button):
        """Handle calculate button click."""
        with self.output_area:
            try:
                self.status_label.value = "Calculating..."

                # Parse manual input if no file was uploaded
                if self.net_benefits is None and self.net_benefits_text.value:
                    self._parse_net_benefits_text()

                if self.parameters is None and self.parameters_text.value and self.analysis_type.value == 'EVPPI':
                    self._parse_parameters_text()

                # Validate data
                if self.net_benefits is None:
                    raise ValueError("Net benefits data is required")

                if self.analysis_type.value == 'EVPPI' and self.parameters is None:
                    raise ValueError("Parameter data is required for EVPPI analysis")

                # Create configuration
                config = VOIAnalysisConfig(
                    use_jit=self.use_jit_checkbox.value,
                    chunk_size=self.chunk_size_input.value or None
                )

                # Create analysis
                self.analysis = DecisionAnalysis(
                    nb_array=self.net_benefits,
                    parameter_samples=self.parameters,
                    use_jit=config.use_jit,
                    enable_caching=True
                )

                # Calculate VOI
                if self.analysis_type.value == 'EVPI':
                    result = self.analysis.evpi(
                        population=self.population_input.value or None,
                        time_horizon=self.time_horizon_input.value or None,
                        discount_rate=self.discount_rate_input.value or None,
                        chunk_size=config.chunk_size
                    )
                    method = "EVPI"
                else:
                    result = self.analysis.evppi(
                        population=self.population_input.value or None,
                        time_horizon=self.time_horizon_input.value or None,
                        discount_rate=self.discount_rate_input.value or None,
                        chunk_size=config.chunk_size
                    )
                    method = "EVPPI"

                # Display result
                with self.result_area:
                    self.result_area.clear_output()
                    print(f"{method} Result: {result:.6f}")

                self.status_label.value = "Calculation completed"

            except Exception as e:
                self.status_label.value = "Error occurred"
                with self.result_area:
                    self.result_area.clear_output()
                    print(f"Error: {e!s}")

    def _on_clear_clicked(self, button):
        """Handle clear button click."""
        # Clear data
        self.net_benefits = None
        self.parameters = None
        self.analysis = None

        # Clear inputs
        self.net_benefits_text.value = ''
        self.parameters_text.value = ''

        # Clear outputs
        with self.output_area:
            self.output_area.clear_output()
        with self.result_area:
            self.result_area.clear_output()

        self.status_label.value = "Data cleared"

    def _on_net_benefits_upload(self, change):
        """Handle net benefits file upload."""
        if change['new']:
            try:
                # Get uploaded file content
                uploaded_file = list(change['new'].values())[0]
                filename = uploaded_file['name']
                content = uploaded_file['content']

                # Parse based on file type
                if filename.endswith('.npy'):
                    # Numpy file
                    import io
                    np_data = np.load(io.BytesIO(content))
                    self.net_benefits = ValueArray.from_numpy(np_data)
                else:
                    # Text/CSV file
                    import io

                    import pandas as pd
                    text_content = content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(text_content))
                    np_data = df.values.astype(np.float64)
                    self.net_benefits = ValueArray.from_numpy(np_data)

                self.status_label.value = f"Net benefits loaded from {filename}"

            except Exception as e:
                self.status_label.value = "Error loading net benefits"
                with self.output_area:
                    print(f"Error loading net benefits: {e!s}")

    def _on_parameters_upload(self, change):
        """Handle parameters file upload."""
        if change['new']:
            try:
                # Get uploaded file content
                uploaded_file = list(change['new'].values())[0]
                filename = uploaded_file['name']
                content = uploaded_file['content']

                # Parse based on file type
                if filename.endswith('.npy'):
                    # Numpy file
                    import io
                    np_data = np.load(io.BytesIO(content))
                    # Convert to parameter dict format
                    param_dict = {f"param_{i}": np_data[:, i] for i in range(np_data.shape[1])}
                    self.parameters = ParameterSet.from_numpy_or_dict(param_dict)
                else:
                    # Text/CSV file
                    import io

                    import pandas as pd
                    text_content = content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(text_content))
                    param_dict = {col: df[col].values.astype(np.float64) for col in df.columns}
                    self.parameters = ParameterSet.from_numpy_or_dict(param_dict)

                self.status_label.value = f"Parameters loaded from {filename}"

            except Exception as e:
                self.status_label.value = "Error loading parameters"
                with self.output_area:
                    print(f"Error loading parameters: {e!s}")

    def _parse_net_benefits_text(self):
        """Parse net benefits from text input."""
        try:
            lines = self.net_benefits_text.value.strip().split('\n')
            data = []
            for line in lines:
                if line.strip():
                    values = [float(x.strip()) for x in line.split(',')]
                    data.append(values)

            np_data = np.array(data, dtype=np.float64)
            self.net_benefits = ValueArray.from_numpy(np_data)

        except Exception as e:
            raise ValueError(f"Error parsing net benefits: {e!s}")

    def _parse_parameters_text(self):
        """Parse parameters from text input."""
        try:
            lines = self.parameters_text.value.strip().split('\n')
            param_dict = {}

            # Parse first line to get parameter names
            if lines:
                first_line = lines[0]
                if ':' in first_line:
                    # Key:value format
                    for line in lines:
                        if line.strip():
                            parts = line.split(':')
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = float(parts[1].strip())
                                if key not in param_dict:
                                    param_dict[key] = []
                                param_dict[key].append(value)
                else:
                    # Comma-separated values with assumed parameter names
                    param_names = [f"param_{i}" for i in range(len(lines[0].split(',')))]
                    for line in lines:
                        if line.strip():
                            values = [float(x.strip()) for x in line.split(',')]
                            for i, value in enumerate(values):
                                if i < len(param_names):
                                    key = param_names[i]
                                    if key not in param_dict:
                                        param_dict[key] = []
                                    param_dict[key].append(value)

            # Convert lists to numpy arrays
            for key in param_dict:
                param_dict[key] = np.array(param_dict[key], dtype=np.float64)

            self.parameters = ParameterSet.from_numpy_or_dict(param_dict)

        except Exception as e:
            raise ValueError(f"Error parsing parameters: {e!s}")

    def display(self):
        """Display the widget."""
        display(self.widget_layout)


# Convenience function to create and display the widget
def create_voi_widget():
    """Create and display a VOI analysis widget."""
    widget = VOIAnalysisWidget()
    widget.display()
    return widget


# Example usage in Jupyter notebook:
#
# from voiage.widgets.voi_widgets import create_voi_widget
# widget = create_voi_widget()

"""Tests for Jupyter widgets."""

import pytest

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from voiage.widgets.voi_widgets import VOIAnalysisWidget


@pytest.mark.skipif(not HAS_WIDGETS, reason="ipywidgets not installed")
def test_widget_creation():
    """Test that the widget can be created."""
    widget = VOIAnalysisWidget()
    assert widget is not None
    assert hasattr(widget, 'widget_layout')


@pytest.mark.skipif(not HAS_WIDGETS, reason="ipywidgets not installed")
def test_widget_components():
    """Test that widget components are created correctly."""
    widget = VOIAnalysisWidget()

    # Check that key components exist
    assert hasattr(widget, 'calculate_button')
    assert hasattr(widget, 'clear_button')
    assert hasattr(widget, 'analysis_type')
    assert hasattr(widget, 'population_input')
    assert hasattr(widget, 'time_horizon_input')
    assert hasattr(widget, 'discount_rate_input')
    assert hasattr(widget, 'chunk_size_input')
    assert hasattr(widget, 'use_jit_checkbox')
    assert hasattr(widget, 'net_benefits_text')
    assert hasattr(widget, 'parameters_text')
    assert hasattr(widget, 'upload_net_benefits_button')
    assert hasattr(widget, 'upload_parameters_button')
    assert hasattr(widget, 'output_area')
    assert hasattr(widget, 'result_area')
    assert hasattr(widget, 'status_label')


@pytest.mark.skipif(not HAS_WIDGETS, reason="ipywidgets not installed")
def test_data_parsing():
    """Test parsing of manual input data."""
    widget = VOIAnalysisWidget()

    # Test net benefits parsing
    net_benefits_text = "1.0, 2.0\n3.0, 1.5\n2.5, 2.0"
    widget.net_benefits_text.value = net_benefits_text

    # This would normally be called internally
    # We're just testing that it doesn't raise an exception
    try:
        widget._parse_net_benefits_text()
        parsing_successful = True
    except Exception:
        parsing_successful = False

    assert parsing_successful


@pytest.mark.skipif(not HAS_WIDGETS, reason="ipywidgets not installed")
def test_widget_display():
    """Test that the widget can be displayed."""
    widget = VOIAnalysisWidget()

    # This would normally display the widget in a Jupyter notebook
    # We're just testing that it doesn't raise an exception
    try:
        widget.display()
        display_successful = True
    except Exception:
        display_successful = False

    assert display_successful


if __name__ == "__main__":
    if HAS_WIDGETS:
        test_widget_creation()
        test_widget_components()
        test_data_parsing()
        test_widget_display()
        print("All widget tests passed!")
    else:
        print("Skipping widget tests - ipywidgets not installed")

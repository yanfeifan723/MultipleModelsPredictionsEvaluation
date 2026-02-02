"""
Plotting Module
Provides intelligent plotting functionality similar to Origin software
"""

# Import output configuration
from ..config.output_config import (
    DATA_OUTPUT_DIR, PLOT_OUTPUT_DIR,
    get_data_output_path, get_plot_output_path,
    get_standard_filename
)

# Smart plotting system (main interface)
from .smart_plotter import (
    auto_plot,
    analyze_data_structure,
    plot_scatter,
    plot_line,
    plot_histogram,
    plot_distribution,
    plot_bar
)

# Specialized plotting modules
from .heatmap import (
    plot_heatmap,
    plot_correlation_heatmap,
    plot_multi_value_heatmap,
    plot_comparison_heatmap
)

from .boxplot import (
    plot_boxplot,
    plot_grouped_boxplot
)

# Statistical plots (generic)
from .statistical_plots import (
    plot_timeseries,
    plot_boxplot as plot_statistical_boxplot,
    plot_scatter as plot_statistical_scatter,
    plot_histogram as plot_statistical_histogram
)

# EOF plots
from .eof_plots import (
    plot_eof_modes,
    plot_pc_timeseries,
    plot_eof_variance,
    plot_common_eof_comparison,
    plot_eof_summary,
    plot_ensemble_pc_timeseries,
    plot_ensemble_eof_comparison,
    plot_ensemble_variance_comparison
)

# Spatial plots
from .spatial_plots import (
    plot_spatial_field,
    plot_spatial_comparison,
    plot_spatial_anomaly,
    plot_spatial_correlation
)

# Taylor diagrams
from .taylor_plots import (
    plot_taylor_diagram,
    plot_taylor_grid,
    plot_taylor_summary,
    calculate_taylor_metrics,
    TaylorDiagram
)

# Spectrum analysis
from .spectrum_plots import (
    plot_power_spectrum,
    plot_spectrum_comparison,
    plot_spectrum_grid,
    plot_spectrum_analysis,
    plot_spectrum_heatmap,
    calculate_power_spectrum
)

__all__ = [
    # Smart plotting system (main interface)
    'auto_plot',
    'analyze_data_structure',
    'plot_scatter',
    'plot_line',
    'plot_histogram',
    'plot_distribution',
    'plot_bar',
    
    # Specialized plotting modules
    'plot_heatmap',
    'plot_correlation_heatmap',
    'plot_multi_value_heatmap',
    'plot_comparison_heatmap',
    'plot_boxplot',
    'plot_grouped_boxplot',
    
    # Statistical plots
    'plot_timeseries',
    'plot_statistical_boxplot',
    'plot_statistical_scatter',
    'plot_statistical_histogram',
    
    # EOF plots
    'plot_eof_modes',
    'plot_pc_timeseries',
    'plot_eof_variance',
    'plot_common_eof_comparison',
    'plot_eof_summary',
    'plot_ensemble_pc_timeseries',
    'plot_ensemble_eof_comparison',
    'plot_ensemble_variance_comparison',
    
    # Spatial plots
    'plot_spatial_field',
    'plot_spatial_comparison',
    'plot_spatial_anomaly',
    'plot_spatial_correlation',
    
    # Taylor diagrams
    'plot_taylor_diagram',
    'plot_taylor_grid',
    'plot_taylor_summary',
    'calculate_taylor_metrics',
    'TaylorDiagram',
    
    # Spectrum analysis
    'plot_power_spectrum',
    'plot_spectrum_comparison',
    'plot_spectrum_grid',
    'plot_spectrum_analysis',
    'plot_spectrum_heatmap',
    'calculate_power_spectrum',
]

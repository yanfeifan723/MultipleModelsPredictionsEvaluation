"""
Smart Plotter Module
Provides intelligent plotting functionality similar to Origin software
Automatically detects data types and chooses appropriate plot types
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Tuple, Union, Dict, Any
import logging
from pathlib import Path
from .heatmap import plot_heatmap
from .boxplot import plot_boxplot
from .eof_plots import plot_eof_modes, plot_pc_timeseries, plot_eof_variance
from .spatial_plots import plot_spatial_field, plot_spatial_comparison
from .taylor_plots import plot_taylor_diagram, calculate_taylor_metrics
from .spectrum_plots import plot_power_spectrum, calculate_power_spectrum

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Setup professional plotting style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def save_plot(fig: plt.Figure, save_path, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save figure with professional settings"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Figure saved to: {save_path}")

def analyze_data_structure(data: Union[pd.DataFrame, np.ndarray, List, Dict]) -> Dict[str, Any]:
    """
    Analyze data structure to determine optimal plot type
    
    Returns:
        Dictionary with data analysis results
    """
    analysis = {
        'data_type': type(data).__name__,
        'dimensions': None,
        'shape': None,
        'n_samples': None,
        'n_features': None,
        'has_categorical': False,
        'has_numerical': False,
        'has_time_series': False,
        'correlation_structure': None,
        'suggested_plots': []
    }
    
    if isinstance(data, pd.DataFrame):
        analysis['dimensions'] = 2
        analysis['shape'] = data.shape
        analysis['n_samples'] = len(data)
        analysis['n_features'] = len(data.columns)
        
        # Check for categorical and numerical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        analysis['has_categorical'] = len(categorical_cols) > 0
        analysis['has_numerical'] = len(numerical_cols) > 0
        
        # Check for time series
        if len(numerical_cols) >= 1:
            # Simple heuristic: if first column looks like time series
            first_col = data.iloc[:, 0]
            if first_col.dtype in [np.number] and len(first_col.unique()) > 10:
                analysis['has_time_series'] = True
        
        # Analyze correlation structure
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            analysis['correlation_structure'] = 'correlation_matrix'
        
        # Suggest plot types
        if analysis['has_categorical'] and analysis['has_numerical']:
            if len(categorical_cols) == 1 and len(numerical_cols) >= 1:
                analysis['suggested_plots'].append('boxplot')
                analysis['suggested_plots'].append('violin_plot')
            if len(categorical_cols) >= 2:
                analysis['suggested_plots'].append('heatmap')
        
        if analysis['correlation_structure'] == 'correlation_matrix':
            analysis['suggested_plots'].append('correlation_heatmap')
        
        if analysis['has_time_series']:
            analysis['suggested_plots'].append('line_plot')
            analysis['suggested_plots'].append('scatter_plot')
        
        if len(numerical_cols) >= 2:
            analysis['suggested_plots'].append('scatter_plot')
        
        if len(numerical_cols) >= 1:
            analysis['suggested_plots'].append('histogram')
            analysis['suggested_plots'].append('distribution_plot')
        
        # Check for potential EOF data structure
        if len(numerical_cols) >= 3 and analysis['n_samples'] > 10:
            # If data looks like it could be EOF results (multiple modes)
            analysis['suggested_plots'].append('eof_modes')
            analysis['suggested_plots'].append('pc_timeseries')
        
        # Check for potential spatial data structure
        if len(numerical_cols) >= 2 and analysis['n_samples'] > 5:
            # If data looks like it could be spatial data
            analysis['suggested_plots'].append('spatial_field')
        
        # Check for potential time series data
        if analysis['has_time_series'] and len(numerical_cols) >= 1:
            analysis['suggested_plots'].append('power_spectrum')
    
    elif isinstance(data, np.ndarray):
        analysis['dimensions'] = data.ndim
        analysis['shape'] = data.shape
        
        if data.ndim == 1:
            analysis['n_samples'] = len(data)
            analysis['suggested_plots'].extend(['histogram', 'line_plot', 'distribution_plot'])
        elif data.ndim == 2:
            analysis['n_samples'], analysis['n_features'] = data.shape
            if data.shape[0] == data.shape[1]:
                analysis['suggested_plots'].append('heatmap')
            analysis['suggested_plots'].extend(['scatter_plot', 'correlation_heatmap'])
            
            # Check for potential EOF structure
            if data.shape[1] >= 3 and data.shape[0] > 10:
                analysis['suggested_plots'].extend(['eof_modes', 'pc_timeseries'])
            
            # Check for potential spatial structure
            if data.shape[1] >= 2 and data.shape[0] > 5:
                analysis['suggested_plots'].append('spatial_field')
            
            # Check for potential time series structure
            if data.shape[0] > 10:
                analysis['suggested_plots'].append('power_spectrum')
    
    elif isinstance(data, list):
        analysis['dimensions'] = 1
        analysis['n_samples'] = len(data)
        analysis['suggested_plots'].extend(['boxplot', 'histogram', 'line_plot'])
    
    elif isinstance(data, dict):
        analysis['dimensions'] = 1
        analysis['n_samples'] = len(data)
        analysis['suggested_plots'].extend(['boxplot', 'bar_plot'])
    
    return analysis

def auto_plot(data: Union[pd.DataFrame, np.ndarray, List, Dict],
              plot_type: Optional[str] = None,
              title: Optional[str] = None,
              figsize: Optional[Tuple[float, float]] = None,
              save_path: Optional[str] = None,
              **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Automatically plot data with intelligent plot type selection
    
    Args:
        data: Input data
        plot_type: Specific plot type (if None, auto-select)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for specific plot types
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Analyze data structure
    analysis = analyze_data_structure(data)
    logger.info(f"Data analysis: {analysis}")
    
    # Auto-select plot type if not specified
    if plot_type is None:
        if analysis['suggested_plots']:
            plot_type = analysis['suggested_plots'][0]
            logger.info(f"Auto-selected plot type: {plot_type}")
        else:
            plot_type = 'line_plot'  # Default fallback
            logger.warning("No specific plot type suggested, using line plot")
    
    # Auto-generate title if not provided
    if title is None:
        title = f"{plot_type.replace('_', ' ').title()} of {analysis['data_type']}"
    
    # Auto-determine figure size
    if figsize is None:
        if plot_type in ['heatmap', 'correlation_heatmap']:
            if analysis['shape']:
                n_rows, n_cols = analysis['shape']
                figsize = (max(10, n_cols * 0.8), max(8, n_rows * 0.6))
            else:
                figsize = (12, 8)
        elif plot_type in ['boxplot', 'violin_plot']:
            if analysis['n_samples']:
                figsize = (max(10, analysis['n_samples'] * 0.8), 6)
            else:
                figsize = (10, 6)
        else:
            figsize = (10, 6)
    
    # Create the plot based on type
    if plot_type == 'heatmap':
        return plot_heatmap(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'correlation_heatmap':
        if isinstance(data, pd.DataFrame):
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                return plot_heatmap(data[numerical_cols].corr(), 
                                  title="Correlation Matrix", 
                                  figsize=figsize, save_path=save_path, **kwargs)
        return plot_heatmap(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'boxplot':
        return plot_boxplot(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'scatter_plot':
        return plot_scatter(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'line_plot':
        return plot_line(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'histogram':
        return plot_histogram(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'distribution_plot':
        return plot_distribution(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'bar_plot':
        return plot_bar(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'eof_modes':
        # For EOF modes, assume data is in the format (n_modes, n_spatial_points)
        return plot_eof_modes(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'pc_timeseries':
        # For PC time series, assume data is in the format (n_time, n_modes)
        return plot_pc_timeseries(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
    
    elif plot_type == 'spatial_field':
        # For spatial data, assume data is 2D with lat/lon coordinates
        if data.ndim == 2:
            # Create dummy coordinates if not provided
            lats = np.arange(data.shape[0])
            lons = np.arange(data.shape[1])
            return plot_spatial_field(data, lats, lons, title=title, figsize=figsize, save_path=save_path, **kwargs)
        else:
            raise ValueError("Spatial field data must be 2D")
    
    elif plot_type == 'power_spectrum':
        # For power spectrum, assume data is 1D time series
        if data.ndim == 1:
            return plot_power_spectrum(data, title=title, figsize=figsize, save_path=save_path, **kwargs)
        else:
            raise ValueError("Power spectrum data must be 1D time series")
    
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

def plot_scatter(data: Union[pd.DataFrame, np.ndarray],
                x_col: Optional[str] = None,
                y_col: Optional[str] = None,
                title: str = "Scatter Plot",
                figsize: Tuple[float, float] = (10, 6),
                save_path: Optional[str] = None,
                **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot scatter plot"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        if x_col is None:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                x_col, y_col = numerical_cols[0], numerical_cols[1]
            else:
                raise ValueError("Need at least 2 numerical columns for scatter plot")
        
        ax.scatter(data[x_col], data[y_col], alpha=0.6, **kwargs)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    
    elif isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] >= 2:
            ax.scatter(data[:, 0], data[:, 1], alpha=0.6, **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            raise ValueError("Need 2D array with at least 2 columns for scatter plot")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_line(data: Union[pd.DataFrame, np.ndarray, List],
              title: str = "Line Plot",
              figsize: Tuple[float, float] = (10, 6),
              save_path: Optional[str] = None,
              **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot line plot"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 1:
            y_data = data[numerical_cols[0]]
            x_data = range(len(y_data))
            ax.plot(x_data, y_data, **kwargs)
            ax.set_xlabel("Index")
            ax.set_ylabel(numerical_cols[0])
    
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            ax.plot(data, **kwargs)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
        elif data.ndim == 2:
            ax.plot(data[:, 0], data[:, 1], **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
    
    elif isinstance(data, list):
        ax.plot(data, **kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_histogram(data: Union[pd.DataFrame, np.ndarray, List],
                  title: str = "Histogram",
                  figsize: Tuple[float, float] = (10, 6),
                  save_path: Optional[str] = None,
                  **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histogram"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 1:
            ax.hist(data[numerical_cols[0]], bins=30, alpha=0.7, edgecolor='black', **kwargs)
            ax.set_xlabel(numerical_cols[0])
    
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black', **kwargs)
            ax.set_xlabel("Value")
    
    elif isinstance(data, list):
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', **kwargs)
        ax.set_xlabel("Value")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_distribution(data: Union[pd.DataFrame, np.ndarray, List],
                     title: str = "Distribution Plot",
                     figsize: Tuple[float, float] = (10, 6),
                     save_path: Optional[str] = None,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot distribution plot using seaborn"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 1:
            sns.histplot(data=data, x=numerical_cols[0], kde=True, ax=ax, **kwargs)
            ax.set_xlabel(numerical_cols[0])
    
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            sns.histplot(data=data, kde=True, ax=ax, **kwargs)
            ax.set_xlabel("Value")
    
    elif isinstance(data, list):
        sns.histplot(data=data, kde=True, ax=ax, **kwargs)
        ax.set_xlabel("Value")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_bar(data: Union[pd.DataFrame, Dict],
             title: str = "Bar Plot",
             figsize: Tuple[float, float] = (10, 6),
             save_path: Optional[str] = None,
             **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot bar plot"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, dict):
        categories = list(data.keys())
        values = list(data.values())
        ax.bar(categories, values, alpha=0.7, **kwargs)
        ax.set_xlabel("Categories")
        ax.set_ylabel("Values")
    
    elif isinstance(data, pd.DataFrame):
        # For DataFrame, use first categorical column vs first numerical column
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
            cat_col = categorical_cols[0]
            num_col = numerical_cols[0]
            
            grouped_data = data.groupby(cat_col)[num_col].mean()
            ax.bar(grouped_data.index, grouped_data.values, alpha=0.7, **kwargs)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"Mean {num_col}")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

if __name__ == "__main__":
    # Example usage
    print("Smart Plotter module loaded")

"""
Boxplot Plotting Module
Provides general boxplot visualization functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Tuple, Union, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Setup plotting style"""
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
    """Save figure"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Figure saved to: {save_path}")

def plot_boxplot(data: Union[pd.DataFrame, List[np.ndarray], Dict[str, np.ndarray]],
                x: Optional[str] = None,
                y: Optional[str] = None,
                hue: Optional[str] = None,
                title: str = "Boxplot",
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                palette: Optional[str] = None,
                show_stats: Optional[bool] = None,
                reference_line: Optional[float] = None,
                save_path: Optional[str] = None,
                auto_adjust: bool = True,
                **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a general boxplot
    
    Args:
        data: Input data (DataFrame, list of arrays, or dict)
        x: Column name for x-axis grouping (for DataFrame)
        y: Column name for y-axis values (for DataFrame)
        hue: Column name for additional grouping (for DataFrame)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        palette: Color palette
        show_stats: Whether to show statistics
        reference_line: Reference line value
        save_path: Path to save the plot
        **kwargs: Additional arguments for seaborn.boxplot
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-adjust parameters based on data characteristics
    if auto_adjust:
        # Auto-determine figure size based on number of groups
        if figsize is None:
            if isinstance(data, pd.DataFrame):
                n_groups = len(data[x].unique()) if x else 1
            elif isinstance(data, list):
                n_groups = len(data)
            elif isinstance(data, dict):
                n_groups = len(data)
            else:
                n_groups = 1
            
            if n_groups <= 3:
                figsize = (8, 6)
            elif n_groups <= 8:
                figsize = (10, 6)
            else:
                figsize = (max(12, n_groups * 0.8), 6)
        
        # Auto-determine palette based on number of groups
        if palette is None:
            if isinstance(data, pd.DataFrame):
                n_groups = len(data[x].unique()) if x else 1
            elif isinstance(data, list):
                n_groups = len(data)
            elif isinstance(data, dict):
                n_groups = len(data)
            else:
                n_groups = 1
            
            if n_groups <= 4:
                palette = "Set2"
            elif n_groups <= 8:
                palette = "husl"
            else:
                palette = "tab20"
        
        # Auto-determine whether to show stats
        if show_stats is None:
            if isinstance(data, pd.DataFrame):
                n_groups = len(data[x].unique()) if x else 1
            elif isinstance(data, list):
                n_groups = len(data)
            elif isinstance(data, dict):
                n_groups = len(data)
            else:
                n_groups = 1
            show_stats = n_groups <= 6  # Only show stats for reasonable number of groups
        
        # Auto-determine labels
        if xlabel is None and isinstance(data, pd.DataFrame) and x:
            xlabel = x
        elif xlabel is None:
            xlabel = "Groups"
        
        if ylabel is None and isinstance(data, pd.DataFrame) and y:
            ylabel = y
        elif ylabel is None:
            ylabel = "Values"
    else:
        # Use default values if not auto-adjusting
        if figsize is None:
            figsize = (10, 6)
        if palette is None:
            palette = "Set2"
        if show_stats is None:
            show_stats = True
        if xlabel is None:
            xlabel = "Groups"
        if ylabel is None:
            ylabel = "Values"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        # DataFrame input
        if x is None or y is None:
            raise ValueError("x and y must be specified for DataFrame input")
        
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=palette, **kwargs)
        
        # Add statistics if requested
        if show_stats:
            for i, group in enumerate(data[x].unique()):
                group_data = data[data[x] == group][y].dropna()
                if not group_data.empty:
                    mean_val = group_data.mean()
                    ax.text(i, ax.get_ylim()[1] * 0.95, f'Mean: {mean_val:.3f}', 
                           ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    elif isinstance(data, list):
        # List of arrays input
        labels = [f'Group_{i+1}' for i in range(len(data))]
        plot_data = []
        plot_labels = []
        
        for i, arr in enumerate(data):
            arr_clean = np.array(arr)[~np.isnan(arr)]
            if len(arr_clean) > 0:
                plot_data.extend(arr_clean)
                plot_labels.extend([labels[i]] * len(arr_clean))
        
        if plot_data:
            plot_df = pd.DataFrame({'Values': plot_data, 'Groups': plot_labels})
            sns.boxplot(data=plot_df, x='Groups', y='Values', ax=ax, palette=palette, **kwargs)
            
            # Add statistics
            if show_stats:
                for i, label in enumerate(labels):
                    group_data = plot_df[plot_df['Groups'] == label]['Values']
                    if not group_data.empty:
                        mean_val = group_data.mean()
                        ax.text(i, ax.get_ylim()[1] * 0.95, f'Mean: {mean_val:.3f}', 
                               ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    elif isinstance(data, dict):
        # Dictionary input
        labels = list(data.keys())
        plot_data = []
        plot_labels = []
        
        for label, arr in data.items():
            arr_clean = np.array(arr)[~np.isnan(arr)]
            if len(arr_clean) > 0:
                plot_data.extend(arr_clean)
                plot_labels.extend([label] * len(arr_clean))
        
        if plot_data:
            plot_df = pd.DataFrame({'Values': plot_data, 'Groups': plot_labels})
            sns.boxplot(data=plot_df, x='Groups', y='Values', ax=ax, palette=palette, **kwargs)
            
            # Add statistics
            if show_stats:
                for i, label in enumerate(labels):
                    group_data = plot_df[plot_df['Groups'] == label]['Values']
                    if not group_data.empty:
                        mean_val = group_data.mean()
                        ax.text(i, ax.get_ylim()[1] * 0.95, f'Mean: {mean_val:.3f}', 
                               ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    else:
        raise ValueError("data must be DataFrame, list of arrays, or dict")
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels if needed
    if len(ax.get_xticklabels()) > 5:
        ax.tick_params(axis='x', rotation=45)
    
    # Add reference line if specified
    if reference_line is not None:
        ax.axhline(y=reference_line, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_grouped_boxplot(data: pd.DataFrame,
                        group_col: str,
                        value_col: str,
                        subgroup_col: Optional[str] = None,
                        title: str = "Grouped Boxplot",
                        figsize: Tuple[float, float] = (12, 8),
                        save_path: Optional[str] = None,
                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot grouped boxplot with optional subgrouping
    
    Args:
        data: Input DataFrame
        group_col: Column name for main grouping
        value_col: Column name for values
        subgroup_col: Column name for subgrouping (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for seaborn.boxplot
        
    Returns:
        Figure and axes objects
    """
    return plot_boxplot(data, x=group_col, y=value_col, hue=subgroup_col,
                       title=title, xlabel=group_col, ylabel=value_col,
                       figsize=figsize, save_path=save_path, **kwargs)

if __name__ == "__main__":
    # Example usage
    print("Boxplot plotting module loaded")

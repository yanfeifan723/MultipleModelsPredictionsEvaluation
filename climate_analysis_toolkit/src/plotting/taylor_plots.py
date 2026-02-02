"""
Taylor Diagram Module
Provides Taylor diagram functionality for model performance evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import matplotlib.patches as patches
from ..config.output_config import get_plot_output_path, get_standard_filename

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Setup professional plotting style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def save_plot(fig: plt.Figure, save_path, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save figure with professional settings"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Figure saved to: {save_path}")

def get_taylor_plot_path(var_type: str, plot_method: str, model: str = None, 
                        leadtime: int = None) -> Path:
    """Get standardized Taylor plot output path"""
    filename = get_standard_filename("taylor", var_type, plot_method, model, leadtime)
    return get_plot_output_path("taylor", var_type, plot_method, filename)

def calculate_taylor_metrics(obs: np.ndarray, model: np.ndarray) -> Dict[str, float]:
    """
    Calculate Taylor diagram metrics
    
    Args:
        obs: observation data
        model: model data
        
    Returns:
        Dictionary with correlation, std_ratio, rmse, and centered_rmse
    """
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs_clean = obs[mask]
    model_clean = model[mask]
    
    if len(obs_clean) < 2:
        return {'correlation': np.nan, 'std_ratio': np.nan, 'rmse': np.nan, 'centered_rmse': np.nan}
    
    # Calculate metrics
    correlation = np.corrcoef(obs_clean, model_clean)[0, 1]
    std_obs = np.std(obs_clean)
    std_model = np.std(model_clean)
    std_ratio = std_model / std_obs if std_obs > 0 else np.nan
    
    # RMSE
    rmse = np.sqrt(np.mean((model_clean - obs_clean) ** 2))
    
    # Centered RMSE (RMSE of anomalies)
    obs_anom = obs_clean - np.mean(obs_clean)
    model_anom = model_clean - np.mean(model_clean)
    centered_rmse = np.sqrt(np.mean((model_anom - obs_anom) ** 2))
    
    return {
        'correlation': correlation,
        'std_ratio': std_ratio,
        'rmse': rmse,
        'centered_rmse': centered_rmse
    }

class TaylorDiagram:
    """Taylor Diagram class for model performance visualization"""
    
    def __init__(self, ref_std: float = 1.0, fig: Optional[plt.Figure] = None, 
                 rect: int = 111, label: str = 'Reference', 
                 srange: Tuple[float, float] = (0, 1.5), ax: Optional[plt.Axes] = None):
        """
        Initialize Taylor diagram
        
        Args:
            ref_std: reference standard deviation
            fig: matplotlib figure
            rect: subplot position
            label: reference label
            srange: standard deviation range
            ax: matplotlib axes
        """
        self.ref_std = ref_std
        self.srange = srange
        
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            self.fig = fig or plt.figure()
            self.ax = self.fig.add_subplot(rect, projection='polar')
        
        self.smin = srange[0] * ref_std
        self.smax = srange[1] * ref_std
        
        # Setup the plot
        self._setup_plot()
        
        # Add reference point
        self.ax.plot([0], [ref_std], 'ko', markersize=8, label=label)
        
        # Add correlation contours
        self._add_correlation_contours()
        
        # Add std ratio contours
        self._add_std_contours()
        
        # Add RMSE contours
        self._add_rmse_contours()
    
    def _setup_plot(self):
        """Setup the polar plot"""
        # Set limits
        self.ax.set_ylim(self.smin, self.smax)
        self.ax.set_xlim(0, np.pi/2)
        
        # Set ticks
        self.ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2])
        self.ax.set_xticklabels(['1', '0.87', '0.5', '0'])
        
        # Set y ticks
        yticks = np.linspace(self.smin, self.smax, 5)
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels([f'{x:.2f}' for x in yticks])
        
        # Add reference line
        self.ax.plot([0, 0], [self.smin, self.smax], 'k-', linewidth=1)
        
        # Add reference circle
        circle = Circle((0, 0), self.ref_std, fill=False, color='k', linestyle='--', alpha=0.5)
        self.ax.add_patch(circle)
    
    def _add_correlation_contours(self, levels: Optional[List[float]] = None):
        """Add correlation contours"""
        if levels is None:
            levels = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
        
        for corr in levels:
            if corr < 1:
                theta = np.linspace(0, np.arccos(corr), 100)
                r = self.ref_std * np.cos(theta) / corr
                self.ax.plot(theta, r, 'k--', alpha=0.3, linewidth=0.5)
    
    def _add_std_contours(self, levels: Optional[List[float]] = None):
        """Add standard deviation ratio contours"""
        if levels is None:
            levels = [0.5, 0.7, 1.0, 1.3, 1.5]
        
        for std_ratio in levels:
            if std_ratio != 1.0:
                theta = np.linspace(0, np.pi/2, 100)
                r = std_ratio * self.ref_std * np.ones_like(theta)
                self.ax.plot(theta, r, 'k:', alpha=0.3, linewidth=0.5)
    
    def _add_rmse_contours(self, levels: Optional[List[float]] = None):
        """Add RMSE contours"""
        if levels is None:
            levels = [0.2*self.ref_std, 0.4*self.ref_std, 0.6*self.ref_std, 0.8*self.ref_std, 1.0*self.ref_std]
        
        theta = np.linspace(0, np.pi/2, 200)
        for rmse in levels:
            valid = rmse**2 >= (self.ref_std**2) * (np.sin(theta)**2)
            r = np.full_like(theta, np.nan)
            r[valid] = (self.ref_std * np.cos(theta[valid]) + 
                       np.sqrt(rmse**2 - (self.ref_std**2) * (np.sin(theta[valid])**2)))
            self.ax.plot(theta, r, color='gray', ls='-', alpha=0.3, linewidth=0.5)
    
    def add_sample(self, std: float, corr: float, label: str, 
                  marker: str = 'o', color: str = 'b', markersize: int = 6) -> plt.Line2D:
        """
        Add a sample point to the Taylor diagram
        
        Args:
            std: standard deviation
            corr: correlation coefficient
            label: sample label
            marker: marker style
            color: marker color
            markersize: marker size
            
        Returns:
            Line2D object
        """
        theta = np.arccos(np.clip(corr, -1, 1))
        h = self.ax.plot([theta], [std], marker=marker, color=color, 
                        label=label, markersize=markersize, linestyle='None')[0]
        return h
    
    def add_legend(self, **kwargs):
        """Add legend to the plot"""
        self.ax.legend(**kwargs)

def plot_taylor_diagram(obs_data: np.ndarray,
                       model_data_dict: Dict[str, np.ndarray],
                       ref_std: Optional[float] = None,
                       title: str = "Taylor Diagram",
                       figsize: Tuple[float, float] = (10, 8),
                       save_path: Optional[str] = None,
                       **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a Taylor diagram comparing multiple models
    
    Args:
        obs_data: observation data
        model_data_dict: dictionary of model data {model_name: data}
        ref_std: reference standard deviation (if None, uses obs std)
        title: plot title
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for TaylorDiagram
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Calculate reference standard deviation
    if ref_std is None:
        ref_std = np.std(obs_data)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create Taylor diagram
    taylor = TaylorDiagram(ref_std=ref_std, fig=fig, **kwargs)
    
    # Add samples for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_data_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
    
    for i, (model_name, model_data) in enumerate(model_data_dict.items()):
        # Calculate metrics
        metrics = calculate_taylor_metrics(obs_data, model_data)
        
        if not np.isnan(metrics['correlation']) and not np.isnan(metrics['std_ratio']):
            taylor.add_sample(
                std=metrics['std_ratio'] * ref_std,
                corr=metrics['correlation'],
                label=f"{model_name} (r={metrics['correlation']:.2f}, σ={metrics['std_ratio']:.2f})",
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)]
            )
    
    # Add legend
    taylor.add_legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Set title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, taylor.ax

def plot_taylor_grid(obs_data: np.ndarray,
                    model_data_dict: Dict[str, np.ndarray],
                    ref_std: Optional[float] = None,
                    title: str = "Taylor Diagram Grid",
                    figsize: Optional[Tuple[float, float]] = None,
                    save_path: Optional[str] = None,
                    **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of Taylor diagrams for different variables or time periods
    
    Args:
        obs_data: observation data (can be 2D with multiple variables/periods)
        model_data_dict: dictionary of model data
        ref_std: reference standard deviation
        title: plot title
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for TaylorDiagram
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Determine grid dimensions
    if obs_data.ndim == 1:
        n_vars = 1
        obs_data = obs_data.reshape(1, -1)
    else:
        n_vars = obs_data.shape[0]
    
    # Auto-determine figure size
    if figsize is None:
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            subplot_kw={'projection': 'polar'})
    
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Calculate reference standard deviation
    if ref_std is None:
        ref_std = np.std(obs_data[0])
    
    # Create Taylor diagrams for each variable
    for i in range(n_vars):
        ax = axes[i]
        
        # Create Taylor diagram
        taylor = TaylorDiagram(ref_std=ref_std, ax=ax, **kwargs)
        
        # Add samples for each model
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_data_dict)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
        
        for j, (model_name, model_data) in enumerate(model_data_dict.items()):
            # Handle 2D model data
            if model_data.ndim > 1:
                model_data_var = model_data[i]
            else:
                model_data_var = model_data
            
            # Calculate metrics
            metrics = calculate_taylor_metrics(obs_data[i], model_data_var)
            
            if not np.isnan(metrics['correlation']) and not np.isnan(metrics['std_ratio']):
                taylor.add_sample(
                    std=metrics['std_ratio'] * ref_std,
                    corr=metrics['correlation'],
                    label=f"{model_name}",
                    color=colors[j % len(colors)],
                    marker=markers[j % len(markers)]
                )
        
        # Set title for subplot
        ax.set_title(f"Variable {i+1}", fontsize=12, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    # Add legend to the last subplot
    if n_vars > 0:
        axes[n_vars-1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    # Set main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_vars]

def plot_taylor_summary(obs_data: np.ndarray,
                       model_data_dict: Dict[str, np.ndarray],
                       ref_std: Optional[float] = None,
                       title: str = "Taylor Diagram Summary",
                       figsize: Tuple[float, float] = (12, 10),
                       save_path: Optional[str] = None,
                       **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a comprehensive Taylor diagram with summary statistics
    
    Args:
        obs_data: observation data
        model_data_dict: dictionary of model data
        ref_std: reference standard deviation
        title: plot title
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for TaylorDiagram
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Calculate reference standard deviation
    if ref_std is None:
        ref_std = np.std(obs_data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main Taylor diagram
    ax_taylor = fig.add_subplot(gs[0, :], projection='polar')
    taylor = TaylorDiagram(ref_std=ref_std, ax=ax_taylor, **kwargs)
    
    # Add samples and collect metrics
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_data_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
    all_metrics = {}
    
    for i, (model_name, model_data) in enumerate(model_data_dict.items()):
        metrics = calculate_taylor_metrics(obs_data, model_data)
        all_metrics[model_name] = metrics
        
        if not np.isnan(metrics['correlation']) and not np.isnan(metrics['std_ratio']):
            taylor.add_sample(
                std=metrics['std_ratio'] * ref_std,
                corr=metrics['correlation'],
                label=model_name,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)]
            )
    
    # Add legend
    taylor.add_legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax_taylor.set_title("Taylor Diagram", fontsize=14, fontweight='bold', pad=20)
    
    # Correlation bar plot
    ax_corr = fig.add_subplot(gs[1, 0])
    model_names = list(all_metrics.keys())
    correlations = [all_metrics[name]['correlation'] for name in model_names]
    
    bars = ax_corr.bar(model_names, correlations, color=colors[:len(model_names)])
    ax_corr.set_ylabel('Correlation Coefficient')
    ax_corr.set_title('Correlation Comparison')
    ax_corr.tick_params(axis='x', rotation=45)
    ax_corr.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        if not np.isnan(corr):
            ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{corr:.2f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE bar plot
    ax_rmse = fig.add_subplot(gs[1, 1])
    rmses = [all_metrics[name]['rmse'] for name in model_names]
    
    bars = ax_rmse.bar(model_names, rmses, color=colors[:len(model_names)])
    ax_rmse.set_ylabel('RMSE')
    ax_rmse.set_title('RMSE Comparison')
    ax_rmse.tick_params(axis='x', rotation=45)
    ax_rmse.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        if not np.isnan(rmse):
            ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rmse:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Set main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, np.array([ax_taylor, ax_corr, ax_rmse])

if __name__ == "__main__":
    print("Taylor diagram module loaded")

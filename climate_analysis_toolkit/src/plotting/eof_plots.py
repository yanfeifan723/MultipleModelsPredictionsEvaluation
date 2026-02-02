"""
EOF Plotting Module
Provides general EOF (Empirical Orthogonal Function) visualization functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from typing import Optional, List, Tuple, Union, Dict, Any
import logging
from pathlib import Path
import geopandas as gpd
from ..config.output_config import get_plot_output_path, get_standard_filename

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

def get_eof_plot_path(var_type: str, plot_method: str, model: str = None, 
                     leadtime: int = None, mode: int = None) -> Path:
    """Get standardized EOF plot output path"""
    filename = get_standard_filename("eof", var_type, plot_method, model, leadtime, mode)
    return get_plot_output_path("eof", var_type, plot_method, filename)

def load_boundary_data(boundary_file: Optional[str] = None):
    """Load country boundary data"""
    if boundary_file and Path(boundary_file).exists():
        try:
            gdf = gpd.read_file(boundary_file)
            logger.info(f"Loaded boundary data: {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.warning(f"Failed to load boundary file: {e}")
    
    return None

def plot_eof_modes(eof_modes: Union[np.ndarray, List[np.ndarray]],
                   spatial_coords: Optional[Dict[str, np.ndarray]] = None,
                   mode_indices: Optional[List[int]] = None,
                   n_modes: int = 4,
                   title: str = "EOF Modes",
                   figsize: Optional[Tuple[float, float]] = None,
                   cmap: str = 'RdBu_r',
                   boundary_file: Optional[str] = None,
                   explained_variance: Optional[np.ndarray] = None,
                   save_path: Optional[str] = None,
                   **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot EOF spatial modes
    
    Args:
        eof_modes: EOF modes array (n_modes, n_spatial_points) or list of arrays
        spatial_coords: Dictionary with spatial coordinates {'lat': lats, 'lon': lons}
        mode_indices: Specific mode indices to plot (0-based)
        n_modes: Number of modes to plot (if mode_indices not specified)
        title: Plot title
        figsize: Figure size
        cmap: Color map for spatial patterns
        boundary_file: Path to boundary shapefile
        explained_variance: Array of explained variance ratios
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert to numpy array if needed
    if isinstance(eof_modes, list):
        eof_modes = np.array(eof_modes)
    
    # Determine number of modes to plot
    if mode_indices is None:
        mode_indices = list(range(min(n_modes, eof_modes.shape[0])))
    
    n_plot_modes = len(mode_indices)
    
    # Auto-determine figure size with optimal square-like arrangement
    if figsize is None:
        # Find the optimal arrangement that is closest to square
        if n_plot_modes <= 4:
            # For small numbers, use simple arrangements
            if n_plot_modes == 1:
                n_cols, n_rows = 1, 1
            elif n_plot_modes == 2:
                n_cols, n_rows = 2, 1
            elif n_plot_modes == 3:
                n_cols, n_rows = 3, 1
            elif n_plot_modes == 4:
                n_cols, n_rows = 2, 2
        else:
            # For larger numbers, find the arrangement closest to square
            # Calculate the square root to get an approximate square arrangement
            sqrt_n = np.sqrt(n_plot_modes)
            
            # Find the best arrangement by testing different combinations
            best_ratio = float('inf')
            best_cols = int(sqrt_n)
            best_rows = int(np.ceil(n_plot_modes / best_cols))
            
            # Test different column numbers around the square root
            for test_cols in range(max(1, int(sqrt_n) - 2), min(n_plot_modes + 1, int(sqrt_n) + 3)):
                test_rows = int(np.ceil(n_plot_modes / test_cols))
                if test_rows * test_cols >= n_plot_modes:  # Ensure we can fit all modes
                    # Calculate aspect ratio (closer to 1 is better)
                    aspect_ratio = abs(test_cols / test_rows - 1)
                    if aspect_ratio < best_ratio:
                        best_ratio = aspect_ratio
                        best_cols = test_cols
                        best_rows = test_rows
            
            n_cols, n_rows = best_cols, best_rows
        
        # Calculate figure size based on the arrangement
        # Adjust aspect ratio for better visualization
        base_width = 3.5  # Reduced from 4.0
        base_height = 3.0  # Reduced from 3.5
        
        # For square-like arrangements, use more balanced dimensions
        if abs(n_cols - n_rows) <= 1:
            base_width = 3.3  # Reduced from 3.8
            base_height = 2.8  # Reduced from 3.2
        
        # Add extra width for colorbar to prevent overlap
        extra_width = 0.8 if n_plot_modes > 0 else 0.0  # Reduced from 1.2
        
        figsize = (base_width * n_cols + extra_width, base_height * n_rows)
    
    # Create subplots with map projection if spatial coordinates are available
    if spatial_coords is not None:
        # Use map projection for geographical plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        # Use regular subplots for non-geographical plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_plot_modes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Remove subplot borders
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Auto-determine color range
    vmax = np.nanpercentile(np.abs(eof_modes), 95)
    vmin = -vmax
    
    # Load boundary data if provided
    boundary_gdf = load_boundary_data(boundary_file)
    
    for i, mode_idx in enumerate(mode_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        mode_data = eof_modes[mode_idx]
        
        if spatial_coords is not None:
            # 2D spatial plot with map projection
            lats = spatial_coords.get('lat', spatial_coords.get('latitude', None))
            lons = spatial_coords.get('lon', spatial_coords.get('longitude', None))
            
            if lats is not None and lons is not None:
                # Reshape to 2D if needed
                if mode_data.ndim == 1:
                    # Try to infer 2D shape
                    n_points = len(mode_data)
                    if len(lats) * len(lons) == n_points:
                        mode_data_2d = mode_data.reshape(len(lats), len(lons))
                    else:
                        # Fallback to 1D plot
                        mode_data_2d = mode_data
                else:
                    mode_data_2d = mode_data
                
                if mode_data_2d.ndim == 2:
                    # Plot with map projection
                    im = ax.contourf(lons, lats, mode_data_2d, 
                                   levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(), **kwargs)
                    
                    # Add map features
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                    ax.add_feature(cfeature.OCEAN, alpha=0.1)
                    ax.add_feature(cfeature.LAND, alpha=0.1)
                    
                    # Add custom boundaries if provided
                    if boundary_gdf is not None:
                        ax.add_geometries(
                            boundary_gdf.geometry,
                            crs=ccrs.PlateCarree(),
                            edgecolor='black',
                            facecolor='none',
                            linewidth=0.8
                        )
                    else:
                        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
                    
                    # Add gridlines
                    gl = ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=0.5,
                        color='gray',
                        alpha=0.5,
                        linestyle='--'
                    )
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                    
                    # Only show axis labels for leftmost and bottom plots
                    if i % n_cols == 0:  # Leftmost column
                        ax.set_ylabel('Latitude', fontsize=10)
                    else:
                        ax.set_ylabel('')
                    
                    if i >= n_plot_modes - n_cols or i == n_plot_modes - 1:  # Bottom row or last plot
                        ax.set_xlabel('Longitude', fontsize=10)
                    else:
                        ax.set_xlabel('')
                    
                    # Remove tick labels except for leftmost and bottom plots
                    if i % n_cols != 0:  # Not leftmost column
                        gl.left_labels = False
                    if i < n_plot_modes - n_cols and i != n_plot_modes - 1:  # Not bottom row
                        gl.bottom_labels = False
                        
                else:
                    # 1D spatial plot
                    ax.plot(mode_data_2d, **kwargs)
                    ax.set_xlabel('Spatial Index')
            else:
                # 1D plot
                ax.plot(mode_data, **kwargs)
                ax.set_xlabel('Spatial Index')
        else:
            # 1D plot without coordinates
            ax.plot(mode_data, **kwargs)
            ax.set_xlabel('Spatial Index')
        
        # Remove title
        # ax.set_title(f'EOF Mode {mode_idx + 1}', fontweight='bold')
        # ax.set_ylabel('Amplitude')
        if spatial_coords is None:
            ax.grid(True, alpha=0.3)
        
        # Add mode number annotation
        if spatial_coords is not None:
            # For map plots, add text annotation above the subplot
            mode_label = f'Mode {mode_idx + 1}'
            if explained_variance is not None and mode_idx < len(explained_variance):
                variance_pct = explained_variance[mode_idx] * 100
                mode_label += f' ({variance_pct:.1f}%)'
            
            ax.text(0.5, 1.02, mode_label, 
                   transform=ax.transAxes, 
                   fontsize=10, fontweight='bold',
                   verticalalignment='bottom',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            # For non-map plots, add title
            mode_label = f'Mode {mode_idx + 1}'
            if explained_variance is not None and mode_idx < len(explained_variance):
                variance_pct = explained_variance[mode_idx] * 100
                mode_label += f' ({variance_pct:.1f}%)'
            
            ax.set_title(mode_label, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_plot_modes, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar for 2D plots with better positioning
    if spatial_coords is not None and 'im' in locals():
        # Position colorbar in the right margin area, avoid overlap
        cbar = plt.colorbar(im, ax=axes[:n_plot_modes], shrink=0.9, pad=0.05, 
                           location='right', fraction=0.05)
        cbar.set_label('EOF Amplitude', fontsize=9)  # Reduced font size
        cbar.ax.tick_params(labelsize=7)  # Reduced tick label size
        cbar.ax.set_ylabel('EOF Amplitude', fontsize=9)  # Reduced font size
    
    # Remove main title
    # fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout with more vertical spacing and right margin for colorbar
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.82, top=0.85, 
                       hspace=0.05, wspace=0.1)  # Reduced top and bottom margins
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_plot_modes]

def plot_pc_timeseries(pc_data: Union[np.ndarray, List[np.ndarray]],
                      time_coords: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                      mode_indices: Optional[List[int]] = None,
                      n_modes: int = 4,
                      title: str = "Principal Component Time Series",
                      figsize: Optional[Tuple[float, float]] = None,
                      colors: Optional[List[str]] = None,
                      save_path: Optional[str] = None,
                      **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot Principal Component time series
    
    Args:
        pc_data: PC time series array (n_time, n_modes) or list of arrays
        time_coords: Time coordinates (array or DatetimeIndex)
        mode_indices: Specific mode indices to plot (0-based)
        n_modes: Number of modes to plot (if mode_indices not specified)
        title: Plot title
        figsize: Figure size
        colors: List of colors for different modes
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert to numpy array if needed
    if isinstance(pc_data, list):
        pc_data = np.array(pc_data)
    
    # Transpose if needed (ensure shape is n_time x n_modes)
    if pc_data.shape[0] < pc_data.shape[1]:
        pc_data = pc_data.T
    
    # Determine number of modes to plot
    if mode_indices is None:
        mode_indices = list(range(min(n_modes, pc_data.shape[1])))
    
    n_plot_modes = len(mode_indices)
    
    # Auto-determine figure size
    if figsize is None:
        n_cols = min(2, n_plot_modes)
        n_rows = (n_plot_modes + n_cols - 1) // n_cols
        figsize = (6 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plot_modes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Auto-generate colors if not provided
    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, n_plot_modes))
    
    # Generate time axis if not provided
    if time_coords is None:
        time_coords = np.arange(pc_data.shape[0])
    
    for i, mode_idx in enumerate(mode_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        pc_series = pc_data[:, mode_idx]
        
        ax.plot(time_coords, pc_series, color=colors[i], linewidth=1.5, **kwargs)
        ax.set_title(f'PC {mode_idx + 1}', fontweight='bold')
        ax.set_ylabel('PC Amplitude')
        
        # Format time axis if it's datetime
        if isinstance(time_coords, pd.DatetimeIndex):
            ax.set_xlabel('Time')
            # Auto-format date labels
            if len(time_coords) > 20:
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        else:
            ax.set_xlabel('Time Index')
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plot_modes, len(axes)):
        axes[i].set_visible(False)
    
    # Remove main title
    # fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_plot_modes]

def plot_eof_variance(variance_ratio: Union[np.ndarray, List[float]],
                     n_modes: Optional[int] = None,
                     title: str = "EOF Variance Explained",
                     figsize: Tuple[float, float] = (10, 6),
                     cumulative: bool = True,
                     save_path: Optional[str] = None,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot EOF variance explained ratio
    
    Args:
        variance_ratio: Variance explained ratio for each mode
        n_modes: Number of modes to plot (if None, plot all)
        title: Plot title
        figsize: Figure size
        cumulative: Whether to plot cumulative variance
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert to numpy array if needed
    if isinstance(variance_ratio, list):
        variance_ratio = np.array(variance_ratio)
    
    # Limit number of modes to plot
    if n_modes is not None:
        variance_ratio = variance_ratio[:n_modes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(1, len(variance_ratio) + 1)
    
    if cumulative:
        cumulative_var = np.cumsum(variance_ratio)
        ax.plot(x, cumulative_var, 'o-', linewidth=2, markersize=6, **kwargs)
        ax.set_ylabel('Cumulative Variance Explained (%)')
        ax.set_title(f'{title} (Cumulative)', fontweight='bold')
    else:
        ax.bar(x, variance_ratio, alpha=0.7, **kwargs)
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title(title, fontweight='bold')
    
    ax.set_xlabel('EOF Mode')
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(variance_ratio):
        if cumulative:
            label = f'{cumulative_var[i]:.1f}%'
        else:
            label = f'{v:.1f}%'
        ax.annotate(label, (x[i], variance_ratio[i]), 
                   xytext=(0, 5), textcoords='offset points', 
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_common_eof_comparison(eof_results: Dict[str, Dict[str, Any]],
                              mode_idx: int = 0,
                              title: str = "Common EOF Comparison",
                              figsize: Optional[Tuple[float, float]] = None,
                              save_path: Optional[str] = None,
                              **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot Common EOF comparison across different datasets/models
    
    Args:
        eof_results: Dictionary with EOF results for different datasets
                     Format: {'dataset_name': {'modes': modes, 'pcs': pcs, 'variance': variance}}
        mode_idx: Mode index to compare (0-based)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    n_datasets = len(eof_results)
    dataset_names = list(eof_results.keys())
    
    # Auto-determine figure size
    if figsize is None:
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_datasets))
    
    for i, (dataset_name, results) in enumerate(eof_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot spatial mode
        if 'modes' in results and mode_idx < results['modes'].shape[0]:
            mode_data = results['modes'][mode_idx]
            ax.plot(mode_data, color=colors[i], linewidth=2, **kwargs)
            ax.set_title(f'{dataset_name} - Mode {mode_idx + 1}', fontweight='bold')
            ax.set_ylabel('EOF Amplitude')
            ax.set_xlabel('Spatial Index')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)
    
    # Remove main title
    # fig.suptitle(f'{title} - Mode {mode_idx + 1}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_datasets]

def plot_eof_summary(eof_modes: np.ndarray,
                    pc_data: np.ndarray,
                    variance_ratio: np.ndarray,
                    spatial_coords: Optional[Dict[str, np.ndarray]] = None,
                    time_coords: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                    n_modes: int = 4,
                    title: str = "EOF Analysis Summary",
                    figsize: Optional[Tuple[float, float]] = None,
                    save_path: Optional[str] = None,
                    **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive EOF analysis summary plot
    
    Args:
        eof_modes: EOF spatial modes
        pc_data: Principal component time series
        variance_ratio: Variance explained ratio
        spatial_coords: Spatial coordinates
        time_coords: Time coordinates
        n_modes: Number of modes to display
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-determine figure size
    if figsize is None:
        figsize = (16, 12)
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # 1. Variance explained plot (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_eof_variance(variance_ratio, n_modes=n_modes, title="Variance Explained", ax=ax1, **kwargs)
    
    # 2. Spatial modes (middle row)
    for i in range(min(n_modes, 4)):
        ax = fig.add_subplot(gs[1, i])
        mode_data = eof_modes[i]
        ax.plot(mode_data, linewidth=2, **kwargs)
        ax.set_title(f'EOF Mode {i + 1}', fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. PC time series (bottom row)
    for i in range(min(n_modes, 4)):
        ax = fig.add_subplot(gs[2, i])
        if pc_data.shape[0] < pc_data.shape[1]:
            pc_series = pc_data.T[:, i]
        else:
            pc_series = pc_data[:, i]
        
        if time_coords is not None:
            ax.plot(time_coords, pc_series, linewidth=1.5, **kwargs)
            ax.set_xlabel('Time')
        else:
            ax.plot(pc_series, linewidth=1.5, **kwargs)
            ax.set_xlabel('Time Index')
        
        ax.set_title(f'PC {i + 1}', fontweight='bold')
        ax.set_ylabel('PC Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Remove main title
    # fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, fig.axes

def plot_ensemble_pc_timeseries(pc_data_dict: Dict[str, np.ndarray],
                               time_coords: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                               n_modes: int = 4,
                               title: str = "Ensemble PC Time Series",
                               figsize: Optional[Tuple[float, float]] = None,
                               normalize: bool = False,
                               save_path: Optional[str] = None,
                               **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot ensemble PC time series with different models and observations on the same plot
    
    Args:
        pc_data_dict: Dictionary with PC data for different models/observations
                     Format: {'model_name': pc_data_array}
        time_coords: Time coordinates for x-axis
        n_modes: Number of modes to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-determine figure size
    if figsize is None:
        figsize = (12, 8)
    
    # Create subplots for each mode
    fig, axes = plt.subplots(n_modes, 1, figsize=figsize, sharex=True)
    if n_modes == 1:
        axes = [axes]
    
    # Define color and line style combinations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Get model names and sort them (put observations first if any)
    model_names = list(pc_data_dict.keys())
    obs_models = [name for name in model_names if 'obs' in name.lower() or 'observation' in name.lower()]
    other_models = [name for name in model_names if name not in obs_models]
    sorted_models = obs_models + other_models
    
    # Apply normalization if requested
    if normalize:
        normalized_pc_data = {}
        for model_name, pc_data in pc_data_dict.items():
            # Handle different data shapes
            if len(pc_data.shape) == 2:
                if pc_data.shape[0] < pc_data.shape[1]:
                    pc_series = pc_data.T
                else:
                    pc_series = pc_data
            elif len(pc_data.shape) == 1:
                pc_series = pc_data.reshape(-1, 1)
            else:
                logger.warning(f"Unexpected PC data shape for {model_name}: {pc_data.shape}")
                continue
            
            # Normalize each mode
            normalized_series = np.zeros_like(pc_series)
            for mode_idx in range(min(pc_series.shape[1], n_modes)):
                mode_data = pc_series[:, mode_idx]
                valid_data = mode_data[~np.isnan(mode_data)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    if std_val > 0:
                        normalized_series[:, mode_idx] = (mode_data - mean_val) / std_val
                    else:
                        normalized_series[:, mode_idx] = mode_data - mean_val
                else:
                    normalized_series[:, mode_idx] = mode_data
            
            # Reshape back to original shape
            if len(pc_data.shape) == 2 and pc_data.shape[0] < pc_data.shape[1]:
                normalized_pc_data[model_name] = normalized_series.T
            elif len(pc_data.shape) == 1:
                normalized_pc_data[model_name] = normalized_series.flatten()
            else:
                normalized_pc_data[model_name] = normalized_series
    else:
        normalized_pc_data = pc_data_dict
    
    # Plot each mode
    for mode_idx in range(n_modes):
        ax = axes[mode_idx]
        
        # Plot each model's PC time series
        for i, model_name in enumerate(sorted_models):
            pc_data = normalized_pc_data[model_name]
            
            # Handle different data shapes
            if len(pc_data.shape) == 2:
                if pc_data.shape[0] < pc_data.shape[1]:
                    pc_series = pc_data.T[:, mode_idx]
                else:
                    pc_series = pc_data[:, mode_idx]
            elif len(pc_data.shape) == 1:
                pc_series = pc_data
            else:
                logger.warning(f"Unexpected PC data shape for {model_name}: {pc_data.shape}")
                continue
            
            # Create x-axis data for this specific model
            if time_coords is not None:
                # 如果提供了时间坐标，需要确保长度匹配
                if len(time_coords) == len(pc_series):
                    x_data = time_coords
                else:
                    # 长度不匹配，使用索引
                    x_data = np.arange(len(pc_series))
                    logger.warning(f"时间坐标长度与{model_name}数据长度不匹配，使用索引")
                x_label = 'Time'
            else:
                x_data = np.arange(len(pc_series))
                x_label = 'Time Index'
            
            # Choose color and line style
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            
            # Plot the line with model name and data length in label
            label_with_info = f"{model_name} ({len(pc_series)} points)"
            ax.plot(x_data, pc_series, 
                   color=color, linestyle=line_style, linewidth=1.5,
                   label=label_with_info, alpha=0.8)
        
        # Customize subplot
        ax.set_title(f'Mode {mode_idx + 1}', fontweight='bold', fontsize=12)
        if normalize:
            ax.set_ylabel('Normalized PC Amplitude', fontsize=10)
        else:
            ax.set_ylabel('PC Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend for the first subplot only, positioned to avoid overlap
        if mode_idx == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
                     frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis label for the last subplot only
        if mode_idx == n_modes - 1:
            ax.set_xlabel(x_label, fontsize=10)
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.82, top=0.95, hspace=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes

def plot_ensemble_eof_comparison(eof_data_dict: Dict[str, np.ndarray],
                                spatial_coords: Optional[Dict[str, np.ndarray]] = None,
                                n_modes: int = 4,
                                title: str = "Ensemble EOF Comparison",
                                figsize: Optional[Tuple[float, float]] = None,
                                save_path: Optional[str] = None,
                                **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot ensemble EOF spatial patterns comparison
    
    Args:
        eof_data_dict: Dictionary with EOF data for different models/observations
                      Format: {'model_name': eof_modes_array}
        spatial_coords: Dictionary with spatial coordinates {'lat': lats, 'lon': lons}
        n_modes: Number of modes to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-determine figure size
    if figsize is None:
        figsize = (12, 8)
    
    # Create subplots for each mode
    fig, axes = plt.subplots(n_modes, 1, figsize=figsize, sharex=True)
    if n_modes == 1:
        axes = [axes]
    
    # Define color combinations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get model names and sort them
    model_names = list(eof_data_dict.keys())
    obs_models = [name for name in model_names if 'obs' in name.lower() or 'observation' in name.lower()]
    other_models = [name for name in model_names if name not in obs_models]
    sorted_models = obs_models + other_models
    
    # Default x_label
    x_label = 'Spatial Index'
    
    # Plot each mode
    for mode_idx in range(n_modes):
        ax = axes[mode_idx]
        
        # Plot each model's EOF spatial pattern
        for i, model_name in enumerate(sorted_models):
            eof_data = eof_data_dict[model_name]
            
            # Handle different data shapes
            if len(eof_data.shape) == 3:
                # 3D data: (modes, lat, lon) or (modes, spatial_points, time)
                if mode_idx < eof_data.shape[0]:
                    # Take a slice along the first spatial dimension
                    eof_series = eof_data[mode_idx, :, 0]  # Take first time point or longitude
                else:
                    continue
            elif len(eof_data.shape) == 2:
                if eof_data.shape[0] < eof_data.shape[1]:
                    eof_series = eof_data.T[:, mode_idx]
                else:
                    eof_series = eof_data[:, mode_idx]
            elif len(eof_data.shape) == 1:
                eof_series = eof_data
            else:
                logger.warning(f"Unexpected EOF data shape for {model_name}: {eof_data.shape}")
                continue
            
            # Create x-axis data based on the actual data length
            x_data = np.arange(len(eof_series))
            x_label = 'Spatial Index'
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Plot the line
            ax.plot(x_data, eof_series, 
                   color=color, linewidth=1.5,
                   label=model_name, alpha=0.8)
        
        # Customize subplot
        ax.set_title(f'Mode {mode_idx + 1}', fontweight='bold', fontsize=12)
        ax.set_ylabel('EOF Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend for the first subplot only
        if mode_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Set x-axis label for the last subplot only
        if mode_idx == n_modes - 1:
            ax.set_xlabel(x_label, fontsize=10)
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95, hspace=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes

def plot_ensemble_variance_comparison(variance_data_dict: Dict[str, np.ndarray],
                                     n_modes: int = 4,
                                     title: str = "Ensemble Variance Explained Comparison",
                                     figsize: Optional[Tuple[float, float]] = None,
                                     save_path: Optional[str] = None,
                                     **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot ensemble variance explained comparison across different models and observations
    
    Args:
        variance_data_dict: Dictionary with variance data for different models/observations
                           Format: {'model_name': variance_array}
        n_modes: Number of modes to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-determine figure size
    if figsize is None:
        figsize = (12, 8)
    
    # Create subplots for individual and cumulative variance
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define color combinations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get model names and sort them (put observations first if any)
    model_names = list(variance_data_dict.keys())
    obs_models = [name for name in model_names if 'obs' in name.lower() or 'observation' in name.lower()]
    other_models = [name for name in model_names if name not in obs_models]
    sorted_models = obs_models + other_models
    
    # Plot individual variance explained
    ax1 = axes[0]
    x = np.arange(1, n_modes + 1)
    
    for i, model_name in enumerate(sorted_models):
        variance_data = variance_data_dict[model_name]
        
        # Handle different data shapes
        if len(variance_data.shape) == 1:
            if len(variance_data) >= n_modes:
                variance_series = variance_data[:n_modes]
            else:
                # Pad with zeros if not enough modes
                variance_series = np.pad(variance_data, (0, n_modes - len(variance_data)), 
                                       mode='constant', constant_values=0)
        else:
            logger.warning(f"Unexpected variance data shape for {model_name}: {variance_data.shape}")
            continue
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Plot individual variance
        ax1.plot(x, variance_series * 100, 'o-', color=color, linewidth=2, 
                markersize=6, label=model_name, alpha=0.8)
    
    ax1.set_title('Individual Variance Explained', fontweight='bold', fontsize=12)
    ax1.set_xlabel('EOF Mode', fontsize=10)
    ax1.set_ylabel('Variance Explained (%)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative variance explained
    ax2 = axes[1]
    
    for i, model_name in enumerate(sorted_models):
        variance_data = variance_data_dict[model_name]
        
        # Handle different data shapes
        if len(variance_data.shape) == 1:
            if len(variance_data) >= n_modes:
                variance_series = variance_data[:n_modes]
            else:
                # Pad with zeros if not enough modes
                variance_series = np.pad(variance_data, (0, n_modes - len(variance_data)), 
                                       mode='constant', constant_values=0)
        else:
            continue
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(variance_series) * 100
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Plot cumulative variance (without label to avoid duplicate legend)
        ax2.plot(x, cumulative_variance, 'o-', color=color, linewidth=2, 
                markersize=6, alpha=0.8)
    
    ax2.set_title('Cumulative Variance Explained', fontweight='bold', fontsize=12)
    ax2.set_xlabel('EOF Mode', fontsize=10)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.02), loc='upper center', 
               fontsize=10, frameon=True, fancybox=True, shadow=True, ncol=4)
    
    # Adjust layout to accommodate the legend at the bottom
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes

if __name__ == "__main__":
    # Example usage
    print("EOF plotting module loaded")

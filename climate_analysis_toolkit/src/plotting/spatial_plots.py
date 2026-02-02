"""
Spatial Plotting Module
Provides geographical plotting functionality with map projections
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
from typing import Optional, Tuple, Union, Dict, Any, List
import logging
from pathlib import Path
import geopandas as gpd
import warnings
from ..config.output_config import get_plot_output_path, get_standard_filename
from ..utils.data_utils import create_land_mask, compute_data_extent

warnings.filterwarnings('ignore')
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

def get_spatial_plot_path(var_type: str, plot_method: str, model: str = None, 
                         leadtime: int = None) -> Path:
    """Get standardized spatial plot output path"""
    filename = get_standard_filename("spatial", var_type, plot_method, model, leadtime)
    return get_plot_output_path("spatial", var_type, plot_method, filename)

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

def plot_spatial_field(data: Union[np.ndarray, xr.DataArray],
                      lats: Union[np.ndarray, xr.DataArray],
                      lons: Union[np.ndarray, xr.DataArray],
                      title: str = "Spatial Field",
                      projection: str = "PlateCarree",
                      extent: Optional[Tuple[float, float, float, float]] = None,
                      cmap: str = "viridis",
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      center: Optional[float] = None,
                      boundary_file: Optional[str] = None,
                      land_mask: Optional[xr.DataArray] = None,
                      auto_extent: bool = True,
                      figsize: Tuple[float, float] = (12, 8),
                      save_path: Optional[str] = None,
                      **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial field on a map
    
    Args:
        data: 2D data array
        lats: latitude coordinates
        lons: longitude coordinates
        title: plot title
        projection: map projection
        extent: map extent (lon_min, lon_max, lat_min, lat_max)
        cmap: color map
        vmin: minimum value for colorbar
        vmax: maximum value for colorbar
        center: center value for diverging colormaps
        boundary_file: path to boundary shapefile
        land_mask: optional land mask (True for land, False for ocean)
        auto_extent: if True, automatically compute extent based on valid data
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert to xr.DataArray if needed for mask application
    if isinstance(data, np.ndarray):
        if isinstance(lats, xr.DataArray) and isinstance(lons, xr.DataArray):
            data_da = xr.DataArray(data, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        else:
            data_da = xr.DataArray(data, dims=['lat', 'lon'], 
                                 coords={'lat': lats, 'lon': lons})
        original_data = data
    else:
        data_da = data
        original_data = data.values
    
    # Apply land mask if provided
    if land_mask is not None:
        # Align mask to data grid
        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
            mask_aligned = land_mask.reindex(
                lat=data_da.lat,
                lon=data_da.lon,
                method='nearest',
                tolerance=0.5
            )
            data_da = data_da.where(mask_aligned)
    
    # Convert to numpy arrays for plotting
    data_plot = data_da.values if isinstance(data_da, xr.DataArray) else original_data
    if isinstance(lats, xr.DataArray):
        lats = lats.values
    if isinstance(lons, xr.DataArray):
        lons = lons.values
    
    # Auto-determine color range (only from valid/masked data)
    if vmin is None:
        vmin = np.nanpercentile(data_plot, 2)
    if vmax is None:
        vmax = np.nanpercentile(data_plot, 98)
    
    # Auto-determine center for diverging colormaps
    if center is None and cmap in ['RdBu_r', 'RdBu', 'coolwarm', 'seismic']:
        center = 0
    
    # Create figure and projection
    fig = plt.figure(figsize=figsize)
    
    if projection == "PlateCarree":
        proj = ccrs.PlateCarree()
    elif projection == "Mercator":
        proj = ccrs.Mercator()
    elif projection == "LambertConformal":
        proj = ccrs.LambertConformal()
    elif projection == "AlbersEqualArea":
        proj = ccrs.AlbersEqualArea()
    else:
        proj = ccrs.PlateCarree()
    
    ax = plt.axes(projection=proj)
    
    # Set map extent
    if extent is None:
        if auto_extent and land_mask is not None:
            # Compute extent based on valid data with mask
            lon_min, lon_max, lat_min, lat_max = compute_data_extent(data_da, land_mask)
            extent = [lon_min, lon_max, lat_min, lat_max]
        elif auto_extent:
            # Compute extent based on valid data without mask
            lon_min, lon_max, lat_min, lat_max = compute_data_extent(data_da, None)
            extent = [lon_min, lon_max, lat_min, lat_max]
        else:
            extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    
    # Add custom boundaries if provided
    boundary_gdf = load_boundary_data(boundary_file)
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
    
    # Plot the data (use masked data)
    if center is not None:
        mesh = ax.contourf(
            lons, lats, data_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=20,
            transform=ccrs.PlateCarree(),
            extend='both',
            **kwargs
        )
    else:
        mesh = ax.pcolormesh(
            lons, lats, data_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            **kwargs
        )
    
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
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label('Value', fontsize=12)
    
    # Set title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_spatial_comparison(data_dict: Dict[str, Union[np.ndarray, xr.DataArray]],
                           lats: Union[np.ndarray, xr.DataArray],
                           lons: Union[np.ndarray, xr.DataArray],
                           title: str = "Spatial Comparison",
                           projection: str = "PlateCarree",
                           extent: Optional[Tuple[float, float, float, float]] = None,
                           cmap: str = "viridis",
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           boundary_file: Optional[str] = None,
                           land_mask: Optional[xr.DataArray] = None,
                           auto_extent: bool = True,
                           figsize: Optional[Tuple[float, float]] = None,
                           save_path: Optional[str] = None,
                           **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple spatial fields for comparison
    
    Args:
        data_dict: dictionary of data arrays {name: data}
        lats: latitude coordinates
        lons: longitude coordinates
        title: plot title
        projection: map projection
        extent: map extent
        cmap: color map
        vmin: minimum value for colorbar
        vmax: maximum value for colorbar
        boundary_file: path to boundary shapefile
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert to numpy arrays if needed
    if isinstance(lats, xr.DataArray):
        lats = lats.values
    if isinstance(lons, xr.DataArray):
        lons = lons.values
    
    # Auto-determine figure size
    n_plots = len(data_dict)
    if figsize is None:
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Apply land mask to all data if provided
    data_dict_masked = {}
    for name, data in data_dict.items():
        if isinstance(data, np.ndarray):
            if isinstance(lats, xr.DataArray) and isinstance(lons, xr.DataArray):
                data_da = xr.DataArray(data, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
            else:
                data_da = xr.DataArray(data, dims=['lat', 'lon'], 
                                     coords={'lat': lats, 'lon': lons})
        else:
            data_da = data
        
        if land_mask is not None:
            if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                mask_aligned = land_mask.reindex(
                    lat=data_da.lat,
                    lon=data_da.lon,
                    method='nearest',
                    tolerance=0.5
                )
                data_da = data_da.where(mask_aligned)
        
        data_dict_masked[name] = data_da
    
    # Auto-determine color range from all masked data
    if vmin is None or vmax is None:
        all_values = []
        for data in data_dict_masked.values():
            if isinstance(data, xr.DataArray):
                all_values.append(data.values.flatten())
            else:
                all_values.append(data.flatten())
        all_values = np.concatenate(all_values)
        if vmin is None:
            vmin = np.nanpercentile(all_values, 2)
        if vmax is None:
            vmax = np.nanpercentile(all_values, 98)
    
    # Create subplots
    fig, axes = plt.subplots(
        nrows=(n_plots + 2) // 3,
        ncols=min(3, n_plots),
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Set map extent
    if extent is None:
        if auto_extent and land_mask is not None and len(data_dict_masked) > 0:
            # Use first data to compute extent
            first_data = list(data_dict_masked.values())[0]
            lon_min, lon_max, lat_min, lat_max = compute_data_extent(first_data, land_mask)
            extent = [lon_min, lon_max, lat_min, lat_max]
        elif auto_extent and len(data_dict_masked) > 0:
            first_data = list(data_dict_masked.values())[0]
            lon_min, lon_max, lat_min, lat_max = compute_data_extent(first_data, None)
            extent = [lon_min, lon_max, lat_min, lat_max]
        else:
            extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    # Plot each field
    for i, (name, data) in enumerate(data_dict_masked.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Convert to numpy if needed
        if isinstance(data, xr.DataArray):
            data = data.values
        
        # Set extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        
        # Add custom boundaries
        boundary_gdf = load_boundary_data(boundary_file)
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
        
        # Plot data
        mesh = ax.pcolormesh(
            lons, lats, data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            **kwargs
        )
        
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
        
        # Set title
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add common colorbar
    if n_plots > 0:
        cbar = plt.colorbar(mesh, ax=axes[:n_plots], shrink=0.8)
        cbar.set_label('Value', fontsize=12)
    
    # Set main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_plots]

def plot_spatial_anomaly(data: Union[np.ndarray, xr.DataArray],
                        lats: Union[np.ndarray, xr.DataArray],
                        lons: Union[np.ndarray, xr.DataArray],
                        climatology: Union[np.ndarray, xr.DataArray],
                        title: str = "Spatial Anomaly",
                        projection: str = "PlateCarree",
                        extent: Optional[Tuple[float, float, float, float]] = None,
                        cmap: str = "RdBu_r",
                        boundary_file: Optional[str] = None,
                        figsize: Tuple[float, float] = (12, 8),
                        save_path: Optional[str] = None,
                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial anomaly field
    
    Args:
        data: current data
        lats: latitude coordinates
        lons: longitude coordinates
        climatology: climatological mean
        title: plot title
        projection: map projection
        extent: map extent
        cmap: color map for anomalies
        boundary_file: path to boundary shapefile
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    # Calculate anomaly
    if isinstance(data, xr.DataArray):
        data = data.values
    if isinstance(climatology, xr.DataArray):
        climatology = climatology.values
    
    anomaly = data - climatology
    
    # Auto-determine color range for anomaly
    vmax = np.nanpercentile(np.abs(anomaly), 95)
    vmin = -vmax
    
    return plot_spatial_field(
        anomaly, lats, lons,
        title=title,
        projection=projection,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        boundary_file=boundary_file,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )

def plot_spatial_correlation(data1: Union[np.ndarray, xr.DataArray],
                            data2: Union[np.ndarray, xr.DataArray],
                            lats: Union[np.ndarray, xr.DataArray],
                            lons: Union[np.ndarray, xr.DataArray],
                            title: str = "Spatial Correlation",
                            projection: str = "PlateCarree",
                            extent: Optional[Tuple[float, float, float, float]] = None,
                            cmap: str = "RdBu_r",
                            boundary_file: Optional[str] = None,
                            figsize: Tuple[float, float] = (12, 8),
                            save_path: Optional[str] = None,
                            **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial correlation field
    
    Args:
        data1: first dataset
        data2: second dataset
        lats: latitude coordinates
        lons: longitude coordinates
        title: plot title
        projection: map projection
        extent: map extent
        cmap: color map for correlation
        boundary_file: path to boundary shapefile
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for plotting
        
    Returns:
        Figure and axes objects
    """
    # Convert to numpy arrays
    if isinstance(data1, xr.DataArray):
        data1 = data1.values
    if isinstance(data2, xr.DataArray):
        data2 = data2.values
    
    # Calculate correlation
    correlation = np.full_like(data1, np.nan)
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            if not (np.isnan(data1[i, j]) or np.isnan(data2[i, j])):
                correlation[i, j] = np.corrcoef([data1[i, j]], [data2[i, j]])[0, 1]
    
    return plot_spatial_field(
        correlation, lats, lons,
        title=title,
        projection=projection,
        extent=extent,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        boundary_file=boundary_file,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )

if __name__ == "__main__":
    print("Spatial plotting module loaded")

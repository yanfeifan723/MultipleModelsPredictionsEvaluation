"""
Spectrum Analysis Plotting Module
Provides power spectral density analysis and visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
from ..config.output_config import get_plot_output_path, get_standard_filename

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

def get_spectrum_plot_path(var_type: str, plot_method: str, model: str = None, 
                          leadtime: int = None) -> Path:
    """Get standardized spectrum plot output path"""
    filename = get_standard_filename("spectrum", var_type, plot_method, model, leadtime)
    return get_plot_output_path("spectrum", var_type, plot_method, filename)

def calculate_power_spectrum(data: np.ndarray, 
                           fs: float = 1.0,
                           method: str = 'welch',
                           nperseg: Optional[int] = None,
                           noverlap: Optional[int] = None,
                           window: str = 'hann',
                           detrend: str = 'constant') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density
    
    Args:
        data: time series data
        fs: sampling frequency
        method: spectral estimation method ('welch', 'periodogram', 'fft')
        nperseg: number of points per segment for Welch method
        noverlap: number of points to overlap for Welch method
        window: window function
        detrend: detrending method
        
    Returns:
        frequencies and power spectral density
    """
    # Remove NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) < 10:
        logger.warning("Data too short for spectral analysis")
        return np.array([]), np.array([])
    
    if method == 'welch':
        if nperseg is None:
            nperseg = min(256, len(data_clean) // 4)
        if noverlap is None:
            noverlap = nperseg // 2
            
        freqs, psd = signal.welch(
            data_clean, fs=fs, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=detrend
        )
        
    elif method == 'periodogram':
        freqs, psd = signal.periodogram(
            data_clean, fs=fs, window=window, detrend=detrend
        )
        
    elif method == 'fft':
        # Simple FFT-based power spectrum
        n = len(data_clean)
        fft_vals = fft(data_clean)
        freqs = fftfreq(n, 1/fs)
        
        # Take only positive frequencies
        pos_freqs = freqs[:n//2]
        psd = np.abs(fft_vals[:n//2])**2 / n
        
        return pos_freqs, psd
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return freqs, psd

def plot_power_spectrum(data: np.ndarray,
                       fs: float = 1.0,
                       title: str = "Power Spectral Density",
                       xlabel: str = "Frequency",
                       ylabel: str = "Power Spectral Density",
                       method: str = 'welch',
                       figsize: Tuple[float, float] = (10, 6),
                       save_path: Optional[str] = None,
                       **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot power spectral density
    
    Args:
        data: time series data
        fs: sampling frequency
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        method: spectral estimation method
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for calculate_power_spectrum
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Calculate power spectrum
    freqs, psd = calculate_power_spectrum(data, fs=fs, method=method, **kwargs)
    
    if len(freqs) == 0:
        logger.error("Failed to calculate power spectrum")
        return None, None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(freqs, psd, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_spectrum_comparison(data_dict: Dict[str, np.ndarray],
                            fs: float = 1.0,
                            title: str = "Power Spectrum Comparison",
                            method: str = 'welch',
                            figsize: Optional[Tuple[float, float]] = None,
                            save_path: Optional[str] = None,
                            **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot power spectral density comparison for multiple datasets
    
    Args:
        data_dict: dictionary of time series data {name: data}
        fs: sampling frequency
        title: plot title
        method: spectral estimation method
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for calculate_power_spectrum
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Auto-determine figure size
    if figsize is None:
        figsize = (12, 8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(data_dict)))
    
    # Plot each dataset
    for i, (name, data) in enumerate(data_dict.items()):
        freqs, psd = calculate_power_spectrum(data, fs=fs, method=method, **kwargs)
        
        if len(freqs) > 0:
            ax.semilogy(freqs, psd, label=name, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_spectrum_grid(data_dict: Dict[str, np.ndarray],
                       fs: float = 1.0,
                       title: str = "Power Spectrum Grid",
                       method: str = 'welch',
                       figsize: Optional[Tuple[float, float]] = None,
                       save_path: Optional[str] = None,
                       **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot power spectral density in a grid layout
    
    Args:
        data_dict: dictionary of time series data {name: data}
        fs: sampling frequency
        title: plot title
        method: spectral estimation method
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for calculate_power_spectrum
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    n_datasets = len(data_dict)
    
    # Auto-determine figure size and grid
    if figsize is None:
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 4 * n_rows)
    
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each dataset
    for i, (name, data) in enumerate(data_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        freqs, psd = calculate_power_spectrum(data, fs=fs, method=method, **kwargs)
        
        if len(freqs) > 0:
            ax.semilogy(freqs, psd, linewidth=2)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Only add labels to leftmost and bottom plots
            if i % n_cols == 0:
                ax.set_ylabel("Power Spectral Density", fontsize=10)
            if i >= n_datasets - n_cols:
                ax.set_xlabel("Frequency", fontsize=10)
    
    # Hide unused subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, axes[:n_datasets]

def plot_spectrum_analysis(data: np.ndarray,
                          fs: float = 1.0,
                          title: str = "Spectral Analysis",
                          method: str = 'welch',
                          figsize: Tuple[float, float] = (15, 10),
                          save_path: Optional[str] = None,
                          **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Comprehensive spectral analysis plot
    
    Args:
        data: time series data
        fs: sampling frequency
        title: plot title
        method: spectral estimation method
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for calculate_power_spectrum
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Remove NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) < 10:
        logger.error("Data too short for spectral analysis")
        return None, None
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series
    ax_time = fig.add_subplot(gs[0, :])
    time_axis = np.arange(len(data_clean)) / fs
    ax_time.plot(time_axis, data_clean, linewidth=1)
    ax_time.set_xlabel("Time", fontsize=12)
    ax_time.set_ylabel("Amplitude", fontsize=12)
    ax_time.set_title("Time Series", fontsize=14, fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    
    # 2. Power spectral density
    ax_psd = fig.add_subplot(gs[1, 0])
    freqs, psd = calculate_power_spectrum(data_clean, fs=fs, method=method, **kwargs)
    
    if len(freqs) > 0:
        ax_psd.semilogy(freqs, psd, linewidth=2)
        ax_psd.set_xlabel("Frequency", fontsize=12)
        ax_psd.set_ylabel("Power Spectral Density", fontsize=12)
        ax_psd.set_title("Power Spectral Density", fontsize=14, fontweight='bold')
        ax_psd.grid(True, alpha=0.3)
        
        # Find dominant frequency
        dominant_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_idx]
        ax_psd.axvline(dominant_freq, color='red', linestyle='--', alpha=0.7,
                      label=f'Dominant: {dominant_freq:.3f}')
        ax_psd.legend()
    
    # 3. Periodogram
    ax_period = fig.add_subplot(gs[1, 1])
    if len(freqs) > 0:
        periods = 1 / freqs[1:]  # Avoid division by zero
        period_psd = psd[1:]
        ax_period.semilogy(periods, period_psd, linewidth=2)
        ax_period.set_xlabel("Period", fontsize=12)
        ax_period.set_ylabel("Power Spectral Density", fontsize=12)
        ax_period.set_title("Periodogram", fontsize=14, fontweight='bold')
        ax_period.grid(True, alpha=0.3)
        
        # Find dominant period
        dominant_period_idx = np.argmax(period_psd)
        dominant_period = periods[dominant_period_idx]
        ax_period.axvline(dominant_period, color='red', linestyle='--', alpha=0.7,
                         label=f'Dominant: {dominant_period:.2f}')
        ax_period.legend()
    
    # 4. Autocorrelation
    ax_autocorr = fig.add_subplot(gs[2, 0])
    if len(data_clean) > 1:
        lags = np.arange(min(50, len(data_clean)//2))
        autocorr = [np.corrcoef(data_clean[:-lag], data_clean[lag:])[0, 1] 
                   if lag > 0 else 1.0 for lag in lags]
        ax_autocorr.plot(lags, autocorr, linewidth=2)
        ax_autocorr.set_xlabel("Lag", fontsize=12)
        ax_autocorr.set_ylabel("Autocorrelation", fontsize=12)
        ax_autocorr.set_title("Autocorrelation Function", fontsize=14, fontweight='bold')
        ax_autocorr.grid(True, alpha=0.3)
        ax_autocorr.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 5. Statistics
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    Time Series Statistics:
    
    Length: {len(data_clean)} points
    Duration: {len(data_clean)/fs:.2f} time units
    Mean: {np.mean(data_clean):.4f}
    Std: {np.std(data_clean):.4f}
    Min: {np.min(data_clean):.4f}
    Max: {np.max(data_clean):.4f}
    """
    
    if len(freqs) > 0:
        stats_text += f"""
    Spectral Statistics:
    
    Sampling freq: {fs:.2f} Hz
    Nyquist freq: {fs/2:.2f} Hz
    Frequency resolution: {freqs[1]-freqs[0]:.4f} Hz
    """
        
        if len(freqs) > 1:
            dominant_idx = np.argmax(psd)
            dominant_freq = freqs[dominant_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
            stats_text += f"""
    Dominant frequency: {dominant_freq:.4f} Hz
    Dominant period: {dominant_period:.2f} time units
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, np.array([ax_time, ax_psd, ax_period, ax_autocorr, ax_stats])

def plot_spectrum_heatmap(data_matrix: np.ndarray,
                         fs: float = 1.0,
                         time_labels: Optional[List[str]] = None,
                         freq_range: Optional[Tuple[float, float]] = None,
                         title: str = "Spectrogram",
                         method: str = 'welch',
                         figsize: Tuple[float, float] = (12, 8),
                         save_path: Optional[str] = None,
                         **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spectrogram heatmap
    
    Args:
        data_matrix: 2D array with time series in rows
        fs: sampling frequency
        time_labels: labels for time axis
        freq_range: frequency range to plot (min_freq, max_freq)
        title: plot title
        method: spectral estimation method
        figsize: figure size
        save_path: path to save the plot
        **kwargs: additional arguments for calculate_power_spectrum
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Calculate power spectra for each time series
    all_freqs = []
    all_psds = []
    
    for i in range(data_matrix.shape[0]):
        data = data_matrix[i]
        freqs, psd = calculate_power_spectrum(data, fs=fs, method=method, **kwargs)
        
        if len(freqs) > 0:
            all_freqs.append(freqs)
            all_psds.append(psd)
    
    if not all_freqs:
        logger.error("No valid power spectra calculated")
        return None, None
    
    # Interpolate to common frequency grid
    min_freq = max(freqs[0] for freqs in all_freqs)
    max_freq = min(freqs[-1] for freqs in all_freqs)
    
    if freq_range is not None:
        min_freq, max_freq = freq_range
    
    # Create common frequency grid
    common_freqs = np.linspace(min_freq, max_freq, 100)
    
    # Interpolate each PSD to common grid
    psd_matrix = np.zeros((len(all_psds), len(common_freqs)))
    
    for i, (freqs, psd) in enumerate(zip(all_freqs, all_psds)):
        # Interpolate to common frequency grid
        psd_interp = np.interp(common_freqs, freqs, psd)
        psd_matrix[i] = psd_interp
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(psd_matrix, aspect='auto', origin='lower',
                   extent=[common_freqs[0], common_freqs[-1], 0, psd_matrix.shape[0]],
                   cmap='viridis')
    
    # Set labels
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Time Index", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set time labels if provided
    if time_labels is not None and len(time_labels) == psd_matrix.shape[0]:
        ax.set_yticks(range(len(time_labels)))
        ax.set_yticklabels(time_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power Spectral Density", fontsize=12)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

if __name__ == "__main__":
    print("Spectrum analysis module loaded")

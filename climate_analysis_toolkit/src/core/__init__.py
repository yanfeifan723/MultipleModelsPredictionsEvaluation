"""
核心分析模块
提供EOF分析、相关性分析、RMSE计算等核心功能
"""

from .eof_analysis import *
from .correlation_analysis import *
from .rmse_analysis import *
from .spectrum_analysis import *
from .common_eof_analysis import *

__all__ = [
    # EOF分析
    'EOFAnalyzer',
    'compute_eof',
    'compute_pcs',
    'compute_explained_variance',
    
    # 相关性分析
    'CorrelationAnalyzer',
    'compute_pearson_correlation',
    'compute_significance',
    
    # RMSE分析
    'RMSEAnalyzer',
    'compute_rmse',
    'compute_bias',
    'compute_mae',
    
    # 功率谱分析
    'SpectrumAnalyzer',
    'compute_power_spectrum',
    'compute_spectral_density',
    
    # Common EOF分析
    'CommonEOFAnalyzer',
    'compute_common_eofs',
    'compute_common_pcs'
]

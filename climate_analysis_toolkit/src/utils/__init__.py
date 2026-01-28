"""
工具函数模块
提供数据处理、坐标转换、质量检查等通用功能
"""

from .data_utils import *
from .coord_utils import *
from .validation import *
from .interpolation import *
from .outlier_detection import *
from .plotting_utils import (
    STANDARD_CONFIG,
    setup_cartopy_axes,
    plot_spatial_field_contour,
    add_significance_stippling,
    create_spatial_distribution_figure,
    create_discrete_colormap_norm,
    create_multi_dataset_spatial_figure,
)

__all__ = [
    # 数据处理
    'find_variable',
    'remove_outliers_iqr',
    'find_valid_data_bounds',
    'dynamic_coord_sel',
    'validate_data',
    'load_netcdf_data',
    'save_netcdf_data',
    'create_land_mask',
    'compute_data_extent',
    
    # 坐标处理
    'standardize_coords',
    'interpolate_to_grid',
    'create_grid',
    
    # 数据验证
    'check_data_quality',
    'detect_outliers',
    'validate_spatial_bounds',
    
    # 异常值检测
    'OutlierDetector',
    'remove_outliers_from_dataframe',
    'get_outlier_statistics',
    'validate_outlier_parameters',
    
    # 插值工具
    'grid_interpolation',
    'temporal_interpolation',
    'bilinear_interpolation',
    'nearest_neighbor_interpolation',
    'cubic_interpolation',
    'conservative_interpolation',
    'spectral_interpolation',
    'adaptive_interpolation',
    'smooth_interpolation',
    
    # 绘图工具
    'STANDARD_CONFIG',
    'setup_cartopy_axes',
    'plot_spatial_field_contour',
    'add_significance_stippling',
    'create_spatial_distribution_figure',
    'create_discrete_colormap_norm',
    'create_multi_dataset_spatial_figure',
]

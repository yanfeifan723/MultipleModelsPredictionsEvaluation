"""
相关性分析模块
提供Pearson相关系数计算和显著性检验功能
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """相关性分析器"""
    
    def __init__(self, min_valid_points: int = 3, significance_level: float = 0.05):
        """
        初始化相关性分析器
        
        Args:
            min_valid_points: 最小有效点数
            significance_level: 显著性水平
        """
        self.min_valid_points = min_valid_points
        self.significance_level = significance_level
    
    def compute_correlation(self, obs_data: xr.DataArray, 
                           fcst_data: xr.DataArray) -> Dict[str, Any]:
        """
        计算相关系数
        
        Args:
            obs_data: 观测数据
            fcst_data: 预测数据
            
        Returns:
            相关性分析结果
        """
        logger.info("开始计算相关系数")
        
        # 确保数据对齐
        common_time = obs_data.time.to_index().intersection(fcst_data.time.to_index())
        if len(common_time) < self.min_valid_points:
            raise ValueError(f"时间点不足: {len(common_time)} < {self.min_valid_points}")
        
        obs_aligned = obs_data.sel(time=common_time)
        fcst_aligned = fcst_data.sel(time=common_time)
        
        # 初始化结果数组
        lat_size, lon_size = obs_aligned.lat.size, obs_aligned.lon.size
        correlation = np.full((lat_size, lon_size), np.nan)
        p_value = np.full((lat_size, lon_size), np.nan)
        t_value = np.full((lat_size, lon_size), np.nan)
        count = np.zeros((lat_size, lon_size), dtype=int)
        
        # 逐点计算相关系数
        for i in range(lat_size):
            for j in range(lon_size):
                obs_series = obs_aligned[:, i, j].values
                fcst_series = fcst_aligned[:, i, j].values
                
                # 移除NaN值
                valid_mask = ~(np.isnan(obs_series) | np.isnan(fcst_series))
                obs_valid = obs_series[valid_mask]
                fcst_valid = fcst_series[valid_mask]
                
                if len(obs_valid) >= self.min_valid_points:
                    # 计算相关系数
                    corr_coef, p_val = stats.pearsonr(obs_valid, fcst_valid)
                    
                    # 计算t统计量
                    n = len(obs_valid)
                    t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                    
                    correlation[i, j] = corr_coef
                    p_value[i, j] = p_val
                    t_value[i, j] = t_stat
                    count[i, j] = n
        
        # 创建结果数据集
        result = xr.Dataset({
            'correlation': xr.DataArray(
                correlation,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Pearson correlation coefficient', 'units': ''}
            ),
            'p_value': xr.DataArray(
                p_value,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'p-value', 'units': ''}
            ),
            't_value': xr.DataArray(
                t_value,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 't-statistic', 'units': ''}
            ),
            'count': xr.DataArray(
                count,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'number of valid points', 'units': ''}
            )
        })
        
        # 添加全局统计信息
        valid_corr = correlation[~np.isnan(correlation)]
        result.attrs['mean_correlation'] = float(np.mean(valid_corr))
        result.attrs['std_correlation'] = float(np.std(valid_corr))
        result.attrs['min_correlation'] = float(np.min(valid_corr))
        result.attrs['max_correlation'] = float(np.max(valid_corr))
        result.attrs['significant_points'] = int(np.sum(p_value < self.significance_level))
        result.attrs['total_points'] = int(np.sum(count > 0))
        
        logger.info(f"相关性计算完成，平均相关系数: {result.attrs['mean_correlation']:.3f}")
        
        return result
    
    def save_results(self, results: xr.Dataset, filepath: str) -> None:
        """保存分析结果"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_netcdf(filepath)
        logger.info(f"相关性分析结果已保存到: {filepath}")

def compute_pearson_correlation(obs_data: xr.DataArray, 
                               fcst_data: xr.DataArray,
                               min_valid_points: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算Pearson相关系数
    
    Args:
        obs_data: 观测数据
        fcst_data: 预测数据
        min_valid_points: 最小有效点数
        
    Returns:
        相关系数和p值
    """
    analyzer = CorrelationAnalyzer(min_valid_points=min_valid_points)
    result = analyzer.compute_correlation(obs_data, fcst_data)
    
    return result['correlation'].values, result['p_value'].values

def compute_significance(correlation: np.ndarray, 
                        p_value: np.ndarray,
                        significance_level: float = 0.05) -> np.ndarray:
    """
    计算显著性
    
    Args:
        correlation: 相关系数
        p_value: p值
        significance_level: 显著性水平
        
    Returns:
        显著性掩码
    """
    return p_value < significance_level

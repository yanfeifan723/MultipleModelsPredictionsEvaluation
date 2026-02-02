"""
RMSE分析模块
提供均方根误差计算功能
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RMSEAnalyzer:
    """RMSE分析器"""
    
    def __init__(self, high_bias_threshold: float = 15, high_rmse_threshold: float = 15):
        """
        初始化RMSE分析器
        
        Args:
            high_bias_threshold: 高偏差阈值
            high_rmse_threshold: 高RMSE阈值
        """
        self.high_bias_threshold = high_bias_threshold
        self.high_rmse_threshold = high_rmse_threshold
    
    def compute_rmse(self, obs_data: xr.DataArray, 
                     fcst_data: xr.DataArray) -> Dict[str, Any]:
        """
        计算RMSE
        
        Args:
            obs_data: 观测数据
            fcst_data: 预测数据
            
        Returns:
            RMSE分析结果
        """
        logger.info("开始计算RMSE")
        
        # 确保数据对齐
        common_time = obs_data.time.to_index().intersection(fcst_data.time.to_index())
        if len(common_time) < 3:
            raise ValueError(f"时间点不足: {len(common_time)} < 3")
        
        obs_aligned = obs_data.sel(time=common_time)
        fcst_aligned = fcst_data.sel(time=common_time)
        
        # 计算偏差
        bias = fcst_aligned - obs_aligned
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((fcst_aligned - obs_aligned)**2, axis=0))
        
        # 计算MAE
        mae = np.mean(np.abs(fcst_aligned - obs_aligned), axis=0)
        
        # 计算标准差
        std_obs = np.std(obs_aligned, axis=0)
        std_fcst = np.std(fcst_aligned, axis=0)
        
        # 计算相关系数
        correlation = self._compute_correlation(obs_aligned, fcst_aligned)
        
        # 创建结果数据集
        result = xr.Dataset({
            'rmse': xr.DataArray(
                rmse,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Root Mean Square Error', 'units': ''}
            ),
            'mae': xr.DataArray(
                mae,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Mean Absolute Error', 'units': ''}
            ),
            'bias': xr.DataArray(
                np.mean(bias, axis=0),
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Mean Bias', 'units': ''}
            ),
            'correlation': xr.DataArray(
                correlation,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Correlation Coefficient', 'units': ''}
            ),
            'std_obs': xr.DataArray(
                std_obs,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Standard Deviation of Observations', 'units': ''}
            ),
            'std_fcst': xr.DataArray(
                std_fcst,
                dims=['lat', 'lon'],
                coords={'lat': obs_aligned.lat, 'lon': obs_aligned.lon},
                attrs={'description': 'Standard Deviation of Forecasts', 'units': ''}
            )
        })
        
        # 添加全局统计信息
        valid_rmse = rmse.values[~np.isnan(rmse.values)]
        valid_bias = result['bias'].values[~np.isnan(result['bias'].values)]
        valid_corr = correlation[~np.isnan(correlation)]
        
        result.attrs['mean_rmse'] = float(np.mean(valid_rmse))
        result.attrs['std_rmse'] = float(np.std(valid_rmse))
        result.attrs['min_rmse'] = float(np.min(valid_rmse))
        result.attrs['max_rmse'] = float(np.max(valid_rmse))
        result.attrs['mean_bias'] = float(np.mean(valid_bias))
        result.attrs['std_bias'] = float(np.std(valid_bias))
        result.attrs['mean_correlation'] = float(np.mean(valid_corr))
        result.attrs['high_rmse_points'] = int(np.sum(valid_rmse > self.high_rmse_threshold))
        result.attrs['high_bias_points'] = int(np.sum(np.abs(valid_bias) > self.high_bias_threshold))
        result.attrs['total_points'] = int(len(valid_rmse))
        
        logger.info(f"RMSE计算完成，平均RMSE: {result.attrs['mean_rmse']:.3f}")
        
        return result
    
#    def _compute_correlation(self, obs_data: xr.DataArray, 
#                           fcst_data: xr.DataArray) -> np.ndarray:
#        """计算逐点相关系数"""
#        lat_size, lon_size = obs_data.lat.size, obs_data.lon.size
#        correlation = np.full((lat_size, lon_size), np.nan)
#        
#        for i in range(lat_size):
#            for j in range(lon_size):
#                obs_series = obs_data[:, i, j].values
#                fcst_series = fcst_data[:, i, j].values
#                
#                # 移除NaN值
#                valid_mask = ~(np.isnan(obs_series) | np.isnan(fcst_series))
#                if np.sum(valid_mask) >= 3:
#                    obs_valid = obs_series[valid_mask]
#                    fcst_valid = fcst_series[valid_mask]
#                    
#                    # 计算相关系数
#                    corr_matrix = np.corrcoef(obs_valid, fcst_valid)
#                    correlation[i, j] = corr_matrix[0, 1]
#        
#        return correlation
    
    def save_results(self, results: xr.Dataset, filepath: str) -> None:
        """保存分析结果"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_netcdf(filepath)
        logger.info(f"RMSE分析结果已保存到: {filepath}")

def compute_rmse(obs_data: xr.DataArray, 
                 fcst_data: xr.DataArray) -> np.ndarray:
    """
    计算RMSE
    
    Args:
        obs_data: 观测数据
        fcst_data: 预测数据
        
    Returns:
        RMSE数组
    """
    analyzer = RMSEAnalyzer()
    result = analyzer.compute_rmse(obs_data, fcst_data)
    
    return result['rmse'].values

def compute_bias(obs_data: xr.DataArray, 
                 fcst_data: xr.DataArray) -> np.ndarray:
    """
    计算偏差
    
    Args:
        obs_data: 观测数据
        fcst_data: 预测数据
        
    Returns:
        偏差数组
    """
    analyzer = RMSEAnalyzer()
    result = analyzer.compute_rmse(obs_data, fcst_data)
    
    return result['bias'].values

def compute_mae(obs_data: xr.DataArray, 
                fcst_data: xr.DataArray) -> np.ndarray:
    """
    计算MAE
    
    Args:
        obs_data: 观测数据
        fcst_data: 预测数据
        
    Returns:
        MAE数组
    """
    analyzer = RMSEAnalyzer()
    result = analyzer.compute_rmse(obs_data, fcst_data)
    
    return result['mae'].values

"""
功率谱分析模块
提供时间序列功率谱密度计算功能
"""

import numpy as np
import xarray as xr
import scipy.signal as signal
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from ..config.settings import get_spectrum_config
from ..utils.aggregation import compute_aggregates

logger = logging.getLogger(__name__)

class SpectrumAnalyzer:
    """功率谱分析器"""
    
    def __init__(self, window_type: str = 'hann', detrend: bool = True):
        """
        初始化功率谱分析器
        
        Args:
            window_type: 窗函数类型
            detrend: 是否去趋势
        """
        self.window_type = window_type
        self.detrend = detrend
    
    def compute_power_spectrum(self, data: xr.DataArray, 
                              region: str = 'global') -> Dict[str, Any]:
        """
        计算功率谱
        
        Args:
            data: 输入数据
            region: 分析区域
            
        Returns:
            功率谱分析结果
        """
        logger.info(f"开始计算功率谱: {region}")
        
        # 区域平均
        if region != 'global':
            data = self._regional_average(data, region)
        else:
            data = data.mean(dim=['lat', 'lon'])
        
        # 移除趋势
        if self.detrend:
            data = signal.detrend(data.values)
        else:
            data = data.values
        
        # 计算功率谱
        frequencies, power_spectrum = signal.welch(
            data,
            fs=1.0,  # 采样频率
            window=self.window_type,
            nperseg=None,  # 自动选择
            noverlap=None,  # 自动选择
            nfft=None,      # 自动选择
            detrend=False   # 已经去趋势
        )
        
        # 计算周期
        periods = 1.0 / frequencies[1:]  # 排除零频率
        
        # 创建结果
        result = {
            'frequencies': frequencies,
            'periods': periods,
            'power_spectrum': power_spectrum,
            'data': data,
            'region': region,
            'window_type': self.window_type,
            'detrend': self.detrend
        }
        
        logger.info(f"功率谱计算完成，频率点数: {len(frequencies)}")
        
        return result

    def compute_time_aggregates(self, data: xr.DataArray, region: str = 'global') -> Dict[str, Any]:
        """对区域平均后的原始时间序列做年/季/月聚合，返回聚合字典。"""
        if region != 'global':
            data = self._regional_average(data, region)
        else:
            data = data.mean(dim=['lat', 'lon'])
        try:
            ts = xr.DataArray(data.values, dims=['time'], coords={'time': data['time']})
        except Exception:
            return {}
        return compute_aggregates(ts)
    
    def _regional_average(self, data: xr.DataArray, region: str) -> xr.DataArray:
        """
        计算区域平均
        
        Args:
            data: 输入数据
            region: 区域名称
        
        Returns:
            区域平均数据
        """
        config = get_spectrum_config()
        regions = config.get('regions', {})
        
        if region not in regions:
            logger.warning(f"未找到区域定义: {region}，使用全局平均")
            return data.mean(dim=['lat', 'lon'])
        
        region_bounds = regions[region]
        
        # 提取区域数据
        region_data = data.sel(
            lat=slice(region_bounds['lat_min'], region_bounds['lat_max']),
            lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
        )
        
        # 计算区域平均
        if 'weights' in region_bounds and region_bounds['weights']:
            # 使用面积权重平均
            weights = np.cos(np.deg2rad(region_data.lat))
            weights = weights / weights.sum()
            weighted_data = region_data * weights
            averaged = weighted_data.sum(dim=['lat', 'lon'])
        else:
            # 简单平均
            averaged = region_data.mean(dim=['lat', 'lon'])
        
        logger.info(f"区域平均完成: {region}")
        return averaged
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """保存分析结果"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建数据集
        ds = xr.Dataset({
            'frequencies': xr.DataArray(
                results['frequencies'],
                dims=['freq'],
                attrs={'description': 'Frequency', 'units': '1/time'}
            ),
            'periods': xr.DataArray(
                results['periods'],
                dims=['period'],
                attrs={'description': 'Period', 'units': 'time'}
            ),
            'power_spectrum': xr.DataArray(
                results['power_spectrum'],
                dims=['freq'],
                attrs={'description': 'Power Spectral Density', 'units': 'power/frequency'}
            )
        })
        
        ds.attrs['region'] = results['region']
        ds.attrs['window_type'] = results['window_type']
        ds.attrs['detrend'] = results['detrend']
        
        ds.to_netcdf(filepath)
        logger.info(f"功率谱分析结果已保存到: {filepath}")

def compute_power_spectrum(data: xr.DataArray, 
                          region: str = 'global',
                          window_type: str = 'hanning',
                          detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算功率谱
    
    Args:
        data: 输入数据
        region: 分析区域
        window_type: 窗函数类型
        detrend: 是否去趋势
        
    Returns:
        频率和功率谱
    """
    analyzer = SpectrumAnalyzer(window_type=window_type, detrend=detrend)
    result = analyzer.compute_power_spectrum(data, region)
    
    return result['frequencies'], result['power_spectrum']

def compute_spectral_density(data: xr.DataArray,
                           region: str = 'global') -> np.ndarray:
    """
    计算谱密度
    
    Args:
        data: 输入数据
        region: 分析区域
        
    Returns:
        谱密度数组
    """
    analyzer = SpectrumAnalyzer()
    result = analyzer.compute_power_spectrum(data, region)
    
    return result['power_spectrum']

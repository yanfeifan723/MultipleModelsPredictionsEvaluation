"""
插值工具模块
提供各种空间和时间插值方法
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from typing import Optional, Tuple, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def grid_interpolation(data: xr.DataArray,
                      target_lats: np.ndarray,
                      target_lons: np.ndarray,
                      method: str = 'linear',
                      fill_value: float = np.nan) -> xr.DataArray:
    """
    网格插值函数
    
    Args:
        data: 输入数据
        target_lats: 目标纬度
        target_lons: 目标经度
        method: 插值方法 ('linear', 'nearest', 'cubic')
        fill_value: 填充值
    
    Returns:
        插值后的数据
    """
    if method == 'conservative':
        return conservative_interpolation(data, target_lats, target_lons, fill_value)
    elif method == 'spectral':
        return spectral_interpolation(data, target_lats, target_lons, fill_value)
    elif method == 'adaptive':
        return adaptive_interpolation(data, target_lats, target_lons, fill_value)
    elif method == 'smooth':
        return smooth_interpolation(data, target_lats, target_lons, fill_value)
    else:
        return _standard_grid_interpolation(data, target_lats, target_lons, method, fill_value)

def _standard_grid_interpolation(data: xr.DataArray,
                                target_lats: np.ndarray,
                                target_lons: np.ndarray,
                                method: str = 'linear',
                                fill_value: float = np.nan) -> xr.DataArray:
    """标准网格插值"""
    # 获取原始坐标
    lats = data.lat.values
    lons = data.lon.values
    
    # 创建网格点
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    
    # 准备插值数据
    valid_mask = ~np.isnan(data.values)
    if not valid_mask.any():
        logger.warning("没有有效数据进行插值")
        return xr.DataArray(np.full((len(target_lats), len(target_lons)), fill_value),
                          coords={'lat': target_lats, 'lon': target_lons})
    
    # 获取有效点
    valid_lats = lats[valid_mask]
    valid_lons = lons[valid_mask]
    valid_values = data.values[valid_mask]
    
    # 执行插值
    interpolated = griddata((valid_lons, valid_lats), valid_values,
                           (lon_grid, lat_grid), method=method, fill_value=fill_value)
    
    # 创建输出DataArray
    result = xr.DataArray(interpolated, coords={'lat': target_lats, 'lon': target_lons})
    
    logger.info(f"标准网格插值完成，方法: {method}")
    return result

def conservative_interpolation(data: xr.DataArray,
                              target_lats: np.ndarray,
                              target_lons: np.ndarray,
                              fill_value: float = np.nan) -> xr.DataArray:
    """
    保守插值方法
    保持质量守恒的插值方法
    """
    # 获取原始坐标
    lats = data.lat.values
    lons = data.lon.values
    
    # 计算网格面积权重
    def calculate_area_weights(lats, lons):
        """计算网格面积权重"""
        dlat = np.diff(lats)
        dlon = np.diff(lons)
        
        # 使用中心差分
        lat_weights = np.concatenate([dlat[:1], (dlat[:-1] + dlat[1:]) / 2, dlat[-1:]])
        lon_weights = np.concatenate([dlon[:1], (dlon[:-1] + dlon[1:]) / 2, dlon[-1:]])
        
        # 创建面积权重网格
        area_weights = np.outer(lat_weights, lon_weights)
        return area_weights
    
    # 计算原始和目标网格的面积权重
    src_weights = calculate_area_weights(lats, lons)
    tgt_weights = calculate_area_weights(target_lats, target_lons)
    
    # 创建目标网格
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    
    # 执行保守插值
    interpolated = np.full((len(target_lats), len(target_lons)), fill_value)
    
    for i, tgt_lat in enumerate(target_lats):
        for j, tgt_lon in enumerate(target_lons):
            # 找到源网格中与目标网格重叠的单元
            lat_overlap = np.abs(lats - tgt_lat) < np.diff(lats).mean() * 1.5
            lon_overlap = np.abs(lons - tgt_lon) < np.diff(lons).mean() * 1.5
            
            if lat_overlap.any() and lon_overlap.any():
                # 计算重叠区域的加权平均
                overlap_data = data.values[lat_overlap][:, lon_overlap]
                overlap_weights = src_weights[lat_overlap][:, lon_overlap]
                
                valid_mask = ~np.isnan(overlap_data)
                if valid_mask.any():
                    weighted_sum = np.sum(overlap_data[valid_mask] * overlap_weights[valid_mask])
                    weight_sum = np.sum(overlap_weights[valid_mask])
                    interpolated[i, j] = weighted_sum / weight_sum if weight_sum > 0 else fill_value
    
    # 创建输出DataArray
    result = xr.DataArray(interpolated, coords={'lat': target_lats, 'lon': target_lons})
    
    logger.info("保守插值完成")
    return result

def spectral_interpolation(data: xr.DataArray,
                          target_lats: np.ndarray,
                          target_lons: np.ndarray,
                          fill_value: float = np.nan) -> xr.DataArray:
    """
    谱插值方法
    使用傅里叶变换进行插值
    """
    # 获取原始坐标
    lats = data.lat.values
    lons = data.lon.values
    
    # 处理缺失值
    data_filled = data.fillna(data.mean())
    
    # 执行二维FFT
    fft_data = np.fft.fft2(data_filled.values)
    
    # 计算频率网格
    freq_lat = np.fft.fftfreq(len(lats), d=np.diff(lats).mean())
    freq_lon = np.fft.fftfreq(len(lons), d=np.diff(lons).mean())
    
    # 创建目标频率网格
    target_freq_lat = np.fft.fftfreq(len(target_lats), d=np.diff(target_lats).mean())
    target_freq_lon = np.fft.fftfreq(len(target_lons), d=np.diff(target_lons).mean())
    
    # 插值到目标频率网格
    freq_lat_grid, freq_lon_grid = np.meshgrid(freq_lat, freq_lon, indexing='ij')
    target_freq_lat_grid, target_freq_lon_grid = np.meshgrid(target_freq_lat, target_freq_lon, indexing='ij')
    
    # 使用线性插值
    interpolated_fft = griddata((freq_lat_grid.flatten(), freq_lon_grid.flatten()),
                               fft_data.flatten(),
                               (target_freq_lat_grid, target_freq_lon_grid),
                               method='linear', fill_value=0)
    
    # 执行逆FFT
    interpolated = np.real(np.fft.ifft2(interpolated_fft))
    
    # 处理边界效应
    interpolated = np.where(np.isnan(data.values), fill_value, interpolated)
    
    # 创建输出DataArray
    result = xr.DataArray(interpolated, coords={'lat': target_lats, 'lon': target_lons})
    
    logger.info("谱插值完成")
    return result

def adaptive_interpolation(data: xr.DataArray,
                          target_lats: np.ndarray,
                          target_lons: np.ndarray,
                          fill_value: float = np.nan) -> xr.DataArray:
    """
    自适应插值方法
    根据数据特征选择最佳插值方法
    """
    # 分析数据特征
    data_std = np.nanstd(data.values)
    data_range = np.nanmax(data.values) - np.nanmin(data.values)
    missing_ratio = np.isnan(data.values).sum() / data.values.size
    
    # 根据数据特征选择插值方法
    if missing_ratio > 0.3:
        # 大量缺失值，使用保守插值
        logger.info("检测到大量缺失值，使用保守插值")
        return conservative_interpolation(data, target_lats, target_lons, fill_value)
    elif data_std / data_range > 0.5:
        # 高变异性数据，使用谱插值
        logger.info("检测到高变异性数据，使用谱插值")
        return spectral_interpolation(data, target_lats, target_lons, fill_value)
    else:
        # 标准情况，使用线性插值
        logger.info("使用标准线性插值")
        return _standard_grid_interpolation(data, target_lats, target_lons, 'linear', fill_value)

def smooth_interpolation(data: xr.DataArray,
                        target_lats: np.ndarray,
                        target_lons: np.ndarray,
                        fill_value: float = np.nan,
                        smoothing_factor: float = 0.1) -> xr.DataArray:
    """
    平滑插值方法
    结合插值和平滑处理
    """
    # 首先进行标准插值
    interpolated = _standard_grid_interpolation(data, target_lats, target_lons, 'linear', fill_value)
    
    # 应用高斯平滑
    from scipy.ndimage import gaussian_filter
    
    # 计算平滑参数
    lat_resolution = np.diff(target_lats).mean()
    lon_resolution = np.diff(target_lons).mean()
    
    # 转换为像素单位的平滑参数
    sigma_lat = smoothing_factor / lat_resolution
    sigma_lon = smoothing_factor / lon_resolution
    
    # 应用平滑
    smoothed_values = gaussian_filter(interpolated.values, sigma=[sigma_lat, sigma_lon])
    
    # 创建输出DataArray
    result = xr.DataArray(smoothed_values, coords={'lat': target_lats, 'lon': target_lons})
    
    logger.info(f"平滑插值完成，平滑因子: {smoothing_factor}")
    return result

def temporal_interpolation(data: xr.DataArray,
                          target_times: np.ndarray,
                          method: str = 'linear',
                          fill_value: float = np.nan) -> xr.DataArray:
    """
    时间插值函数
    
    Args:
        data: 输入数据
        target_times: 目标时间
        method: 插值方法
        fill_value: 填充值
    
    Returns:
        插值后的数据
    """
    # 获取原始时间
    times = data.time.values
    
    # 创建插值函数
    if method == 'linear':
        interp_func = interp1d(times, data.values, axis=0, 
                              kind='linear', fill_value=fill_value, bounds_error=False)
    elif method == 'cubic':
        interp_func = interp1d(times, data.values, axis=0, 
                              kind='cubic', fill_value=fill_value, bounds_error=False)
    elif method == 'nearest':
        interp_func = interp1d(times, data.values, axis=0, 
                              kind='nearest', fill_value=fill_value, bounds_error=False)
    else:
        raise ValueError(f"不支持的时间插值方法: {method}")
    
    # 执行插值
    interpolated_values = interp_func(target_times)
    
    # 创建输出DataArray
    result = xr.DataArray(interpolated_values, 
                         coords={'time': target_times, 'lat': data.lat, 'lon': data.lon})
    
    logger.info(f"时间插值完成，方法: {method}")
    return result

def bilinear_interpolation(data: xr.DataArray,
                          target_lats: np.ndarray,
                          target_lons: np.ndarray,
                          fill_value: float = np.nan) -> xr.DataArray:
    """
    双线性插值
    """
    return _standard_grid_interpolation(data, target_lats, target_lons, 'linear', fill_value)

def nearest_neighbor_interpolation(data: xr.DataArray,
                                  target_lats: np.ndarray,
                                  target_lons: np.ndarray,
                                  fill_value: float = np.nan) -> xr.DataArray:
    """
    最近邻插值
    """
    return _standard_grid_interpolation(data, target_lats, target_lons, 'nearest', fill_value)

def cubic_interpolation(data: xr.DataArray,
                       target_lats: np.ndarray,
                       target_lons: np.ndarray,
                       fill_value: float = np.nan) -> xr.DataArray:
    """
    三次插值
    """
    return _standard_grid_interpolation(data, target_lats, target_lons, 'cubic', fill_value)

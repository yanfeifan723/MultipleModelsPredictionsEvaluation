"""
坐标处理工具函数
提供坐标标准化、插值、网格创建等功能
"""

import xarray as xr
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from typing import Tuple, Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    标准化坐标名称
    
    Args:
        ds: 输入数据集
        
    Returns:
        坐标标准化后的数据集
    """
    coord_mapping = {
        'latitude': 'lat',
        'longitude': 'lon',
        'lats': 'lat',
        'lons': 'lon',
        'ylat': 'lat',
        'xlon': 'lon'
    }
    
    # 重命名坐标
    rename_dict = {}
    for old_name, new_name in coord_mapping.items():
        if old_name in ds.coords:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        ds = ds.rename(rename_dict)
        logger.info(f"坐标重命名: {rename_dict}")
    
    return ds

def create_grid(lat_range: Tuple[float, float],
                lon_range: Tuple[float, float],
                lat_res: float = 1.0,
                lon_res: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建规则网格
    
    Args:
        lat_range: 纬度范围 (min, max)
        lon_range: 经度范围 (min, max)
        lat_res: 纬度分辨率
        lon_res: 经度分辨率
        
    Returns:
        纬度网格和经度网格
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    lats = np.arange(lat_min, lat_max + lat_res, lat_res)
    lons = np.arange(lon_min, lon_max + lon_res, lon_res)
    
    logger.info(f"创建网格: 纬度 {lat_min:.1f}-{lat_max:.1f} ({len(lats)}点), "
                f"经度 {lon_min:.1f}-{lon_max:.1f} ({len(lons)}点)")
    
    return lats, lons

def interpolate_to_grid(data: xr.DataArray,
                       target_lats: np.ndarray,
                       target_lons: np.ndarray,
                       method: str = 'linear',
                       fill_value: float = np.nan) -> xr.DataArray:
    """
    将数据插值到目标网格
    
    Args:
        data: 输入数据
        target_lats: 目标纬度网格
        target_lons: 目标经度网格
        method: 插值方法 ('linear', 'nearest', 'cubic')
        fill_value: 填充值
        
    Returns:
        插值后的数据
    """
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("数据必须包含lat和lon维度")
    
    # 获取原始坐标
    orig_lats = data.lat.values
    orig_lons = data.lon.values
    
    # 创建目标网格
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    
    # 创建原始网格点
    lon_orig, lat_orig = np.meshgrid(orig_lons, orig_lats)
    points = np.column_stack((lat_orig.flatten(), lon_orig.flatten()))
    
    # 提取数据值
    values = data.values.flatten()
    
    # 移除NaN值
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        logger.warning("所有数据值都是NaN")
        return xr.DataArray(
            np.full((len(target_lats), len(target_lons)), fill_value),
            dims=('lat', 'lon'),
            coords={'lat': target_lats, 'lon': target_lons}
        )
    
    points = points[valid_mask]
    values = values[valid_mask]
    
    # 执行插值
    try:
        interpolated = griddata(
            points, values, (lat_grid, lon_grid),
            method=method, fill_value=fill_value
        )
        
        # 创建新的DataArray
        result = xr.DataArray(
            interpolated,
            dims=('lat', 'lon'),
            coords={'lat': target_lats, 'lon': target_lons}
        )
        
        logger.info(f"插值完成: {method}方法, 形状 {data.shape} -> {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"插值失败: {str(e)}")
        raise

def interpolate_like(data: xr.DataArray, 
                    template: xr.DataArray,
                    method: str = 'linear') -> xr.DataArray:
    """
    将数据插值到模板数据的网格
    
    Args:
        data: 要插值的数据
        template: 模板数据
        method: 插值方法
        
    Returns:
        插值后的数据
    """
    if 'lat' not in template.dims or 'lon' not in template.dims:
        raise ValueError("模板数据必须包含lat和lon维度")
    
    target_lats = template.lat.values
    target_lons = template.lon.values
    
    return interpolate_to_grid(data, target_lats, target_lons, method)

def ensure_monotonic_coords(data: xr.DataArray) -> xr.DataArray:
    """
    确保坐标单调递增
    
    Args:
        data: 输入数据
        
    Returns:
        坐标排序后的数据
    """
    result = data.copy()
    
    for coord in ['lat', 'lon']:
        if coord in result.dims:
            coord_values = result[coord].values
            if len(coord_values) > 1 and coord_values[0] > coord_values[1]:
                # 坐标递减，需要排序
                result = result.sortby(coord)
                logger.info(f"坐标 {coord} 已排序为递增")
    
    return result

def get_coord_bounds(data: xr.DataArray) -> Dict[str, Tuple[float, float]]:
    """
    获取数据的坐标边界
    
    Args:
        data: 输入数据
        
    Returns:
        坐标边界字典
    """
    bounds = {}
    
    for coord in ['lat', 'lon']:
        if coord in data.coords:
            coord_values = data[coord].values
            bounds[coord] = (float(coord_values.min()), float(coord_values.max()))
    
    return bounds

def check_coord_consistency(data1: xr.DataArray, 
                           data2: xr.DataArray,
                           tolerance: float = 1e-6) -> bool:
    """
    检查两个数据集的坐标一致性
    
    Args:
        data1: 第一个数据集
        data2: 第二个数据集
        tolerance: 容差
        
    Returns:
        坐标是否一致
    """
    for coord in ['lat', 'lon']:
        if coord in data1.coords and coord in data2.coords:
            coord1 = data1[coord].values
            coord2 = data2[coord].values
            
            if len(coord1) != len(coord2):
                logger.warning(f"坐标 {coord} 长度不一致: {len(coord1)} vs {len(coord2)}")
                return False
            
            if not np.allclose(coord1, coord2, rtol=tolerance):
                logger.warning(f"坐标 {coord} 值不一致")
                return False
    
    return True

def create_regular_grid_interpolator(data: xr.DataArray) -> RegularGridInterpolator:
    """
    创建规则网格插值器
    
    Args:
        data: 输入数据
        
    Returns:
        插值器对象
    """
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("数据必须包含lat和lon维度")
    
    lats = data.lat.values
    lons = data.lon.values
    
    # 确保坐标单调递增
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data.isel(lat=slice(None, None, -1))
    
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        data = data.isel(lon=slice(None, None, -1))
    
    interpolator = RegularGridInterpolator(
        (lats, lons),
        data.values,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    return interpolator

def resample_coordinates(data: xr.DataArray,
                        target_resolution: float,
                        method: str = 'linear') -> xr.DataArray:
    """
    重采样坐标到指定分辨率
    
    Args:
        data: 输入数据
        target_resolution: 目标分辨率
        method: 重采样方法
        
    Returns:
        重采样后的数据
    """
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("数据必须包含lat和lon维度")
    
    # 获取当前分辨率
    lat_res = abs(data.lat.values[1] - data.lat.values[0])
    lon_res = abs(data.lon.values[1] - data.lon.values[0])
    
    if lat_res == target_resolution and lon_res == target_resolution:
        logger.info("数据已经是目标分辨率")
        return data
    
    # 创建目标网格
    lat_bounds = get_coord_bounds(data)['lat']
    lon_bounds = get_coord_bounds(data)['lon']
    
    target_lats, target_lons = create_grid(
        lat_bounds, lon_bounds, target_resolution, target_resolution
    )
    
    # 插值到目标网格
    return interpolate_to_grid(data, target_lats, target_lons, method)

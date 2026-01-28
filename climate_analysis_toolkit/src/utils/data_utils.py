"""
数据处理工具函数
提供数据加载、保存、变量查找等基础功能
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

def remove_outliers_iqr(data: np.ndarray,
                        axis: int = 0,
                        threshold: Optional[float] = None) -> np.ndarray:
    """
    使用IQR方法去除异常值（替换为NaN），供MMPE与toolkit共享。
    """
    if threshold is None:
        threshold = 1.5

    data = np.asarray(data)
    cleaned = data.copy()

    q1 = np.nanpercentile(data, 25, axis=axis, keepdims=True)
    q3 = np.nanpercentile(data, 75, axis=axis, keepdims=True)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outlier_mask = (data < lower_bound) | (data > upper_bound)

    cleaned[outlier_mask] = np.nan
    return cleaned

def find_valid_data_bounds(data: np.ndarray,
                           lat: np.ndarray,
                           lon: np.ndarray) -> Tuple[float, float, float, float]:
    """
    找到数据有效的经纬度范围，排除全NaN的行列。
    """
    data = np.asarray(data)
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    valid_rows = [i for i in range(data.shape[0]) if np.any(~np.isnan(data[i, :]))]
    valid_cols = [j for j in range(data.shape[1]) if np.any(~np.isnan(data[:, j]))]

    if valid_rows and valid_cols:
        return (
            float(lat[min(valid_rows)]),
            float(lat[max(valid_rows)]),
            float(lon[min(valid_cols)]),
            float(lon[max(valid_cols)])
        )

    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())

def find_variable(ds: xr.Dataset, candidates: List[str]) -> str:
    """
    在数据集中查找变量
    
    Args:
        ds: xarray数据集
        candidates: 候选变量名列表
        
    Returns:
        找到的变量名
        
    Raises:
        ValueError: 未找到任何候选变量
    """
    for var in candidates:
        if var in ds:
            logger.debug(f"找到变量: {var}")
            return var
    raise ValueError(f"未找到候选变量: {candidates}")

def dynamic_coord_sel(ds: xr.Dataset, coords: Dict[str, tuple]) -> xr.Dataset:
    """
    动态坐标选择和裁剪
    
    Args:
        ds: xarray数据集
        coords: 坐标裁剪范围，格式为 {'lat': (min, max), 'lon': (min, max)}
        
    Returns:
        裁剪后的数据集
    """
    coord_map = {
        'lat': ['latitude', 'lats', 'ylat'],
        'lon': ['longitude', 'lons', 'xlon']
    }
    
    result_ds = ds.copy()
    
    for target_coord, (start, end) in coords.items():
        # 查找实际坐标名
        real_coord = next(
            (alias for alias in [target_coord] + coord_map.get(target_coord, []) 
             if alias in result_ds.coords), 
            None
        )
        
        if real_coord is None:
            raise ValueError(f"坐标{target_coord}不存在，可用坐标：{list(result_ds.coords.keys())}")
        
        # 重命名坐标
        if real_coord != target_coord:
            result_ds = result_ds.rename({real_coord: target_coord})
            logger.info(f"已重命名坐标 {real_coord} => {target_coord}")
        
        # 坐标裁剪
        coord_values = result_ds[target_coord].values
        is_decreasing = len(coord_values) > 1 and coord_values[0] > coord_values[1]
        buffer = 0.5 * abs(coord_values[1] - coord_values[0]) if len(coord_values) > 1 else 1.0
        
        if is_decreasing:
            adjusted_start = end + buffer
            adjusted_end = start - buffer
        else:
            adjusted_start = start - buffer
            adjusted_end = end + buffer
        
        result_ds = result_ds.sel({target_coord: slice(adjusted_start, adjusted_end)})
        
        if is_decreasing:
            result_ds = result_ds.sortby(target_coord)
        
        logger.info(f"坐标 {target_coord} 范围: {result_ds[target_coord].min().item():.1f} - {result_ds[target_coord].max().item():.1f}")
    
    return result_ds

def validate_data(ds: xr.Dataset, data_type: str) -> bool:
    """
    验证数据有效性
    
    Args:
        ds: xarray数据集
        data_type: 数据类型标识
        
    Returns:
        数据是否有效
    """
    # 基础验证逻辑
    if ds is None:
        logger.error(f"{data_type} 数据为空")
        return False
    
    if len(ds.data_vars) == 0:
        logger.error(f"{data_type} 数据集中没有数据变量")
        return False
    
    return True

def compute_area_mean(da: xr.DataArray, region: Dict[str, List[float]]) -> float:
    """
    计算区域加权平均
    
    - 自动平均除 lat/lon 以外的维度
    - 使用 cos(lat) 权重
    """
    try:
        # 选择区域并确保坐标名标准化
        sub = da.sel(
            lat=slice(region['lat'][0], region['lat'][1]),
            lon=slice(region['lon'][0], region['lon'][1])
        )
        if sub.size == 0:
            logger.warning("选择区域为空")
            return np.nan
        
        # 先对非空间维度取平均，确保只剩 lat/lon
        non_spatial_dims = tuple(d for d in sub.dims if d not in ("lat", "lon"))
        if len(non_spatial_dims) > 0:
            sub = sub.mean(dim=non_spatial_dims, skipna=True)
        
        # 计算纬度权重
        lat_weights = np.cos(np.deg2rad(sub["lat"]))
        
        # 加权平均
        weighted_mean = sub.weighted(lat_weights).mean(dim=("lat", "lon"), skipna=True)
        
        # 返回标量
        val = weighted_mean.values
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return np.nan
            return float(val.reshape(-1)[0])
        return float(val)
    except Exception as e:
        logger.error(f"计算区域平均失败: {e}")
        return np.nan

def load_netcdf_data(file_path: Union[str, Path], 
                    mask_and_scale: bool = False,
                    **kwargs) -> xr.Dataset:
    """
    加载NetCDF数据
    
    Args:
        file_path: 文件路径
        mask_and_scale: 是否应用掩码和缩放
        **kwargs: 传递给xr.open_dataset的参数
        
    Returns:
        加载的数据集
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with xr.open_dataset(file_path, mask_and_scale=mask_and_scale, **kwargs) as ds:
            return ds.load()
    except Exception as e:
        logger.error(f"加载文件 {file_path} 失败: {str(e)}")
        raise

def save_netcdf_data(ds: xr.Dataset, 
                    file_path: Union[str, Path],
                    **kwargs) -> None:
    """
    保存数据到NetCDF文件
    
    Args:
        ds: 要保存的数据集
        file_path: 保存路径
        **kwargs: 传递给to_netcdf的参数
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ds.to_netcdf(file_path, **kwargs)
        logger.info(f"数据已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存文件 {file_path} 失败: {str(e)}")
        raise

def apply_unit_conversion(data: xr.DataArray, 
                         conversion_func: callable,
                         **kwargs) -> xr.DataArray:
    """
    应用单位转换
    
    Args:
        data: 输入数据
        conversion_func: 转换函数
        **kwargs: 传递给转换函数的参数
        
    Returns:
        转换后的数据
    """
    try:
        converted = conversion_func(data, **kwargs)
        logger.info(f"单位转换完成")
        return converted
    except Exception as e:
        logger.error(f"单位转换失败: {str(e)}")
        raise

def handle_missing_values(data: xr.DataArray, 
                         method: str = 'interpolate',
                         **kwargs) -> xr.DataArray:
    """
    处理缺失值
    
    Args:
        data: 输入数据
        method: 处理方法 ('interpolate', 'fill', 'drop')
        **kwargs: 处理参数
        
    Returns:
        处理后的数据
    """
    if method == 'interpolate':
        return data.interpolate_na(**kwargs)
    elif method == 'fill':
        fill_value = kwargs.get('fill_value', data.mean())
        return data.fillna(fill_value)
    elif method == 'drop':
        return data.dropna(**kwargs)
    else:
        raise ValueError(f"不支持的缺失值处理方法: {method}")

def resample_data(data: xr.DataArray,
                 time_freq: str = '1MS',
                 method: str = 'mean') -> xr.DataArray:
    """
    重采样数据
    
    Args:
        data: 输入数据
        time_freq: 时间频率
        method: 重采样方法
        
    Returns:
        重采样后的数据
    """
    if 'time' not in data.dims:
        logger.warning("数据没有时间维度，无法重采样")
        return data
    
    resampled = getattr(data.resample(time=time_freq), method)()
    logger.info(f"数据重采样完成: {time_freq}, 方法: {method}")
    return resampled

def select_time_range(data: xr.DataArray,
                     start_time: str,
                     end_time: str) -> xr.DataArray:
    """
    选择时间范围
    
    Args:
        data: 输入数据
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        时间范围选择后的数据
    """
    if 'time' not in data.dims:
        logger.warning("数据没有时间维度，无法选择时间范围")
        return data
    
    selected = data.sel(time=slice(start_time, end_time))
    logger.info(f"时间范围选择完成: {start_time} 到 {end_time}")
    return selected

def stack_spatial_dims(data: xr.DataArray,
                      dims: tuple = ('lat', 'lon'),
                      new_dim: str = 'space') -> xr.DataArray:
    """
    将空间维度堆叠为一维
    
    Args:
        data: 输入数据
        dims: 要堆叠的维度
        new_dim: 新的维度名
        
    Returns:
        堆叠后的数据
    """
    stacked = data.stack({new_dim: dims})
    logger.info(f"空间维度堆叠完成: {dims} -> {new_dim}")
    return stacked

def unstack_spatial_dims(data: xr.DataArray,
                        dim: str = 'space',
                        dims: tuple = ('lat', 'lon')) -> xr.DataArray:
    """
    将堆叠的空间维度展开
    
    Args:
        data: 输入数据
        dim: 要展开的维度
        dims: 展开后的维度名
        
    Returns:
        展开后的数据
    """
    unstacked = data.unstack(dim)
    logger.info(f"空间维度展开完成: {dim} -> {dims}")
    return unstacked

def create_land_mask(obs_data: xr.DataArray, 
                    target_data: xr.DataArray,
                    var_type: Optional[str] = None) -> Optional[xr.DataArray]:
    """
    创建陆地掩膜（陆地为True，海洋为False）
    基于观测数据判断陆地和海洋区域
    
    对于温度数据：NaN = 海洋
    对于降水数据：需要检查原始NaN或全0区域（海洋区域降水应该始终为0）
    
    Args:
        obs_data: 观测数据（用于判断陆地和海洋）
        target_data: 目标数据（用于对齐掩膜的网格）
        var_type: 变量类型（'temp' 或 'prec'），用于选择掩膜策略
        
    Returns:
        陆地掩膜（与target_data对齐），陆地为True，海洋为False
    """
    try:
        if obs_data is None:
            logger.warning("观测数据为空，无法创建掩膜")
            return None
        
        # 计算观测数据的时间平均，用于判断哪些格点是陆地
        if 'time' in obs_data.dims:
            obs_mean = obs_data.mean(dim='time', skipna=True)
            # 检查每个格点是否有有效数据
            # 如果某个格点在所有时间都是NaN，则认为是海洋
            # 对于降水数据，如果使用了for_mask=True，则不会fillna(0)，所以NaN就是海洋
            # 对于温度数据，NaN也是海洋
            # 如果某个格点的时间平均为NaN，说明该格点在所有时间都是NaN（海洋）
            land_mask = ~np.isnan(obs_mean.values)
            
            # 对于降水数据，如果数据中包含了fillna(0)的情况（for_mask=False），
            # 还需要检查全0区域（海洋区域降水应该始终为0）
            if var_type == 'prec':
                # 检查是否有全0的格点（可能是fillna(0)后的海洋区域）
                # 如果某个格点在所有时间都是0或NaN，则可能是海洋
                # 但这里我们假设使用for_mask=True，所以不会有fillna(0)的情况
                # 如果确实有全0区域，可以通过检查时间方差来区分
                obs_std = obs_data.std(dim='time', skipna=True)
                # 如果平均值为0且标准差也为0（或接近0），且不是NaN，则可能是海洋
                # 但这种情况应该很少，因为使用了for_mask=True
                all_zero = (obs_mean.values == 0) & ~np.isnan(obs_mean.values) & (np.isnan(obs_std.values) | (obs_std.values < 1e-6))
                # 排除全0区域
                land_mask = land_mask & ~all_zero
        else:
            obs_mean = obs_data
            # 对于降水数据，检查是否为0或NaN
            if var_type == 'prec':
                land_mask = (obs_mean.values != 0) & ~np.isnan(obs_mean.values)
            else:
                land_mask = ~np.isnan(obs_mean.values)
        
        # 将掩膜转换为与target_data对齐的DataArray
        if 'lat' in obs_mean.coords and 'lon' in obs_mean.coords:
            # 如果网格匹配，直接使用
            if (len(obs_mean.lat) == len(target_data.lat) and 
                len(obs_mean.lon) == len(target_data.lon) and
                np.allclose(obs_mean.lat.values, target_data.lat.values, rtol=1e-5) and
                np.allclose(obs_mean.lon.values, target_data.lon.values, rtol=1e-5)):
                mask_da = xr.DataArray(
                    land_mask,
                    dims=['lat', 'lon'],
                    coords={'lat': target_data.lat, 'lon': target_data.lon}
                )
            else:
                # 需要插值到目标网格
                mask_da = xr.DataArray(
                    land_mask,
                    dims=['lat', 'lon'],
                    coords={'lat': obs_mean.lat, 'lon': obs_mean.lon}
                )
                # 插值到目标网格（使用最近邻方法，保持掩膜的布尔性质）
                mask_da = mask_da.reindex(
                    lat=target_data.lat,
                    lon=target_data.lon,
                    method='nearest',
                    tolerance=0.5
                )
        else:
            logger.warning("观测数据缺少坐标信息，无法创建掩膜")
            return None
        
        logger.debug(f"陆地掩膜创建完成，陆地格点数: {np.sum(mask_da.values)}, 海洋格点数: {np.sum(~mask_da.values)}")
        return mask_da
        
    except Exception as exc:
        logger.warning(f"创建陆地掩膜失败: {exc}，将使用数据本身的NaN作为掩膜")
        return None

def compute_data_extent(data: xr.DataArray, 
                       land_mask: Optional[xr.DataArray] = None) -> Tuple[float, float, float, float]:
    """
    计算数据的实际范围（经度、纬度）
    根据有效数据的存在范围来决定，并进行取整
    
    Args:
        data: 数据数组
        land_mask: 可选的陆地掩膜（如果提供，只考虑陆地区域）
        
    Returns:
        (lon_min, lon_max, lat_min, lat_max) 取整后的范围
    """
    try:
        # 应用掩膜（如果提供）
        if land_mask is not None:
            # 确保掩膜与数据对齐
            if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                # 对齐掩膜到数据网格
                mask_aligned = land_mask.reindex(
                    lat=data.lat,
                    lon=data.lon,
                    method='nearest',
                    tolerance=0.5
                )
                # 只考虑陆地区域
                valid_data = data.where(mask_aligned)
            else:
                valid_data = data
        else:
            valid_data = data
        
        # 找到有效数据的空间范围
        valid_mask = ~np.isnan(valid_data.values)
        if not np.any(valid_mask):
            # 如果没有有效数据，返回整个数据范围
            lon_min = float(data.lon.min().values)
            lon_max = float(data.lon.max().values)
            lat_min = float(data.lat.min().values)
            lat_max = float(data.lat.max().values)
        else:
            # 找到有效数据所在的经纬度索引
            # 处理2D或3D数据
            if len(valid_mask.shape) == 2:
                valid_lat_indices = np.where(np.any(valid_mask, axis=1))[0]
                valid_lon_indices = np.where(np.any(valid_mask, axis=0))[0]
            elif len(valid_mask.shape) == 3:
                # 对于3D数据，对时间维度取平均
                valid_mask_2d = np.any(valid_mask, axis=0)
                valid_lat_indices = np.where(np.any(valid_mask_2d, axis=1))[0]
                valid_lon_indices = np.where(np.any(valid_mask_2d, axis=0))[0]
            else:
                # 对于其他维度，尝试找到lat和lon维度
                lat_dim = None
                lon_dim = None
                for dim in data.dims:
                    if dim in ['lat', 'latitude', 'lats', 'ylat']:
                        lat_dim = dim
                    elif dim in ['lon', 'longitude', 'lons', 'xlon']:
                        lon_dim = dim
                
                if lat_dim is not None and lon_dim is not None:
                    # 对非空间维度取平均
                    other_dims = [d for d in data.dims if d not in [lat_dim, lon_dim]]
                    if other_dims:
                        valid_mask_2d = np.any(valid_mask, axis=tuple([data.dims.index(d) for d in other_dims]))
                    else:
                        valid_mask_2d = valid_mask
                    
                    lat_idx = data.dims.index(lat_dim)
                    lon_idx = data.dims.index(lon_dim)
                    valid_lat_indices = np.where(np.any(valid_mask_2d, axis=lon_idx))[0]
                    valid_lon_indices = np.where(np.any(valid_mask_2d, axis=lat_idx))[0]
                else:
                    # 如果找不到lat/lon维度，返回整个范围
                    lon_min = float(data.lon.min().values) if 'lon' in data.coords else float(data[lon_dim].min().values)
                    lon_max = float(data.lon.max().values) if 'lon' in data.coords else float(data[lon_dim].max().values)
                    lat_min = float(data.lat.min().values) if 'lat' in data.coords else float(data[lat_dim].min().values)
                    lat_max = float(data.lat.max().values) if 'lat' in data.coords else float(data[lat_dim].max().values)
                    return lon_min, lon_max, lat_min, lat_max
            
            if len(valid_lat_indices) == 0 or len(valid_lon_indices) == 0:
                # 如果没有有效数据，返回整个数据范围
                lon_min = float(data.lon.min().values)
                lon_max = float(data.lon.max().values)
                lat_min = float(data.lat.min().values)
                lat_max = float(data.lat.max().values)
            else:
                # 获取有效数据的经纬度范围
                lon_min = float(data.lon.isel(lon=valid_lon_indices[0]).values)
                lon_max = float(data.lon.isel(lon=valid_lon_indices[-1]).values)
                lat_min = float(data.lat.isel(lat=valid_lat_indices[0]).values)
                lat_max = float(data.lat.isel(lat=valid_lat_indices[-1]).values)
        
        # 取整：向下取整最小值，向上取整最大值
        lon_min = np.floor(lon_min)
        lon_max = np.ceil(lon_max)
        lat_min = np.floor(lat_min)
        lat_max = np.ceil(lat_max)
        
        return lon_min, lon_max, lat_min, lat_max
        
    except Exception as exc:
        logger.warning(f"计算数据范围失败: {exc}，使用整个数据范围")
        try:
            lon_min = float(data.lon.min().values)
            lon_max = float(data.lon.max().values)
            lat_min = float(data.lat.min().values)
            lat_max = float(data.lat.max().values)
        except:
            # 如果连这个都失败，返回默认值
            lon_min, lon_max, lat_min, lat_max = 0.0, 360.0, -90.0, 90.0
        return lon_min, lon_max, lat_min, lat_max

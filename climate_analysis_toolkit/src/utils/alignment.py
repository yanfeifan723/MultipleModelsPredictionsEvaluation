"""
时间与空间对齐工具
提供在多个计算脚本中重复出现的：
- 时间对齐（标准化到月初，聚合重复月份，取共同月份）
- 多数据集时间对齐（将多个数据集对齐到共同时间范围，时间长度不完全一致是正常的）
- 空间网格对齐（裁剪到经纬度交集，将预报插值到观测网格，线性+最近邻补边）
"""

from typing import Tuple, Optional
import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _normalize_to_month_start(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    periods = pd.to_datetime(times).to_period('M')
    return periods.to_timestamp(how='start')


def align_time_to_monthly(
    obs_data: xr.DataArray,
    fcst_data: xr.DataArray,
    min_common_months: int = 12
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    将观测与预报时间统一到月初并对齐，聚合重复月份并取共同月份。

    Returns: (obs_aligned, fcst_aligned) 或 (None, None) 当共同月份不足。
    """
    try:
        obs_times_norm = _normalize_to_month_start(obs_data.time.to_index())
        fcst_times_norm = _normalize_to_month_start(fcst_data.time.to_index())

        obs_norm = obs_data.copy().assign_coords(time=obs_times_norm)
        fcst_norm = fcst_data.copy().assign_coords(time=fcst_times_norm)

        if obs_norm.indexes['time'].duplicated().any():
            obs_norm = obs_norm.groupby('time').mean('time', skipna=True)
        if fcst_norm.indexes['time'].duplicated().any():
            fcst_norm = fcst_norm.groupby('time').mean('time', skipna=True)

        common_times = obs_norm.time.to_index().intersection(fcst_norm.time.to_index())
        common_times = common_times.sort_values()
        if len(common_times) < min_common_months:
            logger.warning(f"时间对齐失败：共同月份不足（{len(common_times)} < {min_common_months}）")
            return None, None

        obs_aligned = obs_norm.sel(time=common_times)
        fcst_aligned = fcst_norm.sel(time=common_times)
        logger.info(f"时间对齐成功（标准化到月初）: {len(common_times)} 个时间点")
        return obs_aligned, fcst_aligned

    except Exception as e:
        logger.error(f"时间对齐出错: {e}")
        return None, None


def align_spatial_to_obs(
    obs_data: xr.DataArray,
    fcst_data: xr.DataArray,
    no_interp: bool = False
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    将空间网格对齐到观测网格：
    1) 先裁剪到观测与预报经纬度交集
    2) 若网格维度或坐标不同且no_interp=False，将预报插值到观测网格
       使用线性插值 + 最近邻对边界NaN进行回填
    3) 若no_interp=True，只进行裁剪，不进行重采样
    
    Args:
        obs_data: 观测数据
        fcst_data: 预报数据
        no_interp: 如果为True，只裁剪不插值（保持预报数据原始网格）
    """
    try:
        # ===== 空间范围裁剪 =====
        # 说明：
        #   - 网格对齐应只依赖经纬度坐标，不受数据内容（NaN 与否）影响
        #   - 尽量不改变观测数据的空间范围，保持观测作为“基准网格”
        #   - 当 no_interp=True 时，只对预报数据做裁剪，不裁剪观测数据
        if all(dim in obs_data.coords for dim in ['lat', 'lon']) and all(dim in fcst_data.coords for dim in ['lat', 'lon']):
            obs_lat_min = float(obs_data.lat.min())
            obs_lat_max = float(obs_data.lat.max())
            obs_lon_min = float(obs_data.lon.min())
            obs_lon_max = float(obs_data.lon.max())
            fcst_lat_min = float(fcst_data.lat.min())
            fcst_lat_max = float(fcst_data.lat.max())
            fcst_lon_min = float(fcst_data.lon.min())
            fcst_lon_max = float(fcst_data.lon.max())

            # ---- 仅容差匹配（no_interp=True）----
            if no_interp:
                # 直接使用经纬度容差匹配（0.5度容差），找到真正重叠的格点
                # 不进行范围裁剪，只通过经纬度容差匹配找到匹配的格点
                # 保持观测数据完整范围，预报数据只保留与观测格点匹配的格点
                
                # 获取观测和预报的经纬度坐标值
                obs_lat_vals = obs_data.lat.values
                obs_lon_vals = obs_data.lon.values
                fcst_lat_vals = fcst_data.lat.values
                fcst_lon_vals = fcst_data.lon.values
                
                # 容差阈值（0.5度，1度网格间距的一半）
                # 对于1度网格，0.5度容差可以匹配所有合理的网格变体（x.5, x.0, x.05等）
                tolerance = 0.5
                
                # 找到预报网格中与观测网格匹配的格点（0.5度容差内）
                # 对于每个观测格点，找到0.5度容差内的预报格点
                fcst_lat_matched_set = set()
                fcst_lon_matched_set = set()
                
                for obs_lat in obs_lat_vals:
                    # 找到与观测纬度在0.5度容差内的预报纬度格点
                    lat_distances = np.abs(fcst_lat_vals - obs_lat)
                    matched_lat_indices = np.where(lat_distances <= tolerance)[0]
                    if len(matched_lat_indices) > 0:
                        # 选择最近的格点
                        nearest_lat_idx = matched_lat_indices[np.argmin(lat_distances[matched_lat_indices])]
                        fcst_lat_matched_set.add(nearest_lat_idx)
                
                for obs_lon in obs_lon_vals:
                    # 找到与观测经度在0.5度容差内的预报经度格点
                    lon_distances = np.abs(fcst_lon_vals - obs_lon)
                    matched_lon_indices = np.where(lon_distances <= tolerance)[0]
                    if len(matched_lon_indices) > 0:
                        # 选择最近的格点
                        nearest_lon_idx = matched_lon_indices[np.argmin(lon_distances[matched_lon_indices])]
                        fcst_lon_matched_set.add(nearest_lon_idx)
                
                if len(fcst_lat_matched_set) > 0 and len(fcst_lon_matched_set) > 0:
                    # 对索引排序
                    fcst_lat_matched = sorted(list(fcst_lat_matched_set))
                    fcst_lon_matched = sorted(list(fcst_lon_matched_set))
                    
                    # 只保留匹配的格点，保持原始坐标
                    fcst_matched = fcst_data.isel(
                        lat=fcst_lat_matched,
                        lon=fcst_lon_matched
                    )
                    
                    # 计算实际匹配的数据范围
                    matched_lat_min = float(fcst_matched.lat.min())
                    matched_lat_max = float(fcst_matched.lat.max())
                    matched_lon_min = float(fcst_matched.lon.min())
                    matched_lon_max = float(fcst_matched.lon.max())
                    
                    logger.info(
                        f"空间对齐（经纬度容差匹配，0.5度容差）: "
                        f"obs={obs_data.shape}, fcst={fcst_matched.shape}"
                    )
                    logger.info(
                        f"观测数据范围: lat=[{obs_lat_min:.2f}, {obs_lat_max:.2f}], lon=[{obs_lon_min:.2f}, {obs_lon_max:.2f}]"
                    )
                    logger.info(
                        f"预报数据匹配范围: lat=[{matched_lat_min:.2f}, {matched_lat_max:.2f}], "
                        f"lon=[{matched_lon_min:.2f}, {matched_lon_max:.2f}]"
                    )
                    logger.info(
                        f"匹配的格点数: lat={len(fcst_lat_matched)}, lon={len(fcst_lon_matched)}"
                    )
                    
                    # 观测数据保持完整范围，预报数据只保留匹配的格点
                    return obs_data, fcst_matched
                else:
                    logger.warning("预报网格中没有格点在0.5度容差内匹配观测网格，保持原始数据")
                    return obs_data, fcst_data

                # 不做任何插值，直接返回
                return obs_data, fcst_data

            # ---- 允许插值的情况：对观测和预报都裁剪到交集 ----
            lat_min = max(obs_lat_min, fcst_lat_min)
            lat_max = min(obs_lat_max, fcst_lat_max)
            lon_min = max(obs_lon_min, fcst_lon_min)
            lon_max = min(obs_lon_max, fcst_lon_max)

            if lat_max > lat_min and lon_max > lon_min:
                obs_data = obs_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                fcst_data = fcst_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            else:
                logger.warning("观测与预报没有重叠的空间范围，跳过裁剪阶段")

        # 如果 no_interp=True 且前面没有经纬度信息，则直接返回（不做任何空间修改）
        if no_interp:
            logger.info(f"空间对齐（仅时间对齐，不修改空间范围）: obs={obs_data.shape}, fcst={fcst_data.shape}")
            return obs_data, fcst_data

        # 是否需要插值
        need_interp = False
        for dim in ['lat', 'lon']:
            if dim not in obs_data.coords or dim not in fcst_data.coords:
                continue
            if obs_data[dim].size != fcst_data[dim].size:
                need_interp = True
                break
            if not np.allclose(obs_data[dim].values, fcst_data[dim].values, rtol=0, atol=1e-6):
                need_interp = True
                break

        if need_interp:
            # 线性插值 + 最近邻补边界NaN
            fcst_linear = fcst_data.interp(lat=obs_data.lat, lon=obs_data.lon, method='linear')
            fcst_nearest = fcst_data.interp(lat=obs_data.lat, lon=obs_data.lon, method='nearest')
            fcst_on_obs = xr.where(xr.ufuncs.isnan(fcst_linear), fcst_nearest, fcst_linear)
            return obs_data, fcst_on_obs

        return obs_data, fcst_data

    except Exception as e:
        logger.warning(f"空间网格对齐失败，回退到原始网格: {e}")
        return obs_data, fcst_data


def ensure_member_dimension(fcst_data: xr.DataArray) -> xr.DataArray:
    """
    确保预报数据具有 member 维度；若无则创建单成员维度。
    """
    if 'member' not in fcst_data.dims:
        try:
            fcst_data = fcst_data.expand_dims(dim={'member': 1})
        except Exception:
            fcst_data = fcst_data.expand_dims('member')
    return fcst_data


def align_multiple_datasets_to_common_time(
    *datasets: xr.DataArray,
    min_common_times: int = 12,
    return_climatology: bool = False
) -> Tuple[Tuple[xr.DataArray, ...], Optional[xr.DataArray]]:
    """
    将多个数据集对齐到共同的时间范围。
    时间长度不完全一致是正常的，只对能够进行计算的时间范围进行计算。
    
    Args:
        *datasets: 可变数量的数据集（xr.DataArray），每个数据集必须包含'time'维度
        min_common_times: 最小共同时间点数，如果共同时间点少于这个数量，返回None
        return_climatology: 如果为True，返回第一个数据集对齐后的气候态（基于对齐后的时间范围）
    
    Returns:
        Tuple[aligned_datasets, climatology]:
            - aligned_datasets: 对齐后的数据集元组，顺序与输入相同
            - climatology: 如果return_climatology=True，返回第一个数据集的气候态；否则返回None
        
    Example:
        >>> obs_aligned, fcst_aligned, ensemble_aligned = align_multiple_datasets_to_common_time(
        ...     obs_data, fcst_data, ensemble_data, min_common_times=12
        ... )
    """
    if len(datasets) == 0:
        logger.warning("没有提供数据集")
        return tuple(), None
    
    if len(datasets) == 1:
        logger.info("只有一个数据集，无需对齐")
        if return_climatology:
            climatology = datasets[0].mean(dim='time')
            return (datasets[0],), climatology
        return (datasets[0],), None
    
    try:
        # 获取所有数据集的时间索引
        time_indices = []
        for i, data in enumerate(datasets):
            if 'time' not in data.dims:
                logger.warning(f"数据集 {i} 没有时间维度，跳过时间对齐")
                return datasets, None
            time_indices.append(data.time.to_index())
        
        # 计算所有数据集的共同时间范围
        common_times = time_indices[0]
        for time_idx in time_indices[1:]:
            common_times = common_times.intersection(time_idx)
        
        common_times = common_times.sort_values()
        
        # 检查共同时间点数量
        if len(common_times) < min_common_times:
            logger.warning(
                f"共同时间点不足: {len(common_times)} < {min_common_times}，需要至少{min_common_times}个时间点"
            )
            return None, None
        
        # 检查是否需要对齐（如果所有数据集的时间范围完全一致，则不需要对齐）
        time_lengths = [len(idx) for idx in time_indices]
        need_alignment = any(len(common_times) < length for length in time_lengths)
        
        if need_alignment:
            logger.info(
                f"时间对齐: 各数据集时间长度={time_lengths}, "
                f"共同时间={len(common_times)}，仅对共同时间范围进行计算"
            )
        else:
            logger.info(f"所有数据集时间范围完全一致: {len(common_times)} 个时间点")
        
        # 对齐所有数据集到共同时间范围
        aligned_datasets = tuple(data.sel(time=common_times) for data in datasets)
        
        # 如果需要返回气候态，计算第一个数据集的气候态（基于对齐后的时间范围）
        climatology = None
        if return_climatology:
            climatology = aligned_datasets[0].mean(dim='time')
            logger.info(
                f"重新计算气候态（基于对齐后的时间范围），范围: "
                f"[{float(climatology.min()):.2f}, {float(climatology.max()):.2f}]"
            )
        
        return aligned_datasets, climatology
        
    except Exception as e:
        logger.error(f"多数据集时间对齐出错: {e}")
        return None, None



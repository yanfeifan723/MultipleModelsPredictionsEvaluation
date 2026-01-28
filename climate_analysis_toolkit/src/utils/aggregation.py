"""
时间聚合工具
提供统一的年/季/月聚合函数，适用于任意带 time 维度的一维时间序列（或可先对空间取平均）。

聚合口径：
- 年度：先按年份求平均，再对年值求平均
- 季节：DJF 跨年（Dec 归入下一年），按季节-年求平均，再跨年平均
- 月度：各月跨年平均
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

SEASONS: Dict[str, List[int]] = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}

MONTH_NAMES: List[str] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def to_time_series(data: xr.DataArray) -> xr.DataArray:
    """确保数据为一维 time 序列，若包含空间维度请先外部取平均。"""
    if 'time' not in data.dims:
        raise ValueError("输入数据必须包含 time 维度")
    other_dims = tuple(d for d in data.dims if d != 'time')
    if len(other_dims) > 0:
        # 自动对非 time 维度取平均
        data = data.mean(dim=other_dims, skipna=True)
    return data


def aggregate_annual(ts: xr.DataArray) -> float:
    ts = to_time_series(ts)
    try:
        yearly = ts.groupby('time.year').mean('time')
        return float(yearly.mean(skipna=True).values) if yearly.size > 0 else np.nan
    except Exception:
        try:
            return float(ts.mean(skipna=True).values)
        except Exception:
            return np.nan


def aggregate_seasonal(ts: xr.DataArray, seasons: Dict[str, List[int]] = None) -> Dict[str, float]:
    ts = to_time_series(ts)
    seasons = seasons or SEASONS
    result: Dict[str, float] = {}
    for season, months in seasons.items():
        mask = ts['time'].dt.month.isin(months)
        sub = ts.where(mask, drop=True)
        if sub.size == 0:
            result[season] = np.nan
            continue
        if season == 'DJF':
            try:
                season_year = sub['time'].dt.year + xr.where(sub['time'].dt.month == 12, 1, 0)
                season_mean = sub.assign_coords(season_year=season_year).groupby('season_year').mean('time').mean(skipna=True)
                result[season] = float(season_mean.values)
            except Exception:
                result[season] = float(sub.mean(skipna=True).values)
        else:
            try:
                season_mean = sub.groupby('time.year').mean('time').mean(skipna=True)
                result[season] = float(season_mean.values)
            except Exception:
                result[season] = float(sub.mean(skipna=True).values)
    return result


def aggregate_monthly(ts: xr.DataArray, month_names: List[str] = None) -> Dict[str, float]:
    ts = to_time_series(ts)
    month_names = month_names or MONTH_NAMES
    out: Dict[str, float] = {}
    for mi, name in enumerate(month_names, 1):
        sub = ts.where(ts['time'].dt.month == mi, drop=True)
        out[name] = float(sub.mean(skipna=True).values) if sub.size > 0 else np.nan
    return out


def compute_aggregates(ts: xr.DataArray, seasons: Dict[str, List[int]] = None) -> Dict:
    """对一维时间序列计算 annual/seasonal/monthly 聚合，返回 dict。"""
    ts = to_time_series(ts)
    return {
        'annual': aggregate_annual(ts),
        'seasonal': aggregate_seasonal(ts, seasons=seasons or SEASONS),
        'monthly': aggregate_monthly(ts)
    }



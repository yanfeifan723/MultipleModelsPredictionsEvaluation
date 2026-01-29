#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时间种类的Pearson相关系数分析计算模块
计算各个模式与观测的年度、季节、月度空间平均Pearson相关系数

计算每个坐标/维度为(Model,Lead,Time,Lat,Lon)格点的变量
去除对应(Model,Lead,month,Lat,Lon)多年平均的气候态后
得到的每个格点的按照时间顺序拼接的异常值时间序列
与观测的异常值时间序列计算距平相关系数
最终得到number of models x lead time x latitude x longitude的结果。
需要提前.mean('number')来得到ensemble mean

运行环境要求:
- 需要在clim环境中运行: conda activate clim
- 确保已安装所需依赖包

并行运算说明:
- 支持多核并行计算，可显著提高计算效率
- 使用 --parallel 参数启用并行处理
- 使用 --n-jobs 参数指定并行作业数
- 系统会自动检测最优并行作业数

使用示例:
# 串行运行（默认）
python calculate/seasonal_monthly_pearson_analysis.py --var temp

# 并行运行（推荐）
python calculate/seasonal_monthly_pearson_analysis.py --var temp --parallel --n-jobs 8

# 处理所有变量和提前期
python calculate/seasonal_monthly_pearson_analysis.py --parallel

# 仅计算特定提前期
python calculate/seasonal_monthly_pearson_analysis.py --var prec --leadtimes 0 1 2
"""

import sys
import os
from pathlib import Path
import logging
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
# === 添加 cartopy 相关导入（用于高精度地图绘制） ===
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def pearson_r_along_time(x, y):
    """
    计算沿时间轴(axis=0)的 Pearson 相关系数 (仅返回 r)
    用于 xarray.apply_ufunc 的核心函数
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 10:
        return np.nan

    x = x[mask]
    y = y[mask]

    cov = np.sum(x * y)
    var_x = np.sum(x ** 2)
    var_y = np.sum(y ** 2)
    denom = np.sqrt(var_x * var_y)
    if denom == 0:
        return np.nan
    return float(cov / denom)

def pearson_r_along_time_with_p(x, y):
    """
    计算沿时间轴(axis=0)的 Pearson 相关系数和 p-value
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 10:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]
    
    # 使用 scipy.stats.pearsonr 计算
    # 注意：pearsonr 内部会自动去均值，但我们的输入已经是距平。
    # 不过 pearsonr 的输入不要求是距平，它是计算两个序列的相关。
    # 我们的输入是距平，这不影响相关系数的计算结果 (r(x-x_bar, y-y_bar) == r(x, y))。
    # 只要确保输入正确即可。
    # 这里为了保险，直接传原始序列给 pearsonr 也是对的，或者传距平也是对的。
    try:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan

def calculate_p_value_from_r(r, n):
    """根据 r 和 n 计算 p-value (t-test)"""
    if np.isnan(r) or n < 3:
        return np.nan
    if abs(r) >= 1.0:
        return 0.0
    # t-statistic
    t = r * np.sqrt((n - 2) / (1 - r**2))
    # two-sided p-value
    p = 2 * stats.t.sf(np.abs(t), n - 2)
    return p

# 统一导入toolkit路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.parallel_utils import ParallelProcessor, get_optimal_n_jobs
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args
from src.utils.plotting_utils import create_spatial_distribution_figure

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    SEASONS,
)


# 配置参数（从 common_config 导入）
MODELS = MODEL_LIST
# 月份标签
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 配置日志
logger = setup_logging(
    log_file='combined_pearson_analysis.log',
    module_name=__name__
)

class SeasonalMonthlyPearsonAnalyzer:
    """季节和月度Pearson相关系数分析器"""
    
    def __init__(self, var_type: str, n_jobs: Optional[int] = None,
                 use_anomaly_seasonal: bool = True,
                 use_anomaly_monthly: bool = True):
        self.var_type = var_type
        self.data_loader = DataLoader()
        self.n_jobs = n_jobs
        # 是否对季节与月度计算使用距平（ACC）
        self.use_anomaly_seasonal = bool(use_anomaly_seasonal)
        self.use_anomaly_monthly = bool(use_anomaly_monthly)
        # 空间异常相关系数存储路径
        self.spatial_acc_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/spatial_acc/{self.var_type}")
        self.spatial_acc_dir.mkdir(parents=True, exist_ok=True)
        # 绘图存储路径
        self.plot_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/plots/{self.var_type}")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        # === 新增：初始化 boundaries 路径（用于高精度地图绘制） ===
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"
       
    def get_anomalies(self, obs_data: xr.DataArray, fcst_data: xr.DataArray, leadtime: int) -> Tuple[xr.DataArray, xr.DataArray]:
        """计算逐格点月距平场"""
        try:
            # 确保时间坐标是 datetime 类型
            if not np.issubdtype(obs_data.time.dtype, np.datetime64):
                obs_data['time'] = pd.to_datetime(obs_data.time.values)
            if not np.issubdtype(fcst_data.time.dtype, np.datetime64):
                fcst_data['time'] = pd.to_datetime(fcst_data.time.values)

            # 观测的气候态：按月份计算 (1-12月)
            obs_clim = obs_data.groupby('time.month').mean('time')
            
            # 模式的气候态：必须针对特定的 leadtime 计算
            # 注意：这里的 fcst_data 已经是平移过后的目标月时间戳
            # 但是计算气候态时，我们应该确保是针对同一个 leadtime 的所有年份进行平均
            fcst_clim = fcst_data.groupby('time.month').mean('time')
            
            obs_anom = obs_data.groupby('time.month') - obs_clim
            fcst_anom = fcst_data.groupby('time.month') - fcst_clim
            
            return obs_anom, fcst_anom
        except Exception as e:
            logger.error(f"距平场计算失败: {e}")
            return None, None

    def calculate_temporal_acc_monthly_mean(self, obs_anom: xr.DataArray, fcst_anom: xr.DataArray) -> xr.Dataset:
        """
        计算逐月平均的逐格点时间异常相关系数 (Mean Temporal ACC) 及其显著性
        返回 Dataset {'temporal_acc_mean': ..., 'p_value': ...}
        """
        try:
            # 定义权重
            weights = np.cos(np.deg2rad(obs_anom.lat))
            
            monthly_means = []
            monthly_p_values = []
            months = list(range(1, 13))
            
            for month in months:
                # 提取该月的数据 (所有年份)
                obs_m = obs_anom.sel(time=obs_anom.time.dt.month == month)
                fcst_m = fcst_anom.sel(time=fcst_anom.time.dt.month == month)
                
                n_samples = obs_m.sizes['time']
                
                if obs_m.size == 0 or fcst_m.size == 0 or n_samples < 3:
                    monthly_means.append(np.nan)
                    monthly_p_values.append(np.nan)
                    continue
                    
                # 计算逐格点时间相关系数 (lat, lon) - 仅需 r
                tcc_map = xr.apply_ufunc(
                    pearson_r_along_time,
                    obs_m, fcst_m,
                    input_core_dims=[['time'], ['time']],
                    output_core_dims=[[]],
                    vectorize=True,
                    dask='parallelized'
                )
                
                # 计算空间加权平均
                mean_tcc = tcc_map.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
                mean_val = float(mean_tcc.item())
                monthly_means.append(mean_val)
                
                # 基于 Mean TCC 和样本量 N 计算 p-value
                p_val = calculate_p_value_from_r(mean_val, n_samples)
                monthly_p_values.append(p_val)
                
            ds = xr.Dataset(
                {
                    'temporal_acc_mean': (['month'], monthly_means),
                    'p_value': (['month'], monthly_p_values)
                },
                coords={'month': months}
            )
            return ds
        except Exception as e:
            logger.error(f"Temporal ACC 计算失败: {e}")
            return None

    def calculate_temporal_acc_map(self, obs_anom: xr.DataArray, fcst_anom: xr.DataArray) -> xr.Dataset:
        """
        计算逐格点的“全逐月时间序列”异常相关系数（Temporal ACC）及显著性。
        返回 Dataset {'r': ..., 'p': ..., 'significant': ...}
        """
        try:
            res = xr.apply_ufunc(
                pearson_r_along_time_with_p,
                obs_anom,
                fcst_anom,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[[], []],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float, float]
            )
            
            # 确保结果是 DataArray 列表
            if isinstance(res, tuple):
                r_da, p_da = res
            else:
                # 某些版本的 xarray/apply_ufunc 可能行为不同
                logger.error(f"apply_ufunc 返回类型意外: {type(res)}")
                return None

            ds = xr.Dataset({
                'temporal_acc': r_da,
                'p_value': p_da
            })
            ds['temporal_acc'].attrs = {
                "long_name": "Temporal Anomaly Correlation Coefficient",
                "units": "1",
            }
            # 标记显著性 (95%)
            ds['significant'] = ds['p_value'] < 0.05
            
            return ds
        except Exception as e:
            logger.error(f"Temporal ACC map 计算失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def calculate_spatial_acc(self, obs_anom: xr.DataArray, fcst_anom: xr.DataArray) -> xr.DataArray:
        """计算逐时间步的加权空间异常相关系数 (Spatial ACC)"""
        try:
            # 生成权重数组 (与数据形状一致)
            weights = np.cos(np.deg2rad(obs_anom.lat))
            # 广播权重到与经度一致的形状
            weights_2d = weights.broadcast_like(obs_anom.isel(time=0))
            
            acc_list = []
            for t in range(len(obs_anom.time)):
                o = obs_anom.isel(time=t).values.flatten()
                f = fcst_anom.isel(time=t).values.flatten()
                w = weights_2d.values.flatten()
                
                acc = self._weighted_pearson(o, f, w)
                acc_list.append(acc)
                
            return xr.DataArray(
                acc_list, 
                coords={'time': obs_anom.time}, 
                dims=['time'],
                name='spatial_acc'
            )
        except Exception as e:
            logger.error(f"空间加权ACC计算失败: {e}")
            return None

    def _weighted_pearson(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """计算两个向量的加权 Pearson 相关系数"""
        # 掩码处理：确保 x, y, w 都没有 NaN
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
        if np.sum(mask) < 10:  # 有效样本太少
            return np.nan
            
        x, y, w = x[mask], y[mask], w[mask]
        
        # 1. 计算加权平均值
        def weighted_mean(v, weights):
            return np.sum(v * weights) / np.sum(weights)
        
        mean_x = weighted_mean(x, w)
        mean_y = weighted_mean(y, w)
        
        # 2. 计算中心化后的向量
        x_centered = x - mean_x
        y_centered = y - mean_y
        
        # 3. 计算加权协方差和方差
        cov_xy = np.sum(w * x_centered * y_centered)
        var_x = np.sum(w * x_centered**2)
        var_y = np.sum(w * y_centered**2)
        
        # 4. 计算相关系数
        denominator = np.sqrt(var_x * var_y)
        if denominator == 0:
            return np.nan
            
        return float(cov_xy / denominator)

    def load_and_preprocess_data(self, model: str, leadtime: int) -> Tuple[xr.DataArray, xr.DataArray]:
        """加载和预处理数据"""
        try:
            # 加载观测数据
            obs_data = self.data_loader.load_obs_data(self.var_type)
            obs_data = obs_data.resample(time='1MS').mean()
            obs_data = obs_data.sel(time=slice('1993', '2020'))
            
            # 加载模型数据
            fcst_data = self.data_loader.load_forecast_data(model, self.var_type, leadtime)
            if fcst_data is None:
                return None, None
                
            fcst_data = fcst_data.resample(time='1MS').mean()
            # 重要：sfc 文件中 time(6) 本身就是“目标月份”(例如 199301 文件 time=1993-01..1993-06)，
            # load_forecast_data 已按 leadtime 选择对应 index，并保留了正确的目标月份 time 坐标。
            # 因此这里不要再额外做 +leadtime 的时间平移（否则会二次平移导致对齐错误）。
            
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))
            
            # 时间对齐 (使用目标月份进行对齐)
            common_times = obs_data.time.to_index().intersection(fcst_data.time.to_index())
            if len(common_times) < 12:
                logger.warning(f"时间对齐失败: {model} L{leadtime}, 共同时间点不足")
                return None, None
            
            obs_aligned = obs_data.sel(time=common_times)
            fcst_aligned = fcst_data.sel(time=common_times)
            
            
            # 空间插值对齐
            try:
                fcst_interpolated = fcst_aligned.interp(
                    lat=obs_aligned.lat,
                    lon=obs_aligned.lon,
                    method='linear'
                )
            except Exception as e:
                logger.warning(f"空间插值失败: {model} L{leadtime}: {e}")
                return None, None

            logger.info(f"数据加载成功: {model} L{leadtime}, 时间点: {len(common_times)}")
            return obs_aligned, fcst_interpolated
            
        except Exception as e:
            logger.error(f"数据加载失败: {model} L{leadtime}: {e}")
            return None, None
    
    def calculate_spatial_average_correlation(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> float:
        """计算空间平均后的Pearson相关系数"""
        try:
            # 定义权重
            weights = np.cos(np.deg2rad(obs_data.lat))
            # 计算加权空间平均
            obs_mean = obs_data.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            fcst_mean = fcst_data.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            
            # 移除NaN值
            valid_mask = ~(np.isnan(obs_mean.values) | np.isnan(fcst_mean.values))
            if np.sum(valid_mask) < 3:
                return np.nan
            
            obs_valid = obs_mean.values[valid_mask]
            fcst_valid = fcst_mean.values[valid_mask]
            
            # 计算Pearson相关系数
            corr, _ = stats.pearsonr(obs_valid, fcst_valid)
            return corr
            
        except Exception as e:
            logger.error(f"相关系数计算失败: {e}")
            return np.nan
    
    def calculate_interannual_correlation(self, obs_data: xr.DataArray, fcst_data: xr.DataArray,
                                          use_anomaly: bool = True) -> float:
        """计算加权后的年际相关系数"""
        return np.nan
        try:
            # 1. 定义权重
            weights = np.cos(np.deg2rad(obs_data.lat))
            
            # 2. 空间平均（使用 xarray 自带的加权平均功能，更稳健）
            obs_mean_ts = obs_data.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            fcst_mean_ts = fcst_data.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)

            if use_anomaly:
                # 去气候态（按月距平）
                obs_clim = obs_mean_ts.groupby('time.month').mean('time')
                fcst_clim = fcst_mean_ts.groupby('time.month').mean('time')
                obs_anom = obs_mean_ts.groupby('time.month') - obs_clim
                fcst_anom = fcst_mean_ts.groupby('time.month') - fcst_clim
            else:
                obs_anom = obs_mean_ts
                fcst_anom = fcst_mean_ts

            # 3. 年平均
            obs_year = obs_anom.groupby('time.year').mean('time')
            fcst_year = fcst_anom.groupby('time.year').mean('time')

            # 4. 对齐年份并计算常规 Pearson (此时已是标量序列，不需要再次加权)
            common_years = np.intersect1d(obs_year['year'].values, fcst_year['year'].values)
            if common_years.size < 3: return np.nan

            obs_vec = obs_year.sel(year=common_years).values
            fcst_vec = fcst_year.sel(year=common_years).values

            valid = ~(np.isnan(obs_vec) | np.isnan(fcst_vec))
            if np.sum(valid) < 3: return np.nan
            
            corr, _ = stats.pearsonr(obs_vec[valid], fcst_vec[valid])
            return float(corr)
        except Exception as e:
            logger.error(f"年际相关系数计算失败: {e}")
            return np.nan

    def calculate_annual_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[int, float]:
        """计算年度的相关系数，返回 {year:int -> corr:float} 字典（兼容旧版 xarray）"""
        return {}
        annual_corrs: Dict[int, float] = {}
        try:
            years = np.unique(obs_data['time'].dt.year.values)
        except Exception:
            years = []
        for year_val in years:
            try:
                year_int = int(year_val)
            except Exception:
                continue
            obs_year = obs_data.sel(time=obs_data.time.dt.year == year_int)
            fcst_year = fcst_data.sel(time=fcst_data.time.dt.year == year_int)
            corr = self.calculate_spatial_average_correlation(obs_year, fcst_year)
            annual_corrs[year_int] = corr
        return annual_corrs
    
    def calculate_seasonal_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[str, float]:
        """计算各季节的相关系数"""
        return {}
        seasonal_corrs = {}
        
        for season, months in SEASONS.items():
            # 选择季节数据
            season_mask = obs_data['time'].dt.month.isin(months)
            obs_season = obs_data.where(season_mask, drop=True)
            fcst_season = fcst_data.where(season_mask, drop=True)

            # 若使用距平：先按历月去气候态，再按年份聚合季节平均
            if self.use_anomaly_seasonal:
                obs_clim = obs_data.groupby('time.month').mean('time')
                fcst_clim = fcst_data.groupby('time.month').mean('time')
                obs_season = obs_season.groupby('time.month') - obs_clim
                fcst_season = fcst_season.groupby('time.month') - fcst_clim

            # 按季节-年聚合，DJF 采用跨年归属：Dec 归入下一年（season_year = year + (month==12)）
            if 'time' in obs_season.dims and obs_season.sizes.get('time', 0) > 0:
                try:
                    if season == 'DJF':
                        obs_sy = obs_season['time'].dt.year + xr.where(obs_season['time'].dt.month == 12, 1, 0)
                        fcst_sy = fcst_season['time'].dt.year + xr.where(fcst_season['time'].dt.month == 12, 1, 0)
                        obs_season = obs_season.assign_coords(season_year=obs_sy).groupby('season_year').mean('time').rename({'season_year': 'time'})
                        fcst_season = fcst_season.assign_coords(season_year=fcst_sy).groupby('season_year').mean('time').rename({'season_year': 'time'})
                    else:
                        obs_season = obs_season.groupby('time.year').mean('time').rename({'year': 'time'})
                        fcst_season = fcst_season.groupby('time.year').mean('time').rename({'year': 'time'})
                except Exception as e:
                    # 回退：若groupby失败则保持原样
                    logger.error(f"按季节-年聚合失败: {e}")
                    pass

            if obs_season.sizes.get('time', 0) == 0:
                seasonal_corrs[season] = np.nan
                continue
            
            # 计算相关系数
            corr = self.calculate_spatial_average_correlation(obs_season, fcst_season)
            seasonal_corrs[season] = corr
            
        return seasonal_corrs
    
    def calculate_monthly_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[str, float]:
        """计算各月份的相关系数"""
        return {}
        monthly_corrs = {}
        
        for month_idx, month_name in enumerate(MONTHS, 1):
            # 选择月份数据
            month_mask = obs_data['time'].dt.month == month_idx
            obs_month = obs_data.where(month_mask, drop=True)
            fcst_month = fcst_data.where(month_mask, drop=True)

            # 若使用距平：对该历月的序列做距平（跨年去该月气候态）
            if self.use_anomaly_monthly and obs_month.sizes.get('time', 0) > 0:
                # 使用整段数据的该月均值作为气候态
                obs_month_clim = obs_data.where(month_mask, drop=True).mean('time')
                fcst_month_clim = fcst_data.where(month_mask, drop=True).mean('time')
                obs_month = obs_month - obs_month_clim
                fcst_month = fcst_month - fcst_month_clim
            
            if len(obs_month.time) == 0:
                monthly_corrs[month_name] = np.nan
                continue
            
            # 计算相关系数
            corr = self.calculate_spatial_average_correlation(obs_month, fcst_month)
            monthly_corrs[month_name] = corr
            
        return monthly_corrs
    
    def add_china_map_details(self, ax, data, lon, lat, levels, cmap, draw_scs=True):
        """
        添加中国国界线、河流、海岸线和南海子图（复用自 climatology_analysis.py）
        
        注意：此方法仅修改子图内部内容，不影响整体布局
        
        Args:
            ax: matplotlib axes 对象（必须是 GeoAxes，即使用 projection=ccrs.PlateCarree()）
            data: 要绘制的数据数组（用于南海子图）
            lon: 经度坐标
            lat: 纬度坐标
            levels: 等高线级别
            cmap: 色标
            draw_scs: 是否绘制南海子图
        """
        # 指定需要加载的国界线文件列表
        bou_paths = [
            Path("/sas12t1/ffyan/boundaries/中国_省1.shp"),
            Path("/sas12t1/ffyan/boundaries/中国_省2.shp")
        ]
        
        # 河流路径
        hyd_path = self.boundaries_dir / "河流.shp"
        if not hyd_path.exists():
            hyd_path = None
        
        # --- 1. 绘制河流 ---
        if hyd_path:
            try:
                reader = shpreader.Reader(str(hyd_path))
                ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
                                edgecolor='blue', facecolor='none', 
                                linewidth=0.6, alpha=0.6, zorder=5)
            except Exception as e:
                ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6, zorder=5)
        else:
            ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6, zorder=5)
        
        # --- 2. 绘制海岸线 ---
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', zorder=50)
        
        # --- 3. 绘制国界线 ---
        loaded_borders = False
        for bou_path in bou_paths:
            if bou_path.exists():
                try:
                    reader = shpreader.Reader(str(bou_path))
                    geoms = list(reader.geometries())
                    ax.add_geometries(geoms, ccrs.PlateCarree(), 
                                    edgecolor='black', facecolor='none', 
                                    linewidth=0.6, zorder=100)
                    loaded_borders = True
                except Exception:
                    pass
        
        if not loaded_borders:
            ax.add_feature(cfeature.BORDERS, linewidth=1.0, zorder=100)
        
        # --- 4. 绘制南海子图 ---
        if draw_scs:
            try:
                # 位置参数：[x, y, width, height] (相对于父ax的坐标 0-1)
                scs_width = 0.33
                scs_height = 0.35
                sub_ax = ax.inset_axes([0.7548, 0, scs_width, scs_height], 
                                      projection=ccrs.PlateCarree())
                
                # 设置白色背景
                sub_ax.patch.set_facecolor('white')
                
                # 设置南海范围
                sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
                
                # 在子图中绘制相同的数据
                sub_ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, levels=levels, extend='both')
                
                # 在子图中也绘制海岸线
                sub_ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray', zorder=50)
                
                # 在子图中也绘制国界线
                if loaded_borders:
                    for bou_path in bou_paths:
                        if bou_path.exists():
                            try:
                                reader = shpreader.Reader(str(bou_path))
                                geoms_sub = list(reader.geometries())
                                sub_ax.add_geometries(geoms_sub, ccrs.PlateCarree(),
                                                    edgecolor='black', facecolor='none', 
                                                    linewidth=0.6, zorder=100)
                            except Exception:
                                pass
                
                # 移除子图刻度，保留边框
                sub_ax.tick_params(left=False, labelleft=False, 
                                  bottom=False, labelbottom=False)
                
                # 设置子图边框
                for spine in sub_ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.0)
                    
            except Exception as e:
                logger.warning(f"南海子图绘制失败: {e}")
    
    def analyze_all_models_leadtimes(self, models: List[str] = None, leadtimes: List[int] = None) -> Dict:
        """分析所有模式和提前期"""
        if models is None:
            models = MODELS
        if leadtimes is None:
            leadtimes = LEADTIMES
        
        logger.info(f"开始{self.var_type}的Pearson分析")
        
        results = {}
        # 存储用于绘图的“逐月 Mean Temporal ACC”(leadtime, month): {model: [xr.DataArray(leadtime, month)]}
        model_spatial_acc_data = {model: [] for model in models}
        # 存储每个 leadtime 的 Temporal ACC 空间分布图(lat, lon): {model: {leadtime: xr.Dataset}}
        model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]] = {model: {} for model in models}
        
        for leadtime in leadtimes:
            logger.info(f"处理预报时效: L{leadtime}")
            
            annual_data = {}  # {model: {year: corr}}  逐年诊断用
            annual_interannual = {}  # {model: corr}    年际相关（主指标）
            seasonal_data = {}  # {model: {season: corr}}
            monthly_data = {}   # {model: {month: corr}}
            
            for model in models:
                logger.info(f"处理模型: {model}")
                
                # 加载数据
                obs_data, fcst_data = self.load_and_preprocess_data(model, leadtime)
                if obs_data is None or fcst_data is None:
                    logger.warning(f"跳过 {model} L{leadtime}: 数据加载失败")
                    continue
                
                # --- 新增：计算逐格点时间异常相关系数 (Temporal ACC) 并求空间平均 ---
                obs_anom_field, fcst_anom_field = self.get_anomalies(obs_data, fcst_data, leadtime)
                if obs_anom_field is not None and fcst_anom_field is not None:
                    # 1) 全逐月序列的逐格点 Temporal ACC map (lat, lon)
                    acc_map = self.calculate_temporal_acc_map(obs_anom_field, fcst_anom_field)
                    if acc_map is not None:
                        model_temporal_acc_maps[model][int(leadtime)] = acc_map

                    # 使用新的 Temporal ACC Map 计算方法
                    acc_monthly = self.calculate_temporal_acc_monthly_mean(obs_anom_field, fcst_anom_field)
                    if acc_monthly is not None:
                        # 扩展维度以便后续合并
                        acc_monthly = acc_monthly.expand_dims(leadtime=[leadtime])
                        model_spatial_acc_data[model].append(acc_monthly)

                # 计算年度相关系数
                annual_corrs = self.calculate_annual_correlations(obs_data, fcst_data)
                annual_data[model] = annual_corrs

                # 计算年际相关（按年平均的年际时间序列相关，默认用距平）
                interannual_corr = self.calculate_interannual_correlation(obs_data, fcst_data, use_anomaly=True)
                annual_interannual[model] = interannual_corr
                
                # 计算季节相关系数
                seasonal_corrs = self.calculate_seasonal_correlations(obs_data, fcst_data)
                seasonal_data[model] = seasonal_corrs
                
                # 计算月度相关系数
                monthly_corrs = self.calculate_monthly_correlations(obs_data, fcst_data)
                monthly_data[model] = monthly_corrs
                
                logger.info(f"完成 {model} L{leadtime}")
            
            results[leadtime] = {
                'annual': annual_data,
                'annual_interannual': annual_interannual,
                'seasonal': seasonal_data,
                'monthly': monthly_data
            }
        
        # 保存空间ACC到NetCDF
        self.save_spatial_acc_to_nc(model_spatial_acc_data)
        # 保存逐格点 Temporal ACC 空间分布(每个模式 6 个 leadtime)
        self.save_temporal_acc_maps_to_nc(model_temporal_acc_maps)
        
        # 新增：绘制结果图表
        self.plot_spatial_acc_monthly_contour(model_spatial_acc_data)
        self.plot_spatial_acc_heatmap_diverging_discrete(model_spatial_acc_data)
        self.plot_spatial_acc_leadtime_timeseries(model_spatial_acc_data)
        # 绘制ACC空间分布图 (Lead 0 和 Lead 3)
        self.plot_acc_spatial_maps(model_temporal_acc_maps)
        
        return results

    def plot_acc_spatial_maps(self, model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]]):
        """
        绘制ACC空间分布图 (Lead 0 和 Lead 3)
        
        重写版本：添加中国国界线、河流和南海子图
        保持原有布局不变，仅修改子图内部的地图细节
        """
        try:
            logger.info(f"开始绘制ACC空间分布图（含中国地图细节）: {self.var_type}")
            
            # 准备数据
            plot_models = list(model_temporal_acc_maps.keys())
            leadtimes = [0, 3]
            
            if not plot_models:
                logger.warning("没有可用的ACC空间分布数据")
                return
            
            # 设置参数
            lon_range = (70, 140)
            lat_range = (15, 55)
            vmin, vmax = -1, 1
            cmap = 'RdBu_r'
            n_levels = 20
            levels = np.linspace(vmin, vmax, n_levels + 1)
            
            # 创建图形 - 保持与原函数相同的布局
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig,
                         hspace=0.25, wspace=0.15,
                         left=0.05, right=0.92, top=0.94, bottom=0.06)
            
            # 计算经纬度刻度
            lon_ticks = np.arange(75, 141, 15)
            lat_ticks = np.arange(20, 56, 10)
            
            # 绘制每个leadtime
            for lt_idx, leadtime in enumerate(leadtimes):
                row_start = lt_idx * 2
                
                # 第一行：空白 + 3个模型
                ax_blank = fig.add_subplot(gs[row_start, 0])
                ax_blank.axis('off')
                
                for col_idx in range(3):
                    if col_idx >= len(plot_models):
                        ax = fig.add_subplot(gs[row_start, col_idx + 1])
                        ax.axis('off')
                        continue
                    
                    model = plot_models[col_idx]
                    if leadtime not in model_temporal_acc_maps[model]:
                        ax = fig.add_subplot(gs[row_start, col_idx + 1])
                        ax.axis('off')
                        continue
                    
                    acc_ds = model_temporal_acc_maps[model][leadtime]
                    data = acc_ds['temporal_acc']
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    # 创建cartopy地图
                    ax = fig.add_subplot(gs[row_start, col_idx + 1], 
                                       projection=ccrs.PlateCarree())
                    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                                 crs=ccrs.PlateCarree())
                    
                    # 添加基础特征（陆地和海洋底色）
                    ax.add_feature(cfeature.LAND, alpha=0.1)
                    ax.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 添加网格
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xformatter = LongitudeFormatter(number_format='.0f')
                    gl.yformatter = LatitudeFormatter(number_format='.0f')
                    
                    # 绘制填色图
                    im = ax.contourf(data.lon, data.lat, data,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap, levels=levels, extend='both')
                    
                    # === 关键：添加中国地图细节 ===
                    self.add_china_map_details(ax, data, data.lon, data.lat, 
                                              levels, cmap, draw_scs=True)
                    
                    # 显著性打点
                    if 'significant' in acc_ds:
                        sig_mask = acc_ds['significant'].values
                        if np.any(sig_mask):
                            X, Y = np.meshgrid(data.lon, data.lat)
                            ax.scatter(X[sig_mask][::2], Y[sig_mask][::2], 
                                     transform=ccrs.PlateCarree(),
                                     s=1, c='black', alpha=0.5, marker='.')
                    
                    # 添加标题
                    title_text = f"({chr(97 + col_idx)}) {display_name}"
                    ax.text(0.02, 0.96, title_text,
                           transform=ax.transAxes, fontsize=18, fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
                    
                    # 添加leadtime标签（第一个模型）
                    if col_idx == 0:
                        ax.text(0.98, 0.96, f'L{leadtime}',
                               transform=ax.transAxes, fontsize=18, fontweight='bold',
                               verticalalignment='top', horizontalalignment='right')
                
                # 第二行：4个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(plot_models):
                        ax = fig.add_subplot(gs[row_start + 1, col_idx])
                        ax.axis('off')
                        continue
                    
                    model = plot_models[model_idx]
                    if leadtime not in model_temporal_acc_maps[model]:
                        ax = fig.add_subplot(gs[row_start + 1, col_idx])
                        ax.axis('off')
                        continue
                    
                    acc_ds = model_temporal_acc_maps[model][leadtime]
                    data = acc_ds['temporal_acc']
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    # 创建cartopy地图
                    ax = fig.add_subplot(gs[row_start + 1, col_idx],
                                       projection=ccrs.PlateCarree())
                    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                                 crs=ccrs.PlateCarree())
                    
                    # 添加基础特征
                    ax.add_feature(cfeature.LAND, alpha=0.1)
                    ax.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 添加网格
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xformatter = LongitudeFormatter(number_format='.0f')
                    gl.yformatter = LatitudeFormatter(number_format='.0f')
                    
                    # 绘制填色图
                    im = ax.contourf(data.lon, data.lat, data,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap, levels=levels, extend='both')
                    
                    # === 关键：添加中国地图细节 ===
                    self.add_china_map_details(ax, data, data.lon, data.lat,
                                              levels, cmap, draw_scs=True)
                    
                    # 显著性打点
                    if 'significant' in acc_ds:
                        sig_mask = acc_ds['significant'].values
                        if np.any(sig_mask):
                            X, Y = np.meshgrid(data.lon, data.lat)
                            ax.scatter(X[sig_mask][::2], Y[sig_mask][::2],
                                     transform=ccrs.PlateCarree(),
                                     s=1, c='black', alpha=0.5, marker='.')
                    
                    # 添加标题
                    title_text = f"({chr(97 + model_idx)}) {display_name}"
                    ax.text(0.02, 0.96, title_text,
                           transform=ax.transAxes, fontsize=18, fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
            
            # 添加colorbar
            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Temporal ACC', fontsize=14, labelpad=10)
            cbar.ax.tick_params(labelsize=12)
            
            # # 添加总标题
            # fig.suptitle(f'{self.var_type.upper()} - Temporal Anomaly Correlation Coefficient (ACC)\n'
            #             f'Black dots: p < 0.05 (95% significance)',
            #             fontsize=16, fontweight='bold', y=0.98)
            
            # 保存图像
            output_file = self.plot_dir / f"acc_spatial_maps_L0_L3_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            logger.info(f"ACC空间分布图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"ACC空间分布图绘制失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spatial_acc_heatmap_diverging_discrete(self, model_spatial_acc_data):
        """
        绘制离散型、对称分布（RdBu_r）的空间ACC热图
        """
        try:
            logger.info(f"开始绘制离散型对称热图: {self.var_type}")
            # 1. 基础数据准备
            months = np.arange(1, 13)
            leadtimes = np.arange(0, 6)
            plot_models = [m for m in MODELS if m in model_spatial_acc_data and model_spatial_acc_data[m]]
            
            # 预计算所有矩阵
            model_matrices = {}
            model_p_matrices = {}  # 存储 p-value 矩阵
            
            for model in plot_models:
                combined_ds = xr.concat(model_spatial_acc_data[model], dim='leadtime').sortby('leadtime')
                
                # 提取 r
                matrix = combined_ds['temporal_acc_mean'].reindex(leadtime=leadtimes, month=months).values
                model_matrices[model] = matrix
                
                # 提取 p-value (如果存在)
                if 'p_value' in combined_ds:
                    p_matrix = combined_ds['p_value'].reindex(leadtime=leadtimes, month=months).values
                    model_p_matrices[model] = p_matrix
                else:
                    model_p_matrices[model] = None

            if not model_matrices: return
            
            # --- 动态计算对称离散色标 ---
            # 范围固定为 -1 到 1，每 0.1 一个段
            n_bins = 20  # -1 到 1 共 20 个 0.1 的区间
            levels = np.linspace(-1, 1, n_bins + 1)
            
            # 创建离散色板：RdBu_r (反转红蓝: 蓝为负, 红为正)
            cmap = plt.get_cmap('RdBu_r', n_bins)
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            # 2. 绘图布局
            fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True, constrained_layout=True)
            axes[0, 0].axis('off')

            # 网格边界定义
            x_edges = np.arange(0.5, 13.5, 1)
            y_edges = np.arange(-0.5, 6.5, 1)
            X, Y = np.meshgrid(x_edges, y_edges)
            
            # 打点用的中心坐标
            X_center, Y_center = np.meshgrid(months, leadtimes)

            mesh = None
            for i, model in enumerate(plot_models):
                row, col = (0, i + 1) if i < 3 else (1, i - 3)
                ax = axes[row, col]
                
                matrix = model_matrices[model]
                p_matrix = model_p_matrices[model]
                
                # 绘制热图
                mesh = ax.pcolormesh(X, Y, matrix, cmap=cmap, norm=norm,
                                     edgecolor='face', linewidth=0.1, shading='flat')

                # --- 显著性打点 ---
                if p_matrix is not None:
                    # 找到 p < 0.05 的位置 (且数据不是 NaN)
                    sig_mask = (p_matrix < 0.05) & np.isfinite(matrix)
                    if np.any(sig_mask):
                        ax.scatter(X_center[sig_mask], Y_center[sig_mask], 
                                   marker='.', s=15, color='black', alpha=0.6, label='p<0.05')

                # 装饰细节
                display_name = model.split('-')[0]
                ax.set_title(f"({chr(97+i)}) {display_name}", fontsize=15, fontweight='bold', loc='left', pad=15)
                ax.set_xticks(months)
                ax.set_yticks(leadtimes)
                
                if col == 0 or (row == 0 and col == 1):
                    ax.set_ylabel('Lead Time', fontsize=14)
                if row == 1:
                    ax.set_xlabel('Month', fontsize=14)
                ax.grid(False)

            # 3. 全局 Colorbar 设置
            if mesh:
                # spacing='proportional' 确保色条刻度与 levels 一致
                cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', 
                                    fraction=0.02, pad=0.04, ticks=levels)
                cbar.set_label('Mean Temporal ACC', fontsize=14, labelpad=15)
                
            output_file = self.plot_dir / f"spatial_acc_heatmap_diverging_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"离散型对称热图已保存: {output_file}")

        except Exception as e:
            logger.error(f"离散型对称热图绘制失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def save_temporal_acc_maps_to_nc(self, model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]]):
        """保存每个模式的 Temporal ACC 空间分布图 (leadtime, lat, lon) 到 NetCDF。"""
        out_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/temporal_acc_maps/{self.var_type}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for model, lt_to_map in model_temporal_acc_maps.items():
            if not lt_to_map:
                continue
            try:
                leadtimes_sorted = sorted(int(k) for k in lt_to_map.keys())
                # map 是 Dataset，包含 r, p, significant
                maps = [lt_to_map[lt].expand_dims(leadtime=[lt]) for lt in leadtimes_sorted]
                ds = xr.concat(maps, dim='leadtime').sortby('leadtime')
                
                save_path = out_dir / f"temporal_acc_map_{model}_{self.var_type}.nc"
                ds.to_netcdf(save_path)
                logger.info(f"Temporal ACC 空间分布已保存: {save_path}")
            except Exception as e:
                logger.error(f"保存 Temporal ACC 空间分布失败 ({model}): {e}")

    def save_spatial_acc_to_nc(self, model_spatial_acc_data: Dict[str, List[xr.Dataset]]):
        """将空间ACC时间序列保存为NetCDF文件"""
        for model, acc_list in model_spatial_acc_data.items():
            if not acc_list:
                continue
            try:
                # 合并所有leadtime
                combined_ds = xr.concat(acc_list, dim='leadtime')
                # 排序leadtime
                combined_ds = combined_ds.sortby('leadtime')
                
                save_path = self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.nc"
                combined_ds.to_netcdf(save_path)
                logger.info(f"空间ACC已保存到: {save_path}")
                
                # 同时保存一份CSV版本 (转置为 month x leadtime) - 仅保存 r 值
                try:
                    # 提取 r
                    da_r = combined_ds['temporal_acc_mean']
                    df = da_r.to_pandas()
                    # DataArray 转 DataFrame 后，列是 month，行是 leadtime
                    # 我们希望行是 month (1-12)，列是 leadtime
                    if isinstance(df, pd.DataFrame):
                        df = df.T
                    
                    csv_path = self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.csv"
                    df.to_csv(csv_path)
                except Exception as e:
                    logger.warning(f"空间ACC保存CSV失败 ({model}): {e}")
                    
            except Exception as e:
                logger.error(f"空间ACC保存NetCDF失败 ({model}): {e}")

    def save_data_to_csv(self, results: Dict, save_dir: str = None):
        """保存数据到CSV文件"""
        if save_dir is None:
            save_dir = f"/sas12t1/ffyan/output/pearson_analysis/summary/{self.var_type}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        for leadtime, leadtime_data in results.items():
            # 创建数据框
            annual_data = leadtime_data.get('annual', {})
            annual_interannual = leadtime_data.get('annual_interannual', {})
            seasonal_data = leadtime_data.get('seasonal', {})
            monthly_data = leadtime_data.get('monthly', {})
            
            # 创建年度（逐年诊断）数据框
            annual_df = pd.DataFrame(annual_data).T
            annual_df = annual_df.reindex(columns=list(range(1993, 2021)))

            # 创建年际相关数据框（单值）
            interannual_df = pd.DataFrame({'Interannual': annual_interannual})
            
            # 创建季节数据框
            seasonal_df = pd.DataFrame(seasonal_data).T
            seasonal_df = seasonal_df.reindex(columns=list(SEASONS.keys()))
            
            # 创建月度数据框
            monthly_df = pd.DataFrame(monthly_data).T
            monthly_df = monthly_df.reindex(columns=MONTHS)
            
            # 创建年度数据框
            annual_path = os.path.join(save_dir, f"annual_correlations_L{leadtime}.csv")
            interannual_path = os.path.join(save_dir, f"annual_interannual_L{leadtime}.csv")
            seasonal_path = os.path.join(save_dir, f"seasonal_correlations_L{leadtime}.csv")
            monthly_path = os.path.join(save_dir, f"monthly_correlations_L{leadtime}.csv")
             
            annual_df.to_csv(annual_path)
            interannual_df.to_csv(interannual_path)
            seasonal_df.to_csv(seasonal_path)
            monthly_df.to_csv(monthly_path)
            
            logger.info(f"数据已保存: {annual_path}, {interannual_path}, {seasonal_path}, {monthly_path}")

    def plot_spatial_acc_monthly_contour(self, model_spatial_acc_data: Dict[str, List[xr.DataArray]]):
        """绘制空间ACC逐月等高线图"""
        return
        try:
            logger.info(f"开始绘制空间ACC逐月等高线图: {self.var_type}")
            
            # 准备数据：{model: np.array(leadtime, month)}
            model_contour_data = {}
            all_values = []
            
            model_names = [m for m in model_spatial_acc_data.keys() if model_spatial_acc_data[m]]
            if not model_names:
                logger.warning("没有可用的空间ACC数据用于绘图")
                return

            for model in model_names:
                acc_list = model_spatial_acc_data[model]
                combined_acc = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                
                # combined_acc 维度为 (leadtime, month)
                leadtimes = combined_acc.leadtime.values
                months = list(range(1, 13))
                contour_matrix = np.full((len(leadtimes), len(months)), np.nan)
                
                for li, lt in enumerate(leadtimes):
                    # 提取该leadtime的数据
                    lt_data = combined_acc.sel(leadtime=lt)
                    
                    for mi, mon in enumerate(months):
                        if mon in lt_data.month.values:
                            val = float(lt_data.sel(month=mon).item())
                            if np.isfinite(val):
                                contour_matrix[li, mi] = val
                                all_values.append(val)
                
                model_contour_data[model] = contour_matrix
            
            if not all_values:
                logger.warning("没有有效的ACC数据用于绘制等高线图")
                return

            # 创建图形：2行4列布局，使用 constrained_layout 自动处理间距
            fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)
            
            # 隐藏第一行第一个子图 (实现第一排3个右对齐)
            axes[0, 0].axis('off')

            # 排序模型以保证一致性
            plot_models = [m for m in MODELS if m in model_contour_data]
            
            for i, model in enumerate(plot_models):
                if i < 3:
                     # 第一排3个，右对齐。
                     row = 0
                     col = i + 1 # 1, 2, 3
                else:
                     # 第二排4个
                     row = 1
                     col = i - 3 # 0, 1, 2, 3
                
                ax = axes[row, col]
                display_name = model.replace('-mon', '').replace('mon-', '')
                
                data = model_contour_data[model]
                lts = combined_acc.leadtime.values
                mons = list(range(1, 13))
                X, Y = np.meshgrid(mons, lts)
                
                # 确定等高线级别
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    data_min, data_max = np.nanmin(data), np.nanmax(data)
                    if data_max - data_min > 1e-6:
                        # 使用 MaxNLocator 自动选择等间距且“美观”的刻度（类似 0.1, 0.2, 0.5 等）
                        levels = ticker.MaxNLocator(nbins=7, prune=None).tick_values(data_min, data_max)
                    else:
                        levels = 6
                else:
                    levels = 6
                
                contours = ax.contour(X, Y, data, levels=levels, colors='black', linewidths=1.2, alpha=0.8)
                ax.clabel(contours, inline=True, fontsize=12, fmt='%.2f')
                
                # 标题增加 pad，防止贴合坐标轴
                ax.set_title(f"({chr(97+i)}) {display_name}", fontsize=16, fontweight='bold', loc='left', pad=15)
                
                # 网格线
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # 坐标轴刻度设置
                ax.set_xticks(mons)
                ax.set_yticks(lts)
                
                ax.tick_params(axis='both', labelsize=12)
                
                # 只在第二行子图添加 Month 标注
                if row == 1:
                    ax.set_xlabel('Month', fontsize=14)
                
                # 只在每行第一个有图的子图添加 Lead Time 标注
                if col == 0 or (row == 0 and col == 1):
                    ax.set_ylabel('Lead Time', fontsize=14)
            
            output_file = self.plot_dir / f"spatial_acc_monthly_contour_{self.var_type}.png"
            plt.savefig(output_file, dpi=300)
            plt.close()
            logger.info(f"空间ACC逐月等高线图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制空间ACC逐月等高线图出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def plot_spatial_acc_leadtime_timeseries(self, model_spatial_acc_data: Dict[str, List[xr.DataArray]]):
        """绘制空间ACC随leadtime变化的折线图"""
        try:
            logger.info(f"开始绘制空间ACC随leadtime变化的折线图: {self.var_type}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 设置颜色
            plot_models = [m for m in MODELS if m in model_spatial_acc_data and model_spatial_acc_data[m]]
            if not plot_models:
                logger.warning("没有可用的空间ACC数据用于绘制折线图")
                plt.close()
                return

            cmap = plt.get_cmap('tab10')
            all_leadtimes = set()
            
            for i, model in enumerate(plot_models):
                acc_list = model_spatial_acc_data[model]  # List[Dataset]
                combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                
                # 提取 temporal_acc_mean DataArray
                combined_acc = combined_ds['temporal_acc_mean']
                
                # 记录所有出现的leadtime
                all_leadtimes.update(combined_acc.leadtime.values)
                
                # 计算每个leadtime的全年平均ACC (按月份取平均)
                leadtime_means = combined_acc.mean(dim='month', skipna=True)
                
                # 确保数据维度正确 (处理单个值或数组的情况)
                # 使用 to_numpy() 方法而不是 .values 属性，确保获得 NumPy 数组
                x_vals = leadtime_means.leadtime.to_numpy()
                y_vals = leadtime_means.to_numpy()
                
                # 如果是 0 维数组 (单个数值)，需要转为 1D 数组以便绘图
                x_vals = np.atleast_1d(x_vals)
                y_vals = np.atleast_1d(y_vals)
                
                ax.plot(x_vals, y_vals, 
                        marker='o', linewidth=2, label=model.replace('-mon', '').replace('mon-', ''),
                        color=cmap(i % cmap.N))
            
            ax.set_xlabel('Lead Time', fontsize=14)
            ax.set_ylabel('Mean Temporal ACC', fontsize=14)
            if all_leadtimes:
                # 转换 set 为 sorted list
                xticks_list = sorted(list(all_leadtimes))
                # 确保是 int 类型
                xticks_list = [int(x) for x in xticks_list]
                ax.set_xticks(xticks_list)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 图例放在图像横轴下方，限制四列
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=4, fontsize=12, frameon=False)
            
            output_file = self.plot_dir / f"spatial_acc_leadtime_timeseries_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"空间ACC随leadtime折线图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制空间ACC折线图出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def load_spatial_acc_from_nc(self, models: List[str]) -> Dict[str, List[xr.Dataset]]:
        """从已保存的NetCDF文件加载空间ACC数据"""
        model_spatial_acc_data = {model: [] for model in models}
        for model in models:
            save_path = self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.nc"
            if save_path.exists():
                try:
                    with xr.open_dataset(save_path) as ds:
                        # 兼容新旧变量名 - 现在保存的是 Dataset
                        if 'temporal_acc_mean' not in ds:
                            logger.warning(f"文件中未找到 temporal_acc_mean 变量: {save_path}")
                            logger.warning(f"可用变量: {list(ds.data_vars)}")
                            continue
                        
                        # 加载整个 Dataset 并将其按leadtime拆分回列表
                        ds_loaded = ds.load()
                        for lt in ds_loaded.leadtime.values:
                            # 选择该 leadtime 的数据(Dataset)并重新扩展维度
                            lt_ds = ds_loaded.sel(leadtime=lt).expand_dims(leadtime=[lt])
                            model_spatial_acc_data[model].append(lt_ds)
                    logger.info(f"成功加载已有ACC数据: {model}")
                except Exception as e:
                    logger.warning(f"从 {save_path} 加载数据失败: {repr(e)}")
                    import traceback
                    logger.warning(traceback.format_exc())
        return model_spatial_acc_data
    
    def load_temporal_acc_maps_from_nc(self, models: List[str]) -> Dict[str, Dict[int, xr.Dataset]]:
        """从已保存的NetCDF文件加载ACC空间分布数据"""
        model_temporal_acc_maps = {model: {} for model in models}
        out_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/temporal_acc_maps/{self.var_type}")
        
        for model in models:
            save_path = out_dir / f"temporal_acc_map_{model}_{self.var_type}.nc"
            if save_path.exists():
                try:
                    with xr.open_dataset(save_path) as ds:
                        ds_loaded = ds.load()
                        # 按leadtime拆分
                        for lt in ds_loaded.leadtime.values:
                            lt_key = int(lt)
                            # 提取该leadtime的数据
                            lt_ds = ds_loaded.sel(leadtime=lt).drop_vars('leadtime', errors='ignore')
                            model_temporal_acc_maps[model][lt_key] = lt_ds
                    logger.info(f"成功加载ACC空间分布数据: {model}")
                except Exception as e:
                    logger.warning(f"从 {save_path} 加载ACC空间分布数据失败: {repr(e)}")
        
        return model_temporal_acc_maps

    def run_analysis(self, models: List[str] = None, leadtimes: List[int] = None,
                     parallel: bool = False, n_jobs: Optional[int] = None,
                     plot_only: bool = False):
        """运行Pearson相关分析计算"""
        models = models or MODELS
        leadtimes = leadtimes or LEADTIMES

        logger.info(f"开始 {self.var_type} Pearson相关分析计算 (plot_only={plot_only})")

        if plot_only:
            logger.info("Plot-only 模式：尝试加载已有数据并绘图...")
            model_spatial_acc_data = self.load_spatial_acc_from_nc(models)
            
            # 加载ACC空间分布数据
            model_temporal_acc_maps = self.load_temporal_acc_maps_from_nc(models)
            
            # 加载现有的汇总数据结果以便能够运行绘图（部分绘图可能依赖 results）
            # 注意：这里的 results 主要用于 save_data_to_csv，如果是 plot_only，我们只绘图
            # self.plot_spatial_acc_monthly_contour(model_spatial_acc_data)
            self.plot_spatial_acc_heatmap_diverging_discrete(model_spatial_acc_data)
            self.plot_spatial_acc_leadtime_timeseries(model_spatial_acc_data)
            # 绘制ACC空间分布图
            self.plot_acc_spatial_maps(model_temporal_acc_maps)
            logger.info("绘图完成。")
            return None

        # 计算相关系数
        if not parallel:
            results = self.analyze_all_models_leadtimes(models, leadtimes)
        else:
            logger.info("使用并行模式计算相关...")
            # 与原文件一致：默认使用 CPU 一半的核，最多 32
            max_workers = min(n_jobs or max(1, cpu_count() // 2), 32)
            tasks = [(self.var_type, model, lt) for lt in leadtimes for model in models]
            results: Dict[int, Dict[str, Dict]] = {
                lt: {'annual': {}, 'annual_interannual': {}, 'seasonal': {}, 'monthly': {}} for lt in leadtimes
            }
            # 并行模式下的空间ACC收集
            model_spatial_acc_data = {model: [] for model in models}
            # 并行模式下的逐格点 Temporal ACC 空间分布收集
            model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]] = {model: {} for model in models}
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(_compute_correlations_task, *t): t for t in tasks}
                completed = 0
                for future in as_completed(future_to_task):
                    var_type, model, lt = future_to_task[future]
                    try:
                        out = future.result()
                    except Exception as e:
                        logger.error(f"任务失败: {model} L{lt}: {e}")
                        continue
                    if out is None:
                        continue
                    # 子进程返回:
                    # (leadtime, model, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map)
                    lt_out, model_out, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map = out
                    results[lt_out]['annual'][model_out] = annual_corrs
                    results[lt_out]['annual_interannual'][model_out] = interannual_corr
                    results[lt_out]['seasonal'][model_out] = seasonal_corrs
                    results[lt_out]['monthly'][model_out] = monthly_corrs
                    
                    if acc_ts is not None:
                        model_spatial_acc_data[model_out].append(acc_ts)

                    if acc_map is not None:
                        try:
                            # acc_map 是 Dataset, 包含 r, p, significant
                            # 确保 leadtime 维度处理正确
                            lt_key = int(lt_out)
                            model_temporal_acc_maps[model_out][lt_key] = acc_map.sel(leadtime=lt_key).drop_vars('leadtime')
                        except Exception:
                            # 兜底：若维度不一致，直接尝试 squeeze
                            try:
                                model_temporal_acc_maps[model_out][int(lt_out)] = acc_map.squeeze()
                            except Exception:
                                pass
                        
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"并行任务进度: {completed}/{len(tasks)}")
            
            # 并行模式下保存空间ACC
            self.save_spatial_acc_to_nc(model_spatial_acc_data)
            # 并行模式下保存逐格点 Temporal ACC 空间分布
            self.save_temporal_acc_maps_to_nc(model_temporal_acc_maps)
            
            # 新增：绘制结果图表
            # self.plot_spatial_acc_monthly_contour(model_spatial_acc_data)
            self.plot_spatial_acc_heatmap_diverging_discrete(model_spatial_acc_data)
            self.plot_spatial_acc_leadtime_timeseries(model_spatial_acc_data)
            # 绘制ACC空间分布图 (Lead 0 和 Lead 3)
            self.plot_acc_spatial_maps(model_temporal_acc_maps)

        # 保存其他计算结果到CSV
        self.save_data_to_csv(results)
        logger.info(f"{self.var_type}Pearson分析计算完成")
        
        return results


def _compute_correlations_task(var_type: str, model: str, leadtime: int):
    """子进程任务：加载数据并计算相关系数。"""
    try:
        # 由于在子进程中，需要重新构造 DataLoader 和 Analyzer 的最小逻辑
        sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
        from src.utils.data_loader import DataLoader  # noqa: F401

        analyzer = SeasonalMonthlyPearsonAnalyzer(var_type)
        obs_data, fcst_data = analyzer.load_and_preprocess_data(model, leadtime)
        if obs_data is None or fcst_data is None:
            return None

        # 计算距平场
        obs_anom_field, fcst_anom_field = analyzer.get_anomalies(obs_data, fcst_data, leadtime)
        acc_ts = None
        acc_map = None
        if obs_anom_field is not None and fcst_anom_field is not None:
            acc_ts = analyzer.calculate_temporal_acc_monthly_mean(obs_anom_field, fcst_anom_field)
            if acc_ts is not None:
                acc_ts = acc_ts.expand_dims(leadtime=[leadtime])
            # 逐格点 Temporal ACC 空间分布 (lat, lon)
            acc_map = analyzer.calculate_temporal_acc_map(obs_anom_field, fcst_anom_field)
            if acc_map is not None:
                acc_map = acc_map.expand_dims(leadtime=[leadtime])

        annual_corrs = analyzer.calculate_annual_correlations(obs_data, fcst_data)
        interannual_corr = analyzer.calculate_interannual_correlation(obs_data, fcst_data, use_anomaly=True)
        seasonal_corrs = analyzer.calculate_seasonal_correlations(obs_data, fcst_data)
        monthly_corrs = analyzer.calculate_monthly_correlations(obs_data, fcst_data)
        
        # 子进程返回:
        # (leadtime, model, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map)
        return (leadtime, model, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map)
    except Exception:
        return None


def main():
    """主函数"""
    parser = create_parser(
        description="Pearson相关系数分析计算（默认使用距平ACC计算季节与月度相关）",
        include_pearson=True,
        var_default=None,  # 允许不指定，默认处理temp和prec
        var_required=False
    )
    args = parser.parse_args()
    
    # 解析参数
    models = parse_models(args.models, MODELS) if args.models else MODELS
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    logger.info(f"将处理变量: {var_list}")
    logger.info(f"将处理模型: {models}")
    logger.info(f"将处理提前期: {leadtimes}")

    for var_type in var_list:
        logger.info(f"开始 {var_type} 的Pearson相关分析计算")
        analyzer = SeasonalMonthlyPearsonAnalyzer(
            var_type,
            n_jobs=args.n_jobs,
            use_anomaly_seasonal=not args.no_anomaly_seasonal,
            use_anomaly_monthly=not args.no_anomaly_monthly
        )
        # 处理并行参数：如果指定了--parallel或--n-jobs，则启用并行
        parallel = normalize_parallel_args(args) or (args.n_jobs is not None and args.n_jobs > 1)
        
        analyzer.run_analysis(
            models=models, 
            leadtimes=leadtimes,
            parallel=parallel,
            n_jobs=args.n_jobs,
            plot_only=args.plot_only
        )
    
    logger.info("所有分析计算完成！")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时间种类的Pearson相关系数分析计算模块
计算各个模式与观测的年度、季节、月度空间平均Pearson相关系数

修改：
1. 增加 Ensemble Member 维度的 ACC 计算。
2. 绘图增加所有模式所有成员的 Spread 阴影。
3. 保存时增加 member 维度的变量。
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
    SPATIAL_BOUNDS,
)


# 配置参数（从 common_config 导入）
MODELS = MODEL_LIST
# 月份标签
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# === 定义区域（Global + 9个规则气候区划） ===
def generate_regions():
    """
    生成分析区域：Global（全域）+ 9个规则气候区划
    基于方案二：中国气候9大规则区划
    """
    regions = {'Global': None}  # None 表示使用全域

    # 定义9个区域的经纬度范围 (lat: min-max, lon: min-max)
    # 依据方案二：中国气候 9 大规则区划
    regions.update({
        # --- 北部行 ---
        'Z1-Northwest':     {'lat': (39, 49), 'lon': (73, 105)},   # 西北干旱区
        'Z2-InnerMongolia': {'lat': (39, 50), 'lon': (106, 118)},  # 内蒙半干旱区
        'Z3-Northeast':     {'lat': (40, 54), 'lon': (119, 135)},  # 东北湿润区
        # --- 中部行 ---
        'Z4-Tibetan':       {'lat': (27, 39), 'lon': (73, 95)},    # 青藏高寒区
        'Z5-NorthChina':    {'lat': (34, 39), 'lon': (106, 122)},  # 黄土-华北区
        'Z6-Yangtze':       {'lat': (26, 34), 'lon': (109, 123)},  # 长江中下游区
        # --- 南部行 ---
        'Z7-Southwest':     {'lat': (23, 33), 'lon': (96, 108)},   # 四川-西南区
        'Z8-SouthChina':    {'lat': (21, 25), 'lon': (106, 120)},  # 华南湿润区
        'Z9-SouthSea':      {'lat': (18, 21), 'lon': (105, 125)}   # 南海热带区
    })

    return regions

REGIONS = generate_regions()

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
        # 区域 Index ACC 存储路径（用于 plot_only 时加载热图/Global 折线图）
        self.region_index_acc_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/region_index_acc/{self.var_type}")
        self.region_index_acc_dir.mkdir(parents=True, exist_ok=True)
        # 绘图存储路径
        self.plot_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/plots/{self.var_type}")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        # === 新增：初始化 boundaries 路径（用于高精度地图绘制） ===
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"
       
    def get_anomalies(self, obs_data: xr.DataArray, fcst_data: xr.DataArray, leadtime: int) -> Tuple[xr.DataArray, xr.DataArray]:
        """计算逐格点月距平场 (支持 ensemble members)"""
        try:
            # 确保时间坐标是 datetime 类型
            if not np.issubdtype(obs_data.time.dtype, np.datetime64):
                obs_data['time'] = pd.to_datetime(obs_data.time.values)
            if not np.issubdtype(fcst_data.time.dtype, np.datetime64):
                fcst_data['time'] = pd.to_datetime(fcst_data.time.values)

            # 观测的气候态
            obs_clim = obs_data.groupby('time.month').mean('time')
            obs_anom = obs_data.groupby('time.month') - obs_clim
            
            # 模式的气候态
            # 如果 fcst_data 有 'number' 维度，groupby('time.month').mean('time') 会对每个 member 计算气候态
            # 这里的做法是：每个 member 减去该 member 自己的气候态 (或者 ensemble mean 的气候态)。
            # 通常 ACC 计算中，每个 member 减去其自身的气候态是比较标准的做法。
            fcst_clim = fcst_data.groupby('time.month').mean('time')
            fcst_anom = fcst_data.groupby('time.month') - fcst_clim
            
            return obs_anom, fcst_anom
        except Exception as e:
            logger.error(f"距平场计算失败: {e}")
            return None, None

    def calculate_temporal_acc_monthly_mean(self, obs_anom: xr.DataArray, fcst_anom: xr.DataArray) -> xr.Dataset:
        """
        计算逐月平均的逐格点时间异常相关系数 (Mean Temporal ACC) 及其显著性
        注：此函数用于空间分布的平均统计，目前仅计算 Ensemble Mean 的 ACC。
        """
        try:
            # 如果有 number 维度，先求 Ensemble Mean 用于此计算 (Map相关)
            if 'number' in fcst_anom.dims:
                fcst_input = fcst_anom.mean(dim='number')
            else:
                fcst_input = fcst_anom

            weights = np.cos(np.deg2rad(obs_anom.lat))
            
            monthly_means = []
            monthly_p_values = []
            months = list(range(1, 13))
            
            for month in months:
                obs_m = obs_anom.sel(time=obs_anom.time.dt.month == month)
                fcst_m = fcst_input.sel(time=fcst_input.time.dt.month == month)
                
                n_samples = obs_m.sizes['time']
                
                if obs_m.size == 0 or fcst_m.size == 0 or n_samples < 3:
                    monthly_means.append(np.nan)
                    monthly_p_values.append(np.nan)
                    continue
                    
                tcc_map = xr.apply_ufunc(
                    pearson_r_along_time,
                    obs_m, fcst_m,
                    input_core_dims=[['time'], ['time']],
                    output_core_dims=[[]],
                    vectorize=True,
                    dask='parallelized'
                )
                
                mean_tcc = tcc_map.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
                mean_val = float(mean_tcc.item())
                monthly_means.append(mean_val)
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
        计算逐格点的“全逐月时间序列”异常相关系数。仅计算 Ensemble Mean。
        """
        try:
            if 'number' in fcst_anom.dims:
                fcst_input = fcst_anom.mean(dim='number')
            else:
                fcst_input = fcst_anom

            res = xr.apply_ufunc(
                pearson_r_along_time_with_p,
                obs_anom,
                fcst_input,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[[], []],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float, float]
            )
            
            if isinstance(res, tuple):
                r_da, p_da = res
            else:
                return None

            ds = xr.Dataset({
                'temporal_acc': r_da,
                'p_value': p_da
            })
            ds['temporal_acc'].attrs = {"long_name": "Temporal ACC (Ensemble Mean)", "units": "1"}
            ds['significant'] = ds['p_value'] < 0.05
            
            return ds
        except Exception as e:
            logger.error(f"Temporal ACC map 计算失败: {e}")
            return None

    def calculate_regional_index_acc(self, obs_anom: xr.DataArray, fcst_anom: xr.DataArray, 
                                     region_bounds: Optional[Dict]) -> xr.Dataset:
        """
        计算区域平均后的指数相关系数 (Regional Index ACC)。
        同时计算：
        1. Ensemble Mean ACC (deterministic skill)
        2. Individual Member ACC (probabilistic skill spread)
        """
        try:
            # 1. 区域截取
            obs_reg = obs_anom
            fcst_reg = fcst_anom
            
            if region_bounds is not None:
                lat_b = region_bounds['lat']
                lon_b = region_bounds['lon']
                if obs_reg.lat[0] < obs_reg.lat[-1]:
                    lat_slice = slice(lat_b[0], lat_b[1])
                else:
                    lat_slice = slice(lat_b[1], lat_b[0])
                obs_reg = obs_reg.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
                fcst_reg = fcst_reg.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
            
            # 2. 纬度加权空间平均
            weights = np.cos(np.deg2rad(obs_reg.lat))
            weights.name = "weights"
            
            obs_ts = obs_reg.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            fcst_ts = fcst_reg.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            
            # 分离 Ensemble Mean 和 Members
            if 'number' in fcst_ts.dims:
                fcst_ts_mean = fcst_ts.mean(dim='number')
                fcst_ts_members = fcst_ts # (time, number)
                members = fcst_ts.number.values
            else:
                fcst_ts_mean = fcst_ts
                fcst_ts_members = None
                members = []

            # ---------------------------
            # 计算 Ensemble Mean 的 ACC
            # ---------------------------
            monthly_corrs = []
            monthly_p_values = []
            seasonal_corrs = []
            seasonal_p_values = []
            
            months = list(range(1, 13))
            seasons_list = list(SEASONS.keys())

            def calc_acc_for_ts(o_ts, f_ts):
                # Monthly
                m_corrs, m_ps = [], []
                for month in months:
                    o = o_ts.sel(time=o_ts.time.dt.month == month).values
                    f = f_ts.sel(time=f_ts.time.dt.month == month).values
                    mask = np.isfinite(o) & np.isfinite(f)
                    if np.sum(mask) < 3:
                        m_corrs.append(np.nan); m_ps.append(np.nan)
                    else:
                        r, p = stats.pearsonr(o[mask], f[mask])
                        m_corrs.append(r); m_ps.append(p)
                
                # Seasonal
                s_corrs, s_ps = [], []
                for season in seasons_list:
                    month_idxs = SEASONS[season]
                    season_mask_obs = o_ts.time.dt.month.isin(month_idxs)
                    obs_season_subset = o_ts.sel(time=season_mask_obs)
                    season_mask_fcst = f_ts.time.dt.month.isin(month_idxs)
                    fcst_season_subset = f_ts.sel(time=season_mask_fcst)

                    if season == 'DJF':
                        obs_sy = obs_season_subset.time.dt.year + (obs_season_subset.time.dt.month == 12)
                        fcst_sy = fcst_season_subset.time.dt.year + (fcst_season_subset.time.dt.month == 12)
                        obs_yearly = obs_season_subset.groupby(obs_sy.rename('year')).mean('time')
                        fcst_yearly = fcst_season_subset.groupby(fcst_sy.rename('year')).mean('time')
                    else:
                        obs_yearly = obs_season_subset.groupby('time.year').mean('time')
                        fcst_yearly = fcst_season_subset.groupby('time.year').mean('time')

                    common_years = np.intersect1d(obs_yearly.year, fcst_yearly.year)
                    if len(common_years) < 3:
                        s_corrs.append(np.nan); s_ps.append(np.nan)
                    else:
                        o_vec = obs_yearly.sel(year=common_years).values
                        f_vec = fcst_yearly.sel(year=common_years).values
                        mask = np.isfinite(o_vec) & np.isfinite(f_vec)
                        if np.sum(mask) < 3:
                            s_corrs.append(np.nan); s_ps.append(np.nan)
                        else:
                            r, p = stats.pearsonr(o_vec[mask], f_vec[mask])
                            s_corrs.append(r); s_ps.append(p)
                return m_corrs, m_ps, s_corrs, s_ps

            # 计算 Mean 的 ACC
            m_r, m_p, s_r, s_p = calc_acc_for_ts(obs_ts, fcst_ts_mean)
            monthly_corrs, monthly_p_values = m_r, m_p
            seasonal_corrs, seasonal_p_values = s_r, s_p

            # ---------------------------
            # 计算 Members 的 ACC
            # ---------------------------
            monthly_corrs_mem = [] # list of lists (n_members, n_months)
            seasonal_corrs_mem = [] # list of lists (n_members, n_seasons)

            if fcst_ts_members is not None:
                for mem_idx in range(len(members)):
                    # 提取单个 member 的时间序列
                    mem_ts = fcst_ts_members.isel(number=mem_idx)
                    mm_r, _, ss_r, _ = calc_acc_for_ts(obs_ts, mem_ts)
                    monthly_corrs_mem.append(mm_r)
                    seasonal_corrs_mem.append(ss_r)
            else:
                # 兼容无 members 的情况 (dummy dim)
                monthly_corrs_mem = [monthly_corrs]
                seasonal_corrs_mem = [seasonal_corrs]
                members = [0]

            # 5. 创建结果数据集
            ds = xr.Dataset(
                {
                    'regional_index_acc': (['month'], monthly_corrs),
                    'p_value': (['month'], monthly_p_values),
                    'regional_index_acc_seasonal': (['season'], seasonal_corrs),
                    'p_value_seasonal': (['season'], seasonal_p_values),
                    
                    # Member data
                    'regional_index_acc_members': (['number', 'month'], monthly_corrs_mem),
                    'regional_index_acc_seasonal_members': (['number', 'season'], seasonal_corrs_mem),
                    
                    'significant': (['month'], [p < 0.05 for p in monthly_p_values])
                },
                coords={
                    'month': months,
                    'season': seasons_list,
                    'number': members
                }
            )
            
            return ds
            
        except Exception as e:
            logger.error(f"区域 ACC 计算失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def load_and_preprocess_data(self, model: str, leadtime: int) -> Tuple[xr.DataArray, xr.DataArray]:
        """加载和预处理数据 (使用 Ensemble Loader)"""
        try:
            # 加载观测数据
            obs_data = self.data_loader.load_obs_data(self.var_type)
            obs_data = obs_data.resample(time='1MS').mean()
            obs_data = obs_data.sel(time=slice('1993', '2020'))
            
            # 加载模型数据 (Ensemble)
            # 使用 load_forecast_data_ensemble 替代 load_forecast_data
            fcst_data = self.data_loader.load_forecast_data_ensemble(model, self.var_type, leadtime)
            
            if fcst_data is None:
                return None, None
                
            fcst_data = fcst_data.resample(time='1MS').mean()
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))
            
            # 时间对齐
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

            return obs_aligned, fcst_interpolated
            
        except Exception as e:
            logger.error(f"数据加载失败: {model} L{leadtime}: {e}")
            return None, None
    
    def calculate_annual_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[int, float]:
        return {}
    
    def calculate_seasonal_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[str, float]:
        return {}
    
    def calculate_monthly_correlations(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Dict[str, float]:
        return {}
    
    def calculate_interannual_correlation(self, obs_data: xr.DataArray, fcst_data: xr.DataArray,
                                          use_anomaly: bool = True) -> float:
        return np.nan

    def add_china_map_details(self, ax, data, lon, lat, levels, cmap, draw_scs=True):
        # ... (保持原有代码不变)
        bou_paths = [
            Path("/sas12t1/ffyan/boundaries/中国_省1.shp"),
            Path("/sas12t1/ffyan/boundaries/中国_省2.shp")
        ]
        hyd_path = self.boundaries_dir / "河流.shp"
        if hyd_path.exists():
            try:
                reader = shpreader.Reader(str(hyd_path))
                ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
                                edgecolor='blue', facecolor='none', 
                                linewidth=0.6, alpha=0.6, zorder=5)
            except Exception:
                ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6, zorder=5)
        else:
            ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6, zorder=5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', zorder=50)
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
        if draw_scs:
            try:
                scs_width = 0.33
                scs_height = 0.35
                sub_ax = ax.inset_axes([0.7548, 0, scs_width, scs_height], 
                                      projection=ccrs.PlateCarree())
                sub_ax.patch.set_facecolor('white')
                sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
                sub_ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, levels=levels, extend='both')
                sub_ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray', zorder=50)
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
                sub_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                for spine in sub_ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.0)
            except Exception as e:
                logger.warning(f"南海子图绘制失败: {e}")
    
    def analyze_all_models_leadtimes(self, models: List[str] = None, leadtimes: List[int] = None) -> Dict:
        # ... (保持原有逻辑，内部调用已更新的 load_and_preprocess_data)
        if models is None: models = MODELS
        if leadtimes is None: leadtimes = LEADTIMES
        
        logger.info(f"开始{self.var_type}的Pearson分析 (含 Ensemble Members)")
        
        results = {}
        model_spatial_acc_data = {model: [] for model in models}
        model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]] = {model: {} for model in models}
        region_spatial_acc_data = {r: {m: [] for m in models} for r in REGIONS.keys()}
        
        for leadtime in leadtimes:
            logger.info(f"处理预报时效: L{leadtime}")
            annual_data = {}; annual_interannual = {}; seasonal_data = {}; monthly_data = {}
            for model in models:
                obs_data, fcst_data = self.load_and_preprocess_data(model, leadtime)
                if obs_data is None or fcst_data is None: continue
                
                obs_anom_field, fcst_anom_field = self.get_anomalies(obs_data, fcst_data, leadtime)
                if obs_anom_field is not None and fcst_anom_field is not None:
                    # Map 和 Global Mean ACC 仍基于 Ensemble Mean 计算，避免数据量过大
                    acc_map = self.calculate_temporal_acc_map(obs_anom_field, fcst_anom_field)
                    if acc_map is not None:
                        model_temporal_acc_maps[model][int(leadtime)] = acc_map

                    acc_monthly = self.calculate_temporal_acc_monthly_mean(obs_anom_field, fcst_anom_field)
                    if acc_monthly is not None:
                        acc_monthly = acc_monthly.expand_dims(leadtime=[leadtime])
                        model_spatial_acc_data[model].append(acc_monthly)
                    
                    # 区域 ACC (含 Member 计算)
                    for reg_name, reg_bounds in REGIONS.items():
                        reg_acc_ds = self.calculate_regional_index_acc(obs_anom_field, fcst_anom_field, reg_bounds)
                        if reg_acc_ds is not None:
                            reg_acc_ds = reg_acc_ds.expand_dims(leadtime=[leadtime])
                            region_spatial_acc_data[reg_name][model].append(reg_acc_ds)

                annual_data[model] = {}; annual_interannual[model] = np.nan
                seasonal_data[model] = {}; monthly_data[model] = {}
                
            results[leadtime] = {'annual': annual_data, 'annual_interannual': annual_interannual, 'seasonal': seasonal_data, 'monthly': monthly_data}
        
        self.save_spatial_acc_to_nc(model_spatial_acc_data)
        self.save_temporal_acc_maps_to_nc(model_temporal_acc_maps)
        self.save_region_index_acc_to_nc(region_spatial_acc_data)
        
        self.plot_spatial_acc_heatmap_diverging_discrete(region_spatial_acc_data)
        self.plot_spatial_acc_leadtime_timeseries(region_spatial_acc_data)
        self.plot_acc_spatial_maps(model_temporal_acc_maps)
        self.plot_regional_index_acc_leadtime_timeseries(region_spatial_acc_data)
        
        return results

    def plot_acc_spatial_maps(self, model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]]):
        # ... (保持不变)
        try:
            plot_models = list(model_temporal_acc_maps.keys())
            leadtimes = [0, 3]
            if not plot_models: return
            lon_range = (70, 140); lat_range = (15, 55)
            vmin, vmax = -1, 1; cmap = 'RdBu_r'
            n_levels = 20; levels = np.linspace(vmin, vmax, n_levels + 1)
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15, left=0.05, right=0.92, top=0.94, bottom=0.06)
            lon_ticks = np.arange(75, 141, 15); lat_ticks = np.arange(20, 56, 10)
            for lt_idx, leadtime in enumerate(leadtimes):
                row_start = lt_idx * 2
                ax_blank = fig.add_subplot(gs[row_start, 0]); ax_blank.axis('off')
                for col_idx in range(3):
                    if col_idx >= len(plot_models): continue
                    self._plot_single_map(fig, gs, row_start, col_idx + 1, plot_models[col_idx], leadtime,
                                          model_temporal_acc_maps, levels, cmap, lon_ticks, lat_ticks, chr(97 + col_idx))
                    if col_idx == 0:
                         ax = fig.axes[-1]
                         ax.text(0.98, 0.96, f'L{leadtime}', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(plot_models): continue
                    self._plot_single_map(fig, gs, row_start + 1, col_idx, plot_models[model_idx], leadtime,
                                          model_temporal_acc_maps, levels, cmap, lon_ticks, lat_ticks, chr(97 + model_idx))
            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(levels, 256), cmap=cmap), 
                                cax=cbar_ax, orientation='vertical', extend='both')
            cbar.set_label('Temporal ACC', fontsize=14, labelpad=10)
            output_file = self.plot_dir / f"acc_spatial_maps_L0_L3_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        except Exception as e:
            logger.error(f"ACC空间分布图绘制失败: {e}")

    def _plot_single_map(self, fig, gs, row, col, model, leadtime, maps, levels, cmap, xticks, yticks, char_label):
        # ... (保持不变)
        if leadtime not in maps[model]:
            ax = fig.add_subplot(gs[row, col]); ax.axis('off'); return
        acc_ds = maps[model][leadtime]
        data = acc_ds['temporal_acc']
        display_name = model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, alpha=0.1); ax.add_feature(cfeature.OCEAN, alpha=0.1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = FixedLocator(xticks); gl.ylocator = FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter(number_format='.0f'); gl.yformatter = LatitudeFormatter(number_format='.0f')
        im = ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both')
        self.add_china_map_details(ax, data, data.lon, data.lat, levels, cmap, draw_scs=True)
        if 'significant' in acc_ds:
            sig_mask = acc_ds['significant'].values
            if np.any(sig_mask):
                X, Y = np.meshgrid(data.lon, data.lat)
                ax.scatter(X[sig_mask][::2], Y[sig_mask][::2], transform=ccrs.PlateCarree(), s=1, c='black', alpha=0.5, marker='.')
        ax.text(0.02, 0.96, f"({char_label}) {display_name}", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

    def plot_spatial_acc_heatmap_diverging_discrete(self, region_spatial_acc_data: Optional[Dict] = None):
        # ... (保持不变)
        try:
            if region_spatial_acc_data is None or 'Global' not in region_spatial_acc_data: return
            global_data = region_spatial_acc_data['Global']
            plot_models = [m for m in MODELS if m in global_data and global_data[m]]
            if not plot_models: return
            months = np.arange(1, 13); leadtimes = np.arange(0, 6)
            model_matrices = {}; model_p_matrices = {}
            for model in plot_models:
                acc_list = global_data[model]
                combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                matrix = combined_ds['regional_index_acc'].reindex(leadtime=leadtimes, month=months).values
                model_matrices[model] = matrix
                if 'p_value' in combined_ds:
                    p_matrix = combined_ds['p_value'].reindex(leadtime=leadtimes, month=months).values
                    model_p_matrices[model] = p_matrix
                else:
                    model_p_matrices[model] = None
            n_bins = 20; levels = np.linspace(-1, 1, n_bins + 1)
            cmap = plt.get_cmap('RdBu_r', n_bins); norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=False, sharey=False)
            axes[0, 0].axis('off')
            x_edges = np.arange(0.5, 13.5, 1); y_edges = np.arange(-0.5, 6.5, 1)
            X, Y = np.meshgrid(x_edges, y_edges)
            X_center, Y_center = np.meshgrid(months, leadtimes)
            mesh = None
            for i, model in enumerate(plot_models):
                row, col = (0, i + 1) if i < 3 else (1, i - 3)
                ax = axes[row, col]
                matrix = model_matrices[model]; p_matrix = model_p_matrices[model]
                mesh = ax.pcolormesh(X, Y, matrix, cmap=cmap, norm=norm, edgecolor='face', linewidth=0.1, shading='flat')
                if p_matrix is not None and np.isfinite(p_matrix).any():
                    sig_mask = (p_matrix < 0.05) & np.isfinite(matrix)
                    if np.any(sig_mask):
                        ax.scatter(X_center[sig_mask], Y_center[sig_mask], marker='.', s=18, color='black', alpha=0.8)
                ax.set_title(f"({chr(97+i)}) {model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')}", fontsize=18, fontweight='bold', loc='left', pad=10)
                ax.set_xticks(months); ax.set_yticks(leadtimes)
                ax.set_xlabel('Month', fontsize=16); ax.set_ylabel('Lead Time', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.grid(False)
            plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.10, wspace=0.3, hspace=0.35)
            if mesh:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='vertical', ticks=levels)
                cbar.set_label('Global Index ACC', fontsize=14, labelpad=15)
            output_file = self.plot_dir / f"spatial_acc_heatmap_diverging_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"离散型对称热图绘制失败: {e}")

    def save_temporal_acc_maps_to_nc(self, model_temporal_acc_maps: Dict[str, Dict[int, xr.Dataset]]):
        # ... (保持不变)
        out_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/temporal_acc_maps/{self.var_type}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for model, lt_to_map in model_temporal_acc_maps.items():
            if not lt_to_map: continue
            try:
                maps = [lt_to_map[lt].expand_dims(leadtime=[lt]) for lt in sorted(int(k) for k in lt_to_map.keys())]
                ds = xr.concat(maps, dim='leadtime').sortby('leadtime')
                ds.to_netcdf(out_dir / f"temporal_acc_map_{model}_{self.var_type}.nc")
            except Exception as e:
                logger.error(f"保存 Temporal ACC 空间分布失败 ({model}): {e}")

    def save_region_index_acc_to_nc(self, region_spatial_acc_data: Dict):
        # ... (保持不变，因为 dataset 结构已经包含 members 变量)
        if not region_spatial_acc_data: return
        for region_name, model_data in region_spatial_acc_data.items():
            for model, acc_list in model_data.items():
                if not acc_list: continue
                try:
                    combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                    safe_region = region_name.replace(' ', '_')
                    save_path = self.region_index_acc_dir / f"region_index_acc_{safe_region}_{model}_{self.var_type}.nc"
                    combined_ds.to_netcdf(save_path)
                except Exception as e:
                    logger.warning(f"保存区域 Index ACC 失败 ({region_name} {model}): {e}")

    def load_region_index_acc_from_nc(self, models: List[str]) -> Dict:
        # ... (保持不变)
        region_spatial_acc_data = {r: {m: [] for m in models} for r in REGIONS.keys()}
        for region_name in REGIONS.keys():
            safe_region = region_name.replace(' ', '_')
            for model in models:
                save_path = self.region_index_acc_dir / f"region_index_acc_{safe_region}_{model}_{self.var_type}.nc"
                if not save_path.exists(): continue
                try:
                    with xr.open_dataset(save_path) as ds:
                        ds_loaded = ds.load()
                    for lt in ds_loaded.leadtime.values:
                        lt_ds = ds_loaded.sel(leadtime=lt).expand_dims(leadtime=[lt])
                        region_spatial_acc_data[region_name][model].append(lt_ds)
                except Exception as e:
                    pass
        return region_spatial_acc_data

    def save_spatial_acc_to_nc(self, model_spatial_acc_data: Dict[str, List[xr.Dataset]]):
        # ... (保持不变)
        for model, acc_list in model_spatial_acc_data.items():
            if not acc_list: continue
            try:
                combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                combined_ds.to_netcdf(self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.nc")
                try:
                    df = combined_ds['temporal_acc_mean'].to_pandas()
                    if isinstance(df, pd.DataFrame): df = df.T
                    df.to_csv(self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.csv")
                except: pass
            except Exception as e:
                logger.error(f"空间ACC保存NetCDF失败 ({model}): {e}")

    def save_data_to_csv(self, results: Dict, save_dir: str = None):
        pass

    def plot_spatial_acc_leadtime_timeseries(self, region_spatial_acc_data: Optional[Dict] = None):
        """绘制 Global 区域的 Index ACC 随 Leadtime 变化的折线图 (增加 Spread)"""
        try:
            if region_spatial_acc_data is None or 'Global' not in region_spatial_acc_data: return
            global_data = region_spatial_acc_data['Global']
            plot_models = [m for m in MODELS if m in global_data and global_data[m]]
            if not plot_models: return

            fig, ax = plt.subplots(figsize=(10, 6))
            cmap = plt.get_cmap('tab10')
            all_leadtimes = set()

            # --- 计算 Spread (所有模式所有成员) ---
            spread_min = {}
            spread_max = {}
            
            for model in plot_models:
                acc_list = global_data[model]
                combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                if 'regional_index_acc_members' in combined_ds:
                    # (leadtime, number, month) -> mean over month -> (leadtime, number)
                    da_mem = combined_ds['regional_index_acc_members'].mean(dim='month', skipna=True)
                    for lt in da_mem.leadtime.values:
                        vals = da_mem.sel(leadtime=lt).values.flatten()
                        vals = vals[np.isfinite(vals)]
                        if len(vals) > 0:
                            lt_int = int(lt)
                            spread_min.setdefault(lt_int, []).extend(vals)
                            spread_max.setdefault(lt_int, []).extend(vals)
            
            sorted_lts = sorted(spread_min.keys())
            if sorted_lts:
                y_min = [np.min(spread_min[lt]) for lt in sorted_lts]
                y_max = [np.max(spread_max[lt]) for lt in sorted_lts]
                ax.fill_between(sorted_lts, y_min, y_max, color='gray', alpha=0.2, label='Multi-model Member Spread')

            # --- 绘制 Ensemble Mean ---
            for i, model in enumerate(plot_models):
                acc_list = global_data[model]
                combined_ds = xr.concat(acc_list, dim='leadtime').sortby('leadtime')
                mean_acc = combined_ds['regional_index_acc'].mean(dim='month', skipna=True)
                x_vals = np.atleast_1d(mean_acc.leadtime.values)
                y_vals = np.atleast_1d(mean_acc.values)
                all_leadtimes.update(int(x) for x in mean_acc.leadtime.values)
                ax.plot(x_vals, y_vals, marker='o', linewidth=2, label=model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC'), color=cmap(i % cmap.N))

            ax.set_ylabel('All Regions Mean ACC', fontsize=14)
            ax.set_xlabel('Lead Time', fontsize=14)
            if all_leadtimes: ax.set_xticks(sorted(all_leadtimes))
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=14, frameon=False)
            plt.tight_layout()
            plt.savefig(self.plot_dir / f"spatial_acc_leadtime_timeseries_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Global Index ACC 折线图绘制失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def plot_regional_index_acc_leadtime_timeseries(self, region_spatial_acc_data: Dict):
        """绘制分区域的 Index ACC 随 Leadtime 变化的折线图 (增加 Spread)"""
        try:
            region_order = [
                'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
                'Z4-Tibetan',   'Z5-NorthChina',    'Z6-Yangtze',
                'Z7-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
            ]
            regions = [r for r in region_order if r in region_spatial_acc_data]
            if not regions: return
            
            n_cols = 3; n_rows = 3
            fig = plt.figure(figsize=(18, 15))
            subplot_positions = {
                'Z1-Northwest': (0, 0), 'Z2-InnerMongolia': (0, 1), 'Z3-Northeast': (0, 2),
                'Z4-Tibetan': (1, 0), 'Z5-NorthChina': (1, 1), 'Z6-Yangtze': (1, 2),
                'Z7-Southwest': (2, 0), 'Z8-SouthChina': (2, 1), 'Z9-SouthSea': (2, 2)
            }
            cmap = plt.get_cmap('tab10')
            axes_dict = {}
            for reg_name in regions:
                if reg_name in subplot_positions:
                    row, col = subplot_positions[reg_name]
                    ax = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
                    axes_dict[reg_name] = ax
            
            for reg_name in regions:
                ax = axes_dict[reg_name]
                model_data = region_spatial_acc_data[reg_name]
                
                # --- 计算 Spread ---
                spread_min = {}
                spread_max = {}
                for model in MODELS:
                    if model not in model_data or not model_data[model]: continue
                    combined_ds = xr.concat(model_data[model], dim='leadtime').sortby('leadtime')
                    if 'regional_index_acc_members' in combined_ds:
                        da_mem = combined_ds['regional_index_acc_members'].mean(dim='month', skipna=True)
                        for lt in da_mem.leadtime.values:
                            vals = da_mem.sel(leadtime=lt).values.flatten()
                            vals = vals[np.isfinite(vals)]
                            if len(vals) > 0:
                                lt_int = int(lt)
                                spread_min.setdefault(lt_int, []).extend(vals)
                                spread_max.setdefault(lt_int, []).extend(vals)
                
                sorted_lts = sorted(spread_min.keys())
                if sorted_lts:
                    y_min = [np.min(spread_min[lt]) for lt in sorted_lts]
                    y_max = [np.max(spread_max[lt]) for lt in sorted_lts]
                    ax.fill_between(sorted_lts, y_min, y_max, color='gray', alpha=0.2, label='Multi-model Member Spread' if reg_name == regions[0] else "")

                # --- 绘制 Mean ---
                for mi, model in enumerate(MODELS):
                    if model not in model_data or not model_data[model]: continue
                    combined_ds = xr.concat(model_data[model], dim='leadtime').sortby('leadtime')
                    da_mean = combined_ds['regional_index_acc'].mean(dim='month', skipna=True)
                    ax.plot(da_mean.leadtime.values, da_mean.values, marker='o', linewidth=2, markersize=6,
                           label=model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC') if reg_name == regions[0] else "",
                           color=cmap(mi % cmap.N))
                
                ax.set_title(reg_name, fontsize=16, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlabel('Lead Time', fontsize=16)
            
            for ax in axes_dict.values(): ax.set_xticks(LEADTIMES)
            
            handles, labels = axes_dict[regions[0]].get_legend_handles_labels()
            # 去重 legend
            by_label = dict(zip(labels, handles))
            if by_label: 
                fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fontsize=16, frameon=False)
            
            plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3, wspace=0.15)
            plt.savefig(self.plot_dir / f"regional_index_acc_leadtime_timeseries_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"分区域折线图绘制失败: {e}")

    def load_spatial_acc_from_nc(self, models: List[str]) -> Dict[str, List[xr.Dataset]]:
        # ... (保持不变)
        model_spatial_acc_data = {model: [] for model in models}
        for model in models:
            save_path = self.spatial_acc_dir / f"spatial_acc_timeseries_{model}_{self.var_type}.nc"
            if save_path.exists():
                try:
                    with xr.open_dataset(save_path) as ds:
                        ds_loaded = ds.load()
                        for lt in ds_loaded.leadtime.values:
                            lt_ds = ds_loaded.sel(leadtime=lt).expand_dims(leadtime=[lt])
                            model_spatial_acc_data[model].append(lt_ds)
                except Exception: pass
        return model_spatial_acc_data
    
    def load_temporal_acc_maps_from_nc(self, models: List[str]) -> Dict[str, Dict[int, xr.Dataset]]:
        # ... (保持不变)
        model_temporal_acc_maps = {model: {} for model in models}
        out_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/temporal_acc_maps/{self.var_type}")
        for model in models:
            save_path = out_dir / f"temporal_acc_map_{model}_{self.var_type}.nc"
            if save_path.exists():
                try:
                    with xr.open_dataset(save_path) as ds:
                        ds_loaded = ds.load()
                        for lt in ds_loaded.leadtime.values:
                            model_temporal_acc_maps[model][int(lt)] = ds_loaded.sel(leadtime=lt).drop_vars('leadtime', errors='ignore')
                except Exception: pass
        return model_temporal_acc_maps

    def run_analysis(self, models: List[str] = None, leadtimes: List[int] = None,
                     parallel: bool = False, n_jobs: Optional[int] = None,
                     plot_only: bool = False):
        """运行Pearson相关分析计算"""
        models = models or MODELS
        leadtimes = leadtimes or LEADTIMES

        if plot_only:
            logger.info("Plot-only 模式：尝试加载已有数据并绘图...")
            region_spatial_acc_data = self.load_region_index_acc_from_nc(models)
            model_temporal_acc_maps = self.load_temporal_acc_maps_from_nc(models)
            
            if any(len(d)>0 for m in region_spatial_acc_data.values() for d in m.values()):
                self.plot_spatial_acc_heatmap_diverging_discrete(region_spatial_acc_data)
                self.plot_spatial_acc_leadtime_timeseries(region_spatial_acc_data)
                self.plot_regional_index_acc_leadtime_timeseries(region_spatial_acc_data)
            
            self.plot_acc_spatial_maps(model_temporal_acc_maps)
            return None

        # 计算相关系数
        if not parallel:
            results = self.analyze_all_models_leadtimes(models, leadtimes)
        else:
            logger.info("使用并行模式计算相关...")
            max_workers = min(n_jobs or max(1, cpu_count() // 2), 32)
            tasks = [(self.var_type, model, lt) for lt in leadtimes for model in models]
            results = {lt: {'annual': {}, 'annual_interannual': {}, 'seasonal': {}, 'monthly': {}} for lt in leadtimes}
            
            model_spatial_acc_data = {model: [] for model in models}
            model_temporal_acc_maps = {model: {} for model in models}
            region_spatial_acc_data = {r: {m: [] for m in models} for r in REGIONS.keys()}
            
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
                    if out is None: continue
                    
                    lt_out, model_out, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map, region_acc_dict = out
                    results[lt_out]['annual'][model_out] = annual_corrs
                    results[lt_out]['annual_interannual'][model_out] = interannual_corr
                    results[lt_out]['seasonal'][model_out] = seasonal_corrs
                    results[lt_out]['monthly'][model_out] = monthly_corrs
                    
                    if acc_ts is not None: model_spatial_acc_data[model_out].append(acc_ts)
                    if acc_map is not None: 
                        try:
                            model_temporal_acc_maps[model_out][int(lt_out)] = acc_map.sel(leadtime=int(lt_out)).drop_vars('leadtime')
                        except:
                             model_temporal_acc_maps[model_out][int(lt_out)] = acc_map.squeeze()

                    if region_acc_dict:
                        for reg_name, reg_ds in region_acc_dict.items():
                            if reg_ds is not None: region_spatial_acc_data[reg_name][model_out].append(reg_ds)
                        
                    completed += 1
                    if completed % 10 == 0: logger.info(f"并行任务进度: {completed}/{len(tasks)}")
            
            self.save_spatial_acc_to_nc(model_spatial_acc_data)
            self.save_temporal_acc_maps_to_nc(model_temporal_acc_maps)
            self.save_region_index_acc_to_nc(region_spatial_acc_data)
            
            self.plot_spatial_acc_heatmap_diverging_discrete(region_spatial_acc_data)
            self.plot_spatial_acc_leadtime_timeseries(region_spatial_acc_data)
            self.plot_acc_spatial_maps(model_temporal_acc_maps)
            self.plot_regional_index_acc_leadtime_timeseries(region_spatial_acc_data)

        self.save_data_to_csv(results)
        return results


def _compute_correlations_task(var_type: str, model: str, leadtime: int):
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
        from src.utils.data_loader import DataLoader # noqa

        analyzer = SeasonalMonthlyPearsonAnalyzer(var_type)
        obs_data, fcst_data = analyzer.load_and_preprocess_data(model, leadtime)
        if obs_data is None or fcst_data is None: return None

        obs_anom_field, fcst_anom_field = analyzer.get_anomalies(obs_data, fcst_data, leadtime)
        acc_ts = None; acc_map = None; region_acc_dict = {}
        
        if obs_anom_field is not None and fcst_anom_field is not None:
            # Global Mean ACC (Mean only)
            acc_ts = analyzer.calculate_temporal_acc_monthly_mean(obs_anom_field, fcst_anom_field)
            if acc_ts is not None: acc_ts = acc_ts.expand_dims(leadtime=[leadtime])
            
            # Spatial Map (Mean only)
            acc_map = analyzer.calculate_temporal_acc_map(obs_anom_field, fcst_anom_field)
            if acc_map is not None: acc_map = acc_map.expand_dims(leadtime=[leadtime])
            
            # Regional Index ACC (Mean + Members)
            for reg_name, reg_bounds in REGIONS.items():
                reg_acc_ds = analyzer.calculate_regional_index_acc(obs_anom_field, fcst_anom_field, reg_bounds)
                if reg_acc_ds is not None:
                    reg_acc_ds = reg_acc_ds.expand_dims(leadtime=[leadtime])
                    region_acc_dict[reg_name] = reg_acc_ds

        annual_corrs = analyzer.calculate_annual_correlations(obs_data, fcst_data)
        interannual_corr = analyzer.calculate_interannual_correlation(obs_data, fcst_data, use_anomaly=True)
        seasonal_corrs = analyzer.calculate_seasonal_correlations(obs_data, fcst_data)
        monthly_corrs = analyzer.calculate_monthly_correlations(obs_data, fcst_data)
        
        return (leadtime, model, annual_corrs, interannual_corr, seasonal_corrs, monthly_corrs, acc_ts, acc_map, region_acc_dict)
    except Exception:
        return None


def main():
    parser = create_parser(description="Pearson相关系数分析计算", include_pearson=True, var_default=None, var_required=False)
    args = parser.parse_args()
    
    models = parse_models(args.models, MODELS) if args.models else MODELS
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    for var_type in var_list:
        analyzer = SeasonalMonthlyPearsonAnalyzer(var_type, n_jobs=args.n_jobs, use_anomaly_seasonal=not args.no_anomaly_seasonal, use_anomaly_monthly=not args.no_anomaly_monthly)
        parallel = normalize_parallel_args(args) or (args.n_jobs is not None and args.n_jobs > 1)
        analyzer.run_analysis(models=models, leadtimes=leadtimes, parallel=parallel, n_jobs=args.n_jobs, plot_only=args.plot_only)

if __name__ == "__main__":
    main()
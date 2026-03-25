#!/usr/bin/env python3
"""
Index Analysis

功能：
1. 计算和分析 EAWM (East Asian Winter Monsoon) 指数
2. 计算和分析 Nino3.4 指数
3. 绘制多模式平均 (MMM) 时间序列图 (参考 nino34_eawm_index_calculation.ipynb 的绘图逻辑)
4. 计算 Nino 3.4 Heidke Skill Score (基于 NOAA/CMA 严格业务标准的 HSS 评分)
   - 输出详细列联表数据（正确率、空报率等），严格分离 Model 和 Leadtime
   - 绘制不同模式随 Leadtime 变化的 HSS 折线图 (样式已与 MMSPE 模块全面对齐)
5. 支持 --plot-only 模式，利用 pickle 缓存解耦计算与绘图，实现极速重绘。

使用方法：
首次计算并绘图：python index_analysis.py --models all --leadtimes 0 1 2 3 4 5
修改图表快速重绘：python index_analysis.py --models all --leadtimes 0 1 2 3 4 5 --plot-only
"""

import sys
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    CLIMATOLOGY_PERIOD,
    SPATIAL_BOUNDS as COMMON_SPATIAL_BOUNDS,
)

from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, normalize_parallel_args

warnings.filterwarnings('ignore')

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

logger = setup_logging(
    log_file='index_analysis.log',
    module_name=__name__
)

MODELS = MODEL_LIST
SPATIAL_BOUNDS = COMMON_SPATIAL_BOUNDS

def _worker_task(model, leadtime):
    """
    独立函数用于多进程执行。
    """
    analyzer = IndexAnalyzer() 
    return (model, leadtime), analyzer._process_single_model_leadtime(model, leadtime)

class IndexAnalyzer:
    """
    指数分析器：负责EAWM和Nino3.4指数的计算与绘图
    """
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader or DataLoader()
        logger.info(f"初始化指数分析器")
        
        # 统一输出路径配置
        self.base_dir = Path("/sas12t1/ffyan/output/index_analysis")
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        self.cache_file = self.base_dir / "cache" / "index_analysis_cache.pkl"
        
        for d in [self.results_dir, self.plots_dir, self.cache_file.parent]:
            d.mkdir(parents=True, exist_ok=True)

    def area_weighted_mean(self, da: xr.DataArray, lat_name: str = 'lat') -> xr.DataArray:
        """计算面积加权平均（使用 cos(latitude) 权重）"""
        try:
            weights = np.cos(np.deg2rad(da[lat_name]))
            if 'lon' in da.dims:
                da_lon_mean = da.mean(dim='lon')
            else:
                da_lon_mean = da
            
            weighted_mean = (da_lon_mean * weights).sum(dim=lat_name) / weights.sum()
            return weighted_mean
        except Exception as e:
            logger.error(f"面积加权平均失败: {e}")
            raise

    def compute_monthly_anomaly(self, series: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """计算逐月气候态异常"""
        try:
            start_year, end_year = baseline.split('-')
            clim_data = series.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            climatology = clim_data.groupby('time.month').mean(dim='time')
            anomaly = series.groupby('time.month') - climatology
            return anomaly
        except Exception as e:
            logger.error(f"计算月异常失败: {e}")
            return None

    def load_obs_pressure_level_data(self, var_name: str, pressure_level: int = 500,
                                     year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """从MonthlyPressureLevel加载观测数据"""
        try:
            obs_dir = Path("/sas12t1/ffyan/MonthlyPressureLevel")
            if not obs_dir.exists(): return None
            
            logger.info(f"从MonthlyPressureLevel加载观测数据: {var_name} @ {pressure_level}hPa")
            monthly_da_list = []
            start_year, end_year = year_range
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = obs_dir / f"era5_pressure_levels_{year}{month:02d}.nc"
                    if not file_path.exists(): continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            var_candidates = [var_name, var_name.upper(), var_name.lower()]
                            actual_var = None
                            for candidate in var_candidates:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            if actual_var is None: continue
                            
                            da = ds[actual_var]
                            level_coord = 'pressure_level' if 'pressure_level' in da.dims else 'level'
                            if level_coord in da.coords:
                                if pressure_level in ds[level_coord].values:
                                    da = da.sel({level_coord: pressure_level}).drop_vars(level_coord, errors='ignore')
                                else:
                                    continue
                            
                            time_coord = 'valid_time' if 'valid_time' in da.dims else ('time' if 'time' in da.dims else None)
                            if time_coord and da[time_coord].size > 0:
                                da = da.isel({time_coord: 0})
                            
                            if 'latitude' in da.coords: da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords: da = da.rename({'longitude': 'lon'})
                            
                            if 'lat' in da.coords and da.lat.values[0] > da.lat.values[-1]:
                                da = da.sortby('lat')
                            
                            time_stamp = pd.Timestamp(year, month, 1)
                            da = da.expand_dims(time=[time_stamp])
                            monthly_da_list.append(da.load())
                    except Exception:
                        continue
            
            if not monthly_da_list: return None
            data = xr.concat(monthly_da_list, dim='time').sortby('time')
            if 'lat' in data.coords and data.lat.values[0] > data.lat.values[-1]:
                data = data.sortby('lat')
            
            return data
        except Exception as e:
            logger.error(f"加载观测气压层数据失败: {e}")
            return None

    def load_model_sst_monthly(self, model: str, leadtime: int,
                                year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """【内存优化版】加载模式SST并直接计算区域平均"""
        try:
            forecast_dir = Path("/raid62/EC-C3S/month")
            model_dir = forecast_dir / model
            
            if not model_dir.exists(): return None
            
            suffix = self.data_loader.models[model].get('sfc', None)
            if suffix is None: return None
            
            logger.info(f"加载模式 SST 数据: {model} L{leadtime}")
            monthly_values_list = []
            start_year, end_year = year_range
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = model_dir / f"{year}{month:02d}.{suffix}.nc"
                    if not file_path.exists(): continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            sst_vars = ['sst', 'tos', 'sea_surface_temperature', 'SST', 'TOS']
                            actual_var = None
                            for candidate in sst_vars:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            if actual_var is None: continue
                            
                            da = ds[actual_var]
                            
                            if 'time' in da.dims and da.time.size > leadtime:
                                da = da.isel(time=leadtime)
                            elif 'time' in da.dims:
                                init_time = pd.Timestamp(year, month, 1)
                                try:
                                    da = da.sel(time=init_time, method='nearest', tolerance='15D')
                                except Exception: continue
                            
                            if 'latitude' in da.coords: da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords: da = da.rename({'longitude': 'lon'})
                            
                            if da.lon.min() < 0:
                                da.coords['lon'] = (da.coords['lon'] + 360) % 360
                                da = da.sortby('lon')
                            
                            da_nino = da.sel(lat=slice(-5, 5), lon=slice(190, 240))
                            if da_nino.size == 0:
                                da_nino = da.sel(lat=slice(5, -5), lon=slice(190, 240))
                            
                            if 'number' not in da_nino.dims:
                                da_nino = da_nino.expand_dims('number')
                            
                            weights = np.cos(np.deg2rad(da_nino.lat))
                            da_weighted = da_nino.weighted(weights)
                            spatial_mean = da_weighted.mean(dim=['lat', 'lon']).load()
                            
                            forecast_time = pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                            spatial_mean = spatial_mean.expand_dims(time=[forecast_time])
                            monthly_values_list.append(spatial_mean)
                            
                    except Exception:
                        continue
            
            if not monthly_values_list: return None
            data = xr.concat(monthly_values_list, dim='time').sortby('time').astype(np.float32)
            return data
            
        except Exception as e:
            logger.error(f"加载模式 SST 数据失败: {e}")
            return None

    def compute_eawm_index(self, u_500: xr.DataArray) -> Optional[xr.DataArray]:
        """计算东亚冬季季风指数（I_EAWM）"""
        try:
            if 'number' in u_500.dims:
                u_500_mean = u_500.mean(dim='number')
            else:
                u_500_mean = u_500
            
            south_region = {'lat': slice(25, 35), 'lon': slice(80, 120)}
            north_region = {'lat': slice(45, 55), 'lon': slice(80, 120)}
            
            u_south = self.area_weighted_mean(u_500_mean.sel(**south_region), lat_name='lat')
            u_north = self.area_weighted_mean(u_500_mean.sel(**north_region), lat_name='lat')
            
            index_raw = u_south - u_north
            djf_months = [12, 1, 2]
            index_djf = index_raw.sel(time=index_raw.time.dt.month.isin(djf_months))
            
            if len(index_djf.time) < 3: return None
            
            index_values = index_djf.values
            index_mean = np.nanmean(index_values)
            index_std = np.nanstd(index_values)
            
            if index_std < 1e-10: return None
            index_normalized = (index_values - index_mean) / index_std
            
            index_result = xr.DataArray(
                index_normalized, coords={'time': index_djf.time}, dims=['time'], name='eawm_index'
            )
            index_result.attrs = {'long_name': 'EAWM Index', 'season': 'DJF'}
            return index_result
        except Exception as e:
            logger.error(f"计算EAWM指数失败: {e}")
            return None

    def compute_nino34_index_optimized(self, baseline: str = CLIMATOLOGY_PERIOD,
                                       era5_root: str = '/sas12t1/ffyan/ERA5/daily-nc/single-level/',
                                       year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """内存优化版：逐月处理并即时释放内存"""
        try:
            logger.info("启动内存优化模式计算 Nino3.4...")
            var_name = 'sst'
            var_dir = Path(era5_root) / var_name
            if not var_dir.exists(): return None
            
            start_year, end_year = year_range
            nino34_timeseries = []
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = var_dir / f"era5_daily_{var_name}_{year}{month:02d}.nc"
                    if not file_path.exists(): continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            sst_vars = ['sst', 'tos', 'sea_surface_temperature', 'SST', 'TOS']
                            actual_var = None
                            for candidate in sst_vars:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            if actual_var is None: continue
                            
                            da = ds[actual_var]
                            if 'latitude' in da.coords: da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords: da = da.rename({'longitude': 'lon'})
                            
                            if da.lon.min() < 0:
                                da.coords['lon'] = (da.coords['lon'] + 360) % 360
                                da = da.sortby('lon')
                            
                            da = da.sortby('lat')
                            da_sub = da.sel(lat=slice(-5, 5), lon=slice(190, 240))
                            if da_sub.size == 0: continue
                            
                            da_mon = da_sub.resample(time='1MS').mean('time')
                            weights = np.cos(np.deg2rad(da_mon.lat))
                            da_weighted = da_mon.weighted(weights)
                            mean_val = da_weighted.mean(dim=['lat', 'lon']).load()
                            nino34_timeseries.append(mean_val)
                            
                    except Exception:
                        continue
            
            if not nino34_timeseries: return None
            nino34_da = xr.concat(nino34_timeseries, dim='time').sortby('time')
            
            start_year_str, end_year_str = baseline.split('-')
            clim_data = nino34_da.sel(time=slice(f"{start_year_str}-01-01", f"{end_year_str}-12-31"))
            clim = clim_data.groupby('time.month').mean('time')
            nino34_anom = nino34_da.groupby('time.month') - clim
            
            nino34_anom.name = 'nino34_index'
            return nino34_anom
            
        except Exception as e:
            logger.error(f"ERA5 计算失败: {e}")
            return None

    # ========================== 绘图逻辑 ==========================

    def plot_nino34_mmm(self, nino34_indices: Dict[str, xr.DataArray], output_file: Path, time_resolution: str = 'annual'):
        """绘制Nino3.4指数多模式平均时间序列图"""
        try:
            def aggregate_data(times, values, resolution, season=None):
                df = pd.DataFrame({'time': pd.DatetimeIndex(times), 'value': values})
                if resolution == 'monthly':
                    return df['time'].values, df['value'].values
                elif resolution == 'annual':
                    df['year'] = df['time'].dt.year
                    yearly = df.groupby('year')['value'].mean()
                    plot_times = pd.DatetimeIndex([pd.Timestamp(f'{y}-01-01') for y in yearly.index])
                    return plot_times.values, yearly.values
                elif resolution == 'seasonal':
                    df['year'] = df['time'].dt.year
                    df['month'] = df['time'].dt.month
                    def get_season(month):
                        if month in [12, 1, 2]: return 'DJF'
                        elif month in [3, 4, 5]: return 'MAM'
                        elif month in [6, 7, 8]: return 'JJA'
                        else: return 'SON'
                    df['season'] = df['month'].apply(get_season)
                    df['season_year'] = df['year']
                    df.loc[df['month'] == 12, 'season_year'] = df.loc[df['month'] == 12, 'year'] + 1
                    
                    if season is not None: df = df[df['season'] == season]
                    
                    df['season_key'] = df['season_year'].astype(str) + '-' + df['season']
                    seasonal = df.groupby('season_key')['value'].mean()
                    plot_times = []
                    for key in seasonal.index:
                        year, s = key.split('-')
                        month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
                        plot_times.append(pd.Timestamp(f'{year}-{month_map[s]:02d}-15'))
                    return pd.DatetimeIndex(plot_times).values, seasonal.values
                else:
                    raise ValueError(f"不支持的时间分辨率: {resolution}")
            
            era5_data = nino34_indices.get('ERA5')
            if era5_data is None: return
            
            model_data_l0 = {k.replace('_L0', ''): v for k, v in nino34_indices.items() if '_L0' in k}
            model_data_l3 = {k.replace('_L3', ''): v for k, v in nino34_indices.items() if '_L3' in k}
            
            if time_resolution == 'seasonal':
                seasons = ['DJF', 'MAM', 'JJA', 'SON']
                for season in seasons:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    era5_t, era5_v = aggregate_data(era5_data.time.values, era5_data.values, 'seasonal', season)
                    era5_t = pd.DatetimeIndex(era5_t)
                    ax.plot(era5_t, era5_v, color='black', linewidth=2.5, marker='o', markersize=4, label='ERA5', zorder=10)
                    
                    for leadtime, model_data, color, name in [(0, model_data_l0, 'red', 'L0'), (3, model_data_l3, 'blue', 'L3')]:
                        if len(model_data) > 1:
                            model_aggs = {}
                            for m, d in model_data.items():
                                t, v = aggregate_data(d.time.values, d.values, 'seasonal', season)
                                model_aggs[m] = pd.Series(v, index=pd.DatetimeIndex(t))
                            
                            df_models = pd.DataFrame(model_aggs)
                            mmm = df_models.mean(axis=1)
                            std = df_models.std(axis=1)
                            common_t = mmm.index
                            ax.plot(common_t, mmm, color=color, linewidth=2, linestyle='-' if leadtime==0 else '--', 
                                    marker='o' if leadtime==0 else 's', markersize=4, label=f'MMM ({name})', zorder=5)
                            ax.fill_between(common_t, mmm-std, mmm+std, color=color, alpha=0.2)

                    ax.tick_params(axis='both', labelsize=20)
                    ax.set_ylabel('Nino3.4 Index (K)', fontsize=20, fontweight='bold')
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    plt.tight_layout()
                    
                    s_out = output_file.parent / f"{output_file.stem}_{season.lower()}{output_file.suffix}"
                    plt.savefig(s_out, dpi=300, bbox_inches='tight')
                    plt.close()
                return

            fig, ax = plt.subplots(figsize=(14, 7))
            era5_t, era5_v = aggregate_data(era5_data.time.values, era5_data.values, time_resolution)
            era5_t = pd.DatetimeIndex(era5_t)
            ax.plot(era5_t, era5_v, color='black', linewidth=2.5, marker='o', markersize=4, label='ERA5', zorder=10)
            
            for leadtime, model_data, color, name in [(0, model_data_l0, 'red', 'L0'), (3, model_data_l3, 'blue', 'L3')]:
                if len(model_data) > 1:
                    model_aggs = {}
                    for m, d in model_data.items():
                        t, v = aggregate_data(d.time.values, d.values, time_resolution)
                        model_aggs[m] = pd.Series(v, index=pd.DatetimeIndex(t))
                    
                    df_models = pd.DataFrame(model_aggs)
                    mmm = df_models.mean(axis=1)
                    std = df_models.std(axis=1)
                    common_t = mmm.index
                    
                    ax.plot(common_t, mmm, color=color, linewidth=2, linestyle='-' if leadtime==0 else '--', 
                            marker='o' if leadtime==0 else 's', markersize=4, label=f'MMM ({name})', zorder=5)
                    ax.fill_between(common_t, mmm-std, mmm+std, color=color, alpha=0.2)

            ax.tick_params(axis='both', labelsize=20)
            ax.set_ylabel('Nino3.4 Index (K)', fontsize=20, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"绘制Nino3.4图失败: {e}")

    def plot_eawm_index_timeseries(self, index_dict: Dict[str, xr.DataArray], output_file: Path):
        """绘制EAWM指数时间序列图"""
        try:
            data_frames = {}
            for name, da in index_dict.items():
                if da is None or da.size == 0: continue
                times = pd.DatetimeIndex(da.time.values)
                values = da.values
                years = [t.year if t.month == 12 else t.year - 1 for t in times]
                s = pd.Series(values, index=years, name=name).groupby(level=0).mean()
                data_frames[name] = s
            
            era5_s = data_frames.get('ERA5')
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if era5_s is not None and not era5_s.dropna().empty:
                era5_valid = era5_s.dropna()
                ax.plot(era5_valid.index, era5_valid.values, color='black', linewidth=2.5, marker='o', label='ERA5', zorder=10)
            
            l0_models = [k for k in data_frames.keys() if '_L0' in k]
            l3_models = [k for k in data_frames.keys() if '_L3' in k]
            
            for leadtime, models, color, name in [(0, l0_models, 'red', 'L0'), (3, l3_models, 'blue', 'L3')]:
                if models:
                    df_m = pd.DataFrame({m: data_frames[m] for m in models})
                    mmm = df_m.mean(axis=1)
                    std = df_m.std(axis=1)
                    ax.plot(mmm.index, mmm.values, color=color, linewidth=2, linestyle='-' if leadtime == 0 else '--',
                            marker='o' if leadtime == 0 else 's', markersize=4, label=f'MMM ({name})', zorder=5)
                    ax.fill_between(mmm.index, mmm - std, mmm + std, color=color, alpha=0.2)

            ax.tick_params(axis='both', labelsize=20)
            ax.set_ylabel('EAWM Index (Standardized)', fontsize=20, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            plt.tight_layout()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"绘制 EAWM 图失败: {e}")

    def compute_nino34_hss(self, nino34_indices: Dict[str, xr.DataArray], output_dir: Path):
        """
        计算各模式与 ERA5 在 Nino3.4 指数上的 Heidke Skill Score (HSS)。
        计算列联表：正确率、空报率、漏报率等详细信息。
        """
        try:
            logger.info("开始计算 Nino3.4 Heidke Skill Score 及列联表数据...")
            if 'ERA5' not in nino34_indices:
                logger.warning("缺少 ERA5 数据，无法计算 HSS")
                return
            
            era5_da = nino34_indices['ERA5']
            
            def classify_series_standard(da: xr.DataArray) -> pd.Series:
                s = pd.Series(da.values, index=pd.DatetimeIndex(da.time.values))
                # 3个月滑动平均（ONI/Z值）
                oni = s.rolling(window=3, min_periods=3, center=True).mean()
                
                is_el = oni >= 0.5
                is_la = oni <= -0.5

                el_5 = is_el.rolling(window=5, min_periods=5).sum() == 5
                la_5 = is_la.rolling(window=5, min_periods=5).sum() == 5

                el_mask = el_5.copy()
                la_mask = la_5.copy()
                for i in range(1, 5):
                    el_mask = el_mask | el_5.shift(-i, fill_value=False)
                    la_mask = la_mask | la_5.shift(-i, fill_value=False)
                
                classes = pd.Series(0, index=oni.index)
                classes[el_mask] = 1   # El Nino 事件
                classes[la_mask] = -1  # La Nina 事件
                classes[oni.isna()] = np.nan
                return classes

            obs_classes = classify_series_standard(era5_da)
            
            hss_results = []
            for model_name, model_da in nino34_indices.items():
                if model_name == 'ERA5': continue
                
                mod_classes = classify_series_standard(model_da)
                common_times = obs_classes.dropna().index.intersection(mod_classes.dropna().index)
                if len(common_times) == 0: continue
                    
                obs_align = obs_classes.loc[common_times]
                mod_align = mod_classes.loc[common_times]

                categories = [-1, 0, 1]
                conf_matrix = pd.crosstab(obs_align, mod_align, dropna=False)
                conf_matrix = conf_matrix.reindex(index=categories, columns=categories, fill_value=0)
                
                T = conf_matrix.values.sum()
                if T == 0: continue
                H = np.trace(conf_matrix.values)
                E = np.sum((conf_matrix.sum(axis=1).values * conf_matrix.sum(axis=0).values)) / T
                
                hss = 0.0 if T == E else (H - E) / (T - E)
                accuracy = H / T if T > 0 else 0.0

                obs_el = conf_matrix.loc[1].sum()
                pred_el = conf_matrix[1].sum()
                hit_el = conf_matrix.loc[1, 1]
                far_el = (pred_el - hit_el) / pred_el if pred_el > 0 else np.nan  
                pod_el = hit_el / obs_el if obs_el > 0 else np.nan                
                
                obs_la = conf_matrix.loc[-1].sum()
                pred_la = conf_matrix[-1].sum()
                hit_la = conf_matrix.loc[-1, -1]
                far_la = (pred_la - hit_la) / pred_la if pred_la > 0 else np.nan
                pod_la = hit_la / obs_la if obs_la > 0 else np.nan
                
                obs_nu = conf_matrix.loc[0].sum()
                pred_nu = conf_matrix[0].sum()
                hit_nu = conf_matrix.loc[0, 0]
                far_nu = (pred_nu - hit_nu) / pred_nu if pred_nu > 0 else np.nan
                pod_nu = hit_nu / obs_nu if obs_nu > 0 else np.nan

                base_model = model_name.split('_L')[0] if '_L' in model_name else model_name
                leadtime_val = int(model_name.split('_L')[-1]) if '_L' in model_name else 0

                hss_results.append({
                    'Model': base_model,
                    'Leadtime': leadtime_val,
                    'HSS': hss,
                    'Accuracy': accuracy,
                    'Total_Samples': T,
                    'Obs_El_Nino': obs_el,
                    'Pred_El_Nino': pred_el,
                    'Hit_El_Nino': hit_el,
                    'POD_El_Nino': pod_el,
                    'FAR_El_Nino': far_el,
                    'Obs_La_Nina': obs_la,
                    'Pred_La_Nina': pred_la,
                    'Hit_La_Nina': hit_la,
                    'POD_La_Nina': pod_la,
                    'FAR_La_Nina': far_la,
                    'Obs_Neutral': obs_nu,
                    'Pred_Neutral': pred_nu,
                    'Hit_Neutral': hit_nu,
                    'POD_Neutral': pod_nu,
                    'FAR_Neutral': far_nu
                })
                
            if hss_results:
                df_hss = pd.DataFrame(hss_results)
                output_csv = output_dir / "nino34_contingency_and_hss_standard.csv"
                df_hss.to_csv(output_csv, index=False)
                logger.info(f"HSS 及其列联表计算完成，保存至: {output_csv}")
                
                self.plot_hss_lines(df_hss, output_dir / "nino34_hss_lines.png")
                
        except Exception as e:
            logger.error(f"计算 HSS 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def plot_hss_lines(self, df_hss: pd.DataFrame, output_file: Path):
        """绘制横轴为 Leadtime 的多模式 HSS 折线图 (兼容 MMSPE 样式)"""
        try:
            logger.info("绘制 HSS 折线图...")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = [m for m in MODELS if m in df_hss['Model'].unique()]
            for m in df_hss['Model'].unique():
                if m not in models: models.append(m)
                    
            leadtimes = sorted(df_hss['Leadtime'].unique())
            cmap = plt.get_cmap('tab10')
            
            for idx, model in enumerate(MODELS):
                if model not in df_hss['Model'].values: continue
                
                model_data = df_hss[df_hss['Model'] == model].sort_values('Leadtime')
                if model_data.empty: continue
                
                display_name = model.replace('-mon','').replace('Meteo-France','MF').replace('ECCC-Canada', 'ECCC-3')
                color = cmap(idx % 10)
                
                ax.plot(model_data['Leadtime'], model_data['HSS'], marker='o', 
                        linewidth=2.5, markersize=8, label=display_name, color=color, alpha=0.9, 
                        markeredgecolor='white', markeredgewidth=0.5)
            

            # ax.set_xlabel('Lead Time (Months)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Heidke Skill Score (HSS)', fontsize=14, fontweight='bold')
            
            ax.set_xticks(leadtimes)
            ax.set_xticklabels([f"Lead {lt}" for lt in leadtimes], fontsize=12)

            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', 
                       bbox_to_anchor=(0.5, -0.05), ncol=min(len(models), 6), fontsize=12, frameon=False)
            
            ax.grid(True, linestyle=':', alpha=0.6)
            
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"HSS 折线图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制 HSS 折线图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _process_single_model_leadtime(self, model: str, leadtime: int) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
        try:
            eawm_index, nino34_index = None, None
            u500_model = self.data_loader.load_pressure_level_data(model=model, leadtime=leadtime, var_name='u', pressure_level=500)
            if u500_model is not None:
                eawm_index = self.compute_eawm_index(u500_model)
            
            sst_index_raw = self.load_model_sst_monthly(model, leadtime)
            if sst_index_raw is not None:
                if 'number' in sst_index_raw.dims:
                    sst_index_raw = sst_index_raw.mean(dim='number')
                nino34_index = self.compute_monthly_anomaly(sst_index_raw)
                if nino34_index is not None:
                    nino34_index.name = 'nino34_index'
            
            return eawm_index, nino34_index
        except Exception:
            return None, None

    def _load_cache(self):
        """从缓存加载计算结果"""
        if not self.cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.cache_file}. 请先不带 --plot-only 运行一次。")
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_cache(self, eawm_indices, nino34_indices):
        """保存计算结果到缓存"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'eawm_indices': eawm_indices,
                'nino34_indices': nino34_indices,
            }, f)
        logger.info(f"已保存缓存: {self.cache_file}")

    def run_analysis(self, models: List[str], leadtimes: List[int], parallel: bool = False, n_jobs: Optional[int] = None, plot_only: bool = False):
        if plot_only:
            logger.info("--plot-only 模式: 从缓存加载数据并仅绘图")
            cache = self._load_cache()
            eawm_indices = cache.get('eawm_indices', {})
            nino34_indices = cache.get('nino34_indices', {})
            
            if len(eawm_indices) > 0:
                self.plot_eawm_index_timeseries(eawm_indices, self.plots_dir / "circulation_eawm_index_mmm.png")
            
            if len(nino34_indices) > 0:
                self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_monthly.png", 'monthly')
                self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_annual.png", 'annual')
                self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_seasonal.png", 'seasonal')
                self.compute_nino34_hss(nino34_indices, self.plots_dir)
            return

        eawm_era5, nino34_era5 = None, None
        try:
            eawm_file = self.results_dir / "circulation_obs_eawm_index.nc"
            if eawm_file.exists(): eawm_era5 = xr.open_dataarray(eawm_file).load()
            else:
                u500_era5 = self.load_obs_pressure_level_data('u', 500)
                if u500_era5 is not None:
                    eawm_era5 = self.compute_eawm_index(u500_era5)
                    if eawm_era5 is not None: eawm_era5.to_netcdf(eawm_file)
        except Exception: pass

        try:
            nino34_file = self.results_dir / "circulation_nino34_era5.nc"
            if nino34_file.exists(): nino34_era5 = xr.open_dataarray(nino34_file)
            else:
                nino34_era5 = self.compute_nino34_index_optimized()
                if nino34_era5 is not None: nino34_era5.to_netcdf(nino34_file)
        except Exception: pass
        
        eawm_indices, nino34_indices = {}, {}
        if eawm_era5 is not None: eawm_indices['ERA5'] = eawm_era5
        if nino34_era5 is not None: nino34_indices['ERA5'] = nino34_era5
        
        model_tasks = [(model, leadtime) for leadtime in leadtimes for model in models]
        
        if parallel and n_jobs and n_jobs > 1 and len(model_tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            parallel_failed = False
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_task = {executor.submit(_worker_task, model, leadtime): (model, leadtime) for model, leadtime in model_tasks}
                    for future in as_completed(future_to_task):
                        try:
                            (model, leadtime), (eawm_index, nino34_index) = future.result(timeout=1800)
                            if eawm_index is not None: eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                            if nino34_index is not None: nino34_indices[f"{model}_L{leadtime}"] = nino34_index
                        except Exception: pass
            except Exception:
                parallel_failed = True
            
            if parallel_failed:
                for model, leadtime in model_tasks:
                    eawm_index, nino34_index = self._process_single_model_leadtime(model, leadtime)
                    if eawm_index is not None: eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                    if nino34_index is not None: nino34_indices[f"{model}_L{leadtime}"] = nino34_index
        else:
            for model, leadtime in model_tasks:
                eawm_index, nino34_index = self._process_single_model_leadtime(model, leadtime)
                if eawm_index is not None: eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                if nino34_index is not None: nino34_indices[f"{model}_L{leadtime}"] = nino34_index

        self._save_cache(eawm_indices, nino34_indices)

        if len(eawm_indices) > 0:
            self.plot_eawm_index_timeseries(eawm_indices, self.plots_dir / "circulation_eawm_index_mmm.png")
        
        if len(nino34_indices) > 0:
            self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_monthly.png", 'monthly')
            self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_annual.png", 'annual')
            self.plot_nino34_mmm(nino34_indices, self.plots_dir / "circulation_nino34_index_mmm_seasonal.png", 'seasonal')
            
            self.compute_nino34_hss(nino34_indices, self.plots_dir)

def main():
    parser = create_parser(description="指数分析：EAWM 和 Nino3.4", var_required=False)
    args = parser.parse_args()
    
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, [0, 1, 2, 3, 4, 5]) if args.leadtimes else [0, 1, 2, 3, 4, 5]
    parallel = normalize_parallel_args(args)
    plot_only = getattr(args, 'plot_only', False)
    
    analyzer = IndexAnalyzer()
    analyzer.run_analysis(models=models, leadtimes=leadtimes, parallel=parallel, n_jobs=args.n_jobs, plot_only=plot_only)

if __name__ == "__main__":
    main()
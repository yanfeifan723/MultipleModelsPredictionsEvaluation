#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节性预报方案误差分析模块 (RMSE, MAE, Bias) (V4 - Seasonal Boxplots & Dynamic Time)
修改内容：
1. 集成命令行参数 --plot-only，支持跳过计算直接绘图。
2. 引入数据缓存机制 (pickle)。
3. 移除硬编码的 MONTH_MAPPING，使用模运算动态计算 lead_time。
4. 将硬编码的跨年逻辑 (DJF 12月 +1) 下沉为动态的 seasonal_year 分配，彻底解耦 time 坐标处理。
5. 将月度折线图替换为以季节为横轴的箱线散点图 (Boxplot + Scatter)，全面展示成员不确定性 (Spread)、单模式和MMM。
6. 区域数据计算升级为：先空间拼接成季节场，求面积加权区域平均，最后沿“年份(Year)”计算季节误差指标。

定义：
- Short-term (L0-2): 使用起报当月(L0)、+1月(L1)、+2月(L2)的数据合成目标季节。
- Long-term (L3-5): 使用起报+3月(L3)、+4月(L4)、+5月(L5)的数据合成目标季节。
"""

import sys
import os
import pickle
from pathlib import Path
import logging
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import pandas as pd
import regionmask
import seaborn as sns  # 引入 seaborn 用于绘制箱线图
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
# === Cartopy 相关导入 ===
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 统一导入toolkit路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_vars, normalize_parallel_args

from common_config import (
    MODEL_LIST,
    SEASONS,
    SPATIAL_BOUNDS,
)

# === 配置参数 ===
MODELS = MODEL_LIST
SCHEMES = ['Short-term', 'Long-term']

# === 动态时间计算配置 ===
# 方案相对于季节起始月份的初始化提前量 (月)
SCHEME_OFFSETS = {
    'Short-term': 0, 
    'Long-term': 3   
}

def get_season_for_month(target_month: int) -> str:
    """根据目标月份动态判断所属季节"""
    for season, months in SEASONS.items():
        if target_month in months:
            return season
    return 'DJF'

def get_dynamic_lead_time(target_month: int, season: str, scheme: str) -> int:
    """利用数学取模动态计算所需预报时效，消除查表硬编码"""
    season_start_month = SEASONS[season][0]
    # 取模处理能完美解决跨年月份计算 (如 1月 - 12月 -> (1 - 12) % 12 = 1)
    month_diff = (target_month - season_start_month) % 12
    return month_diff + SCHEME_OFFSETS.get(scheme, 0)

# 区域定义
def generate_regions():
    regions = {'Global': None}
    regions.update({
        'Z1-Northwest':     {'lat': (39, 49), 'lon': (73, 105)},
        'Z2-InnerMongolia': {'lat': (39, 50), 'lon': (106, 118)},
        'Z3-Northeast':     {'lat': (40, 54), 'lon': (119, 135)},
        'Z4-Tibetan':       {'lat': (27, 39), 'lon': (73, 95)},
        'Z5-NorthChina':    {'lat': (34, 39), 'lon': (106, 122)},
        'Z6-Yangtze':       {'lat': (26, 34), 'lon': (109, 123)},
        'Z7-Southwest':     {'lat': (23, 33), 'lon': (96, 108)},
        'Z8-SouthChina':    {'lat': (21, 25), 'lon': (106, 120)},
        'Z9-SouthSea':      {'lat': (18, 21), 'lon': (105, 125)}
    })
    return regions

REGIONS = generate_regions()

# 配置日志
logger = setup_logging(
    log_file='seasonal_scheme_error_analysis.log',
    module_name=__name__
)

class SeasonalSchemeErrorAnalyzer:
    """季节预报方案误差分析器 (RMSE, MAE, Bias)"""
    
    def __init__(self, var_type: str, n_jobs: Optional[int] = None):
        self.var_type = var_type
        self.data_loader = DataLoader()
        self.n_jobs = n_jobs
        self.unit_label = "°C" if var_type == 'temp' else "mm/day"
        
        # 输出路径配置
        self.base_dir = Path("/sas12t1/ffyan/output/seasonal_scheme_error_analysis")
        self.map_dir = self.base_dir / f"spatial_maps/{self.var_type}"
        self.plot_dir = self.base_dir / f"plots/{self.var_type}"
        # 缓存文件路径
        self.cache_file = self.base_dir / "cache" / f"{self.var_type}_error_data_cache.pkl"
        
        for d in [self.map_dir, self.plot_dir, self.cache_file.parent]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"

    def convert_temp_units(self, ds: xr.DataArray) -> xr.DataArray:
        """仅对温度进行单位换算 (K -> C)"""
        if self.var_type == 'temp':
            if ds.mean(skipna=True) > 200:
                ds = ds - 273.15
        return ds

    def apply_land_mask(self, ds: xr.DataArray) -> xr.DataArray:
        """应用海陆掩模 (保留陆地)"""
        try:
            if 'number' in ds.dims:
                sample = ds.isel(number=0, drop=True) if ds.sizes['number'] > 0 else ds
                land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(sample)
            else:
                land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds)
            ds_masked = ds.where(land_mask == 0)
            return ds_masked
        except Exception:
            return ds

    def load_monthly_data(self, model: str, scheme: str, target_month: int):
        """加载月度数据 (包含单位转换、Mask和动态年份处理)"""
        try:
            # === 1. 动态获取 Lead Time ===
            season = get_season_for_month(target_month)
            lead_time = get_dynamic_lead_time(target_month, season, scheme)
            
            # 加载观测
            obs_full = self.data_loader.load_obs_data(self.var_type)
            if obs_full is None: return None, None
            obs_full = obs_full.resample(time='1MS').mean().sel(time=slice('1993', '2020'))
            
            # 加载模式
            ds_lead = self.data_loader.load_forecast_data_ensemble(model, self.var_type, lead_time)
            if ds_lead is None: return None, None
            ds_lead = ds_lead.resample(time='1MS').mean().sel(time=slice('1993', '2020'))
            
            # 筛选月份
            obs_m = obs_full.sel(time=obs_full.time.dt.month == target_month)
            fcst_m = ds_lead.sel(time=ds_lead.time.dt.month == target_month)
            
            # 对齐
            try:
                fcst_m = fcst_m.interp(lat=obs_m.lat, lon=obs_m.lon, method='linear')
            except: return None, None
            
            common_times = obs_m.time.to_index().intersection(fcst_m.time.to_index())
            if len(common_times) == 0: return None, None
            
            obs_m = obs_m.sel(time=common_times)
            fcst_m = fcst_m.sel(time=common_times)
            
            # 单位转换和Mask
            obs_m = self.apply_land_mask(self.convert_temp_units(obs_m))
            fcst_m = self.apply_land_mask(self.convert_temp_units(fcst_m))
            
            # === 2. 动态年份(季节)映射消除硬编码跨年逻辑 ===
            is_cross_year = (season == 'DJF' and target_month == 12)
            year_offset = 1 if is_cross_year else 0
            
            obs_m = obs_m.assign_coords(year=('time', obs_m.time.dt.year.values + year_offset))
            fcst_m = fcst_m.assign_coords(year=('time', fcst_m.time.dt.year.values + year_offset))
            
            obs_m = obs_m.swap_dims({'time': 'year'}).drop_vars('time')
            fcst_m = fcst_m.swap_dims({'time': 'year'}).drop_vars('time')
            
            return obs_m, fcst_m
        except Exception as e:
            logger.debug(f"加载月数据失败 {model} {scheme} {target_month}: {e}")
            return None, None

    def construct_seasonal_data(self, model: str, scheme: str, season: str):
        """构建季节平均场"""
        try:
            target_months = SEASONS[season]
            obs_list, fcst_list = [], []
            
            for m in target_months:
                o, f = self.load_monthly_data(model, scheme, m)
                if o is None or f is None: return None, None
                
                # 由于底层 load_monthly_data 已经动态处理了跨年偏移，此处直接合并
                obs_list.append(o)
                fcst_list.append(f)
            
            common_years = obs_list[0].year.values
            for da in obs_list[1:] + fcst_list:
                common_years = np.intersect1d(common_years, da.year.values)
            
            if len(common_years) < 5: return None, None
            
            obs_final = [da.sel(year=common_years) for da in obs_list]
            fcst_final = [da.sel(year=common_years) for da in fcst_list]
            
            obs_seasonal = xr.concat(obs_final, dim='component').mean(dim='component')
            fcst_seasonal = xr.concat(fcst_final, dim='component').mean(dim='component')
            
            return obs_seasonal, fcst_seasonal
        except Exception as e:
            logger.error(f"构建季节数据失败 {model} {scheme} {season}: {e}")
            return None, None

    def calculate_pointwise_metrics(self, obs: xr.DataArray, fcst: xr.DataArray) -> xr.Dataset:
        """计算逐格点指标 (Time Mean) 用作空间绘图"""
        if 'number' in fcst.dims: fcst = fcst.mean(dim='number')
        diff = fcst - obs
        rmse = np.sqrt((diff ** 2).mean(dim='year', skipna=True))
        bias = diff.mean(dim='year', skipna=True)
        mae = np.abs(diff).mean(dim='year', skipna=True)
        return xr.Dataset({'rmse': rmse, 'bias': bias, 'mae': mae})

    def calculate_seasonal_regional_metrics(self, model: str, scheme: str):
        """计算区域的季节平均误差，包含 Ensemble Mean 和 Members"""
        results = {reg: {} for reg in REGIONS}
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        
        for season in seasons:
            obs_seasonal, fcst_seasonal = self.construct_seasonal_data(model, scheme, season)
            if obs_seasonal is None: continue
            
            weights = np.cos(np.deg2rad(obs_seasonal.lat))
            if 'number' in fcst_seasonal.dims:
                fcst_mean = fcst_seasonal.mean(dim='number')
                fcst_mems = fcst_seasonal
            else:
                fcst_mean = fcst_seasonal
                fcst_mems = None
            
            for reg_name, bounds in REGIONS.items():
                if bounds:
                    lat_sl = slice(bounds['lat'][0], bounds['lat'][1]) if obs_seasonal.lat[0] < obs_seasonal.lat[-1] else slice(bounds['lat'][1], bounds['lat'][0])
                    lon_sl = slice(bounds['lon'][0], bounds['lon'][1])
                    obs_reg = obs_seasonal.sel(lat=lat_sl, lon=lon_sl)
                    fcst_reg = fcst_mean.sel(lat=lat_sl, lon=lon_sl)
                    fcst_mem_reg = fcst_mems.sel(lat=lat_sl, lon=lon_sl) if fcst_mems is not None else None
                else:
                    obs_reg = obs_seasonal; fcst_reg = fcst_mean; fcst_mem_reg = fcst_mems
                
                w_reg = weights.sel(lat=obs_reg.lat)
                
                # Mean Metrics
                diff_mean = fcst_reg - obs_reg
                rmse = np.sqrt((diff_mean**2).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True)).mean(dim='year', skipna=True).item()
                mae = np.abs(diff_mean).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                bias = diff_mean.weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                
                # Member Metrics (flattened into lists)
                mem_metrics = {'rmse': [], 'mae': [], 'bias': []}
                if fcst_mem_reg is not None:
                    diff_mem = fcst_mem_reg - obs_reg # (year, number, lat, lon)
                    # MSE over space first, then mean over time
                    mse_mem = (diff_mem**2).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True) # (number)
                    mae_mem = np.abs(diff_mem).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True)
                    bias_mem = diff_mem.weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True)
                    
                    mem_metrics['rmse'] = np.sqrt(mse_mem).values.tolist()
                    mem_metrics['mae'] = mae_mem.values.tolist()
                    mem_metrics['bias'] = bias_mem.values.tolist()
                
                results[reg_name][season] = {
                    'rmse': rmse, 'mae': mae, 'bias': bias,
                    'members': mem_metrics
                }
        return results

    def calculate_mmm_seasonal_metrics(self, scheme: str):
        """计算 Multi-Model Mean (MMM) 的季节区域误差"""
        results = {reg: {} for reg in REGIONS}
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        
        for season in seasons:
            obs_sample, fcst_list = None, []
            for model in MODELS:
                o, f = self.construct_seasonal_data(model, scheme, season)
                if o is None: continue
                if obs_sample is None: obs_sample = o
                common = np.intersect1d(obs_sample.year, f.year)
                if len(common) < 5: continue
                if 'number' in f.dims: f = f.mean(dim='number')
                fcst_list.append(f.sel(year=common))
            
            if not fcst_list: continue
            final_years = obs_sample.year.values
            for f in fcst_list: final_years = np.intersect1d(final_years, f.year.values)
            
            obs_final = obs_sample.sel(year=final_years)
            mmm = xr.concat([f.sel(year=final_years) for f in fcst_list], dim='model').mean(dim='model')
            weights = np.cos(np.deg2rad(obs_final.lat))
            
            for reg, bounds in REGIONS.items():
                if bounds:
                    lat_sl = slice(bounds['lat'][0], bounds['lat'][1]) if obs_final.lat[0] < obs_final.lat[-1] else slice(bounds['lat'][1], bounds['lat'][0])
                    lon_sl = slice(bounds['lon'][0], bounds['lon'][1])
                    o_r = obs_final.sel(lat=lat_sl, lon=lon_sl)
                    f_r = mmm.sel(lat=lat_sl, lon=lon_sl)
                else:
                    o_r = obs_final; f_r = mmm
                
                w_r = weights.sel(lat=o_r.lat)
                diff = f_r - o_r
                
                rmse = np.sqrt((diff**2).weighted(w_r).mean(dim=['lat', 'lon'], skipna=True)).mean(dim='year', skipna=True).item()
                mae = np.abs(diff).weighted(w_r).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                bias = diff.weighted(w_r).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                
                results[reg][season] = {'rmse': rmse, 'mae': mae, 'bias': bias}
        return results

    def get_plotting_params(self, metric: str):
        n_bins = 20
        if metric in ['rmse', 'mae']:
            if self.var_type == 'temp':
                colors = ['white'] + list(plt.get_cmap('Reds')(np.linspace(0, 1, n_bins)))
                cmap = mcolors.LinearSegmentedColormap.from_list('WhiteReds', colors, N=n_bins)
                levels = np.linspace(0, 4, n_bins + 1)
            else:
                colors = ['white'] + list(plt.get_cmap('Blues')(np.linspace(0, 1, n_bins)))
                cmap = mcolors.LinearSegmentedColormap.from_list('WhiteBlues', colors, N=n_bins)
                levels = np.linspace(0, 5, n_bins + 1)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            extend = 'max'
        elif metric == 'bias':
            cmap = plt.get_cmap('RdBu_r' if self.var_type == 'temp' else 'RdBu', n_bins)
            levels = np.linspace(-3, 3, n_bins + 1)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            extend = 'both'
        return cmap, levels, norm, extend

    def add_china_map_details(self, ax, data, levels, cmap, norm, extend):
        bou_paths = [self.boundaries_dir / "中国_省1.shp", self.boundaries_dir / "中国_省2.shp"]
        hyd_path = self.boundaries_dir / "河流.shp"
        if hyd_path.exists():
            try: ax.add_geometries(shpreader.Reader(str(hyd_path)).geometries(), ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6)
            except: pass
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        loaded = False
        for p in bou_paths:
            if p.exists():
                try: ax.add_geometries(shpreader.Reader(str(p)).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6); loaded = True
                except: pass
        if not loaded: ax.add_feature(cfeature.BORDERS, linewidth=1.0)
        try:
            sub = ax.inset_axes([0.7548, 0, 0.33, 0.35], projection=ccrs.PlateCarree())
            sub.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
            sub.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, norm=norm, extend=extend)
            sub.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray')
            if loaded:
                for p in bou_paths:
                    if p.exists():
                        try: sub.add_geometries(shpreader.Reader(str(p)).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6)
                        except: pass
            sub.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in sub.spines.values(): spine.set_edgecolor('black')
        except: pass

    def _plot_single_map(self, fig, gs, row, col, model, scheme, maps, metric, cmap, levels, norm, extend, xticks, yticks, label_char):
        key = (scheme, model)
        if key not in maps:
            ax = fig.add_subplot(gs[row, col]); ax.axis('off'); return
        
        ds = maps[key]
        data = ds[metric]
        
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = FixedLocator(xticks); gl.ylocator = FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter(number_format='.0f')
        gl.yformatter = LatitudeFormatter(number_format='.0f')
        
        ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, norm=norm, extend=extend)
        self.add_china_map_details(ax, data, levels, cmap, norm, extend)
        
        display_name = model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
        ax.text(0.02, 0.96, f"({label_char}) {display_name}", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

    def plot_seasonal_spatial_maps(self, results_map, metric, title_metric):
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        cmap, levels, norm, extend = self.get_plotting_params(metric)
        lon_ticks = np.arange(75, 141, 15); lat_ticks = np.arange(20, 56, 10)
        
        for season in seasons:
            current_maps = {}
            for s in SCHEMES:
                for m in MODELS:
                    if (s, season, m) in results_map:
                        current_maps[(s, m)] = results_map[(s, season, m)]
            if not current_maps: continue

            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15, left=0.05, right=0.92, top=0.94, bottom=0.06)
            
            for s_idx, scheme in enumerate(SCHEMES):
                row_start = s_idx * 2
                ax_blank = fig.add_subplot(gs[row_start, 0]); ax_blank.axis('off')
                for col_idx in range(3):
                    if col_idx >= len(MODELS): continue
                    self._plot_single_map(fig, gs, row_start, col_idx + 1, MODELS[col_idx], scheme, current_maps, metric, cmap, levels, norm, extend, lon_ticks, lat_ticks, chr(97 + col_idx))
                    if col_idx == 0:
                        ax = fig.axes[-1]
                        ax.text(0.98, 0.96, scheme, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(MODELS): continue
                    self._plot_single_map(fig, gs, row_start + 1, col_idx, MODELS[model_idx], scheme, current_maps, metric, cmap, levels, norm, extend, lon_ticks, lat_ticks, chr(97 + model_idx))

            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks=levels[::2], extend=extend, label=f'{title_metric} ({self.unit_label})')
            plt.suptitle(f"{season} {title_metric} - {self.var_type}", fontsize=22, fontweight='bold')
            plt.savefig(self.plot_dir / f"{metric}_map_{season}_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_seasonal_regional_boxplot(self, seasonal_reg_res, mmm_seasonal_res, metric, title_metric):
        """
        绘制季节区域误差指标的箱线散点图 (Boxplot + Scatter)
        展示 All members spread, 各模式均值, 以及 MMM
        """
        regions_order = ['Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast', 
                         'Z4-Tibetan', 'Z5-NorthChina', 'Z6-Yangtze', 
                         'Z7-Southwest', 'Z8-SouthChina', 'Z9-SouthSea']
        regions_to_plot = [r for r in regions_order if r in REGIONS]
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        
        for scheme in SCHEMES:
            logger.info(f"绘制季节箱线图 ({metric}): {scheme}")
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            axes = axes.flatten()
            cmap = plt.get_cmap('tab10')
            
            for i, reg in enumerate(regions_to_plot):
                if i >= len(axes): break
                ax = axes[i]
                
                spread_data = []
                spread_labels = []
                
                # 1. 收集所有 member 的值以绘制底层箱线图
                for season in seasons:
                    season_mems = []
                    for model in MODELS:
                        data = seasonal_reg_res[scheme].get(model, {}).get(reg, {}).get(season, {})
                        if 'members' in data and metric in data['members']:
                            valid = [v for v in data['members'][metric] if np.isfinite(v)]
                            season_mems.extend(valid)
                    
                    spread_data.extend(season_mems)
                    spread_labels.extend([season] * len(season_mems))
                
                if spread_data:
                    sns.boxplot(x=spread_labels, y=spread_data, ax=ax, 
                                color='lightgray', width=0.5, 
                                boxprops=dict(alpha=0.6), showfliers=False, zorder=1)
                
                if metric == 'bias':
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=0)

                # 2. 绘制各个模式 Ensemble Mean 的散点 (加入微小抖动防重叠)
                for idx, model in enumerate(MODELS):
                    model_vals = []
                    for season in seasons:
                        val = seasonal_reg_res[scheme].get(model, {}).get(reg, {}).get(season, {}).get(metric, np.nan)
                        model_vals.append(val)
                    
                    x_pos = np.arange(len(seasons)) + np.random.uniform(-0.15, 0.15, size=len(seasons))
                    display_name = model.replace('-mon','').replace('Meteo-France','MF').replace('ECCC-Canada', 'ECCC-3')
                    
                    ax.scatter(x_pos, model_vals, color=cmap(idx % 10), s=60, alpha=0.9, 
                               edgecolors='white', linewidth=0.5, 
                               label=display_name if i == 0 else "", zorder=3)

                # 3. 绘制 Multi-Model Mean (MMM) 的黑色大星星
                mmm_vals = [mmm_seasonal_res[scheme].get(reg, {}).get(season, {}).get(metric, np.nan) for season in seasons]
                ax.scatter(np.arange(len(seasons)), mmm_vals, color='black', marker='*', 
                           s=250, edgecolors='white', linewidth=1, 
                           label='MMM' if i == 0 else "", zorder=4)
                
                # 样式设置
                ax.set_title(reg, fontsize=16, fontweight='bold')
                ax.set_xticks(np.arange(len(seasons)))
                ax.set_xticklabels(seasons, fontsize=14)
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                
                if metric in ['rmse', 'mae']:
                    ax.set_ylim(bottom=0)
                
                if i % 3 == 0: 
                    ax.set_ylabel(f'Seasonal {title_metric} ({self.unit_label})', fontsize=14)
                ax.tick_params(labelsize=12)

            # 提取并重组图例，确保 MMM 星星在最显眼位置
            handles, labels = axes[0].get_legend_handles_labels()
            if 'MMM' in labels:
                mmm_idx = labels.index('MMM')
                handles.append(handles.pop(mmm_idx))
                labels.append(labels.pop(mmm_idx))
                
            fig.legend(handles, labels, loc='lower center', 
                       bbox_to_anchor=(0.5, 0.02), ncol=len(MODELS)+1, fontsize=14, frameon=False)
            
            plt.subplots_adjust(top=0.92, bottom=0.10, hspace=0.3, wspace=0.2)
            plt.suptitle(f"Regional Seasonal {title_metric} Distribution ({scheme}) - {self.var_type}", fontsize=22, fontweight='bold', y=0.97)
            
            plt.savefig(self.plot_dir / f"seasonal_{metric}_boxplot_{scheme}_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()


    def _process_spatial_task(self, model, scheme, season):
        obs, fcst = self.construct_seasonal_data(model, scheme, season)
        if obs is None: return None
        return (model, scheme, season, self.calculate_pointwise_metrics(obs, fcst))

    def _process_seasonal_regional_task(self, model, scheme):
        return (model, scheme, self.calculate_seasonal_regional_metrics(model, scheme))

    def _load_cache(self):
        """从缓存加载计算结果"""
        if not self.cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.cache_file}. 请先不带 --plot-only 运行一次。")
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_cache(self, results_map, seasonal_reg_res, mmm_res):
        """保存计算结果到缓存"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'results_map': results_map,
                'seasonal_reg_res': seasonal_reg_res,
                'mmm_res': mmm_res,
            }, f)
        logger.info(f"已保存缓存: {self.cache_file}")

    def run_analysis(self, models=None, parallel=False, n_jobs=None, plot_only=False):
        models = models or MODELS
        if plot_only:
            logger.info("--plot-only 模式: 从缓存加载数据并仅绘图")
            cache = self._load_cache()
            for metric, title in [('rmse', 'RMSE'), ('mae', 'MAE'), ('bias', 'Bias')]:
                self.plot_seasonal_spatial_maps(cache['results_map'], metric, title)
                self.plot_seasonal_regional_boxplot(cache['seasonal_reg_res'], cache['mmm_res'], metric, title)
            return

        logger.info("开始计算空间分布数据 (四季)...")
        results_map = {}
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        tasks_spatial = [(model, scheme, season) for scheme in SCHEMES for season in seasons for model in models]
        
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_spatial) or 1)) as executor:
            for future in as_completed({executor.submit(self._process_spatial_task, *t): t for t in tasks_spatial}):
                if future.result(): results_map[(future.result()[1], future.result()[2], future.result()[0])] = future.result()[3]
        
        logger.info("开始计算季节区域指标时间序列数据...")
        seasonal_reg_res = {s: {} for s in SCHEMES}
        tasks_regional = [(model, scheme) for model in models for scheme in SCHEMES]
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_regional))) as executor:
            for future in as_completed({executor.submit(self._process_seasonal_regional_task, *t): t for t in tasks_regional}):
                res = future.result()
                if res: seasonal_reg_res[res[1]][res[0]] = res[2]
        
        logger.info("计算 Multi-Model Mean (MMM) 季节区域指标...")
        mmm_res = {s: self.calculate_mmm_seasonal_metrics(s) for s in SCHEMES}
        
        self._save_cache(results_map, seasonal_reg_res, mmm_res)
        
        for metric, title in [('rmse', 'RMSE'), ('mae', 'MAE'), ('bias', 'Bias')]:
            self.plot_seasonal_spatial_maps(results_map, metric, title)
            self.plot_seasonal_regional_boxplot(seasonal_reg_res, mmm_res, metric, title)

def main():
    parser = create_parser(description="季节预报方案误差分析", var_default=None, var_required=False)
    args = parser.parse_args()
    models = parse_models(args.models, MODELS) if args.models else MODELS
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    plot_only = getattr(args, 'plot_only', False)

    for var_type in var_list:
        analyzer = SeasonalSchemeErrorAnalyzer(var_type, n_jobs=args.n_jobs)
        analyzer.run_analysis(models=models, parallel=normalize_parallel_args(args),
                              n_jobs=args.n_jobs, plot_only=plot_only)

if __name__ == "__main__":
    main()
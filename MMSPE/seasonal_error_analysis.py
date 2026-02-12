#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节性预报方案误差分析模块 (RMSE, MAE, Bias) (Style Updated & Cached)
修改内容：
1. 集成命令行参数 --plot-only，支持跳过计算直接绘图。
2. 引入数据缓存机制 (pickle)。
3. plot_monthly_timeseries 风格完全对齐 combined_error_analysis.py。
4. 引入 Multi-model Member Spread (灰色背景)。

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

# 月份映射关系 (Month -> Season, Short Lead, Long Lead)
MONTH_MAPPING = {
    1:  ('DJF', 1, 4), 2:  ('DJF', 2, 5), 3:  ('MAM', 0, 3),
    4:  ('MAM', 1, 4), 5:  ('MAM', 2, 5), 6:  ('JJA', 0, 3),
    7:  ('JJA', 1, 4), 8:  ('JJA', 2, 5), 9:  ('SON', 0, 3),
    10: ('SON', 1, 4), 11: ('SON', 2, 5), 12: ('DJF', 0, 3)
}

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
        """加载月度数据 (包含单位转换和Mask)"""
        try:
            scheme_idx = 1 if scheme == 'Short-term' else 2 
            lead_time = MONTH_MAPPING[target_month][scheme_idx]
            
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
            
            obs_m = obs_m.assign_coords(year=('time', obs_m.time.dt.year.values))
            fcst_m = fcst_m.assign_coords(year=('time', fcst_m.time.dt.year.values))
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
                
                if season == 'DJF' and m == 12:
                    o = o.assign_coords(year=o.year + 1)
                    f = f.assign_coords(year=f.year + 1)
                
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
        """计算逐格点指标 (Time Mean)"""
        if 'number' in fcst.dims: fcst = fcst.mean(dim='number')
        diff = fcst - obs
        rmse = np.sqrt((diff ** 2).mean(dim='year', skipna=True))
        bias = diff.mean(dim='year', skipna=True)
        mae = np.abs(diff).mean(dim='year', skipna=True)
        return xr.Dataset({'rmse': rmse, 'bias': bias, 'mae': mae})

    def calculate_monthly_regional_metrics(self, model: str, scheme: str):
        """计算月度区域指标"""
        results = {reg: {} for reg in REGIONS}
        for m in range(1, 13):
            obs, fcst = self.load_monthly_data(model, scheme, m)
            if obs is None: continue
            
            weights = np.cos(np.deg2rad(obs.lat))
            if 'number' in fcst.dims:
                fcst_mean = fcst.mean(dim='number')
                fcst_mems = fcst
            else:
                fcst_mean = fcst
                fcst_mems = None
            
            for reg_name, bounds in REGIONS.items():
                if bounds:
                    lat_sl = slice(bounds['lat'][0], bounds['lat'][1]) if obs.lat[0] < obs.lat[-1] else slice(bounds['lat'][1], bounds['lat'][0])
                    lon_sl = slice(bounds['lon'][0], bounds['lon'][1])
                    obs_reg = obs.sel(lat=lat_sl, lon=lon_sl)
                    fcst_reg = fcst_mean.sel(lat=lat_sl, lon=lon_sl)
                    fcst_mem_reg = fcst_mems.sel(lat=lat_sl, lon=lon_sl) if fcst_mems is not None else None
                else:
                    obs_reg = obs; fcst_reg = fcst_mean; fcst_mem_reg = fcst_mems
                
                w_reg = weights.sel(lat=obs_reg.lat)
                
                # Mean Metrics
                diff_mean = fcst_reg - obs_reg
                rmse = np.sqrt((diff_mean**2).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True)).mean(dim='year', skipna=True).item()
                mae = np.abs(diff_mean).weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                bias = diff_mean.weighted(w_reg).mean(dim=['lat', 'lon'], skipna=True).mean(dim='year', skipna=True).item()
                
                # Member Metrics (flattened)
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
                
                results[reg_name][m] = {
                    'rmse': rmse, 'mae': mae, 'bias': bias,
                    'members': mem_metrics
                }
        return results

    def calculate_mmm_monthly_metrics(self, scheme: str):
        """计算 MMM 的月度区域指标"""
        results = {reg: {} for reg in REGIONS}
        for m in range(1, 13):
            obs_sample, fcst_list = None, []
            for model in MODELS:
                o, f = self.load_monthly_data(model, scheme, m)
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
                
                results[reg][m] = {'rmse': rmse, 'mae': mae, 'bias': bias}
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

    def plot_monthly_timeseries(self, monthly_reg_res, mmm_res, metric, title_metric):
        """
        绘制区域月度指标折线图
        风格: Grey Spread, Colored Lines, Bold MMM
        """
        regions_order = ['Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast', 
                         'Z4-Tibetan', 'Z5-NorthChina', 'Z6-Yangtze', 
                         'Z7-Southwest', 'Z8-SouthChina', 'Z9-SouthSea']
        regions_to_plot = [r for r in regions_order if r in REGIONS]
        
        for scheme in SCHEMES:
            logger.info(f"绘制折线图 ({metric}): {scheme}")
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            axes = axes.flatten()
            months = np.arange(1, 13)
            cmap = plt.get_cmap('tab10')

            for i, reg in enumerate(regions_to_plot):
                if i >= len(axes): break
                ax = axes[i]
                
                # --- Step 1: Spread ---
                spread_min, spread_max = [], []
                for m in months:
                    all_mems = []
                    for model in MODELS:
                        data = monthly_reg_res[scheme].get(model, {}).get(reg, {}).get(m, {})
                        if 'members' in data and metric in data['members']:
                             valid = [v for v in data['members'][metric] if np.isfinite(v)]
                             all_mems.extend(valid)
                    
                    if all_mems:
                        spread_min.append(np.min(all_mems))
                        spread_max.append(np.max(all_mems))
                    else:
                        spread_min.append(np.nan)
                        spread_max.append(np.nan)
                
                ax.fill_between(months, spread_min, spread_max, color='gray', alpha=0.2, 
                                label='Multi-model Member Spread' if i == 0 else "")
                
                # --- Step 2: Models ---
                for idx, model in enumerate(MODELS):
                    model_vals = []
                    for m in months:
                        val = monthly_reg_res[scheme].get(model, {}).get(reg, {}).get(m, {}).get(metric, np.nan)
                        model_vals.append(val)
                    
                    label = model.replace('-mon','').replace('Meteo-France','MF').replace('ECCC-Canada', 'ECCC-3') if i == 0 else ""
                    ax.plot(months, model_vals, marker='o', markersize=5, linewidth=2, 
                            color=cmap(idx % 10), label=label)
                
                # --- Step 3: MMM ---
                mmm_vals = [mmm_res[scheme][reg].get(m, {}).get(metric, np.nan) for m in months]
                ax.plot(months, mmm_vals, color='black', linewidth=3, label='MMM' if i == 0 else "")
                
                # --- Styling ---
                ax.set_title(reg, fontsize=16, fontweight='bold')
                ax.set_xticks(months); ax.set_xticklabels([str(m) for m in months])
                ax.grid(True, linestyle=':', alpha=0.6)
                if i >= 6: ax.set_xlabel('Month', fontsize=16)
                if i % 3 == 0: ax.set_ylabel(f'{title_metric} ({self.unit_label})', fontsize=16)
                ax.tick_params(labelsize=12)

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                       bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=14, frameon=False)
            
            plt.subplots_adjust(top=0.95, bottom=0.12, hspace=0.3, wspace=0.2)
            plt.savefig(self.plot_dir / f"monthly_{metric}_timeseries_{scheme}_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _process_spatial_task(self, model, scheme, season):
        obs, fcst = self.construct_seasonal_data(model, scheme, season)
        if obs is None: return None
        return (model, scheme, season, self.calculate_pointwise_metrics(obs, fcst))

    def _process_monthly_task(self, model, scheme):
        return (model, scheme, self.calculate_monthly_regional_metrics(model, scheme))

    def _load_cache(self):
        """从缓存加载计算结果"""
        if not self.cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.cache_file}. 请先不带 --plot-only 运行一次。")
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_cache(self, results_map, monthly_reg_res, mmm_res):
        """保存计算结果到缓存"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'results_map': results_map,
                'monthly_reg_res': monthly_reg_res,
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
                self.plot_monthly_timeseries(cache['monthly_reg_res'], cache['mmm_res'], metric, title)
            return

        logger.info("开始计算空间分布数据...")
        results_map = {}
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        tasks_spatial = [(model, scheme, season) for scheme in SCHEMES for season in seasons for model in models]
        
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_spatial) or 1)) as executor:
            for future in as_completed({executor.submit(self._process_spatial_task, *t): t for t in tasks_spatial}):
                if future.result(): results_map[(future.result()[1], future.result()[2], future.result()[0])] = future.result()[3]
        
        logger.info("开始计算月度区域数据...")
        monthly_reg_res = {s: {} for s in SCHEMES}
        tasks_monthly = [(model, scheme) for model in models for scheme in SCHEMES]
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_monthly))) as executor:
            for future in as_completed({executor.submit(self._process_monthly_task, *t): t for t in tasks_monthly}):
                if future.result(): monthly_reg_res[future.result()[1]][future.result()[0]] = future.result()[2]
        
        logger.info("计算 MMM...")
        mmm_res = {s: self.calculate_mmm_monthly_metrics(s) for s in SCHEMES}
        
        self._save_cache(results_map, monthly_reg_res, mmm_res)
        
        for metric, title in [('rmse', 'RMSE'), ('mae', 'MAE'), ('bias', 'Bias')]:
            self.plot_seasonal_spatial_maps(results_map, metric, title)
            self.plot_monthly_timeseries(monthly_reg_res, mmm_res, metric, title)

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节性预报方案 Pearson 相关系数分析模块 (V3 - Style Updated)
修改内容：
1. plot_monthly_timeseries 风格完全对齐 combined_pearson_analysis.py。
2. 引入 Multi-model Member Spread (灰色背景)。
3. 调整配色、标记、网格和图例布局。
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
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
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

# 月份映射关系
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
    log_file='seasonal_scheme_analysis.log',
    module_name=__name__
)

# === 统计函数 ===
def pearson_r_along_time_with_p(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 5: return np.nan, np.nan
    try:
        r, p = stats.pearsonr(x[mask], y[mask])
        return float(r), float(p)
    except: return np.nan, np.nan

class SeasonalSchemePearsonAnalyzerV3:
    """季节预报方案 ACC 分析器 (V3)"""
    
    def __init__(self, var_type: str, n_jobs: Optional[int] = None):
        self.var_type = var_type
        self.data_loader = DataLoader()
        self.n_jobs = n_jobs
        
        self.base_dir = Path("/sas12t1/ffyan/output/seasonal_scheme_analysis")
        self.map_dir = self.base_dir / f"acc_maps/{self.var_type}"
        self.plot_dir = self.base_dir / f"plots/{self.var_type}"
        self.cache_file = self.base_dir / "cache" / f"{self.var_type}.pkl"
        
        for d in [self.map_dir, self.plot_dir, self.cache_file.parent]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"

    def load_monthly_data(self, model: str, scheme: str, target_month: int):
        """加载特定模式、方案、目标月份的观测和预测数据"""
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
            
            # 计算距平
            obs_clim = obs_full.groupby('time.month').mean('time')
            obs_anom = obs_full.groupby('time.month') - obs_clim
            obs_m = obs_anom.sel(time=obs_anom.time.dt.month == target_month)
            
            fcst_clim = ds_lead.groupby('time.month').mean('time')
            fcst_anom = ds_lead.groupby('time.month') - fcst_clim
            fcst_m = fcst_anom.sel(time=fcst_anom.time.dt.month == target_month)
            
            # 对齐
            try:
                fcst_m = fcst_m.interp(lat=obs_m.lat, lon=obs_m.lon, method='linear')
            except: return None, None
            
            common_times = obs_m.time.to_index().intersection(fcst_m.time.to_index())
            if len(common_times) == 0: return None, None
            
            obs_m = obs_m.sel(time=common_times)
            fcst_m = fcst_m.sel(time=common_times)
            
            obs_m = obs_m.assign_coords(year=('time', obs_m.time.dt.year.values))
            fcst_m = fcst_m.assign_coords(year=('time', fcst_m.time.dt.year.values))
            obs_m = obs_m.swap_dims({'time': 'year'}).drop_vars('time')
            fcst_m = fcst_m.swap_dims({'time': 'year'}).drop_vars('time')
            
            return obs_m, fcst_m
        except Exception as e:
            logger.debug(f"加载月数据失败 {model} {scheme} {target_month}: {e}")
            return None, None

    def construct_seasonal_data(self, model: str, scheme: str, season: str):
        """构建季节距平场"""
        try:
            target_months = SEASONS[season]
            obs_list, fcst_list = [], []
            
            for m in target_months:
                o, f = self.load_monthly_data(model, scheme, m)
                if o is None or f is None: return None, None
                
                # 跨年处理
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

    def calculate_acc_spatial_map(self, obs: xr.DataArray, fcst: xr.DataArray) -> xr.Dataset:
        if 'number' in fcst.dims: fcst = fcst.mean(dim='number')
        res = xr.apply_ufunc(
            pearson_r_along_time_with_p, obs, fcst,
            input_core_dims=[['year'], ['year']], output_core_dims=[[], []],
            vectorize=True, dask='parallelized', output_dtypes=[float, float]
        )
        return xr.Dataset({'acc': res[0], 'p_value': res[1], 'significant': res[1] < 0.05})

    def calculate_monthly_regional_acc(self, model: str, scheme: str):
        """计算区域 ACC，包含 Ensemble Mean 和 Members"""
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
                obs_ts = obs_reg.weighted(w_reg).mean(['lat', 'lon'], skipna=True)
                fcst_ts = fcst_reg.weighted(w_reg).mean(['lat', 'lon'], skipna=True)
                
                mask = np.isfinite(obs_ts) & np.isfinite(fcst_ts)
                r_mean = stats.pearsonr(obs_ts[mask], fcst_ts[mask])[0] if mask.sum() > 3 else np.nan
                
                mem_rs = []
                if fcst_mem_reg is not None:
                    mem_ts_all = fcst_mem_reg.weighted(w_reg).mean(['lat', 'lon'], skipna=True)
                    for i in range(len(mem_ts_all.number)):
                        mts = mem_ts_all.isel(number=i)
                        m_mask = np.isfinite(obs_ts) & np.isfinite(mts)
                        if m_mask.sum() > 3: mem_rs.append(stats.pearsonr(obs_ts[m_mask], mts[m_mask])[0])
                
                results[reg_name][m] = {'acc': r_mean, 'members': mem_rs}
        return results

    def calculate_mmm_monthly_acc(self, scheme: str):
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
            mmm_anom = xr.concat([f.sel(year=final_years) for f in fcst_list], dim='model').mean(dim='model')
            weights = np.cos(np.deg2rad(obs_final.lat))
            
            for reg, bounds in REGIONS.items():
                if bounds:
                    lat_sl = slice(bounds['lat'][0], bounds['lat'][1]) if obs_final.lat[0] < obs_final.lat[-1] else slice(bounds['lat'][1], bounds['lat'][0])
                    lon_sl = slice(bounds['lon'][0], bounds['lon'][1])
                    o_r = obs_final.sel(lat=lat_sl, lon=lon_sl)
                    f_r = mmm_anom.sel(lat=lat_sl, lon=lon_sl)
                else:
                    o_r = obs_final; f_r = mmm_anom
                
                w_r = weights.sel(lat=o_r.lat)
                o_ts = o_r.weighted(w_r).mean(['lat', 'lon'], skipna=True)
                f_ts = f_r.weighted(w_r).mean(['lat', 'lon'], skipna=True)
                mask = np.isfinite(o_ts) & np.isfinite(f_ts)
                if mask.sum() > 3: results[reg][m] = stats.pearsonr(o_ts[mask], f_ts[mask])[0]
                else: results[reg][m] = np.nan
        return results

    def add_china_map_details(self, ax, data, levels, cmap, draw_scs=True):
        bou_paths = [self.boundaries_dir / "中国_省1.shp", self.boundaries_dir / "中国_省2.shp"]
        hyd_path = self.boundaries_dir / "河流.shp"
        
        # 河流
        if hyd_path.exists():
            try: ax.add_geometries(shpreader.Reader(str(hyd_path)).geometries(), ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6, zorder=5)
            except: pass
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', zorder=50)
        
        # 省界
        loaded_borders = False
        for p in bou_paths:
            if p.exists():
                try: 
                    ax.add_geometries(shpreader.Reader(str(p)).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6, zorder=100)
                    loaded_borders = True
                except: pass
        if not loaded_borders:
            ax.add_feature(cfeature.BORDERS, linewidth=1.0, zorder=100)
            
        # 南海子图
        if draw_scs:
            try:
                sub = ax.inset_axes([0.7548, 0, 0.33, 0.35], projection=ccrs.PlateCarree())
                sub.patch.set_facecolor('white')
                sub.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
                sub.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both')
                sub.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray', zorder=50)
                if loaded_borders:
                    for p in bou_paths:
                        if p.exists():
                            try: sub.add_geometries(shpreader.Reader(str(p)).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6, zorder=100)
                            except: pass
                sub.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                for spine in sub.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1.0)
            except: pass

    def _plot_single_map(self, fig, gs, row, col, model, scheme, maps, levels, cmap, xticks, yticks, char_label):
        """单张子图绘制辅助函数"""
        key = (scheme, model)
        if key not in maps:
            ax = fig.add_subplot(gs[row, col]); ax.axis('off'); return
        
        ds = maps[key]
        data = ds['acc']
        display_name = model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
        
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = FixedLocator(xticks); gl.ylocator = FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter(number_format='.0f')
        gl.yformatter = LatitudeFormatter(number_format='.0f')
        
        im = ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both')
        self.add_china_map_details(ax, data, levels, cmap, draw_scs=True)
        
        if 'significant' in ds and np.any(ds['significant']):
            X, Y = np.meshgrid(data.lon, data.lat)
            ax.scatter(X[ds['significant'].values][::2], Y[ds['significant'].values][::2], transform=ccrs.PlateCarree(), s=1, c='black', alpha=0.5, marker='.')
            
        ax.text(0.02, 0.96, f"({char_label}) {display_name}", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

    def plot_seasonal_spatial_maps(self, results_map):
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        plot_models = MODELS 
        
        lon_ticks = np.arange(75, 141, 15)
        lat_ticks = np.arange(20, 56, 10)
        vmin, vmax = -1, 1
        n_levels = 20
        levels = np.linspace(vmin, vmax, n_levels + 1)
        cmap = 'RdBu_r'

        for season in seasons:
            logger.info(f"绘制季节空间图: {season}")
            current_season_maps = {}
            for scheme in SCHEMES:
                for model in plot_models:
                    if (scheme, season, model) in results_map:
                        current_season_maps[(scheme, model)] = results_map[(scheme, season, model)]
            
            if not current_season_maps: continue

            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15, left=0.05, right=0.92, top=0.94, bottom=0.06)
            
            for s_idx, scheme in enumerate(SCHEMES):
                row_start = s_idx * 2
                ax_blank = fig.add_subplot(gs[row_start, 0]); ax_blank.axis('off')
                for col_idx in range(3):
                    if col_idx >= len(plot_models): continue
                    self._plot_single_map(fig, gs, row_start, col_idx + 1, 
                                          plot_models[col_idx], scheme, current_season_maps, 
                                          levels, cmap, lon_ticks, lat_ticks, chr(97 + col_idx))
                    if col_idx == 0:
                         ax = fig.axes[-1]
                         ax.text(0.98, 0.96, f'{scheme}', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(plot_models): continue
                    self._plot_single_map(fig, gs, row_start + 1, col_idx, 
                                          plot_models[model_idx], scheme, current_season_maps, 
                                          levels, cmap, lon_ticks, lat_ticks, chr(97 + model_idx))

            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(levels, 256), cmap=cmap), 
                                cax=cbar_ax, orientation='vertical', extend='both')
            cbar.set_label('Temporal ACC', fontsize=14, labelpad=10)
            
            plt.savefig(self.plot_dir / f"acc_map_{season}_{self.var_type}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    def plot_monthly_timeseries(self, monthly_reg_res, mmm_res):
        """
        绘制区域月度 ACC 折线图
        风格更新：参考 regional_index_acc_leadtime_timeseries_temp.jpg
        1. 灰色背景为 Multi-model Member Spread (所有模式所有成员的极值)
        2. 字体加粗、字号加大、Tab10 颜色、增加 Marker
        """
        regions_order = ['Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast', 
                         'Z4-Tibetan', 'Z5-NorthChina', 'Z6-Yangtze', 
                         'Z7-Southwest', 'Z8-SouthChina', 'Z9-SouthSea']
        
        # 确保只绘制存在的区域
        regions_to_plot = [r for r in regions_order if r in REGIONS]
        
        for scheme in SCHEMES:
            logger.info(f"绘制折线图 (Style V2): {scheme}")
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            axes = axes.flatten()
            months = np.arange(1, 13)
            cmap = plt.get_cmap('tab10')

            for i, reg in enumerate(regions_to_plot):
                if i >= len(axes): break
                ax = axes[i]
                
                # --- Step 1: 计算 Multi-model Member Spread (所有模式成员的 Min/Max) ---
                spread_min = []
                spread_max = []
                
                for m in months:
                    all_members_in_month = []
                    # 遍历所有模型，收集该月的所有成员ACC
                    for model in MODELS:
                        data = monthly_reg_res[scheme].get(model, {}).get(reg, {}).get(m, {})
                        if 'members' in data and data['members']:
                            # 过滤 nan
                            valid_mems = [v for v in data['members'] if np.isfinite(v)]
                            all_members_in_month.extend(valid_mems)
                    
                    if all_members_in_month:
                        spread_min.append(np.min(all_members_in_month))
                        spread_max.append(np.max(all_members_in_month))
                    else:
                        spread_min.append(np.nan)
                        spread_max.append(np.nan)
                
                # 绘制灰色背景 Spread
                # 仅在第一个子图添加 label，用于图例生成
                ax.fill_between(months, spread_min, spread_max, color='gray', alpha=0.2, 
                                label='Multi-model Member Spread' if i == 0 else "")

                # --- Step 2: 绘制各个 Model 的 Ensemble Mean ---
                for idx, model in enumerate(MODELS):
                    model_accs = []
                    for m in months:
                        val = monthly_reg_res[scheme].get(model, {}).get(reg, {}).get(m, {}).get('acc', np.nan)
                        model_accs.append(val)
                    
                    display_name = model.replace('-mon','').replace('Meteo-France','MF').replace('ECCC-Canada', 'ECCC-3')
                    # 仅在第一个子图添加 label
                    label = display_name if i == 0 else ""
                    
                    ax.plot(months, model_accs, marker='o', markersize=5, linewidth=2, 
                            color=cmap(idx % 10), label=label)

                # --- Step 3: 绘制 MMM (黑色粗线) ---
                mmm_vals = [mmm_res[scheme][reg].get(m, np.nan) for m in months]
                ax.plot(months, mmm_vals, color='black', linewidth=3, 
                        label='MMM' if i == 0 else "")
                
                # --- Step 4: 样式修饰 ---
                ax.set_title(reg, fontsize=16, fontweight='bold')
                ax.set_xticks(months)
                ax.set_xticklabels([str(m) for m in months])
                # 统一 Y 轴范围，或者根据数据自适应但保持对齐 (此处参考原图范围)
                # ax.set_ylim(-0.4, 0.8) 
                ax.grid(True, linestyle=':', alpha=0.6)
                
                if i >= 6: 
                    ax.set_xlabel('Month', fontsize=16)
                if i % 3 == 0: 
                    ax.set_ylabel('ACC', fontsize=16)
                ax.tick_params(labelsize=12)

            # --- Step 5: 全局图例 ---
            handles, labels = axes[0].get_legend_handles_labels()
            # 简单去重 (虽然上面逻辑已经控制了label只在i==0生成，但防万一)
            by_label = dict(zip(labels, handles))
            
            # 图例放置在底部
            fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                       bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=14, frameon=False)
            
            plt.subplots_adjust(top=0.95, bottom=0.12, hspace=0.3, wspace=0.2)
            plt.savefig(self.plot_dir / f"monthly_acc_timeseries_{scheme}_{self.var_type}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _process_spatial_task(self, model, scheme, season):
        obs, fcst = self.construct_seasonal_data(model, scheme, season)
        if obs is None: return None
        return (model, scheme, season, self.calculate_acc_spatial_map(obs, fcst))

    def _process_monthly_task(self, model, scheme):
        return (model, scheme, self.calculate_monthly_regional_acc(model, scheme))

    def _load_cache(self):
        """从缓存加载计算结果，供 --plot-only 使用"""
        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"缓存文件不存在: {self.cache_file}\n"
                "请先完整运行一次分析以生成缓存，再使用 --plot-only"
            )
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
            self.plot_seasonal_spatial_maps(cache['results_map'])
            self.plot_monthly_timeseries(cache['monthly_reg_res'], cache['mmm_res'])
            return

        logger.info("开始计算空间分布数据 (四季)...")
        results_map = {}
        seasons = ['DJF', 'MAM', 'JJA', 'SON'] 
        tasks_spatial = [(model, scheme, season) for scheme in SCHEMES for season in seasons for model in models]
        
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_spatial) or 1)) as executor:
            for future in as_completed({executor.submit(self._process_spatial_task, *t): t for t in tasks_spatial}):
                if future.result(): results_map[(future.result()[1], future.result()[2], future.result()[0])] = future.result()[3]
        
        self.plot_seasonal_spatial_maps(results_map)
        
        logger.info("开始计算月度区域数据...")
        monthly_reg_res = {s: {} for s in SCHEMES}
        tasks_monthly = [(model, scheme) for model in models for scheme in SCHEMES]
        with ProcessPoolExecutor(max_workers=min(n_jobs or 16, len(tasks_monthly))) as executor:
            for future in as_completed({executor.submit(self._process_monthly_task, *t): t for t in tasks_monthly}):
                if future.result(): monthly_reg_res[future.result()[1]][future.result()[0]] = future.result()[2]
        
        logger.info("计算 MMM...")
        mmm_res = {s: self.calculate_mmm_monthly_acc(s) for s in SCHEMES}
        self._save_cache(results_map, monthly_reg_res, mmm_res)
        self.plot_monthly_timeseries(monthly_reg_res, mmm_res)

def main():
    parser = create_parser(description="季节预报方案 ACC 分析", var_default=None, var_required=False)
    args = parser.parse_args()
    models = parse_models(args.models, MODELS) if args.models else MODELS
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    plot_only = getattr(args, 'plot_only', False)
    for var_type in var_list:
        analyzer = SeasonalSchemePearsonAnalyzerV3(var_type, n_jobs=args.n_jobs)
        analyzer.run_analysis(models=models, parallel=normalize_parallel_args(args), n_jobs=args.n_jobs, plot_only=plot_only)

if __name__ == "__main__":
    main()
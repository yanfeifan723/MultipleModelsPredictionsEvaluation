#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模式误差分析计算模块 (RMSE, MAE, Bias)
绘制三种指标的空间分布图与折线图，包含海陆掩模处理与特定配色方案。

功能特性:
1. 计算并绘制三种指标:
   - RMSE (均方根误差): 反映总误差幅度
   - MAE (平均绝对误差): 反映平均误差大小
   - Bias (偏差): 反映系统性冷暖/干湿倾向
2. 空间分布图配色:
   - RMSE/MAE (Temp): 白->红 (MPL_Reds)
   - RMSE/MAE (Prec): 白->蓝 (MPL_Blues)
   - Bias (Temp): 蓝(冷)->白->红(暖) (RdBu_r)
   - Bias (Prec): 红(干)->白->蓝(湿) (RdBu)
3. 强制应用海陆掩模 (Land-Sea Mask)，仅保留陆地数据
4. 折线图: 包含轴标签和单位

运行环境要求:
- 需要安装 regionmask: pip install regionmask
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
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    SPATIAL_BOUNDS,
)

# 配置参数
MODELS = MODEL_LIST

# === 定义区域 ===
def generate_regions():
    regions = {'Global': None}
    regions.update({
        'Z1-Northwest':     {'lat': (39, 49), 'lon': (73, 105)},
        'Z2-InnerMongolia': {'lat': (39, 50), 'lon': (106, 118)},
        'Z3-Northeast':     {'lat': (40, 54), 'lon': (119, 135)},
        'Z4-Tibetan':       {'lat': (27, 39), 'lon': (73, 95)},
        'Z5-NorthChina':    {'lat': (34, 39), 'lon': (106, 122)},
        'Z7-Yangtze':       {'lat': (26, 34), 'lon': (109, 123)},
        'Z6-Southwest':     {'lat': (23, 33), 'lon': (96, 108)},
        'Z8-SouthChina':    {'lat': (21, 25), 'lon': (106, 120)},
        'Z9-SouthSea':      {'lat': (18, 21), 'lon': (105, 125)}
    })
    return regions

REGIONS = generate_regions()

# 配置日志
logger = setup_logging(
    log_file='combined_error_analysis.log',
    module_name=__name__
)

class MultiModelErrorAnalyzer:
    """多模式误差分析器 (RMSE, MAE, Bias)"""
    
    def __init__(self, var_type: str, n_jobs: Optional[int] = None):
        self.var_type = var_type
        self.data_loader = DataLoader()
        self.n_jobs = n_jobs
        
        # 单位定义
        self.unit_label = "°C" if var_type == 'temp' else "mm/day"
        
        # 输出路径
        base_output = Path(f"/sas12t1/ffyan/output/error_analysis")
        self.spatial_map_dir = base_output / f"spatial_maps/{self.var_type}"
        self.region_metric_dir = base_output / f"region_metrics/{self.var_type}"
        self.plot_dir = base_output / f"plots/{self.var_type}"
        
        for p in [self.spatial_map_dir, self.region_metric_dir, self.plot_dir]:
            p.mkdir(parents=True, exist_ok=True)
            
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"

    def convert_temp_units(self, ds: xr.DataArray) -> xr.DataArray:
        """仅对温度进行单位换算 (K -> C)"""
        if self.var_type == 'temp':
            # 如果均值 > 200，认为是Kelvin，转换为Celsius
            if ds.mean(skipna=True) > 200:
                ds = ds - 273.15
        return ds

    def apply_land_mask(self, ds: xr.DataArray) -> xr.DataArray:
        """
        使用 regionmask 应用海陆掩模 (保留陆地，去除海洋)
        """
        try:
            # 使用 Natural Earth 的陆地掩模 (land_110)
            # mask返回: 0=陆地, NaN=海洋 (取决于版本)
            land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds)
            # 保留 mask == 0 的部分 (陆地)
            ds_masked = ds.where(land_mask == 0)
            return ds_masked
        except Exception as e:
            logger.warning(f"海陆掩模应用失败 (请确保安装 regionmask): {e}")
            return ds

    def load_and_preprocess_data(self, model: str, leadtime: int) -> Tuple[xr.DataArray, xr.DataArray]:
        """加载数据、处理单位、应用掩模"""
        try:
            # 1. 加载观测
            # DataLoader 会自动处理降水单位 (mm/day)
            obs_data = self.data_loader.load_obs_data(self.var_type)
            obs_data = obs_data.resample(time='1MS').mean()
            obs_data = obs_data.sel(time=slice('1993', '2020'))
            
            # 2. 加载模式
            # DataLoader 会自动处理降水单位 (mm/day)
            fcst_data = self.data_loader.load_forecast_data(model, self.var_type, leadtime)
            if fcst_data is None:
                return None, None
            fcst_data = fcst_data.resample(time='1MS').mean()
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))
            
            # 3. 时间对齐
            common_times = obs_data.time.to_index().intersection(fcst_data.time.to_index())
            if len(common_times) < 12:
                return None, None
            
            obs_aligned = obs_data.sel(time=common_times)
            fcst_aligned = fcst_data.sel(time=common_times)
            
            # 4. 温度单位转换 (K -> C)
            # 注意：降水不需要在这里转换，因为 DataLoader 已经转了
            obs_conv = self.convert_temp_units(obs_aligned)
            fcst_conv = self.convert_temp_units(fcst_aligned)
            
            # 5. 空间插值
            fcst_interpolated = fcst_conv.interp(
                lat=obs_conv.lat, lon=obs_conv.lon, method='linear'
            )
            
            # 6. === 应用海陆掩模 ===
            # 计算前剔除海洋
            obs_masked = self.apply_land_mask(obs_conv)
            fcst_masked = self.apply_land_mask(fcst_interpolated)

            return obs_masked, fcst_masked
            
        except Exception as e:
            logger.error(f"数据处理失败 {model} L{leadtime}: {e}")
            return None, None

    def calculate_pointwise_metrics(self, obs: xr.DataArray, fcst: xr.DataArray) -> xr.Dataset:
        """计算逐格点指标"""
        try:
            diff = fcst - obs
            
            # RMSE
            rmse_map = np.sqrt((diff ** 2).mean(dim='time', skipna=True))
            # Bias (偏差)
            bias_map = diff.mean(dim='time', skipna=True)
            # MAE (平均绝对误差)
            mae_map = np.abs(diff).mean(dim='time', skipna=True)
            
            ds = xr.Dataset({'rmse': rmse_map, 'bias': bias_map, 'mae': mae_map})
            return ds
        except Exception as e:
            logger.error(f"格点计算失败: {e}")
            return None

    def calculate_regional_metrics(self, obs: xr.DataArray, fcst: xr.DataArray,
                                   region_bounds: Optional[Dict]) -> xr.Dataset:
        """计算区域的月度、季节和年度指标 (RMSE, MAE, Bias)"""
        try:
            obs_reg = obs
            fcst_reg = fcst

            # 1. 区域截取
            if region_bounds is not None:
                lat_b = region_bounds['lat']
                lon_b = region_bounds['lon']
                lat_slice = slice(lat_b[0], lat_b[1]) if obs_reg.lat[0] < obs_reg.lat[-1] else slice(lat_b[1], lat_b[0])
                obs_reg = obs_reg.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
                fcst_reg = fcst_reg.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))

            # 2. 计算空间加权平均的时间序列
            weights = np.cos(np.deg2rad(obs_reg.lat))
            weights.name = "weights"

            diff = fcst_reg - obs_reg
            diff_sq = diff ** 2
            diff_abs = np.abs(diff)

            # 空间平均 -> 得到时间序列
            mse_ts = diff_sq.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            mae_ts = diff_abs.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
            bias_ts = diff.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)

            # 3. 按月份聚合 (1-12月)
            rmse_mon = np.sqrt(mse_ts.groupby('time.month').mean('time'))
            mae_mon = mae_ts.groupby('time.month').mean('time')
            bias_mon = bias_ts.groupby('time.month').mean('time')

            # 4. 按季节聚合
            rmse_seas = np.sqrt(mse_ts.groupby('time.season').mean('time'))

            # 5. 年度聚合
            rmse_annual = np.sqrt(mse_ts.mean('time'))
            mae_annual = mae_ts.mean('time')
            bias_annual = bias_ts.mean('time')

            # 标量 (供原有折线图 plot_line_metrics 使用)
            rmse_val = float(rmse_annual.values) if hasattr(rmse_annual, 'values') else float(rmse_annual)
            mae_val = float(mae_annual.values) if hasattr(mae_annual, 'values') else float(mae_annual)
            bias_val = float(bias_annual.values) if hasattr(bias_annual, 'values') else float(bias_annual)

            ds = xr.Dataset({
                'rmse_monthly': rmse_mon,
                'mae_monthly': mae_mon,
                'bias_monthly': bias_mon,
                'rmse_seasonal': rmse_seas,
                'rmse_annual': rmse_annual,
                'mae_annual': mae_annual,
                'bias_annual': bias_annual,
                'rmse': rmse_val,
                'mae': mae_val,
                'bias': bias_val,
            })
            return ds
        except Exception as e:
            logger.error(f"区域计算失败: {e}")
            return None

    def save_spatial_maps(self, model_metric_maps: Dict, models: List[str], leadtimes: List[int]):
        """保存空间图数据，供 --plot-only 时加载"""
        for model in models:
            for lt in leadtimes:
                if model not in model_metric_maps or lt not in model_metric_maps[model]:
                    continue
                fpath = self.spatial_map_dir / f"{model}_L{lt}.nc"
                try:
                    model_metric_maps[model][lt].to_netcdf(fpath)
                except Exception as e:
                    logger.warning(f"保存空间图失败 {fpath}: {e}")

    def load_spatial_maps(self, models: List[str], leadtimes: List[int]) -> Dict:
        """从磁盘加载空间图数据"""
        out = {m: {} for m in models}
        for model in models:
            for lt in leadtimes:
                fpath = self.spatial_map_dir / f"{model}_L{lt}.nc"
                if not fpath.exists():
                    logger.warning(f"未找到 {fpath}，跳过")
                    continue
                try:
                    out[model][lt] = xr.open_dataset(fpath)
                except Exception as e:
                    logger.warning(f"加载失败 {fpath}: {e}")
        return out

    def save_region_metrics(self, region_metric_data: Dict, models: List[str]):
        """保存区域指标数据，供 --plot-only 时加载"""
        for r_name in REGIONS:
            if r_name not in region_metric_data:
                continue
            for model in models:
                ds_list = region_metric_data[r_name].get(model, [])
                if not ds_list:
                    continue
                try:
                    combined = xr.concat(ds_list, dim='leadtime').sortby('leadtime')
                    fpath = self.region_metric_dir / f"{r_name}_{model}.nc"
                    combined.to_netcdf(fpath)
                except Exception as e:
                    logger.warning(f"保存区域指标失败 {r_name} {model}: {e}")

    def load_region_metrics(self, models: List[str]) -> Dict:
        """从磁盘加载区域指标数据"""
        out = {r: {m: [] for m in models} for r in REGIONS}
        for r_name in REGIONS:
            for model in models:
                fpath = self.region_metric_dir / f"{r_name}_{model}.nc"
                if not fpath.exists():
                    continue
                try:
                    ds = xr.open_dataset(fpath)
                    for lt in ds.leadtime.values:
                        out[r_name][model].append(ds.sel(leadtime=lt))
                except Exception as e:
                    logger.warning(f"加载区域指标失败 {fpath}: {e}")
        return out

    def get_plotting_params(self, metric: str):
        """
        获取绘图参数 (Colormap, Levels, Norm, Extend)
        """
        n_bins = 20
        
        # 1. RMSE 和 MAE (非负值)
        if metric in ['rmse', 'mae']:
            # 温度: White -> Red
            # 降水: White -> Blue
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
            
        # 2. Bias (有正负)
        elif metric == 'bias':
            if self.var_type == 'temp':
                # 温度: <0 蓝 (冷), 0 白, >0 红 (暖)
                cmap = plt.get_cmap('RdBu_r', n_bins) 
                levels = np.linspace(-3, 3, n_bins + 1)
            else:
                # 降水: <0 红 (干), 0 白, >0 蓝 (湿)
                cmap = plt.get_cmap('RdBu', n_bins) 
                levels = np.linspace(-3, 3, n_bins + 1)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            extend = 'both'
            
        return cmap, levels, norm, extend

    def add_china_map_details(self, ax, data, lon, lat, levels, cmap, norm, extend):
        """添加中国地图细节"""
        bou_paths = [
            Path("/sas12t1/ffyan/boundaries/中国_省1.shp"),
            Path("/sas12t1/ffyan/boundaries/中国_省2.shp")
        ]
        hyd_path = self.boundaries_dir / "河流.shp"
        
        # 河流
        if hyd_path.exists():
            try:
                reader = shpreader.Reader(str(hyd_path))
                ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
                                edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6)
            except: pass
        else:
            ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6)
            
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        
        # 国界
        loaded = False
        for p in bou_paths:
            if p.exists():
                try:
                    reader = shpreader.Reader(str(p))
                    ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
                                    edgecolor='black', facecolor='none', linewidth=0.6)
                    loaded = True
                except: pass
        if not loaded:
            ax.add_feature(cfeature.BORDERS, linewidth=1.0)
            
        # 南海子图
        try:
            sub_ax = ax.inset_axes([0.7548, 0, 0.33, 0.35], projection=ccrs.PlateCarree())
            sub_ax.patch.set_facecolor('white')
            sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
            sub_ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(),
                           cmap=cmap, levels=levels, norm=norm, extend=extend)
            sub_ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray')
            if loaded:
                for p in bou_paths:
                    if p.exists():
                        reader = shpreader.Reader(str(p))
                        sub_ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
                                            edgecolor='black', facecolor='none', linewidth=0.6)
            sub_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in sub_ax.spines.values():
                spine.set_edgecolor('black')
        except Exception as e:
            logger.warning(f"南海子图绘制失败: {e}")

    def plot_metric_spatial_maps(self, model_metric_maps: Dict, metric_key: str, title_metric: str):
        """绘制空间分布图"""
        try:
            plot_models = list(model_metric_maps.keys())
            leadtimes = [0, 3]
            
            cmap, levels, norm, extend = self.get_plotting_params(metric_key)
            
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15,
                         left=0.05, right=0.92, top=0.94, bottom=0.06)
            
            lon_ticks = np.arange(75, 141, 15)
            lat_ticks = np.arange(20, 56, 10)

            for lt_idx, leadtime in enumerate(leadtimes):
                row_start = lt_idx * 2
                ax_blank = fig.add_subplot(gs[row_start, 0]); ax_blank.axis('off')
                
                # 第一行
                for col_idx in range(3):
                    if col_idx >= len(plot_models): break
                    self._plot_single_map(fig, gs, row_start, col_idx + 1, plot_models[col_idx], 
                                        leadtime, model_metric_maps, metric_key, cmap, levels, norm, extend,
                                        lon_ticks, lat_ticks, chr(97 + col_idx))
                    if col_idx == 0:
                        ax = fig.axes[-1]
                        ax.text(0.98, 0.96, f'L{leadtime}', transform=ax.transAxes, 
                               fontsize=22, fontweight='bold', va='top', ha='right')
                # 第二行
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(plot_models): break
                    self._plot_single_map(fig, gs, row_start + 1, col_idx, plot_models[model_idx], 
                                        leadtime, model_metric_maps, metric_key, cmap, levels, norm, extend,
                                        lon_ticks, lat_ticks, chr(97 + model_idx))

            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, 
                               ticks=levels[::2], extend=extend)
            cbar.set_label(f'{title_metric} ({self.unit_label})', fontsize=18)
            
            output_file = self.plot_dir / f"{metric_key}_spatial_maps_L0_L3_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"{title_metric} 空间图已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"绘图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_single_map(self, fig, gs, row, col, model, leadtime, maps, metric, cmap, levels, norm, extend, xticks, yticks, label_char):
        if leadtime not in maps[model]:
            ax = fig.add_subplot(gs[row, col]); ax.axis('off'); return

        ds = maps[model][leadtime]
        if metric not in ds: return
        data = ds[metric]

        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
        
        # 仅绘制陆地轮廓（海洋数据为NaN透明）
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = FixedLocator(xticks); gl.ylocator = FixedLocator(yticks)
        gl.xformatter = LongitudeFormatter(number_format='.0f')
        gl.yformatter = LatitudeFormatter(number_format='.0f')
        
        im = ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, levels=levels, norm=norm, extend=extend)
        
        self.add_china_map_details(ax, data, data.lon, data.lat, levels, cmap, norm, extend)
        
        display_name = model.replace('-mon', '').replace('mon-', '')
        ax.text(0.02, 0.96, f"({label_char}) {display_name}", transform=ax.transAxes,
               fontsize=18, fontweight='bold', va='top')

    def plot_line_metrics(self, region_metric_data: Dict, metric: str, is_global: bool = False):
        """绘制折线图"""
        try:
            if is_global:
                regions = ['Global']
                fig = plt.figure(figsize=(10, 6))
                axes = {'Global': plt.gca()}
                fname = f"global_{metric}_timeseries_{self.var_type}.png"
                title_suffix = ""
            else:
                region_order = [
                    'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
                    'Z4-Tibetan',   'Z5-NorthChina',    'Z7-Yangtze',
                    'Z6-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
                ]
                regions = [r for r in region_order if r in region_metric_data]
                if not regions: return
                fig = plt.figure(figsize=(18, 15))
                axes = {}
                subplot_map = {
                    'Z1-Northwest': (0,0), 'Z2-InnerMongolia': (0,1), 'Z3-Northeast': (0,2),
                    'Z4-Tibetan': (1,0), 'Z5-NorthChina': (1,1), 'Z7-Yangtze': (1,2),
                    'Z6-Southwest': (2,0), 'Z8-SouthChina': (2,1), 'Z9-SouthSea': (2,2)
                }
                for r in regions:
                    row, col = subplot_map[r]
                    axes[r] = plt.subplot(3, 3, row*3 + col + 1)
                fname = f"regional_{metric}_timeseries_{self.var_type}.png"

            cmap = plt.get_cmap('tab10')
            all_vals = []
            
            for reg in regions:
                if reg not in region_metric_data: continue
                ax = axes[reg]
                models_data = region_metric_data[reg]
                
                for i, (model, ds_list) in enumerate(models_data.items()):
                    if not ds_list: continue
                    combined = xr.concat(ds_list, dim='leadtime').sortby('leadtime')
                    if metric not in combined: continue
                    y_vals = combined[metric].values
                    x_vals = combined.leadtime.values
                    all_vals.extend(y_vals)
                    
                    label = model.replace('-mon', '') if (is_global or reg == regions[0]) else ""
                    ax.plot(x_vals, y_vals, marker='o', label=label, color=cmap(i % 10))
                
                if not is_global: ax.set_title(reg, fontweight='bold')
                ax.grid(True, linestyle=':')
                ax.set_xticks(LEADTIMES)
                ax.set_xlabel('Lead Time (months)', fontsize=16)
                ax.set_ylabel(f'{metric.upper()} ({self.unit_label})', fontsize=16)

            # 统一Y轴
            if all_vals:
                ymin, ymax = min(all_vals), max(all_vals)
                margin = (ymax - ymin) * 0.1
                for ax in axes.values():
                    lower = 0 if metric != 'bias' else ymin - margin
                    ax.set_ylim(lower, ymax + margin)

            # Legend
            handles, labels = list(axes.values())[0].get_legend_handles_labels()
            if handles:
                if is_global:
                    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=13, 
                              bbox_to_anchor=(0.5, -0.05))
                else:
                    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18, 
                              bbox_to_anchor=(0.5, -0.05))

            plt.subplots_adjust(top=0.95, bottom=0.15 if is_global else 0.1, hspace=0.3, wspace=0.2)
            plt.savefig(self.plot_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"{fname} 已保存")

        except Exception as e:
            logger.error(f"折线图绘制失败: {e}")

    def run_analysis(self, models=None, leadtimes=None, parallel=False, n_jobs=None, plot_only=False):
        models = models or MODELS
        leadtimes = leadtimes or LEADTIMES
        
        if plot_only:
            logger.info(f"仅绘图模式: {self.var_type}")
            model_metric_maps = self.load_spatial_maps(models, leadtimes)
            region_metric_data = self.load_region_metrics(models)
            if not any(model_metric_maps[m] for m in models):
                logger.warning("未找到已保存的空间图数据，请先运行完整分析")
                return
            self._do_plots(model_metric_maps, region_metric_data)
            logger.info("仅绘图完成")
            return

        model_metric_maps = {m: {} for m in models}
        region_metric_data = {r: {m: [] for m in models} for r in REGIONS}
        
        tasks = [(self.var_type, m, lt) for lt in leadtimes for m in models]
        
        logger.info(f"开始误差分析 (RMSE/MAE/Bias, 含掩模): {self.var_type}")
        
        if parallel:
            max_workers = min(n_jobs or max(1, cpu_count() // 2), 32)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(_compute_errors_task, *t): t for t in tasks}
                for future in as_completed(future_to_task):
                    try:
                        res = future.result()
                        if res:
                            lt, mod, map_ds, reg_dict = res
                            model_metric_maps[mod][lt] = map_ds
                            for r, ds in reg_dict.items():
                                region_metric_data[r][mod].append(ds)
                    except Exception as e:
                        logger.error(f"Task error: {e}")
        else:
            for t in tasks:
                res = _compute_errors_task(*t)
                if res:
                    lt, mod, map_ds, reg_dict = res
                    model_metric_maps[mod][lt] = map_ds
                    for r, ds in reg_dict.items():
                        region_metric_data[r][mod].append(ds)

        self.save_spatial_maps(model_metric_maps, models, leadtimes)
        self.save_region_metrics(region_metric_data, models)
        self._do_plots(model_metric_maps, region_metric_data)
        logger.info("所有分析完成")

    def _do_plots(self, model_metric_maps: Dict, region_metric_data: Dict):
        """执行所有绘图（空间图 + 折线图）"""
        self.plot_metric_spatial_maps(model_metric_maps, 'rmse', 'RMSE')
        self.plot_metric_spatial_maps(model_metric_maps, 'mae', 'MAE')
        self.plot_metric_spatial_maps(model_metric_maps, 'bias', 'Bias (Mean Error)')
        for m in ['rmse', 'mae', 'bias']:
            self.plot_line_metrics(region_metric_data, m, is_global=True)
            self.plot_line_metrics(region_metric_data, m, is_global=False)

def _compute_errors_task(var_type, model, leadtime):
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
        from src.utils.data_loader import DataLoader # noqa
        
        analyzer = MultiModelErrorAnalyzer(var_type)
        obs, fcst = analyzer.load_and_preprocess_data(model, leadtime)
        if obs is None or fcst is None: return None
            
        map_ds = analyzer.calculate_pointwise_metrics(obs, fcst)
        reg_dict = {}
        for r_name, r_bounds in REGIONS.items():
            reg_ds = analyzer.calculate_regional_metrics(obs, fcst, r_bounds)
            if reg_ds:
                reg_ds = reg_ds.expand_dims(leadtime=[leadtime])
                reg_dict[r_name] = reg_ds
        return leadtime, model, map_ds, reg_dict
    except Exception:
        return None

def main():
    parser = create_parser(description="多模式误差分析 (RMSE/MAE/Bias, 含掩模与特定绘图)")
    args = parser.parse_args()
    models = parse_models(args.models, MODELS) if args.models else MODELS
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    parallel = normalize_parallel_args(args)
    
    plot_only = getattr(args, 'plot_only', False)
    for var in var_list:
        analyzer = MultiModelErrorAnalyzer(var, n_jobs=args.n_jobs)
        analyzer.run_analysis(models, leadtimes, parallel=parallel, n_jobs=args.n_jobs, plot_only=plot_only)

if __name__ == "__main__":
    main()

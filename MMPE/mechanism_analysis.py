#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Mechanism Analysis: ENSO -> EAWM/WNPAC -> China Climate
功能特性:
1. 物理机制自动分配：Temp(冬季同期机制) / Prec(夏季跨季机制)。
2. 【EOF 降维提取 + 物理锚点】：使用宽泛区域的 EOF1 代替固定 Box，并用经典物理指数锁定相位，完美规避模式空间漂移与相位翻转。
3. 【显著性打点】：引入 Sobel 检验，对总效应、独立效应和中介效应进行 95% 显著性打点。
4. 年际序列回归：每年提取一个季节均值，物理意义极其明确。
5. 完美对接 index_analysis.py 的输出缓存。
6. 【极致复刻绘图】：完美 4x4 GridSpec 布局，包含 ERA5 及高精度地图要素。
"""

import sys
import os
import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from typing import Dict, List, Optional, Tuple

# === 统一导入 toolkit 路径 ===
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args
from common_config import MODEL_LIST

# 配置日志
logger = setup_logging(
    log_file='climate_mechanism_analysis.log',
    module_name=__name__
)

# ==========================================
# 1. 核心数学模块 (中介效应与 Sobel 检验)
# ==========================================

def process_pixel(args):
    """单个空间格点的计算流：计算效应值及 P 值 (Sobel Test)"""
    y_pixel, X_matrix, enso, bridge = args
    
    valid = ~(np.isnan(y_pixel) | np.isnan(X_matrix).any(axis=1))
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    y_v, X_v = y_pixel[valid], X_matrix[valid, :]
    enso_v, bridge_v = enso[valid], bridge[valid]
    
    try:
        # 1. ENSO 的总效应 (Total Effect, c)
        corr_enso, p_corr_enso = stats.pearsonr(enso_v, y_v)
        
        # 2. 多元线性回归 (控制彼此的影响)
        # X_matrix: [Intercept, ENSO(原始), Bridge(原始)]
        model_fit = sm.OLS(y_v, X_v).fit()
        coef_enso_direct = model_fit.params[1]  # ENSO 的直接效应 (c')
        coef_bridge_indep = model_fit.params[2] # 桥梁的独立效应 (b)
        p_coef_bridge = model_fit.pvalues[2]    # 桥梁独立效应的 P 值
        
        # 3. 中介效应 / 桥梁传递效应 (Indirect Effect)
        indirect_effect = corr_enso - coef_enso_direct
        
        # 4. Sobel 检验 (计算中介效应的显著性)
        # 路径 a: ENSO -> Bridge
        model_a = sm.OLS(bridge_v, sm.add_constant(enso_v)).fit()
        a = model_a.params[1]
        se_a = model_a.bse[1]
        
        # 路径 b: Bridge -> Y (控制 ENSO)
        b = coef_bridge_indep
        se_b = model_fit.bse[2]
        
        # Sobel Z 统计量
        sobel_z = (a * b) / np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
        p_indirect = stats.norm.sf(np.abs(sobel_z)) * 2  # 双侧 P 值
        
        return corr_enso, coef_bridge_indep, indirect_effect, p_corr_enso, p_coef_bridge, p_indirect
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# ==========================================
# 2. 顶层并行 Worker 函数
# ==========================================

def _mechanism_worker_task(var_type: str, model: str, leadtime: int):
    try:
        analyzer = ClimateMechanismAnalyzer(var_type=var_type)
        enso, bridge, y_grid, bridge_name = analyzer.load_and_preprocess_data(model, leadtime)
        ds_out = analyzer.compute_mechanisms(model, leadtime, enso, bridge, y_grid, bridge_name, n_jobs=1)
        return model, leadtime, ds_out
    except Exception as e:
        import traceback
        logger.error(f"[{model} L{leadtime}] 任务失败: {e}\n{traceback.format_exc()}")
        return model, leadtime, None

# ==========================================
# 3. 机制分析主类
# ==========================================

class ClimateMechanismAnalyzer:
    
    def __init__(self, var_type: str):
        self.var_type = var_type
        self.data_loader = DataLoader()
        
        self.base_dir = Path(f"/sas12t1/ffyan/output/mechanism_analysis/{self.var_type}")
        self.nc_dir = self.base_dir / "nc_results"
        self.plot_dir = self.base_dir / "plots"
        self.cache_file = self.base_dir / "cache" / "mechanism_cache.pkl"
        self.boundaries_dir = Path(__file__).parent.parent / "boundaries"
        self.index_cache_file = Path("/sas12t1/ffyan/output/index_analysis/cache/index_analysis_cache.pkl")
        
        for d in[self.nc_dir, self.plot_dir, self.cache_file.parent]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_cache(self):
        if not self.cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.cache_file}。请先关闭 --plot-only 运行一次进行计算。")
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_cache(self, results_dict):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(results_dict, f)
        logger.info(f"已保存计算缓存: {self.cache_file}")

    def get_index_from_cache(self, index_name: str, model: str, leadtime: int) -> xr.DataArray:
        if not self.index_cache_file.exists():
            raise FileNotFoundError("找不到指数缓存，请先运行 index_analysis.py")
            
        with open(self.index_cache_file, 'rb') as f:
            cache = pickle.load(f)
            
        data_dict = cache.get('nino34_indices', {})
        key = 'ERA5' if model in['ERA5', 'Obs'] else f"{model}_L{leadtime}"
        if key not in data_dict:
            raise KeyError(f"在 {index_name} 缓存中找不到 {key}")
        return data_dict[key]

    def area_weighted_mean(self, da: xr.DataArray, lat_name: str = 'lat') -> xr.DataArray:
        weights = np.cos(np.deg2rad(da[lat_name]))
        da_lon_mean = da.mean(dim='lon') if 'lon' in da.dims else da
        return (da_lon_mean * weights).sum(dim=lat_name) / weights.sum()

    def load_obs_pressure_level_data(self, var_name: str, pressure_level: int) -> Optional[xr.DataArray]:
        obs_dir = Path("/sas12t1/ffyan/MonthlyPressureLevel")
        monthly_da_list =[]
        for year in range(1993, 2021):
            for month in range(1, 13):
                file_path = obs_dir / f"era5_pressure_levels_{year}{month:02d}.nc"
                if not file_path.exists(): continue
                try:
                    with xr.open_dataset(file_path) as ds:
                        actual_var = next((v for v in[var_name, var_name.upper(), var_name.lower()] if v in ds), None)
                        if not actual_var: continue
                        da = ds[actual_var]
                        
                        level_coord = 'pressure_level' if 'pressure_level' in da.dims else 'level'
                        if level_coord in da.coords: da = da.sel({level_coord: pressure_level}).drop_vars(level_coord, errors='ignore')
                        
                        time_coord = 'valid_time' if 'valid_time' in da.dims else ('time' if 'time' in da.dims else None)
                        if time_coord and da[time_coord].size > 0: da = da.isel({time_coord: 0})
                        
                        if 'latitude' in da.coords: da = da.rename({'latitude': 'lat'})
                        if 'longitude' in da.coords: da = da.rename({'longitude': 'lon'})
                        if 'lat' in da.coords and da.lat.values[0] > da.lat.values[-1]: da = da.sortby('lat')
                        
                        da = da.expand_dims(time=[pd.Timestamp(year, month, 1)])
                        monthly_da_list.append(da.load())
                except Exception: continue
        return xr.concat(monthly_da_list, dim='time').sortby('time') if monthly_da_list else None

    def get_eof_pc1(self, da: xr.DataArray, lat_slice: slice, lon_slice: slice, ref_ts: xr.DataArray) -> Optional[xr.DataArray]:
        """在宽泛区域内进行 EOF 降维提取 PC1，并利用物理锚点(ref_ts)严格校正正负号"""
        try:
            from eofs.standard import Eof
        except ImportError:
            logger.warning("未安装 eofs 库，将回退到固定 Box 面积平均法。")
            return None

        lat_min, lat_max = sorted([lat_slice.start, lat_slice.stop])
        lon_min, lon_max = sorted([lon_slice.start, lon_slice.stop])
        
        if da.lat.values[0] > da.lat.values[-1]:
            da_region = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        else:
            da_region = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        # 纬度加权
        coslat = np.cos(np.deg2rad(da_region.lat.values))
        wgts = np.sqrt(coslat)[:, np.newaxis] * np.ones((len(da_region.lat), len(da_region.lon)))
        
        solver = Eof(da_region.values, weights=wgts)
        pc1 = solver.pcs(npcs=1, pcscaling=1)[:, 0]
        
        # 【核心修复】：利用经典的物理偶极子指数 (ref_ts) 作为锚点，强制锁定 EOF 的相位
        # 确保 PC1 的正值绝对代表反气旋 (WNPAC) 或强冬季风 (EAWM)
        if stats.pearsonr(pc1, ref_ts.values)[0] < 0:
            pc1 = -pc1
            
        return xr.DataArray(pc1, coords={'time': da.time}, dims=['time'])

    def compute_eawm_index(self, model: str, leadtime: int) -> xr.DataArray:
        if model in['ERA5', 'Obs']:
            u500 = self.load_obs_pressure_level_data('u', 500)
        else:
            u500 = self.data_loader.load_pressure_level_data(model=model, leadtime=leadtime, var_name='u', pressure_level=500)
            
        if u500 is None: raise ValueError(f"无法加载 {model} U500")
        if 'number' in u500.dims: u500 = u500.mean(dim='number')
            
        # 1. 先计算经典的固定 Box 物理指数 (作为锚点)
        south_region = {'lat': slice(25, 35), 'lon': slice(80, 120)}
        north_region = {'lat': slice(45, 55), 'lon': slice(80, 120)}
        if u500.lat.values[0] > u500.lat.values[-1]:
            south_region['lat'], north_region['lat'] = slice(35, 25), slice(55, 45)
            
        u_south = self.area_weighted_mean(u500.sel(**south_region))
        u_north = self.area_weighted_mean(u500.sel(**north_region))
        eawm_raw = u_south - u_north
        eawm_raw.name = 'eawm_index'

        # 2. 尝试 EOF 提取，并传入 eawm_raw 锁定相位
        pc1 = self.get_eof_pc1(u500, lat_slice=slice(20, 60), lon_slice=slice(70, 140), ref_ts=eawm_raw)
        if pc1 is not None:
            pc1.name = 'eawm_index'
            return pc1
            
        return eawm_raw

    def compute_wnpac_index(self, model: str, leadtime: int) -> xr.DataArray:
        if model in ['ERA5', 'Obs']:
            u850 = self.load_obs_pressure_level_data('u', 850)
        else:
            u850 = self.data_loader.load_pressure_level_data(model=model, leadtime=leadtime, var_name='u', pressure_level=850)
            
        if u850 is None: raise ValueError(f"无法加载 {model} U850")
        if 'number' in u850.dims: u850 = u850.mean(dim='number')
            
        # 1. 先计算经典的固定 Box 物理指数 (作为锚点)
        south_region = {'lat': slice(5, 15), 'lon': slice(100, 130)}
        north_region = {'lat': slice(20, 30), 'lon': slice(110, 140)}
        if u850.lat.values[0] > u850.lat.values[-1]:
            south_region['lat'], north_region['lat'] = slice(15, 5), slice(30, 20)
            
        u_south = self.area_weighted_mean(u850.sel(**south_region))
        u_north = self.area_weighted_mean(u850.sel(**north_region))
        wnpac_raw = u_north - u_south
        wnpac_raw.name = 'wnpac_index'

        # 2. 尝试 EOF 提取，并传入 wnpac_raw 锁定相位
        pc1 = self.get_eof_pc1(u850, lat_slice=slice(0, 40), lon_slice=slice(100, 150), ref_ts=wnpac_raw)
        if pc1 is not None:
            pc1.name = 'wnpac_index'
            return pc1
            
        return wnpac_raw

    def extract_season_mean(self, da: xr.DataArray, season: str) -> xr.DataArray:
        if season == 'DJF':
            da_sub = da.where(da.time.dt.month.isin([12, 1, 2]), drop=True)
            years, months = da_sub.time.dt.year.values, da_sub.time.dt.month.values
            season_years = np.where(months == 12, years + 1, years)
            da_sub = da_sub.assign_coords(season_year=('time', season_years))
            return da_sub.groupby('season_year').mean('time')
        elif season == 'JJA':
            da_sub = da.where(da.time.dt.month.isin([6, 7, 8]), drop=True)
            da_sub = da_sub.assign_coords(season_year=('time', da_sub.time.dt.year.values))
            return da_sub.groupby('season_year').mean('time')
        raise ValueError("仅支持 DJF 和 JJA")

    def apply_land_mask(self, da: xr.DataArray) -> xr.DataArray:
        try:
            import regionmask
            if hasattr(regionmask.defined_regions, 'natural_earth_v5_0_0'):
                land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
            else:
                land = regionmask.defined_regions.natural_earth.land_110
            mask = land.mask(da.lon, da.lat)
            return da.where(mask == 0)
        except ImportError:
            try:
                from global_land_mask import globe
                lon_grid, lat_grid = np.meshgrid(da.lon.values, da.lat.values)
                lon_grid_180 = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
                mask = globe.is_land(lat_grid, lon_grid_180)
                return da.where(mask)
            except ImportError:
                return da
        except Exception:
            return da

    def load_and_preprocess_data(self, model: str, leadtime: int) -> Tuple[np.ndarray, np.ndarray, xr.DataArray, str]:
        logger.info(f"加载 {model} L{leadtime} 的数据...")
        
        enso_monthly = self.get_index_from_cache('Nino3.4', model, leadtime)
        
        if self.var_type == 'temp':
            bridge_monthly = self.compute_eawm_index(model, leadtime)
            bridge_season, bridge_name, y_season = 'DJF', 'EAWM', 'DJF'
        elif self.var_type == 'prec':
            bridge_monthly = self.compute_wnpac_index(model, leadtime)
            bridge_season, bridge_name, y_season = 'JJA', 'WNPAC', 'JJA'
        else:
            raise ValueError(f"不支持的变量类型: {self.var_type}")
            
        if model in['ERA5', 'Obs']:
            y_grid_monthly = self.data_loader.load_obs_data(self.var_type)
        else:
            y_grid_monthly = self.data_loader.load_forecast_data_ensemble(model, self.var_type, leadtime)
            if y_grid_monthly is not None and 'number' in y_grid_monthly.dims:
                y_grid_monthly = y_grid_monthly.mean(dim='number')
                
        if y_grid_monthly is None: raise ValueError("目标空间场缺失")
        
        y_grid_monthly = self.apply_land_mask(y_grid_monthly)
        y_grid_monthly = y_grid_monthly.resample(time='1MS').mean()
        
        enso_yr = self.extract_season_mean(enso_monthly, 'DJF')
        bridge_yr = self.extract_season_mean(bridge_monthly, bridge_season)
        y_grid_yr = self.extract_season_mean(y_grid_monthly, y_season)
        
        common_years = set(enso_yr.season_year.values) & set(bridge_yr.season_year.values) & set(y_grid_yr.season_year.values)
        common_years = sorted([y for y in common_years if 1993 <= y <= 2020])
        if len(common_years) < 10: raise ValueError("时间序列重叠年份不足")
            
        enso_yr = enso_yr.sel(season_year=common_years)
        bridge_yr = bridge_yr.sel(season_year=common_years)
        y_grid_yr = y_grid_yr.sel(season_year=common_years)
        
        enso_std = (enso_yr - enso_yr.mean()) / enso_yr.std()
        bridge_std = (bridge_yr - bridge_yr.mean()) / bridge_yr.std()
        y_grid_std = (y_grid_yr - y_grid_yr.mean(dim='season_year')) / y_grid_yr.std(dim='season_year')
        
        return enso_std.values, bridge_std.values, y_grid_std, bridge_name

    def compute_mechanisms(self, model: str, leadtime: int, enso: np.ndarray, bridge: np.ndarray, 
                           y_grid: xr.DataArray, bridge_name: str, n_jobs: int) -> xr.Dataset:
        
        X_matrix = np.column_stack((np.ones_like(enso), enso, bridge))
        
        time_len, lat_len, lon_len = y_grid.shape
        y_reshaped = y_grid.values.reshape(time_len, -1)
        tasks = [(y_reshaped[:, i], X_matrix, enso, bridge) for i in range(y_reshaped.shape[1])]
        results = [None] * len(tasks)
        
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_idx = {executor.submit(process_pixel, task): i for i, task in enumerate(tasks)}
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try: results[idx] = future.result()
                    except Exception: results[idx] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                    completed += 1
                    if completed % 5000 == 0: logger.info(f"[{model} L{leadtime}] 进度: {completed}/{len(tasks)}")
        else:
            for i, task in enumerate(tasks):
                try: results[i] = process_pixel(task)
                except Exception: results[i] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                    
        results = np.array(results)
        ds_out = xr.Dataset(
            {
                "corr_enso": (("lat", "lon"), results[:, 0].reshape(lat_len, lon_len)),
                "coef_bridge_indep": (("lat", "lon"), results[:, 1].reshape(lat_len, lon_len)),
                "indirect_effect": (("lat", "lon"), results[:, 2].reshape(lat_len, lon_len)),
                "p_corr_enso": (("lat", "lon"), results[:, 3].reshape(lat_len, lon_len)),
                "p_coef_bridge": (("lat", "lon"), results[:, 4].reshape(lat_len, lon_len)),
                "p_indirect": (("lat", "lon"), results[:, 5].reshape(lat_len, lon_len))
            }, coords={"lat": y_grid.lat, "lon": y_grid.lon}
        )
        ds_out.attrs['bridge_name'] = bridge_name
        nc_path = self.nc_dir / f"mechanism_results_{model}_L{leadtime}_{self.var_type}.nc"
        ds_out.to_netcdf(nc_path)
        return ds_out

    # ==========================================
    # 制图模块：完全复刻 4x4 GridSpec 布局 + 显著性打点
    # ==========================================

    def add_china_map_details(self, ax, data, levels, cmap, draw_scs=True):
        bou_paths =[self.boundaries_dir / "中国_省1.shp", self.boundaries_dir / "中国_省2.shp"]
        hyd_path = self.boundaries_dir / "河流.shp"
        
        if hyd_path.exists():
            try: ax.add_geometries(shpreader.Reader(str(hyd_path)).geometries(), ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6, zorder=5)
            except: pass
            
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', zorder=50)
        loaded_borders = False
        for p in bou_paths:
            if p.exists():
                try: ax.add_geometries(shpreader.Reader(str(p)).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6, zorder=100); loaded_borders = True
                except: pass
        if not loaded_borders: ax.add_feature(cfeature.BORDERS, linewidth=1.0, zorder=100)
            
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
            except Exception: pass

    def _plot_single_map(self, fig, gs, row, col, model, var_data, p_data, norm, levels, cmap, char_label):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)
        
        im = ax.contourf(var_data.lon, var_data.lat, var_data, transform=ccrs.PlateCarree(), 
                         cmap=cmap, norm=norm, levels=levels, extend='both')
        
        # === 显著性打点 (P < 0.05) ===
        if p_data is not None:
            sig_mask = (p_data < 0.05).values
            if np.any(sig_mask):
                X, Y = np.meshgrid(p_data.lon, p_data.lat)
                stride = 2
                X_s, Y_s, mask_s = X[::stride, ::stride], Y[::stride, ::stride], sig_mask[::stride, ::stride]
                ax.scatter(X_s[mask_s], Y_s[mask_s], transform=ccrs.PlateCarree(), 
                           s=2, c='black', alpha=0.6, marker='.', zorder=4)
        
        self.add_china_map_details(ax, var_data, levels, cmap, draw_scs=True)
        
        lon_ticks = np.arange(75, 141, 15)
        lat_ticks = np.arange(20, 56, 10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = FixedLocator(lon_ticks)
        gl.ylocator = FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(number_format='.0f')
        gl.yformatter = LatitudeFormatter(number_format='.0f')
        
        display_name = model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
        title_text = f"({char_label}) {display_name}" if char_label else f"{display_name}"
        ax.text(0.02, 0.96, title_text, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
        return im

    def plot_variable_cross_model_results(self, models_to_plot: List[str], final_model_results: Dict):
        if len(models_to_plot) == 0:
            logger.warning("没有可绘制的模型结果。")
            return

        bridge_name = "Bridge"
        for m in models_to_plot:
            if final_model_results[m]:
                first_lt = list(final_model_results[m].keys())[0]
                bridge_name = final_model_results[m][first_lt].attrs.get('bridge_name', 'Bridge')
                break

        if self.var_type == 'temp':
            enso_label = "ENSO (DJF)"
            bridge_label = f"{bridge_name} (DJF)"
        else:
            enso_label = "ENSO (Prev DJF)"
            bridge_label = f"{bridge_name} (JJA)"

        vars_configs =[
            ("corr_enso", "p_corr_enso", f"Total Effect: {enso_label} vs {self.var_type.upper()}", "RdBu_r"),
            ("coef_bridge_indep", "p_coef_bridge", f"Indep. Effect: {bridge_label} (Controlling ENSO)", "BrBG"),
            ("indirect_effect", "p_indirect", f"Mediation Effect: ENSO via {bridge_name} (Difference)", "PRGn")
        ]

        if 'ERA5' in models_to_plot:
            models_to_plot.remove('ERA5')
            models_to_plot.insert(0, 'ERA5')
        elif 'Obs' in models_to_plot:
            models_to_plot.remove('Obs')
            models_to_plot.insert(0, 'Obs')

        for var, p_var, title, cmap in vars_configs:
            logger.info(f"正在绘制 4x4 组合图: {title}")
            
            valid_data_list =[]
            for m in models_to_plot:
                for lt in [0, 3]:
                    if lt in final_model_results.get(m, {}):
                        valid_data_list.append(final_model_results[m][lt][var].values)
            
            if not valid_data_list: continue
            vmin_global = min([np.nanmin(data) for data in valid_data_list])
            vmax_global = max([np.nanmax(data) for data in valid_data_list])

            vmax_abs = max(abs(vmin_global), abs(vmax_global))
            vmax_abs = max(vmax_abs, 0.01)
            locator = ticker.MaxNLocator(nbins=20, symmetric=True)
            levels = locator.tick_values(-vmax_abs, vmax_abs)
            norm = TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])

            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15, left=0.05, right=0.92, top=0.94, bottom=0.06)
            im_for_cbar = None

            for lt_idx, leadtime in enumerate([0, 3]):
                row_start = lt_idx * 2
                grid_mapping =[]
                
                if leadtime == 0:
                    grid_mapping.append((0, 0, 0, None, False))
                else:
                    ax_blank = fig.add_subplot(gs[row_start, 0])
                    ax_blank.axis('off')
                    
                for i in range(1, 4):
                    if i < len(models_to_plot):
                        grid_mapping.append((0, i, i, chr(97 + i - 1), i == 1))
                        
                for i in range(4, 8):
                    if i < len(models_to_plot):
                        grid_mapping.append((1, i - 4, i, chr(97 + i - 1), False))
                        
                for row_offset, col_offset, m_idx, char_label, is_first_model in grid_mapping:
                    model = models_to_plot[m_idx]
                    lt_to_use = leadtime if leadtime in final_model_results[model] else list(final_model_results[model].keys())[0]
                    
                    if var not in final_model_results[model][lt_to_use]:
                        continue
                        
                    var_data = final_model_results[model][lt_to_use][var]
                    p_data = final_model_results[model][lt_to_use].get(p_var, None)
                    
                    im = self._plot_single_map(fig, gs, row_start + row_offset, col_offset, model, var_data, p_data, norm, levels, cmap, char_label)
                    if im: im_for_cbar = im
                    
                    if is_first_model:
                        ax = fig.axes[-1]
                        ax.text(0.98, 0.96, f'L{leadtime}', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

            fig.suptitle(f"Mechanism Analysis: {title}", fontsize=24, fontweight='bold', y=0.98)

            if im_for_cbar is not None:
                cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.75])
                cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='vertical', extend='both')
                cbar.locator = ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
                cbar.formatter = ticker.FormatStrFormatter('%g')
                cbar.update_ticks()
                cbar.set_label('Standardized Effect Size', fontsize=18, labelpad=15)
                cbar.ax.tick_params(labelsize=14)

            fig_path = self.plot_dir / f"combined_spatial_maps_L0_L3_{var}_{self.var_type}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            logger.info(f"成功导出组合图: {fig_path}")

    # ==========================================
    # 主流程控制
    # ==========================================

    def run_analysis(self, models: List[str], leadtimes: List[int], parallel: bool, n_jobs: int, plot_only: bool = False):
        final_model_results = {m: {} for m in models}

        if plot_only:
            logger.info("--plot-only 模式: 从缓存加载数据并仅绘图")
            results_cache = self._load_cache()
            for key, ds_out in results_cache.items():
                m, lt_str = key.split('_L')
                if m in final_model_results:
                    final_model_results[m][int(lt_str)] = ds_out
        else:
            tasks =[(model, lt) for model in models for lt in leadtimes]
            results_cache = {}

            if parallel and len(tasks) > 1:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_task = {
                        executor.submit(_mechanism_worker_task, self.var_type, model, lt): (model, lt)
                        for model, lt in tasks
                    }
                    for future in as_completed(future_to_task):
                        res_model, res_lt, ds_out = future.result()
                        if ds_out is not None:
                            results_cache[f"{res_model}_L{res_lt}"] = ds_out
                            final_model_results[res_model][res_lt] = ds_out
            else:
                for model, leadtime in tasks:
                    try:
                        enso, bridge, y_grid, bridge_name = self.load_and_preprocess_data(model, leadtime)
                        ds_out = self.compute_mechanisms(model, leadtime, enso, bridge, y_grid, bridge_name, n_jobs)
                        results_cache[f"{model}_L{leadtime}"] = ds_out
                        final_model_results[model][leadtime] = ds_out
                    except Exception as e:
                        logger.error(f"处理 {model} L{leadtime} 发生错误: {e}")
            
            self._save_cache(results_cache)

        models_to_plot =[m for m in models if final_model_results[m]]
        self.plot_variable_cross_model_results(models_to_plot, final_model_results)

# ==========================================
# 5. CLI 入口
# ==========================================

def main():
    parser = create_parser(description="Climate Mechanism Analysis (ENSO -> EAWM/WNPAC)", var_required=False)
    args = parser.parse_args()
    
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    
    if 'ERA5' not in models:
        models = ['ERA5'] + models
        
    default_leads = [0, 3]
    leadtimes = parse_leadtimes(args.leadtimes, default_leads) if args.leadtimes else default_leads
    var_list = parse_vars(args.var) if args.var else['prec', 'temp']
    
    parallel = normalize_parallel_args(args) or getattr(args, 'parallel', False)
    plot_only = getattr(args, 'plot_only', False)
    n_jobs = args.n_jobs if args.n_jobs else (os.cpu_count() - 1)
    
    for var_type in var_list:
        logger.info(f"=== 开始处理变量: {var_type} ===")
        analyzer = ClimateMechanismAnalyzer(var_type=var_type)
        analyzer.run_analysis(models=models, leadtimes=leadtimes, parallel=parallel, n_jobs=n_jobs, plot_only=plot_only)

if __name__ == "__main__":
    main()
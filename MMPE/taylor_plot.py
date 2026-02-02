#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模式Taylor图分析模块 (Fixed V7 - Ensemble Spread Range)
修改说明：
1. 数据计算：
   - 计算所有 Ensemble Members 的指标。
   - 计算 Ensemble Mean 的指标。
2. 绘图重构：
   - 颜色 (Color) -> 区分 Seasons (Annual, DJF, MAM, JJA, SON)
   - 点型 (Marker) -> 区分 Models
   - 绘制 Spread: 
     - 使用对应 Season 的颜色。
     - 绘制该 Season 下所有 Models 的所有 Members 的数据范围 (Convex Hull)。
     - 替代原本的 Member 散点。
   - 绘制 Mean: 保留 Ensemble Mean 的大点。
3. 数据保存：
   - metrics_taylor_{var}_all.nc 增加 'member' 维度/坐标
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import calendar
from scipy.spatial import ConvexHull, QhullError

# 添加 toolkit 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    SEASONS,
    COLORS,
)

# 配置日志
logger = setup_logging(
    log_file='taylor_plot_v7.log',
    module_name=__name__
)

# 定义区域
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

# 定义绘图用的配置
# Color -> Season
SEASON_COLORS = {
    'Annual': 'black',
    'DJF': '#1f77b4',  # blue
    'MAM': '#2ca02c',  # green
    'JJA': '#d62728',  # red
    'SON': '#ff7f0e'   # orange
}

# Marker -> Model (7 models)
MODEL_MARKERS = {
    "CMCC-35": 'o',
    "DWD-mon-21": 's',
    "ECMWF-51-mon": '^',
    "Meteo-France-8": 'D',
    "NCEP-2": 'v',
    "UKMO-14": 'p',
    "ECCC-Canada-3": 'h'
}

# 辅助函数：获取 marker (如果模型不在列表里，默认用 'o')
def get_model_marker(model_name):
    return MODEL_MARKERS.get(model_name, 'o')

# =============================================================================
# 内置 TaylorDiagram 实现 (保持不变)
# =============================================================================
class TaylorDiagram(object):
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as fa
        import mpl_toolkits.axisartist.grid_finder as gf

        self.refstd = refstd
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        r_locs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            self.tmax = np.pi
            r_locs = np.concatenate((-r_locs[:0:-1], r_locs))
        else:
            self.tmax = np.pi/2

        t_locs = np.arccos(r_locs)
        gl1 = gf.FixedLocator(t_locs)
        tf1 = gf.DictFormatter(dict(zip(t_locs, map(str, r_locs))))

        self.smin = srange[0]
        self.smax = srange[1]

        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1
        )

        if fig is None:
            fig = plt.figure()

        # 根据 rect 类型选择 FloatingAxes 或 FloatingSubplot
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            ax = fa.FloatingAxes(fig, rect, grid_helper=ghelper)
            fig.add_axes(ax)
        else:
            ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
            fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_fontsize(14) # Slightly smaller font
        ax.axis["top"].major_ticklabels.set_fontsize(12)

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation (Normalized)")
        ax.axis["left"].label.set_fontsize(14)
        ax.axis["left"].major_ticklabels.set_fontsize(12)

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].major_ticklabels.set_fontsize(12)

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        self.samplePoints = [l]
        self.ax.grid(True, linestyle=':')

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

# =============================================================================
# 指标计算逻辑
# =============================================================================
def calculate_weighted_metrics(obs_da: xr.DataArray, mod_da: xr.DataArray, region_bounds=None) -> dict:
    try:
        if region_bounds is not None:
            lat_b = region_bounds['lat']
            lon_b = region_bounds['lon']
            if obs_da.lat[0] < obs_da.lat[-1]:
                lat_slice = slice(lat_b[0], lat_b[1])
            else:
                lat_slice = slice(lat_b[1], lat_b[0])
            
            obs_sub = obs_da.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
            mod_sub = mod_da.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
        else:
            obs_sub = obs_da
            mod_sub = mod_da

        weights = np.cos(np.deg2rad(obs_sub.lat))
        weights_broad = xr.broadcast(weights, obs_sub)[0]

        obs_flat = obs_sub.values.flatten()
        mod_flat = mod_sub.values.flatten()
        w_flat = weights_broad.values.flatten()

        valid_mask = np.isfinite(obs_flat) & np.isfinite(mod_flat) & np.isfinite(w_flat)
        if np.sum(valid_mask) < 10:
            return {'corr': np.nan, 'std_ratio': np.nan, 'crmse': np.nan}

        o = obs_flat[valid_mask]
        m = mod_flat[valid_mask]
        w = w_flat[valid_mask]

        mean_o = np.average(o, weights=w)
        mean_m = np.average(m, weights=w)
        
        o_anom = o - mean_o
        m_anom = m - mean_m
        
        var_o = np.average(o_anom**2, weights=w)
        var_m = np.average(m_anom**2, weights=w)
        std_o = np.sqrt(var_o)
        std_m = np.sqrt(var_m)
        
        cov = np.average(o_anom * m_anom, weights=w)
        corr = cov / (std_o * std_m) if (std_o > 0 and std_m > 0) else np.nan
        
        std_ratio = std_m / std_o if std_o > 0 else np.nan
        
        if std_o > 0 and np.isfinite(std_ratio) and np.isfinite(corr):
            crmse_norm = np.sqrt(max(0, 1 + std_ratio**2 - 2 * std_ratio * corr))
        else:
            crmse_norm = np.nan

        return {
            'corr': float(corr),
            'std_ratio': float(std_ratio),
            'crmse': float(crmse_norm),
            'ref_std': 1.0 
        }
    except Exception as e:
        return {'corr': np.nan, 'std_ratio': np.nan, 'crmse': np.nan}

class TaylorAnalyzer:
    def __init__(self, var_type: str, n_jobs: int = None):
        self.var_type = var_type
        self.data_loader = DataLoader()
        self.n_jobs = n_jobs
        
        self.output_dir = Path(f"/sas12t1/ffyan/output/taylor_analysis")
        self.data_dir = self.output_dir / "data" / var_type
        self.plot_dir = self.output_dir / "plots" / var_type
        
        for p in [self.data_dir, self.plot_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def load_and_preprocess(self, model: str, leadtime: int):
        try:
            obs_data = self.data_loader.load_obs_data(self.var_type)
            obs_data = obs_data.resample(time='1MS').mean()
            obs_data = obs_data.sel(time=slice('1993', '2020'))

            # Load ensemble data (should contain 'number' dimension)
            fcst_data = self.data_loader.load_forecast_data_ensemble(model, self.var_type, leadtime)
            if fcst_data is None: return None, None
            
            # Note: Do NOT average over 'number' here. Keep it for member analysis.

            fcst_data = fcst_data.resample(time='1MS').mean()
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))

            common_times = obs_data.time.to_index().intersection(fcst_data.time.to_index())
            if len(common_times) < 12: return None, None

            obs_aligned = obs_data.sel(time=common_times)
            fcst_aligned = fcst_data.sel(time=common_times)

            # Interpolate fcst to obs grid (works for extra dimensions like 'number')
            fcst_interp = fcst_aligned.interp(lat=obs_aligned.lat, lon=obs_aligned.lon, method='linear')

            # Calculate anomalies
            obs_clim = obs_aligned.groupby('time.month').mean('time')
            obs_anom = obs_aligned.groupby('time.month') - obs_clim

            fcst_clim = fcst_interp.groupby('time.month').mean('time')
            fcst_anom = fcst_interp.groupby('time.month') - fcst_clim

            return obs_anom, fcst_anom
        except Exception as e:
            logger.error(f"Data loading failed {model} L{leadtime}: {e}")
            return None, None

    def _calc_metrics_for_mask(self, obs, fcst, mask):
        """
        对指定时间 mask 计算所有区域的指标。
        支持 fcst 包含 'number' 维度。
        返回结构: {'mean': {reg: metrics}, 'members': {0: {reg: metrics}, 1: ...}}
        """
        if isinstance(mask, slice):
            o_sub = obs
            f_sub = fcst
        else:
            o_sub = obs.sel(time=mask)
            f_sub = fcst.sel(time=mask)

        if o_sub.size == 0: return None
        
        # 1. Calculate for Ensemble Mean
        if 'number' in f_sub.dims:
            f_mean = f_sub.mean(dim='number')
        else:
            f_mean = f_sub
            
        reg_res_mean = {}
        for reg_name, reg_bounds in REGIONS.items():
            reg_res_mean[reg_name] = calculate_weighted_metrics(o_sub, f_mean, reg_bounds)
            
        result = {'mean': reg_res_mean, 'members': {}}

        # 2. Calculate for Each Member
        if 'number' in f_sub.dims:
            numbers = f_sub.number.values
            for num in numbers:
                # Select member
                f_mem = f_sub.sel(number=num)
                reg_res_mem = {}
                for reg_name, reg_bounds in REGIONS.items():
                    reg_res_mem[reg_name] = calculate_weighted_metrics(o_sub, f_mem, reg_bounds)
                
                # Use scalar number or string as key
                key = int(num) if np.issubdtype(type(num), np.integer) else str(num)
                result['members'][key] = reg_res_mem
        
        return result

    def compute_metrics_task(self, model, leadtime):
        """计算 Annual, Seasons(DJF..), Months(Jan..) 的指标"""
        obs_anom, fcst_anom = self.load_and_preprocess(model, leadtime)
        if obs_anom is None: return None

        results = {}

        # 1. Annual
        met_annual = self._calc_metrics_for_mask(obs_anom, fcst_anom, slice(None))
        if met_annual: results['Annual'] = met_annual

        # 2. Seasons (DJF, MAM, JJA, SON)
        for season_name, months in SEASONS.items():
            mask = obs_anom.time.dt.month.isin(months)
            met_season = self._calc_metrics_for_mask(obs_anom, fcst_anom, mask)
            if met_season: results[season_name] = met_season

        # 3. Monthly (Jan..Dec)
        for m in range(1, 13):
            mask = obs_anom.time.dt.month == m
            m_name = calendar.month_abbr[m]
            met_month = self._calc_metrics_for_mask(obs_anom, fcst_anom, mask)
            if met_month: results[m_name] = met_month

        return (model, leadtime, results)

    def run_analysis(self, models, leadtimes, parallel=True, plot_only=False, no_plot=False):
        if plot_only:
            ds = self.load_metrics()
            if ds is None:
                logger.warning("Plot-only: no saved metrics found, skip plotting.")
                return
            logger.info(f"Plot-only: loaded metrics for {self.var_type}")
            leadtimes_plot = [int(x) for x in ds.leadtime.values]
            models_plot = [str(x) for x in ds.model.values]
            self.plot_all(ds, leadtimes_plot, models_plot)
            ds.close()
            return

        tasks = [(m, lt) for lt in leadtimes for m in models]
        final_data = {}

        logger.info(f"Start Taylor analysis (Ensemble + Spread) for {self.var_type}...")

        if parallel:
            max_workers = min(self.n_jobs or cpu_count(), 32)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.compute_metrics_task, *t): t for t in tasks}
                for i, future in enumerate(as_completed(futures)):
                    res = future.result()
                    if res: self._organize_results(final_data, res)
                    if (i + 1) % 10 == 0: logger.info(f"Progress: {i+1}/{len(tasks)}")
        else:
            for t in tasks:
                res = self.compute_metrics_task(*t)
                if res: self._organize_results(final_data, res)

        self.save_metrics(final_data)

        if not no_plot:
            self.plot_all(final_data, leadtimes, models)

    def _organize_results(self, final_data, res):
        model, leadtime, period_results = res
        if leadtime not in final_data: final_data[leadtime] = {}
        if model not in final_data[leadtime]: final_data[leadtime][model] = {}
        # period_results structure: {period: {'mean': ..., 'members': ...}}
        for period, data_dict in period_results.items():
            final_data[leadtime][model][period] = data_dict

    def save_metrics(self, data):
        """保存所有计算结果到一个 NetCDF 文件 (增加 member 维度)"""
        rows = []
        for lt, mod_dict in data.items():
            for mod, period_dict in mod_dict.items():
                for period, type_dict in period_dict.items():
                    # type_dict: {'mean': {reg: met}, 'members': {id: {reg: met}}}
                    
                    # 1. Mean
                    mean_dict = type_dict.get('mean', {})
                    for reg, met in mean_dict.items():
                        rows.append({
                            'leadtime': lt, 'model': mod, 'season': period, 'region': reg,
                            'member': 'mean',
                            'corr': met['corr'], 'std_ratio': met['std_ratio'], 'crmse': met['crmse']
                        })
                    
                    # 2. Members
                    mem_dict = type_dict.get('members', {})
                    for mem_id, reg_dict in mem_dict.items():
                        for reg, met in reg_dict.items():
                            rows.append({
                                'leadtime': lt, 'model': mod, 'season': period, 'region': reg,
                                'member': str(mem_id),
                                'corr': met['corr'], 'std_ratio': met['std_ratio'], 'crmse': met['crmse']
                            })
                            
        if not rows: return

        df = pd.DataFrame(rows)
        # Convert member to string to be safe
        df['member'] = df['member'].astype(str)
        ds = df.set_index(['leadtime', 'model', 'season', 'region', 'member']).to_xarray()
        out_file = self.data_dir / f"metrics_taylor_{self.var_type}_all.nc"
        ds.to_netcdf(out_file)
        logger.info(f"Saved all metrics (with members) to: {out_file}")

    def load_metrics(self):
        """加载已保存的指标数据"""
        out_file = self.data_dir / f"metrics_taylor_{self.var_type}_all.nc"
        if not out_file.exists():
            return None
        return xr.open_dataset(out_file)

    # =========================================================================
    # 绘图逻辑：按 Lead Time 分图
    # 颜色=Season, Marker=Model, 大点=Mean, Spread (Hull)=Members
    # =========================================================================
    def plot_all(self, data, leadtimes, models):
        """data: xarray Dataset 或 dict"""
        # Convert dict to xarray if needed
        if isinstance(data, dict):
            # This path is usually not taken if we just saved, but for safety:
            rows = []
            for lt, mod_dict in data.items():
                for mod, period_dict in mod_dict.items():
                    for period, type_dict in period_dict.items():
                        # Mean
                        for reg, met in type_dict.get('mean', {}).items():
                            rows.append({'leadtime': lt, 'model': mod, 'season': period, 'region': reg, 'member': 'mean', **met})
                        # Members
                        for mem_id, reg_dict in type_dict.get('members', {}).items():
                            for reg, met in reg_dict.items():
                                rows.append({'leadtime': lt, 'model': mod, 'season': period, 'region': reg, 'member': str(mem_id), **met})
            df = pd.DataFrame(rows)
            ds = df.set_index(['leadtime', 'model', 'season', 'region', 'member']).to_xarray()
        else:
            ds = data

        plot_seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
        available_lts = [int(x) for x in ds.leadtime.values]

        for lt in leadtimes:
            if lt not in available_lts: continue
            try:
                ds_lt = ds.sel(leadtime=lt)
            except Exception:
                continue
            self.plot_single_leadtime_global(lt, ds_lt, models, plot_seasons)
            self.plot_single_leadtime_regions(lt, ds_lt, models, plot_seasons)

    def _create_legend(self, fig):
        """创建图例：Color=Season, Marker=Model, Spread=Range"""
        handles = []
        labels = []
        
        # 1. Seasons (Colors)
        handles.append(Line2D([0], [0], color='w', label='[ Seasons ]', marker=None))
        labels.append(r"$\bf{Seasons}$")
        for s_name, color in SEASON_COLORS.items():
            h = Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8)
            handles.append(h)
            labels.append(s_name)
            
        handles.append(Line2D([0], [0], color='w', label=' ', marker=None))
        labels.append(" ")
        
        # 2. Models (Markers)
        handles.append(Line2D([0], [0], color='w', label='[ Models ]', marker=None))
        labels.append(r"$\bf{Models}$")
        for m_name, marker in MODEL_MARKERS.items():
            display_name = m_name.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
            h = Line2D([0], [0], color='k', marker=marker, linestyle='', markersize=8)
            handles.append(h)
            labels.append(display_name)
            
        handles.append(Line2D([0], [0], color='w', label=' ', marker=None))
        labels.append(" ")

        # 3. Type
        handles.append(Line2D([0], [0], color='w', label='[ Type ]', marker=None))
        labels.append(r"$\bf{Type}$")
        
        h_mean = Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=12, markeredgecolor='k', label='Ensemble Mean')
        handles.append(h_mean)
        labels.append("Ens Mean")
        
        h_spread = Polygon(np.array([[0,0]]), facecolor='gray', alpha=0.3, label='Member Range')
        handles.append(h_spread)
        labels.append("MME Spread")

        fig.legend(handles, labels, loc='lower center', ncol=6,
                   bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)

    def _plot_convex_hull(self, td, stds, corrs, color, alpha):
        """Draw Convex Hull for a set of (std, corr) points"""
        if len(stds) < 3: return

        # Transform to Cartesian for Hull calculation
        # Taylor coords: theta = arccos(corr), r = std
        # x = r * cos(theta) = r * corr
        # y = r * sin(theta) = r * sqrt(1-corr^2)
        
        x_vals = []
        y_vals = []
        valid_indices = []
        
        for i, (s, c) in enumerate(zip(stds, corrs)):
            if np.isfinite(s) and np.isfinite(c) and -1 <= c <= 1:
                x_vals.append(s * c)
                y_vals.append(s * np.sqrt(max(0, 1 - c**2)))
                valid_indices.append(i)
                
        if len(x_vals) < 3: return
        
        points = np.column_stack((x_vals, y_vals))
        
        try:
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            # Close polygon
            vertices = np.vstack((vertices, vertices[0]))
            
            # Convert back to (theta, r) for polar plot
            vs_x = vertices[:, 0]
            vs_y = vertices[:, 1]
            rs = np.sqrt(vs_x**2 + vs_y**2)
            thetas = np.arctan2(vs_y, vs_x)
            
            # Plot on td.ax (PolarAxes)
            td.ax.fill(thetas, rs, color=color, alpha=alpha, zorder=1, label='_nolegend_')
            
        except QhullError:
            pass # Collinear points
        except Exception as e:
            logger.warning(f"Hull plotting error: {e}")

    def plot_single_leadtime_global(self, leadtime, ds_data, models, seasons):
        """绘制单个 Leadtime 的 Global 图"""
        reg_name = 'Global'
        fig = plt.figure(figsize=(10, 9)) 
        rect = [0.1, 0.25, 0.8, 0.7]
        td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.7))
        td.add_contours(levels=5, colors='gray', alpha=0.4)
        plotted_any = False

        # 1. Draw Spreads (Hulls) for each season (All models aggregated)
        for season in seasons:
            if season not in ds_data.season.values: continue
            color = SEASON_COLORS.get(season, 'k')
            
            season_stds = []
            season_corrs = []
            
            for model in models:
                if model not in ds_data.model.values: continue
                try:
                    subset = ds_data.sel(region=reg_name, model=model, season=season)
                    if 'member' in subset.coords:
                         # Filter 'mean'
                         m_vals = subset.member.values
                         real_members = [m for m in m_vals if str(m) != 'mean']
                         if not real_members: continue
                         
                         sub_mem = subset.sel(member=real_members)
                         corrs = sub_mem['corr'].values.flatten()
                         stds = sub_mem['std_ratio'].values.flatten()
                         
                         mask = np.isfinite(corrs) & np.isfinite(stds)
                         season_corrs.extend(corrs[mask])
                         season_stds.extend(stds[mask])
                except Exception: pass
            
            if season_corrs:
                self._plot_convex_hull(td, season_stds, season_corrs, color=color, alpha=0.2)
                plotted_any = True

        # 2. Draw Ensemble Means (Markers)
        for model in models:
            if model not in ds_data.model.values: continue
            marker = get_model_marker(model)
            
            for season in seasons:
                if season not in ds_data.season.values: continue
                color = SEASON_COLORS.get(season, 'k')
                
                try:
                    subset = ds_data.sel(region=reg_name, model=model, season=season)
                    s_mean = subset.sel(member='mean')
                    corr = float(s_mean['corr'].values)
                    std = float(s_mean['std_ratio'].values)
                    if np.isfinite(corr) and np.isfinite(std):
                        td.add_sample(std, corr, color=color, marker=marker, 
                                    markersize=12, markeredgecolor='k', zorder=10, linestyle='', label="_nolegend_")
                        plotted_any = True
                except: pass

        if not plotted_any:
            plt.close(fig)
            return
        
        self._create_legend(fig)
        out_file = self.plot_dir / f"taylor_Global_{self.var_type}_L{leadtime}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Global plot saved: {out_file}")

    def plot_single_leadtime_regions(self, leadtime, ds_data, models, seasons):
        """绘制单个 Leadtime 的 Regions 九宫格"""
        regions_ordered = [
            'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
            'Z4-Tibetan', 'Z5-NorthChina', 'Z6-Yangtze',
            'Z7-Southwest', 'Z8-SouthChina', 'Z9-SouthSea'
        ]
        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3, top=0.92, bottom=0.12)
        plotted_any = False

        for i, reg_name in enumerate(regions_ordered):
            ax_spec = gs[i // 3, i % 3]
            bbox = ax_spec.get_position(fig)
            rect = [bbox.x0, bbox.y0, bbox.width, bbox.height]
            td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.7))
            td.add_contours(levels=5, colors='gray', alpha=0.4)
            td._ax.set_title(f"{reg_name}", fontsize=18, fontweight='bold', y=1.05)
            
            # 1. Spreads (Hulls)
            for season in seasons:
                if season not in ds_data.season.values: continue
                color = SEASON_COLORS.get(season, 'k')
                
                season_stds = []
                season_corrs = []
                
                for model in models:
                    if model not in ds_data.model.values: continue
                    try:
                        subset = ds_data.sel(region=reg_name, model=model, season=season)
                        if 'member' in subset.coords:
                             m_vals = subset.member.values
                             real_members = [m for m in m_vals if str(m) != 'mean']
                             if not real_members: continue
                             
                             sub_mem = subset.sel(member=real_members)
                             corrs = sub_mem['corr'].values.flatten()
                             stds = sub_mem['std_ratio'].values.flatten()
                             
                             mask = np.isfinite(corrs) & np.isfinite(stds)
                             season_corrs.extend(corrs[mask])
                             season_stds.extend(stds[mask])
                    except Exception: pass
                
                if season_corrs:
                    self._plot_convex_hull(td, season_stds, season_corrs, color=color, alpha=0.2)
                    plotted_any = True

            # 2. Means (Markers)
            for model in models:
                if model not in ds_data.model.values: continue
                marker = get_model_marker(model)

                for season in seasons:
                    if season not in ds_data.season.values: continue
                    color = SEASON_COLORS.get(season, 'k')

                    try:
                        subset = ds_data.sel(region=reg_name, model=model, season=season)
                        s_mean = subset.sel(member='mean')
                        corr = float(s_mean['corr'].values)
                        std = float(s_mean['std_ratio'].values)
                        if np.isfinite(corr) and np.isfinite(std):
                            td.add_sample(std, corr, color=color, marker=marker, 
                                        markersize=10, markeredgecolor='k', zorder=10, linestyle='', label="_nolegend_")
                            plotted_any = True
                    except: pass

        if not plotted_any:
            plt.close(fig)
            return
        
        self._create_legend(fig)
        out_file = self.plot_dir / f"taylor_Regions_{self.var_type}_L{leadtime}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Regions plot saved: {out_file}")

def main():
    parser = create_parser(
        description="Taylor图分析（Ensemble Spread, Color=Season, Marker=Model）",
        include_seasons=False,
        var_default=None,
        var_required=False
    )
    args = parser.parse_args()

    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']

    parallel = normalize_parallel_args(args) or (args.n_jobs is not None and args.n_jobs > 1)

    for var in var_list:
        analyzer = TaylorAnalyzer(var, n_jobs=args.n_jobs)
        analyzer.run_analysis(
            models=models,
            leadtimes=leadtimes,
            parallel=parallel,
            plot_only=args.plot_only,
            no_plot=args.no_plot
        )

if __name__ == "__main__":
    main()
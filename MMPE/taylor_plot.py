#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模式Taylor图分析模块 (Fixed V2)
修复说明：
1. 修复参数解析错误：添加 include_seasons=True。
2. 移除 toolkit 绘图依赖：内置 TaylorDiagram 类实现。
3. 增强健壮性：优化指标计算和绘图布局。
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
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 添加 toolkit 路径 (仅用于数据加载和通用配置)
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    SEASONS,
)

# 配置日志
logger = setup_logging(
    log_file='taylor_plot.log',
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
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# =============================================================================
# 内置 TaylorDiagram 实现 (替代 toolkit)
# =============================================================================
class TaylorDiagram(object):
    """
    Taylor diagram implementation within the script to avoid toolkit dependencies.
    Plot model standard deviation and correlation to reference (observation).
    """

    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes.
        
        Parameters:
        - refstd: Reference standard deviation (usually 1.0 for normalized).
        - fig: Input figure or None.
        - rect: Subplot definition.
        - label: Reference label.
        - srange: (min, max) for radial axis (stddev).
        """
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as fa
        import mpl_toolkits.axisartist.grid_finder as gf

        self.refstd = refstd
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        r_locs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            r_locs = np.concatenate((-r_locs[:0:-1], r_locs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2

        t_locs = np.arccos(r_locs)
        gl1 = gf.FixedLocator(t_locs)
        tf1 = gf.DictFormatter(dict(zip(t_locs, map(str, r_locs))))

        # Standard deviation axis extent
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

        ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_fontsize(12)

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation (Normalized)")
        ax.axis["left"].label.set_fontsize(12)

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Add reference point and circle
        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Grid lines
        self.samplePoints = [l]
        self.ax.grid(True, linestyle=':')

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample points (r, theta)."""
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """Add centered RMSE contours."""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Centered RMSE = sqrt(r^2 + ref^2 - 2*r*ref*cos(theta))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

# =============================================================================
# 指标计算逻辑
# =============================================================================
def calculate_weighted_metrics(obs_da: xr.DataArray, mod_da: xr.DataArray, region_bounds=None) -> dict:
    """计算加权泰勒图指标 (Correlation, Std Ratio, RMSE)"""
    try:
        # 1. 区域截取
        if region_bounds is not None:
            lat_b = region_bounds['lat']
            lon_b = region_bounds['lon']
            # 处理纬度切片顺序
            if obs_da.lat[0] < obs_da.lat[-1]:
                lat_slice = slice(lat_b[0], lat_b[1])
            else:
                lat_slice = slice(lat_b[1], lat_b[0])
            
            obs_sub = obs_da.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
            mod_sub = mod_da.sel(lat=lat_slice, lon=slice(lon_b[0], lon_b[1]))
        else:
            obs_sub = obs_da
            mod_sub = mod_da

        # 2. 计算权重 (cos(lat))
        weights = np.cos(np.deg2rad(obs_sub.lat))
        weights_broad = xr.broadcast(weights, obs_sub)[0]

        # 3. 展平数组
        obs_flat = obs_sub.values.flatten()
        mod_flat = mod_sub.values.flatten()
        w_flat = weights_broad.values.flatten()

        valid_mask = np.isfinite(obs_flat) & np.isfinite(mod_flat) & np.isfinite(w_flat)
        if np.sum(valid_mask) < 10:
            return {'corr': np.nan, 'std_ratio': np.nan, 'crmse': np.nan}

        o = obs_flat[valid_mask]
        m = mod_flat[valid_mask]
        w = w_flat[valid_mask]

        # 4. 计算加权统计量
        # 加权均值
        mean_o = np.average(o, weights=w)
        mean_m = np.average(m, weights=w)
        
        # 去中心化 (Centered Pattern)
        o_anom = o - mean_o
        m_anom = m - mean_m
        
        # 加权标准差
        var_o = np.average(o_anom**2, weights=w)
        var_m = np.average(m_anom**2, weights=w)
        std_o = np.sqrt(var_o)
        std_m = np.sqrt(var_m)
        
        # 加权相关系数
        cov = np.average(o_anom * m_anom, weights=w)
        corr = cov / (std_o * std_m) if (std_o > 0 and std_m > 0) else np.nan
        
        # 归一化标准差比
        std_ratio = std_m / std_o if std_o > 0 else np.nan
        
        # 加权中心化 RMSE (归一化)
        # E'^2 = 1 + sigma_hat^2 - 2 * sigma_hat * R
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
        # logger.debug(f"Metrics calculation failed: {e}")
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
        """加载并预处理数据 (计算月距平)"""
        try:
            # 加载观测
            obs_data = self.data_loader.load_obs_data(self.var_type)
            obs_data = obs_data.resample(time='1MS').mean()
            obs_data = obs_data.sel(time=slice('1993', '2020'))

            # 加载模式 (Ensemble Mean)
            fcst_data = self.data_loader.load_forecast_data_ensemble(model, self.var_type, leadtime)
            if fcst_data is None: return None, None
            
            if 'number' in fcst_data.dims:
                fcst_data = fcst_data.mean(dim='number')
                
            fcst_data = fcst_data.resample(time='1MS').mean()
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))
            
            # 对齐
            common_times = obs_data.time.to_index().intersection(fcst_data.time.to_index())
            if len(common_times) < 12: return None, None
            
            obs_aligned = obs_data.sel(time=common_times)
            fcst_aligned = fcst_data.sel(time=common_times)
            
            # 插值
            fcst_interp = fcst_aligned.interp(lat=obs_aligned.lat, lon=obs_aligned.lon, method='linear')
            
            # 计算距平
            obs_clim = obs_aligned.groupby('time.month').mean('time')
            obs_anom = obs_aligned.groupby('time.month') - obs_clim
            
            fcst_clim = fcst_interp.groupby('time.month').mean('time')
            fcst_anom = fcst_interp.groupby('time.month') - fcst_clim
            
            return obs_anom, fcst_anom
        except Exception as e:
            logger.error(f"Data loading failed {model} L{leadtime}: {e}")
            return None, None

    def compute_metrics_task(self, model, leadtime, seasons):
        """计算任务"""
        obs_anom, fcst_anom = self.load_and_preprocess(model, leadtime)
        if obs_anom is None: return None
        
        results = {} 
        season_list = ['annual'] + list(seasons) if seasons else ['annual']
        
        for season in season_list:
            results[season] = {}
            
            # 时间筛选
            if season == 'annual':
                obs_season = obs_anom
                fcst_season = fcst_anom
            elif season in SEASONS:
                months = SEASONS[season]
                mask = obs_anom.time.dt.month.isin(months)
                obs_season = obs_anom.sel(time=mask)
                fcst_season = fcst_anom.sel(time=mask)
            else:
                continue
            
            if obs_season.size == 0: continue

            for reg_name, reg_bounds in REGIONS.items():
                metrics = calculate_weighted_metrics(obs_season, fcst_season, reg_bounds)
                results[season][reg_name] = metrics
                
        return (model, leadtime, results)

    def run_analysis(self, models, leadtimes, seasons, parallel=True):
        tasks = [(m, lt, seasons) for lt in leadtimes for m in models]
        final_data = {}
        
        logger.info(f"Start Taylor analysis for {self.var_type}...")
        
        if parallel:
            max_workers = min(self.n_jobs or cpu_count(), 32)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.compute_metrics_task, *t): t for t in tasks}
                for i, future in enumerate(as_completed(futures)):
                    res = future.result()
                    if res: self._organize_results(final_data, res)
                    if (i+1) % 10 == 0: logger.info(f"Progress: {i+1}/{len(tasks)}")
        else:
            for t in tasks:
                res = self.compute_metrics_task(*t)
                if res: self._organize_results(final_data, res)

        self.save_metrics(final_data)
        self.plot_all(final_data)

    def _organize_results(self, final_data, res):
        model, leadtime, season_results = res
        for season, reg_dict in season_results.items():
            if season not in final_data: final_data[season] = {}
            if leadtime not in final_data[season]: final_data[season][leadtime] = {}
            
            for reg, metrics in reg_dict.items():
                if reg not in final_data[season][leadtime]: 
                    final_data[season][leadtime][reg] = {}
                final_data[season][leadtime][reg][model] = metrics

    def save_metrics(self, data):
        for season, lt_dict in data.items():
            rows = []
            for lt, reg_dict in lt_dict.items():
                for reg, mod_dict in reg_dict.items():
                    for mod, met in mod_dict.items():
                        rows.append({
                            'leadtime': lt,
                            'region': reg,
                            'model': mod,
                            'corr': met['corr'],
                            'std_ratio': met['std_ratio'],
                            'crmse': met['crmse']
                        })
            if not rows: continue
            df = pd.DataFrame(rows)
            ds = df.set_index(['leadtime', 'region', 'model']).to_xarray()
            out_file = self.data_dir / f"metrics_taylor_{self.var_type}_{season}.nc"
            ds.to_netcdf(out_file)
            logger.info(f"Saved metrics: {out_file}")

    def plot_all(self, data):
        for season, lt_data in data.items():
            for lt, reg_data in lt_data.items():
                self.plot_combined_taylor(season, lt, reg_data)

    def plot_combined_taylor(self, season, leadtime, reg_data):
        regions_ordered = ['Global'] + [
            'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
            'Z4-Tibetan',   'Z5-NorthChina',    'Z6-Yangtze',
            'Z7-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
        ]
        
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3, top=0.92, bottom=0.05)
        
        valid_plot = False
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X']
        model_style = {}
        
        for i, reg_name in enumerate(regions_ordered):
            if reg_name not in reg_data: continue
            valid_plot = True
            
            # TaylorDiagram (rect=[left, bottom, width, height] or integer)
            # GridSpec gives us axes, but TaylorDiagram creates its own FloatingSubplot.
            # We need to compute rect from GridSpec slot.
            
            ax_spec = gs[i // 3, i % 3]
            # Use matplotlib layout mechanism to place TaylorDiagram
            # This is tricky with FloatingAxes. A robust way is to pass the subplotspec or a new figure.
            # Here we pass 'rect' as the subplot position in a slightly simpler way:
            # We create the TaylorDiagram on the figure.
            
            # Note: TaylorDiagram implementation expects 'rect' to be 3-digit int or a list.
            # Since we have 12 plots, we can't use 3-digit int easily for all.
            # We will use the GridSpec to get the bounding box.
            
            # IMPORTANT: FloatingSubplot is complex to use with GridSpec directly.
            # Simpler alternative: Create the TaylorDiagram as a standalone subplot
            # occupying the position.
            
            # Let's use the layout engine to determine position
            bbox = ax_spec.get_position(fig)
            rect = [bbox.x0, bbox.y0, bbox.width, bbox.height]
            
            td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.6))
            td.add_contours(levels=5, colors='gray', alpha=0.4) # Add RMSE contours
            
            metrics_dict = reg_data[reg_name]
            for model, met in metrics_dict.items():
                if model not in model_style:
                    model_style[model] = (colors[len(model_style) % 10], markers[len(model_style) % 10])
                c, m = model_style[model]
                
                if np.isfinite(met['corr']) and np.isfinite(met['std_ratio']):
                    td.add_sample(met['std_ratio'], met['corr'], 
                                  color=c, marker=m, markersize=10, 
                                  label=model if i == 0 else "_nolegend_")
            
            # Title needs to be added to the figure or axis
            # td._ax is the underlying Axes
            td._ax.set_title(f"{reg_name}", fontsize=14, fontweight='bold', y=1.05)

        if not valid_plot: 
            plt.close(fig)
            return

        handles = []
        labels = []
        for model, (c, m) in model_style.items():
            h = plt.Line2D([0], [0], color=c, marker=m, linestyle='', markersize=10)
            handles.append(h)
            display_name = model.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
            labels.append(display_name)
            
        fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), 
                   bbox_to_anchor=(0.5, 0.01), fontsize=14, frameon=True)
        
        fig.suptitle(f"Space-Time Taylor Diagram ({self.var_type.upper()}) - {season} L{leadtime}", 
                     fontsize=20, fontweight='bold', y=0.97)
        
        out_file = self.plot_dir / f"taylor_combined_{self.var_type}_{season}_L{leadtime}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot saved: {out_file}")

def main():
    # 关键修复: include_seasons=True
    parser = create_parser(
        description="Taylor Diagram Analysis (Space-Time Patterns)", 
        include_seasons=True
    )
    args = parser.parse_args()
    
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    seasons = []
    if args.seasons:
        if 'all' in args.seasons:
            seasons = list(SEASONS.keys())
        else:
            seasons = [s for s in args.seasons if s in SEASONS]
    
    parallel = normalize_parallel_args(args)
    
    for var in var_list:
        analyzer = TaylorAnalyzer(var, n_jobs=args.n_jobs)
        analyzer.run_analysis(models, leadtimes, seasons, parallel=parallel)

if __name__ == "__main__":
    main()
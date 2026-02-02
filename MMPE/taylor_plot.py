#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模式Taylor图分析模块 (Fixed V5 - Split Leadtime & Season Markers)
修改说明：
1. 绘图重构：
   - 不同的 Lead Time 保存为不同的图像文件 (Files separated by Lead Time).
   - 同一张图中：
     * 颜色 (Color) -> 区分 Models
     * 点型 (Marker) -> 区分 Seasons (Annual, DJF, MAM, JJA, SON)
2. 数据计算与保存：
   - 计算 Annual, Seasonal, 以及 Monthly (Jan-Dec) 的指标。
   - 保存为包含 'season' 维度的 NetCDF 格式，单文件 metrics_taylor_{var}_all.nc。
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import calendar

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
    log_file='taylor_plot_v5.log',
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

# 定义绘图用的季节 Marker
SEASON_MARKERS = {
    'Annual': '*',
    'DJF': 'o',
    'MAM': '^',
    'JJA': 's',
    'SON': 'D'
}

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
        ax.axis["top"].label.set_fontsize(16)
        ax.axis["top"].major_ticklabels.set_fontsize(16)

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation (Normalized)")
        ax.axis["left"].label.set_fontsize(16)
        ax.axis["left"].major_ticklabels.set_fontsize(16)

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].major_ticklabels.set_fontsize(16)

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
# 指标计算逻辑 (保持不变)
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

            fcst_data = self.data_loader.load_forecast_data_ensemble(model, self.var_type, leadtime)
            if fcst_data is None: return None, None

            if 'number' in fcst_data.dims:
                fcst_data = fcst_data.mean(dim='number')

            fcst_data = fcst_data.resample(time='1MS').mean()
            fcst_data = fcst_data.sel(time=slice('1993', '2020'))

            common_times = obs_data.time.to_index().intersection(fcst_data.time.to_index())
            if len(common_times) < 12: return None, None

            obs_aligned = obs_data.sel(time=common_times)
            fcst_aligned = fcst_data.sel(time=common_times)

            fcst_interp = fcst_aligned.interp(lat=obs_aligned.lat, lon=obs_aligned.lon, method='linear')

            obs_clim = obs_aligned.groupby('time.month').mean('time')
            obs_anom = obs_aligned.groupby('time.month') - obs_clim

            fcst_clim = fcst_interp.groupby('time.month').mean('time')
            fcst_anom = fcst_interp.groupby('time.month') - fcst_clim

            return obs_anom, fcst_anom
        except Exception as e:
            logger.error(f"Data loading failed {model} L{leadtime}: {e}")
            return None, None

    def _calc_metrics_for_mask(self, obs, fcst, mask):
        """对指定时间 mask 计算所有区域的指标"""
        if isinstance(mask, slice):
            o_sub = obs
            f_sub = fcst
        else:
            o_sub = obs.sel(time=mask)
            f_sub = fcst.sel(time=mask)

        if o_sub.size == 0: return None

        reg_res = {}
        for reg_name, reg_bounds in REGIONS.items():
            reg_res[reg_name] = calculate_weighted_metrics(o_sub, f_sub, reg_bounds)
        return reg_res

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

        logger.info(f"Start Taylor analysis for {self.var_type} (Annual + Seasons + Months)...")

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
        for period, reg_dict in period_results.items():
            final_data[leadtime][model][period] = reg_dict

    def save_metrics(self, data):
        """保存所有计算结果到一个 NetCDF 文件"""
        rows = []
        for lt, mod_dict in data.items():
            for mod, period_dict in mod_dict.items():
                for period, reg_dict in period_dict.items():
                    for reg, met in reg_dict.items():
                        rows.append({
                            'leadtime': lt,
                            'model': mod,
                            'season': period,
                            'region': reg,
                            'corr': met['corr'],
                            'std_ratio': met['std_ratio'],
                            'crmse': met['crmse']
                        })
        if not rows: return

        df = pd.DataFrame(rows)
        ds = df.set_index(['leadtime', 'model', 'season', 'region']).to_xarray()
        out_file = self.data_dir / f"metrics_taylor_{self.var_type}_all.nc"
        ds.to_netcdf(out_file)
        logger.info(f"Saved all metrics to: {out_file}")

    def load_metrics(self):
        """加载已保存的指标数据（单文件）"""
        out_file = self.data_dir / f"metrics_taylor_{self.var_type}_all.nc"
        if not out_file.exists():
            return None
        return xr.open_dataset(out_file)

    # =========================================================================
    # 绘图逻辑：按 Lead Time 分图，颜色=Model，点型=Season
    # =========================================================================
    def plot_all(self, data, leadtimes, models):
        """data: xarray Dataset 或 dict [leadtime][model][season][region]"""
        if isinstance(data, dict):
            rows = []
            for lt, mod_dict in data.items():
                for mod, period_dict in mod_dict.items():
                    for period, reg_dict in period_dict.items():
                        for reg, met in reg_dict.items():
                            rows.append({
                                'leadtime': lt, 'model': mod, 'season': period, 'region': reg,
                                'corr': met['corr'], 'std_ratio': met['std_ratio'], 'crmse': met['crmse']
                            })
            df = pd.DataFrame(rows)
            ds = df.set_index(['leadtime', 'model', 'season', 'region']).to_xarray()
        else:
            ds = data

        if len(models) <= 10:
            color_list = COLORS if len(COLORS) >= len(models) else plt.cm.tab10.colors
        else:
            color_list = plt.cm.tab20.colors
        model_colors = {m: color_list[i % len(color_list)] for i, m in enumerate(models)}

        plot_seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
        available_lts = [int(x) for x in ds.leadtime.values]

        for lt in leadtimes:
            if lt not in available_lts: continue
            try:
                ds_lt = ds.sel(leadtime=lt)
            except Exception:
                continue
            self.plot_single_leadtime_global(lt, ds_lt, models, plot_seasons, model_colors)
            self.plot_single_leadtime_regions(lt, ds_lt, models, plot_seasons, model_colors)

    def _create_legend(self, fig, model_colors, season_markers):
        """创建组合图例：Color=Model, Marker=Season"""
        handles = []
        labels = []
        handles.append(Line2D([0], [0], color='w', label='[ Models ]', marker=None))
        labels.append(r"$\bf{Models}$")
        for m_name, color in model_colors.items():
            display_name = m_name.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
            h = Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8)
            handles.append(h)
            labels.append(display_name)
        handles.append(Line2D([0], [0], color='w', label=' ', marker=None))
        labels.append(" ")
        handles.append(Line2D([0], [0], color='w', label='[ Seasons ]', marker=None))
        labels.append(r"$\bf{Seasons}$")
        for s_name in ['Annual', 'DJF', 'MAM', 'JJA', 'SON']:
            marker = SEASON_MARKERS.get(s_name, 'o')
            h = Line2D([0], [0], color='k', marker=marker, linestyle='', markersize=8)
            handles.append(h)
            labels.append(s_name)
        fig.legend(handles, labels, loc='lower center', ncol=min(6, len(handles)),
                   bbox_to_anchor=(0.5, 0.02), fontsize=16, frameon=True)

    def plot_single_leadtime_global(self, leadtime, ds_data, models, seasons, model_colors):
        """绘制单个 Leadtime 的 Global 图"""
        reg_name = 'Global'
        fig = plt.figure(figsize=(10, 10))
        rect = [0.1, 0.2, 0.8, 0.7]
        td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.6))
        td.add_contours(levels=5, colors='gray', alpha=0.4)
        plotted_any = False

        for model in models:
            if model not in ds_data.model.values: continue
            color = model_colors.get(model, 'k')
            for season in seasons:
                if season not in ds_data.season.values: continue
                try:
                    corr = float(ds_data['corr'].sel(region=reg_name, model=model, season=season).values)
                    std_ratio = float(ds_data['std_ratio'].sel(region=reg_name, model=model, season=season).values)
                    if np.isfinite(corr) and np.isfinite(std_ratio):
                        marker = SEASON_MARKERS.get(season, 'o')
                        td.add_sample(std_ratio, corr, color=color, marker=marker, markersize=12, label="_nolegend_")
                        plotted_any = True
                except (KeyError, TypeError):
                    continue

        if not plotted_any:
            plt.close(fig)
            return
        
        # 不需要图片标题
        # td._ax.set_title(f"Global - {self.var_type.upper()} (Lead Time {leadtime})", fontsize=20, fontweight='bold', y=1.08)
        self._create_legend(fig, model_colors, SEASON_MARKERS)
        out_file = self.plot_dir / f"taylor_Global_{self.var_type}_L{leadtime}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Global plot saved: {out_file}")

    def plot_single_leadtime_regions(self, leadtime, ds_data, models, seasons, model_colors):
        """绘制单个 Leadtime 的 Regions 九宫格"""
        regions_ordered = [
            'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
            'Z4-Tibetan', 'Z5-NorthChina', 'Z6-Yangtze',
            'Z7-Southwest', 'Z8-SouthChina', 'Z9-SouthSea'
        ]
        fig = plt.figure(figsize=(20, 19))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3, top=0.92, bottom=0.15)
        plotted_any = False

        for i, reg_name in enumerate(regions_ordered):
            ax_spec = gs[i // 3, i % 3]
            bbox = ax_spec.get_position(fig)
            rect = [bbox.x0, bbox.y0, bbox.width, bbox.height]
            td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.6))
            td.add_contours(levels=5, colors='gray', alpha=0.4)
            td._ax.set_title(f"{reg_name}", fontsize=18, fontweight='bold', y=1.05)

            for model in models:
                if model not in ds_data.model.values: continue
                color = model_colors.get(model, 'k')
                for season in seasons:
                    if season not in ds_data.season.values: continue
                    try:
                        corr = float(ds_data['corr'].sel(region=reg_name, model=model, season=season).values)
                        std_ratio = float(ds_data['std_ratio'].sel(region=reg_name, model=model, season=season).values)
                        if np.isfinite(corr) and np.isfinite(std_ratio):
                            marker = SEASON_MARKERS.get(season, 'o')
                            td.add_sample(std_ratio, corr, color=color, marker=marker, markersize=10, label="_nolegend_")
                            plotted_any = True
                    except (KeyError, TypeError):
                        continue

        if not plotted_any:
            plt.close(fig)
            return
        
        # 不需要图片标题
        # fig.suptitle(f"Regional Performance - {self.var_type.upper()} (Lead Time {leadtime})", fontsize=24, fontweight='bold', y=0.96)
        self._create_legend(fig, model_colors, SEASON_MARKERS)
        out_file = self.plot_dir / f"taylor_Regions_{self.var_type}_L{leadtime}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Regions plot saved: {out_file}")

def main():
    parser = create_parser(
        description="Taylor图分析（按 Lead Time 分图，点型=季节）",
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
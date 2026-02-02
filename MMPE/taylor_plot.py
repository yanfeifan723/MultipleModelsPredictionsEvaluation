#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模式Taylor图分析模块 (Fixed V4 - Global/Regions分离 & 多Leadtime同图)
修改说明：
1. 绘图重构：将 Global 和 Regions 分离为两个独立的绘图函数。
2. 多维展示：在同一张图中展示所有 Lead Times。
   - 颜色 (Color) -> 区分 Models
   - 点型 (Marker) -> 区分 Lead Times
3. 图例增强：分别展示 Model 颜色对应关系和 LeadTime 点型对应关系。
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

# 添加 toolkit 路径
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
    log_file='taylor_plot_v4.log',
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
        # 保持原有数据加载逻辑不变
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

    def compute_metrics_task(self, model, leadtime, seasons):
        # 保持原有计算任务逻辑不变
        obs_anom, fcst_anom = self.load_and_preprocess(model, leadtime)
        if obs_anom is None: return None
        
        results = {} 
        season_list = ['annual'] + list(seasons) if seasons else ['annual']
        
        for season in season_list:
            results[season] = {}
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

    def run_analysis(self, models, leadtimes, seasons, parallel=True, plot_only=False, no_plot=False):
        if plot_only:
            final_data, models_loaded, leadtimes_loaded = self.load_metrics(seasons)
            if not final_data:
                logger.warning("Plot-only: no saved metrics found, skip plotting.")
                return
            logger.info(f"Plot-only: loaded metrics for {self.var_type}, seasons={list(final_data.keys())}")
            self.plot_all(final_data, leadtimes_loaded, models_loaded)
            return

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

        if not no_plot:
            self.plot_all(final_data, leadtimes, models)

    def _organize_results(self, final_data, res):
        model, leadtime, season_results = res
        for season, reg_dict in season_results.items():
            if season not in final_data: final_data[season] = {}
            # 注意结构：final_data[season][leadtime][reg][model]
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

    def load_metrics(self, seasons):
        """从已保存的 NetCDF 加载指标，用于 --plot-only。返回 final_data, models, leadtimes。"""
        final_data = {}
        all_models = set()
        all_leadtimes = set()
        for season in seasons:
            out_file = self.data_dir / f"metrics_taylor_{self.var_type}_{season}.nc"
            if not out_file.exists():
                logger.warning(f"Plot-only: file not found, skip season {season}: {out_file}")
                continue
            ds = xr.open_dataset(out_file)
            final_data[season] = {}
            for lt in ds.coords['leadtime'].values:
                lt_int = int(lt)
                all_leadtimes.add(lt_int)
                final_data[season][lt_int] = {}
                for reg in ds.coords['region'].values:
                    reg_str = str(reg) if not isinstance(reg, str) else reg
                    final_data[season][lt_int][reg_str] = {}
                    for mod in ds.coords['model'].values:
                        mod_str = str(mod) if not isinstance(mod, str) else mod
                        all_models.add(mod_str)
                        final_data[season][lt_int][reg_str][mod_str] = {
                            'corr': float(ds['corr'].sel(leadtime=lt, region=reg, model=mod).values),
                            'std_ratio': float(ds['std_ratio'].sel(leadtime=lt, region=reg, model=mod).values),
                            'crmse': float(ds['crmse'].sel(leadtime=lt, region=reg, model=mod).values)
                        }
            ds.close()
        models = sorted(all_models)
        leadtimes = sorted(all_leadtimes)
        return final_data, models, leadtimes

    # =========================================================================
    # 新的绘图逻辑部分
    # =========================================================================
    def plot_all(self, data, leadtimes, models):
        """
        data: final_data[season][leadtime][reg][model]
        """
        # 准备样式
        # 1. Models 用颜色区分
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]
        model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}
        
        # 2. Leadtimes 用点型区分
        # 常用点型: 圆圈, 正方形, 三角形, 钻石, 倒三角, 左三角, 右三角, 五边形, 星形, X
        markers_list = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X']
        # 对 leadtimes 进行排序以保证图例顺序一致
        sorted_lts = sorted(leadtimes)
        lt_markers = {lt: markers_list[i % len(markers_list)] for i, lt in enumerate(sorted_lts)}

        for season, lt_data in data.items():
            # 这里我们需要重组数据以便于绘图：按 [Region] -> [Leadtime] -> [Model] 访问
            # 但原始结构是 [Leadtime][Region][Model]
            # 为了方便，我们在绘图函数内部遍历
            
            # 1. 绘制 Global 单张图
            self.plot_global_only(season, data[season], model_colors, lt_markers)
            
            # 2. 绘制 Regions 九宫格
            self.plot_regions_grid(season, data[season], model_colors, lt_markers)

    def _create_common_legend(self, fig, model_colors, lt_markers):
        """创建组合图例：分两列，一列Models，一列Leadtimes"""
        handles = []
        labels = []
        
        # Model Legend (Color only, invisible marker or simple dot)
        handles.append(Line2D([0], [0], color='w', label='[ Models ]', marker=None)) # Title
        labels.append(r"$\bf{Models}$")
        
        for m_name, color in model_colors.items():
            display_name = m_name.replace('-mon', '').replace('mon-', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC')
            h = Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8)
            handles.append(h)
            labels.append(display_name)
            
        # Spacer
        handles.append(Line2D([0], [0], color='w', label=' ', marker=None))
        labels.append(" ")
        
        # Leadtime Legend (Black color, varies marker)
        handles.append(Line2D([0], [0], color='w', label='[ Lead Times ]', marker=None))
        labels.append(r"$\bf{Lead\ Times}$")
        
        for lt, marker in lt_markers.items():
            h = Line2D([0], [0], color='k', marker=marker, linestyle='', markersize=8)
            handles.append(h)
            labels.append(f"Lead {lt}")

        # 将图例放在底部，自动换行
        # 为了美观，可以分两个Legend，或者计算好列数
        # 这里使用简单的一行多列流式布局
        fig.legend(handles, labels, loc='lower center', ncol=min(6, len(handles)), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)

    def plot_global_only(self, season, season_data, model_colors, lt_markers):
        """单独绘制 Global 区域"""
        reg_name = 'Global'
        
        fig = plt.figure(figsize=(10, 10))
        # 留出底部给图例
        rect = [0.1, 0.2, 0.8, 0.7] 
        
        td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.6))
        td.add_contours(levels=5, colors='gray', alpha=0.4)
        
        plotted_any = False
        
        # season_data 结构: {leadtime: {region: {model: metrics}}}
        for lt, reg_dict in season_data.items():
            if reg_name not in reg_dict: continue
            
            marker = lt_markers.get(lt, 'o')
            metrics_dict = reg_dict[reg_name]
            
            for model, met in metrics_dict.items():
                color = model_colors.get(model, 'k')
                
                if np.isfinite(met['corr']) and np.isfinite(met['std_ratio']):
                    td.add_sample(met['std_ratio'], met['corr'], 
                                  color=color, marker=marker, markersize=12, 
                                  label="_nolegend_") # 图例单独画
                    plotted_any = True

        if not plotted_any:
            plt.close(fig)
            return

        # td._ax.set_title(f"Global - {self.var_type.upper()} ({season})", fontsize=18, fontweight='bold', y=1.08)
        
        self._create_common_legend(fig, model_colors, lt_markers)
        
        out_file = self.plot_dir / f"taylor_Global_{self.var_type}_{season}_multilead.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Global plot saved: {out_file}")

    def plot_regions_grid(self, season, season_data, model_colors, lt_markers):
        """绘制除 Global 外的区域 (九宫格)"""
        regions_ordered = [
            'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
            'Z4-Tibetan',   'Z5-NorthChina',    'Z6-Yangtze',
            'Z7-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
        ]
        
        fig = plt.figure(figsize=(20, 23))
        # 调整底部空间以容纳更大的图例
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3, top=0.92, bottom=0.15)
        
        plotted_any = False
        
        for i, reg_name in enumerate(regions_ordered):
            # 获取 subplot 位置
            ax_spec = gs[i // 3, i % 3]
            bbox = ax_spec.get_position(fig)
            rect = [bbox.x0, bbox.y0, bbox.width, bbox.height]
            
            td = TaylorDiagram(refstd=1.0, fig=fig, rect=rect, label='Obs', srange=(0, 1.6))
            td.add_contours(levels=5, colors='gray', alpha=0.4)
            td._ax.set_title(f"{reg_name}", fontsize=14, fontweight='bold', y=1.05)
            
            # 遍历所有 Leadtime 和 Model
            for lt, reg_dict in season_data.items():
                if reg_name not in reg_dict: continue
                
                marker = lt_markers.get(lt, 'o')
                metrics_dict = reg_dict[reg_name]
                
                for model, met in metrics_dict.items():
                    color = model_colors.get(model, 'k')
                    
                    if np.isfinite(met['corr']) and np.isfinite(met['std_ratio']):
                        td.add_sample(met['std_ratio'], met['corr'], 
                                      color=color, marker=marker, markersize=10, 
                                      label="_nolegend_")
                        plotted_any = True
        
        if not plotted_any:
            plt.close(fig)
            return

        # fig.suptitle(f"Regional Performance ({self.var_type.upper()}) - {season}", 
        #              fontsize=24, fontweight='bold', y=0.96)
        
        self._create_common_legend(fig, model_colors, lt_markers)
        
        out_file = self.plot_dir / f"taylor_Regions_{self.var_type}_{season}_multilead.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Regions plot saved: {out_file}")

def main():
    parser = create_parser(
        description="Taylor图分析（空间-时间型）",
        include_seasons=True,
        var_default=None,
        var_required=False
    )
    args = parser.parse_args()

    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_list = parse_vars(args.var) if args.var else ['temp', 'prec']

    seasons = []
    if getattr(args, 'all_seasons', False):
        seasons = list(SEASONS.keys())
    elif getattr(args, 'seasons', None):
        if 'all' in args.seasons:
            seasons = list(SEASONS.keys())
        else:
            seasons = [s for s in args.seasons if s in SEASONS]
    if not seasons:
        seasons = list(SEASONS.keys())

    parallel = normalize_parallel_args(args) or (args.n_jobs is not None and args.n_jobs > 1)

    for var in var_list:
        analyzer = TaylorAnalyzer(var, n_jobs=args.n_jobs)
        analyzer.run_analysis(
            models=models,
            leadtimes=leadtimes,
            seasons=seasons,
            parallel=parallel,
            plot_only=args.plot_only,
            no_plot=args.no_plot
        )

if __name__ == "__main__":
    main()
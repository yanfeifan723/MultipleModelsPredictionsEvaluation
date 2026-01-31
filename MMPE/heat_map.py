#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多区域双重指标(RMSE & ACC)组合热图绘制模块 (Updated Layout V3 - Fixed)
功能：
1. 绘制 Global 区域的 2x2 组合图
2. 绘制 9个子区域(Z1-Z9) 的组合大图
3. 布局调整：
   - 去除多余标题
   - 极度紧凑的间距
   - 增大显著性打点 (s=25)
   - 模式名称倾斜 45 度
   - 修复 color bins 数量错误
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib import ticker
from typing import Dict, List

# 添加路径以导入配置
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from common_config import SEASONS, LEADTIMES, MODEL_LIST
except ImportError:
    SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    LEADTIMES = [0, 1, 2, 3, 4, 5]
    MODEL_LIST = ["CMCC-35", "DWD-mon-21", "ECMWF-51-mon", "Meteo-France-8", "NCEP-2", "UKMO-14", "ECCC-Canada-3"]

# 区域定义
REGION_GLOBAL = 'Global'
REGIONS_Z = [
    'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
    'Z4-Tibetan',   'Z5-NorthChina',    'Z7-Yangtze',
    'Z6-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
]

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RMSE/ACC 分段数
N_RMSE_LEVELS = 11
N_ACC_LEVELS = 11

class RegionalHeatMapPlotter:
    def __init__(self, var_type: str):
        self.var_type = var_type
        self.acc_base_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/region_index_acc/{var_type}")
        self.rmse_base_dir = Path(f"/sas12t1/ffyan/output/error_analysis/region_metrics/{var_type}")
        self.output_dir = Path(f"/sas12t1/ffyan/output/heat_map_regional/{var_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_regional_data(self, region: str) -> Dict:
        """加载数据"""
        data = {}
        safe_region = region.replace(' ', '_')

        for lt in LEADTIMES:
            data[lt] = {'rmse': {}, 'acc': {}}

            for model in MODEL_LIST:
                # 1. 加载 ACC
                acc_file = self.acc_base_dir / f"region_index_acc_{safe_region}_{model}_{self.var_type}.nc"
                acc_entry = {'monthly': {}, 'seasonal': {}, 'monthly_p': {}, 'seasonal_p': {}}
                
                if acc_file.exists():
                    try:
                        with xr.open_dataset(acc_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                for m in range(1, 13):
                                    m_name = MONTHS[m-1]
                                    if m in ds_lt.month.values:
                                        val = float(ds_lt.regional_index_acc.sel(month=m).values)
                                        p_val = np.nan
                                        if 'p_value' in ds_lt:
                                            p_val = float(ds_lt.p_value.sel(month=m).values)
                                        acc_entry['monthly'][m_name] = val
                                        acc_entry['monthly_p'][m_name] = p_val
                                    else:
                                        acc_entry['monthly'][m_name] = np.nan
                                        acc_entry['monthly_p'][m_name] = np.nan
                                for seas, m_idxs in SEASONS.items():
                                    vals = [acc_entry['monthly'].get(MONTHS[m-1], np.nan) for m in m_idxs]
                                    acc_entry['seasonal'][seas] = float(np.nanmean(vals))
                                    acc_entry['seasonal_p'][seas] = np.nan 
                                data[lt]['acc'][model] = acc_entry
                    except Exception as e:
                        pass

                # 2. 加载 RMSE
                rmse_file = self.rmse_base_dir / f"{region}_{model}.nc"
                rmse_entry = {'monthly': {}, 'seasonal': {}}
                
                if rmse_file.exists():
                    try:
                        with xr.open_dataset(rmse_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                if 'rmse_monthly' in ds_lt:
                                    da = ds_lt.rmse_monthly
                                    month_coord = da.coords.get('month', getattr(ds_lt, 'month', list(range(1, 13))))
                                    for m in range(1, 13):
                                        if m in month_coord:
                                            val = da.sel(month=m).values
                                            rmse_entry['monthly'][MONTHS[m-1]] = float(np.asarray(val).ravel()[0])
                                        else:
                                            rmse_entry['monthly'][MONTHS[m-1]] = np.nan
                                else:
                                    for m in MONTHS: rmse_entry['monthly'][m] = np.nan

                                if 'rmse_seasonal' in ds_lt and 'season' in ds_lt.dims:
                                    for s in SEASONS.keys():
                                        if s in ds_lt.season.values:
                                            val = ds_lt.rmse_seasonal.sel(season=s).values
                                            rmse_entry['seasonal'][s] = float(np.asarray(val).ravel()[0])
                                        else:
                                            rmse_entry['seasonal'][s] = np.nan
                                else:
                                    for seas, m_idxs in SEASONS.items():
                                        vals = [rmse_entry['monthly'].get(MONTHS[m-1], np.nan) for m in m_idxs]
                                        rmse_entry['seasonal'][seas] = float(np.nanmean(vals))
                                
                                data[lt]['rmse'][model] = rmse_entry
                    except Exception as e:
                        pass
        return data

    def _get_levels_and_cmap(self, all_vals, vtype='rmse'):
        """
        获取 colorbar 配置，修复 bin 数量问题
        """
        if vtype == 'rmse':
            valid = [x for x in all_vals if np.isfinite(x) and x >= 0]
            vmin = 0
            vmax = np.nanmax(valid) if valid else 1.0
            if vmax > 5: vmax = np.ceil(vmax)
            else: vmax = np.ceil(vmax * 10) / 10
            
            levels = np.linspace(vmin, vmax, N_RMSE_LEVELS)
            n_bins = len(levels) - 1
            # extend='max' 需要 n_bins + 1 个颜色
            n_colors = n_bins + 1
            
            if self.var_type == 'temp': cmap = plt.get_cmap('Reds', n_colors)
            else: cmap = plt.get_cmap('Blues', n_colors)
            norm = BoundaryNorm(levels, n_colors, extend='max')
            return norm, cmap, levels
        else:
            vmin, vmax = -1.0, 1.0
            levels = np.linspace(vmin, vmax, N_ACC_LEVELS)
            n_bins = len(levels) - 1
            # extend='both' 需要 n_bins + 2 个颜色
            n_colors = n_bins + 2 
            
            cmap = plt.get_cmap('coolwarm', n_colors)
            norm = BoundaryNorm(levels, n_colors, extend='both')
            return norm, cmap, levels

    def _plot_single_heatmap(self, ax, data_dict, models, mode, metric, norm, cmap, levels, 
                             show_xticklabels=True, show_yticklabels=True):
        y_labels = MONTHS if mode == 'monthly' else list(SEASONS.keys())
        rows = len(y_labels)
        cols = len(models)
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        
        # 网格线
        for x in range(cols + 1):
            ax.axvline(x, color='k', lw=0.5)
        for y in range(rows + 1):
            ax.axhline(y, color='k', lw=0.5)
            
        for i, row_label in enumerate(y_labels):
            for j, model in enumerate(models):
                entry = data_dict.get(model, {})
                val = np.nan
                p_val = np.nan
                
                if metric == 'rmse':
                    val = entry.get(mode, {}).get(row_label, np.nan)
                else: 
                    val = entry.get(mode, {}).get(row_label, np.nan)
                    if mode == 'monthly':
                        p_val = entry.get('monthly_p', {}).get(row_label, np.nan)
                    else:
                        p_val = entry.get('seasonal_p', {}).get(row_label, np.nan)

                if np.isfinite(val):
                    if metric == 'rmse':
                        val_clip = max(levels[0], min(val, levels[-1]))
                    else:
                        val_clip = max(-1, min(val, 1))
                    
                    color = cmap(norm(val_clip))
                    rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                    ax.add_patch(rect)
                    
                    # 增大打点 s=25
                    if metric == 'acc' and np.isfinite(p_val) and p_val < 0.05:
                        ax.scatter(j + 0.5, i + 0.5, s=25, c='black', marker='o', linewidths=0)

        # 轴标签处理
        ax.set_xticks(np.arange(cols) + 0.5)
        if show_xticklabels:
            ax.set_xticklabels([m.replace('-mon', '').replace('Meteo-France', 'MF') for m in models], 
                               rotation=45, ha='right', fontsize=10)
        else:
            ax.set_xticklabels([])
            
        ax.set_yticks(np.arange(rows) + 0.5)
        if show_yticklabels:
            ax.set_yticklabels(y_labels, fontsize=10)
        else:
            ax.set_yticklabels([])
            
        ax.tick_params(axis='both', which='both', length=0)

    def plot_global_figure(self, leadtime: int, global_data: Dict):
        models = [m for m in MODEL_LIST if m in global_data['rmse']]
        if not models: return

        # Global 图 14x12
        fig = plt.figure(figsize=(14, 12))
        # 紧凑间距
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.08)
        
        rmse_vals = []
        acc_vals = []
        for m in models:
            rmse_vals.extend(global_data['rmse'][m].get('monthly', {}).values())
            rmse_vals.extend(global_data['rmse'][m].get('seasonal', {}).values())
            acc_vals.extend(global_data['acc'][m].get('monthly', {}).values())
            acc_vals.extend(global_data['acc'][m].get('seasonal', {}).values())
            
        norm_r, cmap_r, levels_r = self._get_levels_and_cmap(rmse_vals, 'rmse')
        norm_a, cmap_a, levels_a = self._get_levels_and_cmap(acc_vals, 'acc')
        
        layouts = [
            (0, 0, 'monthly', 'rmse', norm_r, cmap_r, levels_r),
            (0, 1, 'monthly', 'acc', norm_a, cmap_a, levels_a),
            (1, 0, 'seasonal', 'rmse', norm_r, cmap_r, levels_r),
            (1, 1, 'seasonal', 'acc', norm_a, cmap_a, levels_a)
        ]
        
        for r, c, mode, metric, norm, cmap, levels in layouts:
            ax = fig.add_subplot(gs[r, c])
            # Global图: 底部行显示X轴，左侧列显示Y轴
            show_x = (r == 1)
            show_y = (c == 0)
            
            self._plot_single_heatmap(ax, global_data[metric], models, mode, metric, norm, cmap, levels, 
                                      show_xticklabels=show_x, show_yticklabels=show_y)
            # 无标题
            
        # Colorbars
        cax_r = fig.add_axes([0.15, 0.05, 0.3, 0.02])
        cb_r = plt.colorbar(ScalarMappable(norm=norm_r, cmap=cmap_r), cax=cax_r, orientation='horizontal', extend='max')
        cb_r.set_ticks(levels_r)
        cb_r.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cb_r.set_label('RMSE', fontsize=12)
        
        cax_a = fig.add_axes([0.55, 0.05, 0.3, 0.02])
        cb_a = plt.colorbar(ScalarMappable(norm=norm_a, cmap=cmap_a), cax=cax_a, orientation='horizontal', extend='both')
        cb_a.set_ticks(levels_a)
        cb_a.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cb_a.set_label('ACC', fontsize=12)
        
        fname = f"Global_{self.var_type}_L{leadtime}.png"
        plt.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fname}")

    def plot_regions_figure(self, leadtime: int, regions_data: Dict[str, Dict]):
        models = MODEL_LIST
        
        all_rmse = []
        all_acc = []
        for reg_d in regions_data.values():
            for m in models:
                if m in reg_d['rmse']:
                    all_rmse.extend(reg_d['rmse'][m].get('monthly', {}).values())
                    all_rmse.extend(reg_d['rmse'][m].get('seasonal', {}).values())
                if m in reg_d['acc']:
                    all_acc.extend(reg_d['acc'][m].get('monthly', {}).values())
                    all_acc.extend(reg_d['acc'][m].get('seasonal', {}).values())
        
        norm_r, cmap_r, levels_r = self._get_levels_and_cmap(all_rmse, 'rmse')
        norm_a, cmap_a, levels_a = self._get_levels_and_cmap(all_acc, 'acc')

        # 区域大图 26x22
        fig = plt.figure(figsize=(26, 22))
        
        # 4大块布局，紧凑
        outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.08, height_ratios=[1, 0.4]) 
        
        quadrants = [
            (0, 0, 'monthly', 'rmse', norm_r, cmap_r, levels_r),
            (0, 1, 'monthly', 'acc', norm_a, cmap_a, levels_a),
            (1, 0, 'seasonal', 'rmse', norm_r, cmap_r, levels_r),
            (1, 1, 'seasonal', 'acc', norm_a, cmap_a, levels_a)
        ]

        region_grid = [
            ['Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast'],
            ['Z4-Tibetan',   'Z5-NorthChina',    'Z7-Yangtze'],
            ['Z6-Southwest', 'Z8-SouthChina',    'Z9-SouthSea']
        ]

        for qr, qc, mode, metric, norm, cmap, levels in quadrants:
            # 内部3x3小图布局，极度紧凑
            inner_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_gs[qr, qc], 
                                                        hspace=0.2, wspace=0.05)
            
            for r in range(3):
                for c in range(3):
                    reg_name = region_grid[r][c]
                    ax = fig.add_subplot(inner_gs[r, c])
                    
                    data_to_plot = regions_data.get(reg_name, {}).get(metric, {})
                    
                    # 仅在最底部的行 (row=2) 显示 X轴标签
                    show_x = (r == 2)
                    # 仅在最左侧的列 (col=0) 显示 Y轴标签
                    show_y = (c == 0)
                    
                    self._plot_single_heatmap(ax, data_to_plot, models, mode, metric, norm, cmap, levels,
                                              show_xticklabels=show_x, show_yticklabels=show_y)
                    
                    # 仅保留 Z1~Z9 标注
                    short_name = reg_name.split('-')[0] # Z1, Z2...
                    ax.set_title(short_name, fontsize=11, pad=3, fontweight='bold')

        # Colorbars
        cax_r = fig.add_axes([0.15, 0.05, 0.3, 0.015])
        cb_r = plt.colorbar(ScalarMappable(norm=norm_r, cmap=cmap_r), cax=cax_r, orientation='horizontal', extend='max')
        cb_r.set_ticks(levels_r)
        cb_r.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cb_r.set_label('RMSE', fontsize=14)
        
        cax_a = fig.add_axes([0.55, 0.05, 0.3, 0.015])
        cb_a = plt.colorbar(ScalarMappable(norm=norm_a, cmap=cmap_a), cax=cax_a, orientation='horizontal', extend='both')
        cb_a.set_ticks(levels_a)
        cb_a.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cb_a.set_label('ACC', fontsize=14)

        fname = f"Regions_Combined_{self.var_type}_L{leadtime}.png"
        plt.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fname}")

    def run(self):
        for lt in LEADTIMES:
            logger.info(f"Processing Leadtime: {lt}")
            global_data = self.load_regional_data(REGION_GLOBAL).get(lt, {})
            if global_data:
                self.plot_global_figure(lt, global_data)
            
            regions_data = {}
            for region in REGIONS_Z:
                r_data = self.load_regional_data(region).get(lt, {})
                regions_data[region] = r_data
            if regions_data:
                self.plot_regions_figure(lt, regions_data)

if __name__ == "__main__":
    for var in ['temp', 'prec']:
        plotter = RegionalHeatMapPlotter(var)
        plotter.run()
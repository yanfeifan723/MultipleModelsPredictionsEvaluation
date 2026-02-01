#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多区域三重指标(Bias, RMSE, ACC)组合热图绘制模块 (Fixed V9)
修复与优化：
1. 移除 MAE 指标绘制。
2. 支持读取并绘制带显著性检验的季节性 ACC (Seasonal ACC & P-value)。
3. 修复 Bias/ACC 在 extend='both' 时的 n_colors 计算错误。
4. 优化 Bias 刻度：使用 MaxNLocator 生成以0为中心的整齐对称刻度。
5. 增大所有刻度字体 (Heatmap axes & Colorbar ticks)。
6. 减小 Regions Colorbar 宽度 (变细)。
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
    'Z4-Tibetan',   'Z5-NorthChina',    'Z6-Yangtze',
    'Z7-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
]

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 目标分段数
N_LEVELS = 11

class RegionalHeatMapPlotter:
    def __init__(self, var_type: str):
        self.var_type = var_type
        self.acc_base_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/region_index_acc/{var_type}")
        self.error_base_dir = Path(f"/sas12t1/ffyan/output/error_analysis/region_metrics/{var_type}")
        self.output_dir = Path(f"/sas12t1/ffyan/output/heat_map_regional/{var_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_regional_data(self, region: str) -> Dict:
        """加载数据 (Bias, RMSE, ACC)"""
        data = {}
        safe_region = region.replace(' ', '_')

        for lt in LEADTIMES:
            # 移除 'mae'
            data[lt] = {'bias': {}, 'rmse': {}, 'acc': {}}

            for model in MODEL_LIST:
                # 1. ACC
                acc_file = self.acc_base_dir / f"region_index_acc_{safe_region}_{model}_{self.var_type}.nc"
                acc_entry = {'monthly': {}, 'seasonal': {}, 'monthly_p': {}, 'seasonal_p': {}}
                
                if acc_file.exists():
                    try:
                        with xr.open_dataset(acc_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                
                                # --- Monthly ACC ---
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
                                
                                # --- Seasonal ACC (尝试读取预计算的季节性 ACC) ---
                                if 'regional_index_acc_seasonal' in ds_lt and 'season' in ds_lt.coords:
                                    # 读取科学计算的季节 ACC 和 P 值
                                    for seas in SEASONS.keys():
                                        if seas in ds_lt.season.values:
                                            val = float(ds_lt.regional_index_acc_seasonal.sel(season=seas).values)
                                            p_val = np.nan
                                            if 'p_value_seasonal' in ds_lt:
                                                p_val = float(ds_lt.p_value_seasonal.sel(season=seas).values)
                                            acc_entry['seasonal'][seas] = val
                                            acc_entry['seasonal_p'][seas] = p_val
                                        else:
                                            acc_entry['seasonal'][seas] = np.nan
                                            acc_entry['seasonal_p'][seas] = np.nan
                                else:
                                    # 回退逻辑：如果没有季节性变量，则使用月平均（无 p-value）
                                    for seas, m_idxs in SEASONS.items():
                                        vals = [acc_entry['monthly'].get(MONTHS[m-1], np.nan) for m in m_idxs]
                                        acc_entry['seasonal'][seas] = float(np.nanmean(vals))
                                        acc_entry['seasonal_p'][seas] = np.nan 
                                
                                data[lt]['acc'][model] = acc_entry
                    except Exception as e:
                        # logger.warning(f"Failed loading ACC for {model} {lt}: {e}")
                        pass

                # 2. Error Metrics (Bias, RMSE) - 移除 MAE
                err_file = self.error_base_dir / f"{region}_{model}.nc"
                bias_entry = {'monthly': {}, 'seasonal': {}}
                rmse_entry = {'monthly': {}, 'seasonal': {}}
                
                if err_file.exists():
                    try:
                        with xr.open_dataset(err_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                
                                def extract_metric(ds_subset, metric_prefix, target_entry):
                                    var_mon = f"{metric_prefix}_monthly"
                                    if var_mon in ds_subset:
                                        da = ds_subset[var_mon]
                                        month_coord = da.coords.get('month', getattr(ds_subset, 'month', list(range(1, 13))))
                                        for m in range(1, 13):
                                            if m in month_coord:
                                                val = da.sel(month=m).values
                                                target_entry['monthly'][MONTHS[m-1]] = float(np.asarray(val).ravel()[0])
                                            else:
                                                target_entry['monthly'][MONTHS[m-1]] = np.nan
                                    else:
                                        for m in MONTHS: target_entry['monthly'][m] = np.nan

                                    var_seas = f"{metric_prefix}_seasonal"
                                    if var_seas in ds_subset and 'season' in ds_subset.dims:
                                        for s in SEASONS.keys():
                                            if s in ds_subset.season.values:
                                                val = ds_subset[var_seas].sel(season=s).values
                                                target_entry['seasonal'][s] = float(np.asarray(val).ravel()[0])
                                            else:
                                                target_entry['seasonal'][s] = np.nan
                                    else:
                                        for seas, m_idxs in SEASONS.items():
                                            vals = [target_entry['monthly'].get(MONTHS[m-1], np.nan) for m in m_idxs]
                                            target_entry['seasonal'][seas] = float(np.nanmean(vals))

                                extract_metric(ds_lt, 'bias', bias_entry)
                                # extract_metric(ds_lt, 'mae', mae_entry) # 移除 MAE
                                extract_metric(ds_lt, 'rmse', rmse_entry)
                                
                                data[lt]['bias'][model] = bias_entry
                                data[lt]['rmse'][model] = rmse_entry
                    except Exception as e:
                        pass
        return data

    def _get_levels_and_cmap(self, all_vals, metric):
        """获取 colorbar 配置"""
        valid = [x for x in all_vals if np.isfinite(x)]
        if not valid:
            valid = [0, 1]
            
        # --- 1. Bias (Diverging) ---
        if metric == 'bias':
            max_abs = np.nanmax(np.abs(valid))
            if max_abs == 0: max_abs = 1.0
            
            locator = ticker.MaxNLocator(nbins=(N_LEVELS-1)//2, steps=[1, 2, 2.5, 5, 10])
            levels_pos = locator.tick_values(0, max_abs)
            levels_pos = levels_pos[levels_pos >= 0]
            
            levels_neg = -levels_pos[1:]
            levels = np.concatenate((levels_neg[::-1], levels_pos))
            
            n_bins = len(levels) - 1
            n_colors = n_bins + 2 
            
            if self.var_type == 'temp':
                cmap = plt.get_cmap('RdBu_r', n_colors)
            else:
                cmap = plt.get_cmap('RdBu', n_colors)
            
            norm = BoundaryNorm(levels, n_colors, extend='both')
            return norm, cmap, levels

        # --- 2. RMSE (Sequential) ---
        elif metric == 'rmse':
            valid_pos = [x for x in valid if x >= 0]
            vmin = 0
            vmax_raw = np.nanmax(valid_pos) if valid_pos else 1.0
            
            locator = ticker.MaxNLocator(nbins=N_LEVELS-1, steps=[1, 2, 2.5, 5, 10])
            levels = locator.tick_values(0, vmax_raw)
            levels = levels[levels >= 0]
            
            if levels[-1] < vmax_raw:
                step = levels[1] - levels[0] if len(levels) > 1 else 1.0
                while levels[-1] < vmax_raw:
                    levels = np.append(levels, levels[-1] + step)
            
            n_colors = len(levels)
            
            if self.var_type == 'temp': cmap = plt.get_cmap('Reds', n_colors)
            else: cmap = plt.get_cmap('Blues', n_colors)
            
            norm = BoundaryNorm(levels, n_colors, extend='max')
            return norm, cmap, levels

        # --- 3. ACC ---
        else:
            vmin, vmax = -1.0, 1.0
            levels = np.linspace(vmin, vmax, N_LEVELS)
            n_bins = len(levels) - 1
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
        
        for x in range(cols + 1):
            ax.axvline(x, color='k', lw=0.5)
        for y in range(rows + 1):
            ax.axhline(y, color='k', lw=0.5)
            
        for i, row_label in enumerate(y_labels):
            for j, model in enumerate(models):
                entry = data_dict.get(model, {})
                
                if metric == 'acc':
                    val = entry.get(mode, {}).get(row_label, np.nan)
                    if mode == 'monthly':
                        p_val = entry.get('monthly_p', {}).get(row_label, np.nan)
                    else:
                        p_val = entry.get('seasonal_p', {}).get(row_label, np.nan)
                else:
                    val = entry.get(mode, {}).get(row_label, np.nan)
                    p_val = np.nan 

                if np.isfinite(val):
                    if metric == 'acc':
                        val_clip = max(-1, min(val, 1))
                    elif metric == 'bias':
                        val_clip = max(levels[0], min(val, levels[-1]))
                    else:
                        val_clip = max(levels[0], min(val, levels[-1]))
                    
                    color = cmap(norm(val_clip))
                    rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                    ax.add_patch(rect)
                    
                    # 显著性打点 (p < 0.05)
                    if metric == 'acc' and np.isfinite(p_val) and p_val < 0.05:
                        ax.scatter(j + 0.5, i + 0.5, s=25, c='black', marker='o', linewidths=0)

        # 轴标签处理 (增大字体)
        ax.set_xticks(np.arange(cols) + 0.5)
        if show_xticklabels:
            ax.set_xticklabels([m.replace('-mon', '').replace('Meteo-France', 'MF').replace('ECCC-Canada', 'ECCC') for m in models], 
                               rotation=45, ha='right', fontsize=14)
        else:
            ax.set_xticklabels([])
            
        ax.set_yticks(np.arange(rows) + 0.5)
        if show_yticklabels:
            ax.set_yticklabels(y_labels, fontsize=14)
        else:
            ax.set_yticklabels([])
            
        ax.tick_params(axis='both', which='both', length=0)

    def plot_global_figure(self, leadtime: int, global_data: Dict):
        models = [m for m in MODEL_LIST if m in global_data['rmse']]
        if not models: return

        # 调整布局：2行 x 3列 (Bias, RMSE, ACC)
        fig = plt.figure(figsize=(18, 12)) # 稍微调窄一点因为少了一列
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05)
        
        # 移除 'mae'
        vals = {m: [] for m in ['bias', 'rmse', 'acc']}
        for met in ['bias', 'rmse', 'acc']:
            for m in models:
                vals[met].extend(global_data[met][m].get('monthly', {}).values())
                vals[met].extend(global_data[met][m].get('seasonal', {}).values())
        
        # 仅使用 RMSE 数据
        norm_rmse, cmap_rmse, levels_rmse = self._get_levels_and_cmap(vals['rmse'], 'rmse')

        configs = {}
        configs['bias'] = self._get_levels_and_cmap(vals['bias'], 'bias')
        configs['acc'] = self._get_levels_and_cmap(vals['acc'], 'acc')
        configs['rmse'] = (norm_rmse, cmap_rmse, levels_rmse)

        metrics_order = ['bias', 'rmse', 'acc']
        titles = {'bias': 'Bias', 'rmse': 'RMSE', 'acc': 'ACC'}
        
        for col, metric in enumerate(metrics_order):
            norm, cmap, levels = configs[metric]
            
            ax_m = fig.add_subplot(gs[0, col])
            self._plot_single_heatmap(ax_m, global_data[metric], models, 'monthly', metric, 
                                      norm, cmap, levels, 
                                      show_xticklabels=False, show_yticklabels=(col==0))
            ax_m.text(0.5, 1.02, titles[metric], transform=ax_m.transAxes, 
                      fontsize=20, fontweight='bold', ha='center', va='bottom')
            
            ax_s = fig.add_subplot(gs[1, col])
            self._plot_single_heatmap(ax_s, global_data[metric], models, 'seasonal', metric, 
                                      norm, cmap, levels, 
                                      show_xticklabels=True, show_yticklabels=(col==0))

        unit = '°C' if self.var_type == 'temp' else 'mm/day'
        for col, metric in enumerate(metrics_order):
            norm, cmap, levels = configs[metric]
            # 重新计算 colorbar 位置 (3列)
            left = 0.13 + col * 0.27
            cax = fig.add_axes([left, 0.02, 0.22, 0.012])
            
            extend = 'both' if metric in ['bias', 'acc'] else 'max'
            cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend=extend)
            
            # 增大刻度字体
            cb.ax.tick_params(labelsize=14)
            cb.set_ticks(levels)
            if all(x.is_integer() for x in levels):
                 cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            else:
                 step = levels[1] - levels[0] if len(levels) > 1 else 0.1
                 fmt = '%.2f' if step < 0.1 else '%.1f'
                 cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            
            label = titles[metric]
            if metric != 'acc': label += f" ({unit})"
            cb.set_label(label, fontsize=16)
        
        fname = f"Global_{self.var_type}_L{leadtime}.png"
        plt.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fname}")

    def plot_regions_figure(self, leadtime: int, regions_data: Dict[str, Dict]):
        models = MODEL_LIST
        
        vals = {m: [] for m in ['bias', 'rmse', 'acc']}
        for reg_d in regions_data.values():
            for met in ['bias', 'rmse', 'acc']:
                for m in models:
                    if m in reg_d[met]:
                        vals[met].extend(reg_d[met][m].get('monthly', {}).values())
                        vals[met].extend(reg_d[met][m].get('seasonal', {}).values())

        # 仅使用 RMSE 数据
        norm_rmse, cmap_rmse, levels_rmse = self._get_levels_and_cmap(vals['rmse'], 'rmse')

        configs = {}
        configs['bias'] = self._get_levels_and_cmap(vals['bias'], 'bias')
        configs['acc'] = self._get_levels_and_cmap(vals['acc'], 'acc')
        configs['rmse'] = (norm_rmse, cmap_rmse, levels_rmse)

        fig = plt.figure(figsize=(30, 22)) 
        # 调整为 2行 x 3列
        outer_gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05, height_ratios=[1, 0.45]) 
        
        metrics_order = ['bias', 'rmse', 'acc']
        titles = {'bias': 'Bias', 'rmse': 'RMSE', 'acc': 'ACC'}
        
        region_grid = [
            ['Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast'],
            ['Z4-Tibetan',   'Z5-NorthChina',    'Z6-Yangtze'],
            ['Z7-Southwest', 'Z8-SouthChina',    'Z9-SouthSea']
        ]

        for col, metric in enumerate(metrics_order):
            norm, cmap, levels = configs[metric]
            
            for row, mode in enumerate(['monthly', 'seasonal']):
                inner_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_gs[row, col], 
                                                            hspace=0.15, wspace=0.05)
                
                if row == 0:
                    ax_title = fig.add_subplot(outer_gs[row, col])
                    ax_title.axis('off')
                    ax_title.text(0.5, 1.03, titles[metric], transform=ax_title.transAxes, 
                                  fontsize=28, fontweight='bold', ha='center', va='bottom')

                for r in range(3):
                    for c in range(3):
                        reg_name = region_grid[r][c]
                        ax = fig.add_subplot(inner_gs[r, c])
                        
                        data_to_plot = regions_data.get(reg_name, {}).get(metric, {})
                        
                        show_x = (row == 1 and r == 2)
                        show_y = (col == 0 and c == 0)
                        
                        self._plot_single_heatmap(ax, data_to_plot, models, mode, metric, norm, cmap, levels,
                                                  show_xticklabels=show_x, show_yticklabels=show_y)
                        
                        short_name = reg_name.split('-')[0]
                        ax.set_title(short_name, fontsize=16, pad=3, fontweight='bold')

        unit = '°C' if self.var_type == 'temp' else 'mm/day'
        for col, metric in enumerate(metrics_order):
            norm, cmap, levels = configs[metric]
            
            # 重新计算 colorbar 位置 (3列)
            left = 0.125 + col * 0.27
            cax = fig.add_axes([left + 0.01, 0.04, 0.22, 0.010])
            
            extend = 'both' if metric in ['bias', 'acc'] else 'max'
            cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend=extend)
            
            # 增大刻度字体
            cb.ax.tick_params(labelsize=16)
            cb.set_ticks(levels)
            if all(x.is_integer() for x in levels):
                 cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            else:
                 step = levels[1] - levels[0] if len(levels) > 1 else 0.1
                 fmt = '%.2f' if step < 0.1 else '%.1f'
                 cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            
            label = titles[metric]
            if metric != 'acc': label += f" ({unit})"
            cb.set_label(label, fontsize=20)

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
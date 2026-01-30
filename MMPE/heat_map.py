#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多区域双重指标(RMSE & ACC)组合热图绘制模块
功能：
1. 遍历Global及9个子区域
2. 读取 combined_error_analysis.py 生成的区域RMSE数据
3. 读取 combined_pearson_analysis.py 生成的区域ACC数据
4. 绘制包含 Month/Season x Models 的组合热图
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter
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

# 区域定义 (与分析脚本保持一致)
REGIONS_LIST = [
    'Global',
    'Z1-Northwest', 'Z2-InnerMongolia', 'Z3-Northeast',
    'Z4-Tibetan',   'Z5-NorthChina',    'Z7-Yangtze',
    'Z6-Southwest', 'Z8-SouthChina',    'Z9-SouthSea'
]

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RMSE/ACC 分段数（colorbar 段数 = N_LEVELS - 1）
N_RMSE_LEVELS = 11   # 0 到 vmax 分 10 段
N_ACC_LEVELS = 11    # -1 到 1 分 10 段


class RegionalHeatMapPlotter:
    def __init__(self, var_type: str):
        self.var_type = var_type
        # 数据路径配置
        self.acc_base_dir = Path(f"/sas12t1/ffyan/output/pearson_analysis/region_index_acc/{var_type}")
        self.rmse_base_dir = Path(f"/sas12t1/ffyan/output/error_analysis/region_metrics/{var_type}")
        self.output_dir = Path(f"/sas12t1/ffyan/output/heat_map_regional/{var_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_regional_data(self, region: str) -> Dict[int, Dict]:
        """
        加载指定区域的所有Leadtime和Model数据
        返回结构: {leadtime: {'rmse': {model: {...}}, 'acc': {model: {...}}}}
        包含 monthly, seasonal, annual 数据
        """
        data = {}
        safe_region = region.replace(' ', '_')

        for lt in LEADTIMES:
            data[lt] = {'rmse': {}, 'acc': {}}

            for model in MODEL_LIST:
                # 1. 加载 ACC (from combined_pearson_analysis)
                acc_file = self.acc_base_dir / f"region_index_acc_{safe_region}_{model}_{self.var_type}.nc"
                if acc_file.exists():
                    try:
                        with xr.open_dataset(acc_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                # 提取月度 ACC
                                mon_dict = {}
                                for m in (ds_lt.month.values if hasattr(ds_lt, 'month') else range(1, 13)):
                                    m = int(m)
                                    if m in ds_lt.month.values:
                                        mon_dict[MONTHS[m - 1]] = float(ds_lt.regional_index_acc.sel(month=m).values)
                                    else:
                                        mon_dict[MONTHS[m - 1]] = np.nan
                                if len(mon_dict) < 12:
                                    for i in range(1, 13):
                                        if MONTHS[i - 1] not in mon_dict:
                                            mon_dict[MONTHS[i - 1]] = np.nan
                                # 计算季节 ACC (按 SEASONS 顺序)
                                seas_dict = {}
                                for seas, m_idxs in SEASONS.items():
                                    vals = [mon_dict.get(MONTHS[m - 1], np.nan) for m in m_idxs]
                                    seas_dict[seas] = float(np.nanmean(vals))
                                ann_val = float(np.nanmean([v for v in mon_dict.values() if np.isfinite(v)])) if mon_dict else np.nan

                                data[lt]['acc'][model] = {
                                    'monthly': mon_dict,
                                    'seasonal': seas_dict,
                                    'annual': ann_val
                                }
                    except Exception as e:
                        logger.warning(f"加载ACC失败 {region} {model}: {e}")

                # 2. 加载 RMSE (from combined_error_analysis - 新版格式含 rmse_monthly 等)
                rmse_file = self.rmse_base_dir / f"{region}_{model}.nc"
                if rmse_file.exists():
                    try:
                        with xr.open_dataset(rmse_file) as ds:
                            if int(lt) in ds.leadtime.values:
                                ds_lt = ds.sel(leadtime=int(lt))
                                mon_dict = {}
                                if 'rmse_monthly' in ds_lt:
                                    da = ds_lt.rmse_monthly
                                    month_coord = da.coords.get('month', None)
                                    if month_coord is None:
                                        month_coord = getattr(ds_lt, 'month', None)
                                    months_iter = list(month_coord.values) if month_coord is not None else list(range(1, 13))
                                    for m in months_iter:
                                        m = int(m)
                                        val = da.sel(month=m).values
                                        mon_dict[MONTHS[m - 1]] = float(np.asarray(val).ravel()[0]) if val.size else np.nan
                                else:
                                    mon_dict = {MONTHS[i - 1]: np.nan for i in range(1, 13)}

                                if 'rmse_seasonal' in ds_lt and 'season' in ds_lt.dims:
                                    seas_dict = {}
                                    season_vals = np.asarray(ds_lt.season.values)
                                    for s in list(SEASONS.keys()):
                                        if np.any(season_vals == s):
                                            val = ds_lt.rmse_seasonal.sel(season=s).values
                                            seas_dict[s] = float(np.asarray(val).ravel()[0]) if np.asarray(val).size else np.nan
                                        else:
                                            seas_dict[s] = np.nan
                                else:
                                    # 从月度计算季节
                                    seas_dict = {}
                                    for seas, m_idxs in SEASONS.items():
                                        vals = [mon_dict.get(MONTHS[m - 1], np.nan) for m in m_idxs]
                                        seas_dict[seas] = float(np.nanmean(vals))

                                if 'rmse_annual' in ds_lt:
                                    v = ds_lt.rmse_annual.values
                                    ann_val = float(np.asarray(v).ravel()[0]) if np.asarray(v).size else np.nan
                                else:
                                    ann_val = float(np.nanmean(list(mon_dict.values()))) if mon_dict else np.nan

                                data[lt]['rmse'][model] = {
                                    'monthly': mon_dict,
                                    'seasonal': seas_dict,
                                    'annual': ann_val
                                }
                    except Exception as e:
                        logger.warning(f"加载RMSE失败 {region} {model}: {e}")
        return data

    def plot_heatmap(self, region: str, leadtime: int, data: Dict, mode: str = 'monthly'):
        """绘制热图 (Month x Model 或 Season/Annual x Model)"""

        rmse_data_all = data[leadtime]['rmse']
        acc_data_all = data[leadtime]['acc']

        models = [m for m in MODEL_LIST if m in rmse_data_all or m in acc_data_all]
        if not models:
            return

        if mode == 'monthly':
            y_labels = MONTHS
            rows = 12
        else:
            y_labels = ['Annual'] + list(SEASONS.keys())
            rows = 5

        cols = len(models)
        fig, ax = plt.subplots(figsize=(max(6, cols), max(5, rows * 0.8)))

        all_rmse = []
        all_acc = []

        for m in models:
            r_dict = rmse_data_all.get(m, {})
            a_dict = acc_data_all.get(m, {})

            if mode == 'monthly':
                vals_r = list(r_dict.get('monthly', {}).values())
                vals_a = list(a_dict.get('monthly', {}).values())
            else:
                vals_r = [r_dict.get('annual', np.nan)] + [r_dict.get('seasonal', {}).get(s, np.nan) for s in SEASONS.keys()]
                vals_a = [a_dict.get('annual', np.nan)] + [a_dict.get('seasonal', {}).get(s, np.nan) for s in SEASONS.keys()]

            all_rmse.extend(vals_r)
            all_acc.extend(vals_a)

        # RMSE: 从 0 开始，温度 Reds、降水 Blues；分段 colorbar，extend='max'
        rmse_vmin = 0
        valid_rmse = [x for x in all_rmse if np.isfinite(x) and x >= 0]
        rmse_vmax = np.nanmax(valid_rmse) if valid_rmse else 1.0
        rmse_levels = np.linspace(rmse_vmin, rmse_vmax, N_RMSE_LEVELS)
        n_rmse_colors = (N_RMSE_LEVELS - 1) + 1   # 分段数 + extend='max' 的 1 个三角
        if self.var_type == 'temp':
            rmse_cmap = plt.get_cmap('Reds', n_rmse_colors)
        else:
            rmse_cmap = plt.get_cmap('Blues', n_rmse_colors)
        rmse_norm = BoundaryNorm(rmse_levels, n_rmse_colors, extend='max')

        # ACC: -1 到 1，红蓝渐变 coolwarm；分段 colorbar，extend='both'
        acc_vmin, acc_vmax = -1.0, 1.0
        acc_levels = np.linspace(acc_vmin, acc_vmax, N_ACC_LEVELS)
        n_acc_colors = (N_ACC_LEVELS - 1) + 2   # 分段数 + extend='both' 的 2 个三角
        acc_cmap = plt.get_cmap('coolwarm', n_acc_colors)
        acc_norm = BoundaryNorm(acc_levels, n_acc_colors, extend='both')

        for i, row_label in enumerate(y_labels):
            for j, model in enumerate(models):
                r_val = np.nan
                a_val = np.nan

                r_d = rmse_data_all.get(model, {})
                a_d = acc_data_all.get(model, {})

                if mode == 'monthly':
                    r_val = r_d.get('monthly', {}).get(row_label, np.nan)
                    a_val = a_d.get('monthly', {}).get(row_label, np.nan)
                else:
                    if row_label == 'Annual':
                        r_val = r_d.get('annual', np.nan)
                        a_val = a_d.get('annual', np.nan)
                    else:
                        r_val = r_d.get('seasonal', {}).get(row_label, np.nan)
                        a_val = a_d.get('seasonal', {}).get(row_label, np.nan)

                # RMSE (左上三角)：从 0 起，分段着色
                if not np.isnan(r_val) and np.isfinite(r_val):
                    r_val_clip = max(rmse_vmin, min(r_val, rmse_levels[-1])) if len(rmse_levels) > 1 else r_val
                    idx_r = rmse_norm(r_val_clip)
                    idx_r = int(np.clip(idx_r, 0, n_rmse_colors - 1))
                    color_r = rmse_cmap(idx_r / max(1, n_rmse_colors - 1))
                    poly_r = patches.Polygon([(j, i), (j, i + 1), (j + 1, i + 1)], facecolor=color_r, edgecolor='none')
                    ax.add_patch(poly_r)
                    tc = 'white' if sum(color_r[:3]) / 3 < 0.5 else 'black'
                    ax.text(j + 0.3, i + 0.7, f"{r_val:.2f}", color=tc, fontsize=7, ha='center', va='center')

                # ACC (右下三角)：-1 到 1，分段着色
                if not np.isnan(a_val) and np.isfinite(a_val):
                    a_val_clip = max(acc_vmin, min(a_val, acc_vmax))
                    idx_a = acc_norm(a_val_clip)
                    idx_a = int(np.clip(idx_a, 0, n_acc_colors - 1))
                    color_a = acc_cmap(idx_a / max(1, n_acc_colors - 1))
                    poly_a = patches.Polygon([(j, i), (j + 1, i), (j + 1, i + 1)], facecolor=color_a, edgecolor='none')
                    ax.add_patch(poly_a)
                    tc = 'white' if sum(color_a[:3]) / 3 < 0.5 else 'black'
                    ax.text(j + 0.7, i + 0.3, f"{a_val:.2f}", color=tc, fontsize=7, ha='center', va='center')

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_xticklabels([m.replace('-mon', '') for m in models], rotation=45, ha='right')
        ax.set_yticks(np.arange(rows) + 0.5)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()

        for x in range(cols + 1):
            ax.axvline(x, color='k', lw=0.5)
        for y in range(rows + 1):
            ax.axhline(y, color='k', lw=0.5)
        if mode != 'monthly':
            ax.axhline(1, color='k', lw=2)

        ax.set_title(f"{region} {self.var_type.upper()} Metrics (L{leadtime}) - {mode.title()}")

        cax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
        sm1 = ScalarMappable(norm=rmse_norm, cmap=rmse_cmap)
        sm1.set_array([])
        cb1 = plt.colorbar(sm1, cax=cax1, extend='max', ticks=rmse_levels)
        cb1.set_label('RMSE')

        cax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        sm2 = ScalarMappable(norm=acc_norm, cmap=acc_cmap)
        sm2.set_array([])
        cb2 = plt.colorbar(sm2, cax=cax2, extend='both', ticks=acc_levels)
        cb2.set_label('ACC')

        fname = f"{region.replace(' ', '_')}_{mode}_L{leadtime}.png"
        plt.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fname}")

    def run(self):
        for region in REGIONS_LIST:
            logger.info(f"Processing region: {region}")
            data = self.load_regional_data(region)
            for lt in LEADTIMES:
                if not data[lt]['rmse'] and not data[lt]['acc']:
                    continue
                self.plot_heatmap(region, lt, data, mode='monthly')
                self.plot_heatmap(region, lt, data, mode='seasonal')


if __name__ == "__main__":
    for var in ['temp', 'prec']:
        plotter = RegionalHeatMapPlotter(var)
        plotter.run()

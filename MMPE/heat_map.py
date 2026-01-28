#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSE和ACC（异常相关系数）双重指标组合热图绘制脚本
在同一个单元格内横向分割为两块，分别显示RMSE和ACC，右侧有两个colorbar

运行环境要求:
- 需要在clim环境中运行: conda activate clim
- 确保已安装所需依赖包

使用示例:
python MMPE/triple_metric_combined_plots.py --var temp
python MMPE/triple_metric_combined_plots.py --var prec --leadtimes 0 1 2
python MMPE/triple_metric_combined_plots.py  # 绘制temp和prec两个变量
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

# 添加工具包路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
sys.path.insert(0, str(Path(__file__).parent))

# 导入MMPE配置
try:
    from common_config import SEASONS, LEADTIMES, MODEL_LIST
except ImportError:
    # 如果导入失败，使用默认值
    SEASONS = {
        'DJF': [12, 1, 2],   # 冬季
        'MAM': [3, 4, 5],    # 春季
        'JJA': [6, 7, 8],    # 夏季
        'SON': [9, 10, 11]   # 秋季
    }
    LEADTIMES = [0, 1, 2, 3, 4, 5]
    MODEL_LIST = [
        "CMCC-35",
        "DWD-mon-21",
        "ECMWF-51-mon",
        "Meteo-France-8",
        "NCEP-2",
        "UKMO-14",
        "ECCC-Canada-3",
    ]

warnings.filterwarnings('ignore')

# 月份标签
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 月份到索引的映射（用于从ACC数据中提取）
MONTH_TO_INDEX = {i+1: month for i, month in enumerate(MONTHS)}

# 日志配置
log_dir = "/sas12t1/ffyan/log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'triple_metric_plotter.log'), mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 创建自定义colormap
# RMSE: 白色(低) -> 红色(高)
rmse_colors = [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]  # 白到红
rmse_cmap_custom = LinearSegmentedColormap.from_list('rmse_white_red', rmse_colors, N=256)

# ACC: 白色(低) -> 蓝色(高)
acc_colors = [(1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]  # 白到蓝
acc_cmap_custom = LinearSegmentedColormap.from_list('acc_blue_white', acc_colors, N=256)

class RMSEACCCombinedPlotter:
    """RMSE和ACC（异常相关系数）双重指标组合热图绘制器"""
    
    def __init__(self, var_type: str, rmse_digits: int = 3, acc_digits: int = 2):
        """
        初始化绘制器
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
            rmse_digits: RMSE数值标注的小数位数，默认3位
            acc_digits: ACC数值标注的小数位数，默认2位
        """
        self.var_type = var_type
        # 数值标注的小数位数
        self.rmse_digits = max(0, int(rmse_digits))
        self.acc_digits = max(0, int(acc_digits))
        
    def load_rmse_data(self, data_dir: str = None) -> Dict:
        """加载RMSE数据（优先使用rmse_temporal目录下的时间序列数据）"""
        if data_dir is None:
            # 优先尝试rmse_temporal目录
            data_dir = f"/sas12t1/ffyan/outputdata/rmse_temporal/{self.var_type}"
            if os.path.isdir(data_dir):
                results = self._load_rmse_temporal_data(data_dir)
                if results:
                    return results
            
            # 回退到rmse_summary目录
            data_dir = f"/sas12t1/ffyan/outputdata/rmse_summary/{self.var_type}"
        
        if not os.path.exists(data_dir):
            logger.warning(f"RMSE数据目录不存在: {data_dir}")
            return {}
        
        results = {}
        
        # 查找所有leadtime的文件
        for filename in os.listdir(data_dir):
            if filename.startswith("rmse_seasonal_L") and filename.endswith(".csv"):
                try:
                    leadtime = int(filename.split("_L")[1].split(".")[0])
                except:
                    continue
                
                # 检查所有必需文件是否存在
                files_to_check = [
                    f"rmse_annual_L{leadtime}.csv",
                    f"rmse_seasonal_L{leadtime}.csv",
                    f"rmse_monthly_L{leadtime}.csv"
                ]
                
                if all(os.path.exists(os.path.join(data_dir, f)) for f in files_to_check):
                    try:
                        rmse_annual_df = pd.read_csv(os.path.join(data_dir, f"rmse_annual_L{leadtime}.csv"), index_col=0)
                        rmse_seasonal_df = pd.read_csv(os.path.join(data_dir, f"rmse_seasonal_L{leadtime}.csv"), index_col=0)
                        rmse_monthly_df = pd.read_csv(os.path.join(data_dir, f"rmse_monthly_L{leadtime}.csv"), index_col=0)
                        
                        results[leadtime] = {
                            'rmse_annual': rmse_annual_df['Annual'].to_dict(),
                            'rmse_seasonal': rmse_seasonal_df.to_dict('index'),
                            'rmse_monthly': rmse_monthly_df.to_dict('index')
                        }
                        
                        logger.info(f"成功加载RMSE L{leadtime}数据")
                        
                    except Exception as e:
                        logger.warning(f"读取RMSE CSV文件失败 L{leadtime}: {e}")
        
        return results
    
    def _load_rmse_temporal_data(self, data_dir: str) -> Dict:
        """从rmse_temporal目录加载RMSE时间序列数据并计算月度、季节、年度统计"""
        results = {}
        
        for leadtime in LEADTIMES:
            leadtime_data = {}
            
            # 查找该leadtime的所有模型文件
            import glob
            pattern = f"rmse_temporal_{self.var_type}_*_lead{leadtime}.csv"
            files = glob.glob(os.path.join(data_dir, pattern))
            
            model_stats = {}
            
            for file_path in files:
                try:
                    # 从文件名提取模型名
                    filename = os.path.basename(file_path)
                    model = filename.replace(f'rmse_temporal_{self.var_type}_', '').replace(f'_lead{leadtime}.csv', '')
                    # 使用原始模型名格式（不带-mon后缀）
                    
                    # 读取时间序列数据
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    rmse_series = df.iloc[:, 0]  # 第一列是RMSE值
                    
                    # 转换索引为pandas datetime
                    time_index = pd.to_datetime(rmse_series.index)
                    rmse_ts = pd.Series(rmse_series.values, index=time_index)
                    
                    # 计算月度平均值
                    monthly_means = rmse_ts.groupby(rmse_ts.index.month).mean()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_data = {}
                    for i, month_name in enumerate(month_names):
                        month_idx = i + 1
                        if month_idx in monthly_means.index:
                            monthly_data[month_name] = float(monthly_means.loc[month_idx])
                        else:
                            monthly_data[month_name] = float(rmse_ts.mean())
                    
                    # 计算季节平均值
                    seasonal_data = {}
                    for season, months in SEASONS.items():
                        season_mask = rmse_ts.index.month.isin(months)
                        if season_mask.any():
                            seasonal_data[season] = float(rmse_ts[season_mask].mean())
                        else:
                            seasonal_data[season] = float(rmse_ts.mean())
                    
                    # 合并月度和季节数据
                    seasonal_monthly_data = {**monthly_data, **seasonal_data}
                    
                    # 计算年度平均值
                    annual_mean = float(rmse_ts.mean())
                    seasonal_monthly_data['Annual'] = annual_mean
                    
                    model_stats[model] = seasonal_monthly_data
                    
                except Exception as e:
                    logger.warning(f"读取RMSE时间序列文件失败 {file_path}: {e}")
                    continue
            
            if model_stats:
                results[leadtime] = {
                    'rmse_annual': {model: data['Annual'] for model, data in model_stats.items()},
                    'rmse_seasonal': model_stats,
                    'rmse_monthly': model_stats
                }
                logger.info(f"成功加载RMSE时间序列数据 L{leadtime}")
        
        return results
    
    def _load_spatial_rmse_data(self, spatial_rmse_dir: str) -> Dict:
        """从新的空间RMSE分析NetCDF文件加载数据"""
        results = {}
        
        for leadtime in LEADTIMES:
            leadtime_data = {'rmse': {}}
            
            # 查找该leadtime的所有模型文件
            rmse_pattern = f"spatial_rmse_*_L{leadtime}.nc"
            import glob
            rmse_files = glob.glob(os.path.join(spatial_rmse_dir, rmse_pattern))
            
            # 处理RMSE文件
            for file_path in rmse_files:
                try:
                    # 从文件名提取模型名
                    filename = os.path.basename(file_path)
                    model = filename.replace(f'spatial_rmse_', '').replace(f'_L{leadtime}.nc', '')
                    
                    # 读取数据
                    ds = xr.open_dataset(file_path)
                    if 'spatial_rmse' in ds:
                        rmse_data = ds['spatial_rmse']
                        
                        # 计算季节和月度平均值
                        seasonal_monthly_data = {}
                        
                        # 计算月度平均值
                        monthly_means = rmse_data.groupby('time.month').mean()
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        for i, month_name in enumerate(month_names):
                            month_idx = i + 1
                            if month_idx in monthly_means.month:
                                seasonal_monthly_data[month_name] = float(monthly_means.sel(month=month_idx).values)
                            else:
                                seasonal_monthly_data[month_name] = float(np.nanmean(rmse_data.values))
                        
                        # 计算季节平均值
                        seasonal_means = rmse_data.groupby('time.season').mean()
                        
                        for season in ['DJF', 'MAM', 'JJA', 'SON']:
                            if season in seasonal_means.season:
                                seasonal_monthly_data[season] = float(seasonal_means.sel(season=season).values)
                            else:
                                seasonal_monthly_data[season] = float(np.nanmean(rmse_data.values))
                        
                        # 计算年平均值
                        annual_mean = float(np.nanmean(rmse_data.values))
                        seasonal_monthly_data['Annual'] = annual_mean
                        
                        leadtime_data['rmse'][model] = seasonal_monthly_data
                    ds.close()
                    
                except Exception as e:
                    logger.warning(f"读取空间RMSE文件失败 {file_path}: {e}")
                    continue
            
            if leadtime_data['rmse']:
                # 构建结果结构
                results[leadtime] = {
                    'rmse_annual': {model: data['Annual'] for model, data in leadtime_data['rmse'].items()},
                    'rmse_seasonal': leadtime_data['rmse'],
                    'rmse_monthly': leadtime_data['rmse']
                }
                logger.info(f"成功加载空间RMSE数据 L{leadtime}")
        
        return results
    
    def load_acc_data(self, data_dir: str = None) -> Dict:
        """
        从Spatial ACC NetCDF文件加载ACC数据
        
        从 `/sas12t1/ffyan/output/pearson_analysis/spatial_acc/{var_type}/` 
        加载 `spatial_acc_timeseries_{model}_{var_type}.nc` 文件。
        这些文件包含了 Mean Temporal ACC，维度为 (leadtime, month)。
        
        Returns:
            Dict: {leadtime: {'monthly': {model: {month: acc_value}}, 
                             'seasonal': {model: {season: acc_value}}, 
                             'annual_interannual': {model: acc_value}}}
        """
        if data_dir is None:
            data_dir = f"/sas12t1/ffyan/output/pearson_analysis/spatial_acc/{self.var_type}"
        
        if not os.path.isdir(data_dir):
            logger.warning(f"ACC数据目录不存在: {data_dir}")
            return {}
        
        # 初始化结果字典，由于我们要遍历模型文件而不是leadtime，这里先用中间结构
        # structure: {leadtime: {'monthly': {model: {}}, 'seasonal': {model: {}}, ...}}
        results = {}
        for lt in LEADTIMES:
            results[lt] = {
                'monthly': {},
                'seasonal': {},
                'annual_interannual': {}
            }
            
        import glob
        pattern = f"spatial_acc_timeseries_*_{self.var_type}.nc"
        files = glob.glob(os.path.join(data_dir, pattern))
        
        loaded_models = 0
        
        for file_path in files:
            try:
                # 从文件名提取模型名称
                # spatial_acc_timeseries_{model}_{var_type}.nc
                filename = os.path.basename(file_path)
                model_name = filename.replace("spatial_acc_timeseries_", "").replace(f"_{self.var_type}.nc", "")
                
                # 统一模型名称
                clean_model = model_name.replace('-mon-', '-').replace('-mon', '')
                
                # 读取NetCDF
                with xr.open_dataset(file_path) as ds:
                    if 'temporal_acc_mean' in ds:
                        acc_da = ds['temporal_acc_mean']
                    elif 'spatial_acc' in ds:
                        # 兼容旧版本变量名
                        acc_da = ds['spatial_acc']
                    else:
                        logger.warning(f"文件中未找到ACC变量: {filename}")
                        continue
                    
                    # 确保数据已加载
                    acc_da = acc_da.load()
                    
                    # 遍历 leadtimes (假设数据维度为 leadtime, month)
                    # 如果只有 month 维度（单leadtime文件），需要不同处理
                    # 但根据 combined_pearson_analysis.py，我们保存的是合并了所有leadtime的文件
                    
                    if 'leadtime' not in acc_da.dims:
                        logger.warning(f"文件缺少leadtime维度: {filename}")
                        continue
                        
                    for lt in acc_da.leadtime.values:
                        lt = int(lt)
                        if lt not in results:
                            continue
                            
                        # 提取该leadtime的数据 (month: 1-12)
                        lt_data = acc_da.sel(leadtime=lt)
                        
                        # 1. 填充月度数据
                        month_dict = {}
                        for m_idx in range(1, 13):
                            month_name = MONTHS[m_idx-1]
                            if m_idx in lt_data.month.values:
                                val = float(lt_data.sel(month=m_idx).item())
                                if np.isfinite(val):
                                    month_dict[month_name] = val
                                else:
                                    month_dict[month_name] = np.nan
                            else:
                                month_dict[month_name] = np.nan
                        
                        results[lt]['monthly'][clean_model] = month_dict
                        
                        # 2. 计算季节平均
                        season_dict = {}
                        for season, months in SEASONS.items():
                            vals = [month_dict[MONTHS[m-1]] for m in months if np.isfinite(month_dict.get(MONTHS[m-1], np.nan))]
                            if vals:
                                season_dict[season] = float(np.mean(vals))
                            else:
                                season_dict[season] = np.nan
                        
                        results[lt]['seasonal'][clean_model] = season_dict
                        
                        # 3. 计算年度平均 (Interannual 实际上在这里对应全年的 Mean Temporal ACC)
                        all_vals = [v for v in month_dict.values() if np.isfinite(v)]
                        if all_vals:
                            results[lt]['annual_interannual'][clean_model] = float(np.mean(all_vals))
                        else:
                            results[lt]['annual_interannual'][clean_model] = np.nan
                            
                loaded_models += 1
                
            except Exception as e:
                logger.warning(f"读取ACC文件失败 {file_path}: {e}")
                continue
        
        logger.info(f"成功从新数据源加载ACC数据，共 {loaded_models} 个模型文件")
        
        # 移除空的结果
        final_results = {k: v for k, v in results.items() if v['monthly']}
        return final_results
    
    def plot_monthly_heatmap(self, rmse_results: Dict, acc_results: Dict, save_dir: str = None):
        """
        绘制月度指标组合热图
        
        Args:
            rmse_results: RMSE数据字典
            acc_results: ACC数据字典
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = f"/sas12t1/ffyan/output/heat_map/monthly/{self.var_type}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        MONTHS_KEYS = MONTHS  # ['Jan',...,'Dec']
        
        # 获取所有共同的leadtime
        common_leadtimes = set(rmse_results.keys()) & set(acc_results.keys())
        
        for leadtime in sorted(common_leadtimes):
            logger.info(f"绘制L{leadtime}的月度指标组合热图")
            
            rmse_data = rmse_results[leadtime]
            acc_data = acc_results[leadtime]
            
            # 获取所有模型列表（仅使用MODEL_LIST中定义的模型）
            available_models = set(rmse_data.get('rmse_monthly', {}).keys()) | set(acc_data.get('monthly', {}).keys())
            # 只保留MODEL_LIST中的模型，自动排除JMA等其他模型
            # 匹配逻辑：检查模型名是否与MODEL_LIST中的模型匹配（处理-mon后缀等差异）
            def is_model_in_list(model_name):
                # 统一模型名格式（移除-mon后缀）
                clean_model = model_name.replace('-mon-', '-').replace('-mon', '')
                for config_model in MODEL_LIST:
                    clean_config = config_model.replace('-mon-', '-').replace('-mon', '')
                    if clean_model == clean_config or clean_model in clean_config or clean_config in clean_model:
                        return True
                return False
            all_models = sorted([m for m in available_models if is_model_in_list(m)])
            
            if not all_models:
                logger.warning(f"L{leadtime}无有效模型，跳过月度绘图")
                continue
            
            n_months = len(MONTHS_KEYS)
            n_rows = n_months
            n_cols = len(all_models)
            
            # 收集所有有效值以确定colormap范围
            all_rmse_vals = []
            all_acc_vals = []
            
            for i in range(n_rows):
                for j, model in enumerate(all_models):
                    # 获取RMSE值
                    month = MONTHS_KEYS[i]
                    rmse_val = rmse_data.get('rmse_monthly', {}).get(model, {}).get(month, np.nan)
                    
                    # 获取ACC值
                    acc_val = acc_data.get('monthly', {}).get(model, {}).get(month, np.nan)
                    if acc_val is None:
                        acc_val = np.nan
                    
                    if not np.isnan(rmse_val):
                        all_rmse_vals.append(rmse_val)
                    if not np.isnan(acc_val):
                        all_acc_vals.append(acc_val)
            
            # 确定colormap的范围
            rmse_vmin, rmse_vmax = (min(all_rmse_vals), max(all_rmse_vals)) if all_rmse_vals else (0, 1)
            # ACC范围：以绘图数字的极值为上下界
            if all_acc_vals:
                acc_data_min, acc_data_max = np.nanmin(all_acc_vals), np.nanmax(all_acc_vals)
                if not np.isfinite(acc_data_min) or not np.isfinite(acc_data_max):
                    acc_vmin, acc_vmax = (-1.0, 1.0)
                elif acc_data_min == acc_data_max:
                    # 全为常数，给一个很小的范围
                    delta = 0.01
                    acc_vmin, acc_vmax = (acc_data_min - delta, acc_data_max + delta)
                else:
                    acc_vmin, acc_vmax = (acc_data_min, acc_data_max)
            else:
                acc_vmin, acc_vmax = (-1.0, 1.0)
            
            # 创建图形，减少宽度为一半，调整字体大小以避免遮挡
            fig, ax = plt.subplots(figsize=(max(6, n_cols * 1), max(8, n_rows * 0.5)))
            
            # 设置背景为白色
            ax.set_facecolor('white')
            
            # 使用自定义colormap；RMSE和ACC都使用分段式colorbar
            # RMSE: 白色(低) -> 红色(高)
            rmse_cmap_base = rmse_cmap_custom
            
            # 创建RMSE分段式colorbar（11段）
            if np.isfinite(rmse_vmin) and np.isfinite(rmse_vmax) and rmse_vmax > rmse_vmin:
                # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                rmse_levels = ticker.MaxNLocator(nbins=10, prune=None).tick_values(rmse_vmin, rmse_vmax)
            else:
                rmse_levels = np.linspace(0.0, 1.0, 11)
                rmse_vmin, rmse_vmax = 0.0, 1.0
            rmse_norm = BoundaryNorm(rmse_levels, rmse_cmap_base.N, clip=True)
            rmse_cmap = rmse_cmap_base
            
            # ACC: 蓝色(低) -> 白色(高)
            acc_cmap_base = acc_cmap_custom
            
            # 创建ACC分段式colorbar（11段）
            if np.isfinite(acc_vmin) and np.isfinite(acc_vmax) and acc_vmax > acc_vmin:
                # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                acc_levels = ticker.MaxNLocator(nbins=10, prune=None).tick_values(acc_vmin, acc_vmax)
            else:
                acc_levels = np.linspace(-1.0, 1.0, 11)
                acc_vmin, acc_vmax = -1.0, 1.0
            acc_norm = BoundaryNorm(acc_levels, acc_cmap_base.N, clip=True)
            acc_cmap = acc_cmap_base
            
            # 绘制每个单元格
            text_artists = []
            for i in range(n_rows):
                for j, model in enumerate(all_models):
                    # 获取两个指标的值（RMSE和ACC）
                    month = MONTHS_KEYS[i]
                    rmse_val = rmse_data.get('rmse_monthly', {}).get(model, {}).get(month, np.nan)
                    acc_val = acc_data.get('monthly', {}).get(model, {}).get(month, np.nan)
                    if acc_val is None:
                        acc_val = np.nan
                    
                    # 创建斜向三角形分割（RMSE左上，ACC右下）
                    cell_width = 1.0
                    cell_height = 1.0
                    
                    # 左上三角形：RMSE
                    if not np.isnan(rmse_val):
                        rmse_color = rmse_cmap(rmse_norm(rmse_val))
                    else:
                        rmse_color = 'lightgray'
                    
                    # 左上三角形顶点：(左下, 左上, 右上)
                    rmse_triangle = patches.Polygon([
                        (j-0.5, i-0.5),      # 左下
                        (j-0.5, i+0.5),      # 左上  
                        (j+0.5, i+0.5)       # 右上
                    ], facecolor=rmse_color, edgecolor='none', linewidth=0)
                    ax.add_patch(rmse_triangle)
                    
                    # 右下三角形：ACC
                    if not np.isnan(acc_val):
                        acc_color = acc_cmap(acc_norm(acc_val))
                    else:
                        acc_color = 'lightgray'
                    
                    # 右下三角形顶点：(左下, 右下, 右上)
                    acc_triangle = patches.Polygon([
                        (j-0.5, i-0.5),      # 左下
                        (j+0.5, i-0.5),      # 右下
                        (j+0.5, i+0.5)       # 右上
                    ], facecolor=acc_color, edgecolor='none', linewidth=0)
                    ax.add_patch(acc_triangle)
                    
                    # 文本颜色选择
                    def _pick_text_color(rgba):
                        try:
                            r, g, b, _ = rgba
                        except:
                            r = g = b = 0.8
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        return 'white' if luminance < 0.5 else 'black'
                    
                    # 添加RMSE文本（左上三角形中心）
                    if not np.isnan(rmse_val):
                        rmse_text = f"{rmse_val:.{self.rmse_digits}f}"
                        rmse_text_color = _pick_text_color(rmse_color)
                    else:
                        rmse_text = "N/A"
                        rmse_text_color = 'black'
                    
                    # 左上三角形重心约在 (j-1/6, i+1/6)
                    t1 = ax.text(j - 0.17, i + 0.17, rmse_text, ha='center', va='center',
                                fontsize=5, color=rmse_text_color)
                    t1.set_path_effects([pe.withStroke(linewidth=1.2, foreground=('black' if rmse_text_color=='white' else 'white'))])
                    
                    # 添加ACC文本（右下三角形中心）
                    if not np.isnan(acc_val):
                        acc_text = f"{acc_val:.{self.acc_digits}f}"
                        acc_text_color = _pick_text_color(acc_color)
                    else:
                        acc_text = "N/A"
                        acc_text_color = 'black'
                    
                    # 右下三角形重心约在 (j+1/6, i-1/6)
                    t2 = ax.text(j + 0.17, i - 0.17, acc_text, ha='center', va='center',
                                fontsize=5, color=acc_text_color)
                    t2.set_path_effects([pe.withStroke(linewidth=1.2, foreground=('black' if acc_text_color=='white' else 'white'))])
                    
                    text_artists.extend([t1, t2])
            
            # 设置x轴（模型名称）
            ax.set_xticks(np.arange(n_cols))
            display_models = []
            for _name in all_models:
                _d = _name.replace('-mon-', '-')
                if _d.endswith('-mon'):
                    _d = _d[:-4]
                display_models.append(_d)
            ax.set_xticklabels(display_models, ha='right', fontsize=12, rotation=45, rotation_mode='anchor')
            ax.set_xlim(-0.5, n_cols-0.5)
            
            # 设置左侧y轴标签
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(MONTHS_KEYS, fontsize=16)
            ax.set_ylim(-0.5, n_rows-0.5)
            
            # 在每个模型列之间添加纵向分割线
            for x in np.arange(-0.5, n_cols + 0.5, 1.0):
                ax.axvline(x, color='k', linewidth=1.0)
            
            # 添加顶部标签
            if n_cols > 0:
                top_y = n_rows - 0.5
                label_y = top_y + 0.05
                j0 = 0
                segment_width = 1.0 / 2.0
                # 左上三角形标签位置
                x_rmse = j0 - 0.3
                # 右下三角形标签位置  
                x_acc = j0 + 0.3
                ax.text(x_rmse, label_y, "RMSE", ha='center', va='bottom', fontsize=8, clip_on=False)
                ax.text(x_acc, label_y, "ACC", ha='center', va='bottom', fontsize=8, clip_on=False)
            
            # 标题
            # ax.set_xlabel("Models", fontsize=12)
            # ax.set_ylabel("Month", fontsize=12)
            plt.title(f"{self.var_type.upper()} Monthly Metrics (L{leadtime}) - RMSE & ACC", 
                     fontsize=14, pad=20)
            
            # 添加两个colorbar
            plt.subplots_adjust(right=0.82)
            
            # RMSE colorbar（分段式）
            cax1 = fig.add_axes([0.84, 0.55, 0.02, 0.35])
            rmse_sm = ScalarMappable(cmap=rmse_cmap, norm=rmse_norm)
            rmse_sm.set_array(np.linspace(rmse_vmin, rmse_vmax, 256))
            rmse_cbar = fig.colorbar(rmse_sm, cax=cax1)
            rmse_cbar.set_label('RMSE', fontsize=10)
            rmse_cbar.ax.yaxis.set_label_position('right')
            # 设置分段边界刻度
            rmse_cbar.set_ticks(rmse_levels)
            rmse_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # ACC colorbar（分段式）
            cax2 = fig.add_axes([0.84, 0.1, 0.02, 0.35])
            acc_sm = ScalarMappable(cmap=acc_cmap, norm=acc_norm)
            acc_sm.set_array(np.linspace(acc_vmin, acc_vmax, 256))
            acc_cbar = fig.colorbar(acc_sm, cax=cax2)
            acc_cbar.set_label('ACC', fontsize=10)
            acc_cbar.ax.yaxis.set_label_position('right')
            # 设置分段边界刻度
            acc_cbar.set_ticks(acc_levels)
            acc_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # 保存图片（PNG和PDF）
            save_path_png = os.path.join(save_dir, f"rmse_acc_monthly_L{leadtime}_{self.var_type}.png")
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            logger.info(f"月度指标组合热图已保存: {save_path_png}")
            
            save_path_pdf = os.path.join(save_dir, f"rmse_acc_monthly_L{leadtime}_{self.var_type}.pdf")
            plt.savefig(save_path_pdf, bbox_inches='tight')
            logger.info(f"月度指标组合热图PDF已保存: {save_path_pdf}")

            # 保存无数值标注版本
            try:
                for ta in text_artists:
                    ta.set_visible(False)
                save_path2_png = os.path.join(save_dir, f"rmse_acc_monthly_L{leadtime}_{self.var_type}_no_text.png")
                plt.savefig(save_path2_png, dpi=300, bbox_inches='tight')
                logger.info(f"月度无标注版本已保存: {save_path2_png}")
                
                save_path2_pdf = os.path.join(save_dir, f"rmse_acc_monthly_L{leadtime}_{self.var_type}_no_text.pdf")
                plt.savefig(save_path2_pdf, bbox_inches='tight')
                logger.info(f"月度无标注版本PDF已保存: {save_path2_pdf}")
            finally:
                plt.close(fig)

    def plot_seasonal_annual_heatmap(self, rmse_results: Dict, acc_results: Dict, save_dir: str = None):
        """
        绘制季节+年度指标组合热图
        
        Args:
            rmse_results: RMSE数据字典
            acc_results: ACC数据字典
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = f"/sas12t1/ffyan/output/heat_map/seasonal_annual/{self.var_type}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        SEASONS_KEYS = list(SEASONS.keys())  # ['DJF','MAM','JJA','SON']
        
        # 获取所有共同的leadtime
        common_leadtimes = set(rmse_results.keys()) & set(acc_results.keys())
        
        for leadtime in sorted(common_leadtimes):
            logger.info(f"绘制L{leadtime}的季节+年度指标组合热图")
            
            rmse_data = rmse_results[leadtime]
            acc_data = acc_results[leadtime]
            
            # 获取所有模型列表（仅使用MODEL_LIST中定义的模型）
            available_models = set(rmse_data.get('rmse_seasonal', {}).keys()) | set(acc_data.get('seasonal', {}).keys())
            # 匹配逻辑：检查模型名是否与MODEL_LIST中的模型匹配（处理-mon后缀等差异）
            def is_model_in_list(model_name):
                # 统一模型名格式（移除-mon后缀）
                clean_model = model_name.replace('-mon-', '-').replace('-mon', '')
                for config_model in MODEL_LIST:
                    clean_config = config_model.replace('-mon-', '-').replace('-mon', '')
                    if clean_model == clean_config or clean_model in clean_config or clean_config in clean_model:
                        return True
                return False
            all_models = sorted([m for m in available_models if is_model_in_list(m)])
            
            if not all_models:
                logger.warning(f"L{leadtime}无有效模型，跳过季节绘图")
                continue
            
            n_annual = 1
            n_seasons = len(SEASONS_KEYS)
            n_rows = n_annual + n_seasons  # 年度 + 季节
            n_cols = len(all_models)
            
            # 收集所有有效值以确定colormap范围
            all_rmse_vals = []
            all_acc_vals = []
            
            for i in range(n_rows):
                for j, model in enumerate(all_models):
                    # 获取RMSE值
                    if i == 0:  # 年度
                        rmse_val = rmse_data.get('rmse_annual', {}).get(model, np.nan)
                    else:  # 季节
                        season = SEASONS_KEYS[i - 1]
                        rmse_val = rmse_data.get('rmse_seasonal', {}).get(model, {}).get(season, np.nan)
                    
                    # 获取ACC值
                    if i == 0:  # 年度（使用年度平均值）
                        acc_val = acc_data.get('annual_interannual', {}).get(model, None)
                        if acc_val is None or np.isnan(acc_val):
                            acc_val = np.nan
                    else:  # 季节
                        season = SEASONS_KEYS[i - 1]
                        acc_val = acc_data.get('seasonal', {}).get(model, {}).get(season, np.nan)
                        if acc_val is None:
                            acc_val = np.nan
                    
                    if not np.isnan(rmse_val):
                        all_rmse_vals.append(rmse_val)
                    if not np.isnan(acc_val):
                        all_acc_vals.append(acc_val)
            
            # 确定colormap的范围
            rmse_vmin, rmse_vmax = (min(all_rmse_vals), max(all_rmse_vals)) if all_rmse_vals else (0, 1)
            # ACC范围：以绘图数字的极值为上下界
            if all_acc_vals:
                acc_data_min, acc_data_max = np.nanmin(all_acc_vals), np.nanmax(all_acc_vals)
                if not np.isfinite(acc_data_min) or not np.isfinite(acc_data_max):
                    acc_vmin, acc_vmax = (-1.0, 1.0)
                elif acc_data_min == acc_data_max:
                    delta = 0.01
                    acc_vmin, acc_vmax = (acc_data_min - delta, acc_data_max + delta)
                else:
                    acc_vmin, acc_vmax = (acc_data_min, acc_data_max)
            else:
                acc_vmin, acc_vmax = (-1.0, 1.0)
            
            # 创建图形，减少宽度为一半，调整字体大小以避免遮挡
            fig, ax = plt.subplots(figsize=(max(5, n_cols * 1), max(4, n_rows * 0.8)))
            
            # 设置背景为白色
            ax.set_facecolor('white')
            
            # 使用自定义colormap；RMSE和ACC都使用分段式colorbar
            # RMSE: 白色(低) -> 红色(高)
            rmse_cmap_base = rmse_cmap_custom
            
            # 创建RMSE分段式colorbar（6段）
            if np.isfinite(rmse_vmin) and np.isfinite(rmse_vmax) and rmse_vmax > rmse_vmin:
                # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                rmse_levels = ticker.MaxNLocator(nbins=5, prune=None).tick_values(rmse_vmin, rmse_vmax)
            else:
                rmse_levels = np.linspace(0.0, 1.0, 11)
                rmse_vmin, rmse_vmax = 0.0, 1.0
            rmse_norm = BoundaryNorm(rmse_levels, rmse_cmap_base.N, clip=True)
            rmse_cmap = rmse_cmap_base
            
            # ACC: 蓝色(低) -> 白色(高)
            acc_cmap_base = acc_cmap_custom
            
            # 创建ACC分段式colorbar（6段）
            if np.isfinite(acc_vmin) and np.isfinite(acc_vmax) and acc_vmax > acc_vmin:
                # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                acc_levels = ticker.MaxNLocator(nbins=5, prune=None).tick_values(acc_vmin, acc_vmax)
            else:
                acc_levels = np.linspace(-1.0, 1.0, 11)
                acc_vmin, acc_vmax = -1.0, 1.0
            acc_norm = BoundaryNorm(acc_levels, acc_cmap_base.N, clip=True)
            acc_cmap = acc_cmap_base
            
            # 绘制每个单元格
            text_artists = []
            for i in range(n_rows):
                for j, model in enumerate(all_models):
                    # 获取两个指标的值（RMSE和ACC）
                    if i == 0:  # 年度
                        rmse_val = rmse_data.get('rmse_annual', {}).get(model, np.nan)
                        acc_val = acc_data.get('annual_interannual', {}).get(model, None)
                        if acc_val is None or np.isnan(acc_val):
                            acc_val = np.nan
                    else:  # 季节
                        season = SEASONS_KEYS[i - 1]
                        rmse_val = rmse_data.get('rmse_seasonal', {}).get(model, {}).get(season, np.nan)
                        acc_val = acc_data.get('seasonal', {}).get(model, {}).get(season, np.nan)
                        if acc_val is None:
                            acc_val = np.nan
                    
                    # 创建斜向三角形分割（RMSE左上，ACC右下）
                    cell_width = 1.0
                    cell_height = 1.0
                    
                    # 左上三角形：RMSE
                    if not np.isnan(rmse_val):
                        rmse_color = rmse_cmap(rmse_norm(rmse_val))
                    else:
                        rmse_color = 'lightgray'
                    
                    # 左上三角形顶点：(左下, 左上, 右上)
                    rmse_triangle = patches.Polygon([
                        (j-0.5, i-0.5),      # 左下
                        (j-0.5, i+0.5),      # 左上  
                        (j+0.5, i+0.5)       # 右上
                    ], facecolor=rmse_color, edgecolor='none', linewidth=0)
                    ax.add_patch(rmse_triangle)
                    
                    # 右下三角形：ACC
                    if not np.isnan(acc_val):
                        acc_color = acc_cmap(acc_norm(acc_val))
                    else:
                        acc_color = 'lightgray'
                    
                    # 右下三角形顶点：(左下, 右下, 右上)
                    acc_triangle = patches.Polygon([
                        (j-0.5, i-0.5),      # 左下
                        (j+0.5, i-0.5),      # 右下
                        (j+0.5, i+0.5)       # 右上
                    ], facecolor=acc_color, edgecolor='none', linewidth=0)
                    ax.add_patch(acc_triangle)
                    
                    # 文本颜色选择
                    def _pick_text_color(rgba):
                        try:
                            r, g, b, _ = rgba
                        except:
                            r = g = b = 0.8
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        return 'white' if luminance < 0.5 else 'black'
                    
                    # 添加RMSE文本（左上三角形中心）
                    if not np.isnan(rmse_val):
                        rmse_text = f"{rmse_val:.{self.rmse_digits}f}"
                        rmse_text_color = _pick_text_color(rmse_color)
                    else:
                        rmse_text = "N/A"
                        rmse_text_color = 'black'
                    
                    # 左上三角形重心约在 (j-1/6, i+1/6)
                    t1 = ax.text(j - 0.17, i + 0.17, rmse_text, ha='center', va='center',
                                fontsize=7, color=rmse_text_color)
                    t1.set_path_effects([pe.withStroke(linewidth=1.2, foreground=('black' if rmse_text_color=='white' else 'white'))])
                    
                    # 添加ACC文本（右下三角形中心）
                    if not np.isnan(acc_val):
                        acc_text = f"{acc_val:.{self.acc_digits}f}"
                        acc_text_color = _pick_text_color(acc_color)
                    else:
                        acc_text = "N/A"
                        acc_text_color = 'black'
                    
                    # 右下三角形重心约在 (j+1/6, i-1/6)
                    t2 = ax.text(j + 0.17, i - 0.17, acc_text, ha='center', va='center',
                                fontsize=7, color=acc_text_color)
                    t2.set_path_effects([pe.withStroke(linewidth=1.2, foreground=('black' if acc_text_color=='white' else 'white'))])
                    
                    text_artists.extend([t1, t2])
            
            # 设置x轴（模型名称）
            ax.set_xticks(np.arange(n_cols))
            display_models = []
            for _name in all_models:
                _d = _name.replace('-mon-', '-')
                if _d.endswith('-mon'):
                    _d = _d[:-4]
                display_models.append(_d)
            ax.set_xticklabels(display_models, ha='right', fontsize=12, rotation=45, rotation_mode='anchor')
            ax.set_xlim(-0.5, n_cols-0.5)
            
            # 设置左侧y轴标签
            all_labels = ["Annual"] + SEASONS_KEYS
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(all_labels, fontsize=10)
            ax.set_ylim(-0.5, n_rows-0.5)
            
            # 在每个模型列之间添加纵向分割线
            for x in np.arange(-0.5, n_cols + 0.5, 1.0):
                ax.axvline(x, color='k', linewidth=1.0)
            
            # 绘制年度和季节之间的分割线
            split_y = 1 - 0.5
            ax.axhline(split_y, color='black', linewidth=2)
            
            # 添加区域标签
            # Year标签位置：年度行的中间（第0行）
            year_y = 0.5 / n_rows  # 第0行的中间位置
            
            # Season标签位置：季节行的中间（第1-4行的中间）
            season_start_y = 1.5 / n_rows  # 第1行的中间位置
            season_end_y = 4.5 / n_rows    # 第4行的中间位置
            season_center_y = (season_start_y + season_end_y) / 2  # 季节区域的中间位置
            
            # 左侧区域标签，距离热图更远
            # left_label_x = -0.15
            # ax.text(left_label_x, year_y, "Year", ha='center', va='center', 
            #        fontsize=10, rotation=90, 
            #        transform=ax.get_yaxis_transform())
            # ax.text(left_label_x, season_center_y, "Season", ha='center', va='center', 
            #        fontsize=10, rotation=90, 
            #        transform=ax.get_yaxis_transform())
            
            # 添加顶部标签
            if n_cols > 0:
                top_y = n_rows - 0.5
                label_y = top_y + 0.1
                j0 = 0
                segment_width = 1.0 / 2.0
                # 左上三角形标签位置
                x_rmse = j0 - 0.3
                # 右下三角形标签位置  
                x_acc = j0 + 0.3
                ax.text(x_rmse, label_y, "RMSE", ha='center', va='bottom', fontsize=8, clip_on=False)
                ax.text(x_acc, label_y, "ACC", ha='center', va='bottom', fontsize=8, clip_on=False)
            
            # 标题
            ax.set_xlabel("Models", fontsize=12)
            plt.title(f"{self.var_type.upper()} Seasonal & Annual Metrics (L{leadtime}) - RMSE & ACC", 
                     fontsize=14, pad=25)
            
            # 添加两个colorbar
            plt.subplots_adjust(right=0.82)
            
            # RMSE colorbar（分段式）
            cax1 = fig.add_axes([0.84, 0.55, 0.02, 0.35])
            rmse_sm = ScalarMappable(cmap=rmse_cmap, norm=rmse_norm)
            rmse_sm.set_array(np.linspace(rmse_vmin, rmse_vmax, 256))
            rmse_cbar = fig.colorbar(rmse_sm, cax=cax1)
            rmse_cbar.set_label('RMSE', fontsize=10)
            rmse_cbar.ax.yaxis.set_label_position('right')
            # 设置分段边界刻度
            rmse_cbar.set_ticks(rmse_levels)
            rmse_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # ACC colorbar（分段式）
            cax2 = fig.add_axes([0.84, 0.1, 0.02, 0.35])
            acc_sm = ScalarMappable(cmap=acc_cmap, norm=acc_norm)
            acc_sm.set_array(np.linspace(acc_vmin, acc_vmax, 256))
            acc_cbar = fig.colorbar(acc_sm, cax=cax2)
            acc_cbar.set_label('ACC', fontsize=10)
            acc_cbar.ax.yaxis.set_label_position('right')
            # 设置分段边界刻度
            acc_cbar.set_ticks(acc_levels)
            acc_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # 保存图片（PNG和PDF）
            save_path_png = os.path.join(save_dir, f"rmse_acc_seasonal_annual_L{leadtime}_{self.var_type}.png")
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            logger.info(f"季节+年度指标组合热图已保存: {save_path_png}")
            
            save_path_pdf = os.path.join(save_dir, f"rmse_acc_seasonal_annual_L{leadtime}_{self.var_type}.pdf")
            plt.savefig(save_path_pdf, bbox_inches='tight')
            logger.info(f"季节+年度指标组合热图PDF已保存: {save_path_pdf}")

            # 保存无数值标注版本
            try:
                for ta in text_artists:
                    ta.set_visible(False)
                save_path2_png = os.path.join(save_dir, f"rmse_acc_seasonal_annual_L{leadtime}_{self.var_type}_no_text.png")
                plt.savefig(save_path2_png, dpi=300, bbox_inches='tight')
                logger.info(f"季节+年度无标注版本已保存: {save_path2_png}")
                
                save_path2_pdf = os.path.join(save_dir, f"rmse_acc_seasonal_annual_L{leadtime}_{self.var_type}_no_text.pdf")
                plt.savefig(save_path2_pdf, bbox_inches='tight')
                logger.info(f"季节+年度无标注版本PDF已保存: {save_path2_pdf}")
            finally:
                plt.close(fig)

    def plot_dual_metric_heatmap(self, rmse_results: Dict, acc_results: Dict):
        """
        绘制双重指标组合热图（包括月度和季节+年度）
        
        Args:
            rmse_results: RMSE数据字典
            acc_results: ACC数据字典
        """
        logger.info("绘制月度指标组合热图")
        self.plot_monthly_heatmap(rmse_results, acc_results)
        
        logger.info("绘制季节+年度指标组合热图")  
        self.plot_seasonal_annual_heatmap(rmse_results, acc_results)
    
    def run_plotting(self, leadtimes: List[int] = None):
        """
        运行绘图
        
        Args:
            leadtimes: 指定预报时效列表，如果为None则使用所有可用的leadtime
        """
        logger.info("开始从CSV文件加载RMSE数据")
        rmse_results = self.load_rmse_data()
        
        logger.info("开始从NetCDF文件加载ACC数据")
        acc_results = self.load_acc_data()
        
        if not rmse_results or not acc_results:
            logger.error("未找到可用的数据，请先运行相关的数据计算脚本")
            return
        
        # 如果指定了leadtimes，过滤结果
        if leadtimes:
            rmse_results = {lt: data for lt, data in rmse_results.items() if lt in leadtimes}
            acc_results = {lt: data for lt, data in acc_results.items() if lt in leadtimes}
            
            if not rmse_results or not acc_results:
                logger.error(f"指定的leadtimes {leadtimes} 没有对应的数据")
                return
        
        logger.info("开始绘制双重指标组合热图")
        self.plot_dual_metric_heatmap(rmse_results, acc_results)
        logger.info(f"{self.var_type} 双重指标组合热图绘制完成")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RMSE和ACC（异常相关系数）双重指标组合热图绘制")
    parser.add_argument("--var", type=str, choices=['temp', 'prec'], 
                       default=None, help="变量类型（不指定则同时处理temp和prec）")
    parser.add_argument("--leadtimes", type=int, nargs="+", help="指定预报时效列表")
    parser.add_argument("--rmse-digits", type=int, default=3, help="RMSE小数位数，默认3位")
    parser.add_argument("--acc-digits", type=int, default=2, help="ACC小数位数，默认2位")
    
    args = parser.parse_args()
    
    # 确定变量列表
    var_list = [args.var] if args.var else ['temp', 'prec']
    
    for var_type in var_list:
        logger.info(f"开始绘制{var_type}的双重指标组合热图")
        
        plotter = RMSEACCCombinedPlotter(
            var_type, 
            rmse_digits=args.rmse_digits, 
            acc_digits=args.acc_digits
        )
        plotter.run_plotting(leadtimes=args.leadtimes)
    
    logger.info("所有双重指标组合热图绘制完成！")

if __name__ == "__main__":
    main()

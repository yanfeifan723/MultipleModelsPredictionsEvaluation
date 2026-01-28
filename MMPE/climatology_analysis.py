#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气候态分析和绘图脚本

绘制观测和模式的温度和降水气候态（年平均和季节平均）
使用1度网格，每个lead time一张组合图

使用方法：
python climatology_analysis.py --var temp prec --leadtimes 0 1 2 3 4 5 --seasons all
python climatology_analysis.py --var temp --leadtimes 0 --seasons annual
"""

import os
import sys
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# common_config已在上面导入

# 统一导入toolkit路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_utils import find_variable
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars
from src.utils.plotting_utils import (
    STANDARD_CONFIG,
    setup_cartopy_axes,
    plot_spatial_field_contour,
    create_discrete_colormap_norm,
)

from common_config import (
    MODEL_FILE_MAP,
    DATA_PATHS as BASE_DATA_PATHS,
    VAR_NAMES,
)

# 配置日志
logger = setup_logging(
    log_file='climatology_analysis.log',
    module_name=__name__
)

warnings.filterwarnings('ignore')

# 模型配置
MODELS = MODEL_FILE_MAP

# 变量配置（使用common_config中的VAR_NAMES）
VAR_CONFIG = {
    "temp": {
        "file_type": "sfc",
        **VAR_NAMES["temp"],  # 使用common_config中的变量名
        "unit": "°C",
        "display_conv": lambda x: x - 273.15,  # 显示时转换：开尔文转摄氏度
    },
    "prec": {
        "file_type": "sfc",
        **VAR_NAMES["prec"],  # 使用common_config中的变量名
        "obs_conv": lambda x: x * 86400,
        "fcst_conv": lambda x: x * 86400 * 1000,
        "unit": "mm/day",
        "display_conv": lambda x: x,  # 降水不需要转换
    }
}

# 数据路径（扩展common_config中的基础路径）
DATA_PATHS = {
    **BASE_DATA_PATHS,  # 从common_config继承基础路径
    "output_dir": "/sas12t1/ffyan/output/climatology_analysis"  # 覆盖输出目录
}

LEADTIMES = [0, 3]
BIAS_TIMESERIES_LEADS = [0, 1, 2, 3, 4, 5]
SPATIAL_BOUNDS = {"lat": [15.0, 55.0], "lon": [70.0, 140.0]}


class ClimatologyAnalyzer:
    """气候态分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.obs_dir = Path(DATA_PATHS["obs_dir"])
        self.forecast_dir = Path(DATA_PATHS["forecast_dir"])
        self.output_dir = Path(DATA_PATHS["output_dir"])
        self.plots_dir = self.output_dir / "plots"
        
        # 创建输出目录
        self.data_dir = self.output_dir / "data"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.plots_dir / "temp").mkdir(exist_ok=True)
        (self.plots_dir / "prec").mkdir(exist_ok=True)
        (self.data_dir / "temp").mkdir(exist_ok=True)
        (self.data_dir / "prec").mkdir(exist_ok=True)
        
        logger.info("ClimatologyAnalyzer 初始化完成")
    
    def load_obs_data(self, var_type: str) -> Optional[xr.DataArray]:
        """加载观测数据（1度网格，不插值）"""
        try:
            obs_path = self.obs_dir / f"{var_type}_1deg_199301-202012.nc"
            
            if not obs_path.exists():
                logger.error(f"观测数据文件不存在: {obs_path}")
                return None
            
            logger.info(f"加载观测数据: {obs_path}")
            ds = xr.open_dataset(obs_path, mask_and_scale=True)
            
            try:
                var_name = find_variable(ds, VAR_CONFIG[var_type]["obs_names"])
            except ValueError:
                var_name = None
            if var_name is None:
                logger.error(f"观测数据中未找到 {var_type} 变量")
                ds.close()
                return None
            
            data = ds[var_name].where(
                ~ds[var_name].isin([1e20, ds[var_name].attrs.get('_FillValue', 1e20)]), 
                np.nan
            )
            
            # 单位转换（降水）
            if var_type == 'prec' and 'obs_conv' in VAR_CONFIG[var_type]:
                data = data * VAR_CONFIG[var_type]['obs_conv'](1)
                data = data.clip(min=0)  # 保留海洋NaN，不使用fillna(0)
            
            # 空间裁剪
            if 'lat' in data.coords and 'lon' in data.coords:
                data = data.sel(
                    lat=slice(SPATIAL_BOUNDS["lat"][0], SPATIAL_BOUNDS["lat"][1]),
                    lon=slice(SPATIAL_BOUNDS["lon"][0], SPATIAL_BOUNDS["lon"][1])
                )
            
            # 时间处理
            if 'time' in data.coords:
                data = data.resample(time='1MS').mean()
                data = data.sel(time=slice('1993-01-01', '2020-12-31'))
            
            data = data.load()
            ds.close()
            
            # 输出数据范围用于诊断
            valid_data = data.values[np.isfinite(data.values)]
            if len(valid_data) > 0:
                logger.info(f"观测数据加载成功: {data.shape}, 数据范围: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}] {VAR_CONFIG[var_type]['unit']}")
            else:
                logger.warning(f"观测数据加载成功: {data.shape}, 但所有值都是NaN")
            return data
            
        except Exception as e:
            logger.error(f"加载观测数据时出错: {e}")
            return None
    
    def load_forecast_data(self, model: str, var_type: str, leadtime: int) -> Optional[xr.DataArray]:
        """加载预报数据（ensemble mean）"""
        try:
            config = VAR_CONFIG[var_type]
            suffix = MODELS[model][config['file_type']]
            model_dir = self.forecast_dir / model
            
            if not model_dir.exists():
                logger.warning(f"模式目录不存在: {model_dir}")
                return None
            
            logger.debug(f"加载 {model} L{leadtime} {var_type}, {model_dir}{suffix}")
            
            monthly_da_list = []
            all_monthly_times = []
            
            for year in range(1993, 2021):
                for month in range(1, 13):
                    fp = model_dir / f"{year}{month:02d}.{suffix}.nc"
                    if not fp.exists():
                        continue
                    
                    try:
                        with xr.open_dataset(fp) as ds:
                            try:
                                var_name = find_variable(ds, config['fcst_names'])
                            except ValueError:
                                var_name = None
                            if var_name is None:
                                logger.debug(f"文件 {fp.name} 中未找到变量: {config['fcst_names']}")
                                continue
                            
                            da = ds[var_name]
                            
                            # 处理时间和ensemble维度
                            if 'number' in da.dims and 'time' in da.dims:
                                if da.time.size <= leadtime:
                                    logger.debug(f"文件 {fp.name} 时间维度 {da.time.size} <= leadtime {leadtime}")
                                    continue
                                da = da.isel(time=leadtime)
                                if 'number' in da.dims:
                                    da = da.mean(dim='number')  # ensemble mean
                            elif 'time' in da.dims:
                                # 仅有time维度：time 已对应 lead=0..N 的有效月份，直接按索引选取
                                if da.time.size <= leadtime:
                                    logger.debug(f"文件 {fp.name} 时间维度 {da.time.size} <= leadtime {leadtime}")
                                    continue
                                da = da.isel(time=leadtime)
                            
                            # 单位转换（降水）
                            if var_type == 'prec' and 'fcst_conv' in config:
                                da = da * config['fcst_conv'](1)
                                da = da.clip(min=0).fillna(0)
                            
                            # 空间裁剪
                            da = self._dynamic_coord_sel(da, {'lat': (15, 55), 'lon': (70, 140)})
                            da = da.squeeze(drop=True)
                            
                            # 检查维度
                            if da.ndim != 2:
                                logger.debug(f"文件 {fp.name} 维度不是2: {da.ndim}, dims={da.dims}")
                                continue
                            if 'lat' not in da.dims or 'lon' not in da.dims:
                                logger.debug(f"文件 {fp.name} 缺少lat或lon维度: dims={da.dims}")
                                continue
                            
                            # 使用实际的预报时间（从文件内time坐标获取）
                            forecast_time = pd.Timestamp(da.time.values) if hasattr(da, 'time') else pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                            monthly_da_list.append((forecast_time, da))
                            all_monthly_times.append(forecast_time)
                            
                    except Exception as e:
                        logger.warning(f"处理文件 {fp.name} 时出错: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue
            
            if not monthly_da_list:
                logger.warning(f"无预测数据 {model} L{leadtime}")
                return None
            
            # 去重
            unique_times = list(dict.fromkeys(all_monthly_times))
            if len(unique_times) != len(all_monthly_times):
                unique_list = []
                seen_times = set()
                for t, da in monthly_da_list:
                    if t not in seen_times:
                        unique_list.append((t, da))
                        seen_times.add(t)
                monthly_da_list = unique_list
            
            # 拼接
            data = xr.concat(
                [da for t, da in monthly_da_list],
                dim=xr.DataArray([t for t, _ in monthly_da_list], dims='time', name='time')
            )
            data = data.sortby('time')
            data = data.load()
            
            logger.info(f"模式数据加载成功 {model} L{leadtime}: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载预报数据时出错 {model} L{leadtime}: {e}")
            return None
    
    def _apply_unit_conversion(self, da: xr.DataArray, var_type: str) -> xr.DataArray:
        """应用单位转换（与现有配置一致）"""
        try:
            config = VAR_CONFIG.get(var_type, {})
            if var_type == 'prec' and 'fcst_conv' in config:
                da = da * config['fcst_conv'](1)
                da = da.clip(min=0).fillna(0)
            return da
        except Exception:
            return da
    
    def _apply_display_conversion(self, data, var_type: str):
        """应用显示单位转换（用于colorbar范围）"""
        config = VAR_CONFIG.get(var_type, {})
        if 'display_conv' in config:
            return config['display_conv'](data)
        return data
    
    def _extract_leadtime_from_file(self, da: xr.DataArray, leadtime: int, var_type: str) -> Optional[xr.DataArray]:
        """从单月文件中提取指定lead的二维格点场（若有ensemble则先lead后ensemble平均）"""
        try:
            work = da
            if 'time' in work.dims:
                if work.time.size <= leadtime:
                    return None
                work = work.isel(time=leadtime)
            else:
                return None
            if 'number' in work.dims:
                work = work.mean(dim='number', skipna=True)
            work = self._apply_unit_conversion(work, var_type)
            work = self._dynamic_coord_sel(work, SPATIAL_BOUNDS)
            work = work.squeeze(drop=True)
            # 维度检查（坐标别名可能被重命名为标准lat/lon）
            if work.ndim != 2:
                return None
            if not (('lat' in work.dims or 'latitude' in work.dims) and ('lon' in work.dims or 'longitude' in work.dims)):
                return None
            return work
        except Exception:
            return None
    
    def _combine_leadtime_data(self, leadtime_data: Dict[int, List[xr.DataArray]], model: str) -> Dict[int, xr.DataArray]:
        """合并各lead的目标月样本为一个按time排序的跨年序列"""
        result: Dict[int, xr.DataArray] = {}
        for lead, lst in leadtime_data.items():
            try:
                if not lst:
                    continue
                combined = xr.concat(lst, dim='time')
                combined = combined.sortby('time')
                result[lead] = combined.load()
                logger.info(f"模式 {model} leadtime {lead}: 加载了 {combined.time.size} 个目标月份样本")
            except Exception as e:
                logger.warning(f"合并lead {lead} 时出错: {e}")
        return result
    
    def load_forecast_data_by_target_month(self, model: str, var_type: str,
                                           target_months: List[pd.Timestamp]) -> Dict[int, xr.DataArray]:
        """
        按目标月份加载预报数据，返回 {lead: DataArray(time=目标月, lat, lon)}
        假定单月文件内time轴即为有效月份的lead序列(0..N)
        """
        try:
            config = VAR_CONFIG[var_type]
            suffix = MODELS[model][config['file_type']]
            model_dir = self.forecast_dir / model
            if not model_dir.exists():
                logger.warning(f"模式目录不存在: {model_dir}")
                return {}
            
            leadtime_data: Dict[int, List[xr.DataArray]] = {}
            for target_month in target_months:
                # 仅考虑lead 0..5
                for lead in range(0, 6):
                    init_month = target_month - pd.DateOffset(months=lead)
                    if init_month.year < 1993 or init_month.year > 2020:
                        continue
                    fp = model_dir / f"{init_month.strftime('%Y%m')}.{suffix}.nc"
                    if not fp.exists():
                        continue
                    try:
                        with xr.open_dataset(fp) as ds:
                            try:
                                var_name = find_variable(ds, config['fcst_names'])
                            except ValueError:
                                var_name = None
                            if var_name is None:
                                continue
                            da = ds[var_name]
                            da_target = self._extract_leadtime_from_file(da, lead, var_type)
                            if da_target is None:
                                continue
                            # 标注目标月份
                            da_target = da_target.expand_dims(time=[pd.Timestamp(target_month)])
                            leadtime_data.setdefault(lead, []).append(da_target)
                        # 日志：init+lead→target
                        logger.debug(f"{model} {init_month.strftime('%Y-%m')} + L{lead} -> target {target_month.strftime('%Y-%m')}")
                    except Exception as e:
                        logger.warning(f"处理文件 {fp.name} L{lead} 时出错: {e}")
                        continue
            return self._combine_leadtime_data(leadtime_data, model)
        except Exception as e:
            logger.error(f"按目标月份加载预报失败: {e}")
            return {}
    def _dynamic_coord_sel(self, da: xr.DataArray, bounds: Dict) -> xr.DataArray:
        """动态坐标选择（处理坐标名称别名）"""
        try:
            # 坐标名称映射
            coord_map = {
                'lat': ['latitude', 'lats', 'ylat'],
                'lon': ['longitude', 'lons', 'xlon']
            }
            
            da = da.copy()
            
            for target, (start, end) in bounds.items():
                # 查找实际的坐标名称
                real_coord = next((alias for alias in [target] + coord_map.get(target, []) 
                                 if alias in da.coords), None)
                
                if real_coord is None:
                    continue  # 跳过不存在的坐标
                
                # 重命名为标准名称
                if real_coord != target:
                    da = da.rename({real_coord: target})
                
                # 获取坐标值并判断是否递减
                vals = da[target].values
                if len(vals) <= 1:
                    continue
                
                is_decreasing = vals[0] > vals[1]
                buffer = 0.5 * abs(vals[1] - vals[0])
                
                # 根据递减/递增选择切片方式
                if is_decreasing:
                    slc = slice(end + buffer, start - buffer)
                else:
                    slc = slice(start - buffer, end + buffer)
                
                da = da.sel({target: slc})
            
            return da
        except Exception as e:
            logger.debug(f"坐标选择失败: {e}")
            return da
    
    def save_climatology_data(self, var_type: str, leadtime: int, season: Optional[str],
                              obs_clim: xr.DataArray, model_clims: Dict[str, xr.DataArray],
                              model_biases: Dict[str, xr.DataArray]):
        """
        保存气候态和偏差数据到NetCDF文件
        
        Args:
            var_type: 变量类型
            leadtime: 提前期
            season: 季节名称
            obs_clim: 观测气候态
            model_clims: 模型气候态字典
            model_biases: 模型偏差字典
        """
        try:
            season_str = season if season else 'annual'
            
            # 保存观测气候态
            obs_file = self.data_dir / var_type / f"climatology_obs_{var_type}_{season_str}.nc"
            obs_clim.to_netcdf(obs_file)
            logger.info(f"观测气候态已保存: {obs_file}")
            
            # 保存模型气候态和偏差
            for model in model_clims.keys():
                # 气候态
                clim_file = self.data_dir / var_type / f"climatology_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                model_clims[model].to_netcdf(clim_file)
                
                # 偏差
                bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                model_biases[model].to_netcdf(bias_file)
            
            logger.info(f"已保存 {len(model_clims)} 个模型的气候态和偏差数据")
            
        except Exception as e:
            logger.error(f"保存气候态数据失败: {e}")
    
    def _adjust_year_for_season(self, year: int, month: int, season: str) -> int:
        """调整跨年季节的年份：DJF 的 12 月归到前一年"""
        if season == 'DJF' and month == 12:
            return year - 1
        return year
    
    def generate_target_months(self, start_year: int = 1993, end_year: int = 2020,
                               season: Optional[str] = None) -> List[pd.Timestamp]:
        """生成需要分析的目标月份列表"""
        targets: List[pd.Timestamp] = []
        if season is None or season == 'annual':
            for y in range(start_year, end_year + 1):
                for m in range(1, 13):
                    targets.append(pd.Timestamp(y, m, 1))
        else:
            months = SEASONS.get(season, [])
            for y in range(start_year, end_year + 1):
                for m in months:
                    actual_year = self._adjust_year_for_season(y, m, season)
                    if start_year <= actual_year <= end_year:
                        targets.append(pd.Timestamp(actual_year, m, 1))
        return sorted(targets)
    
    def load_obs_climatology(self, var_type: str, season: Optional[str]) -> Optional[xr.DataArray]:
        """加载观测的季节气候态"""
        obs = self.load_obs_data(var_type)
        if obs is None:
            return None
        return self.calculate_climatology(obs, season if season else None)
    
    def calculate_seasonal_climatology(self, data: xr.DataArray, season: str) -> Optional[xr.DataArray]:
        """等权月季节平均：对季内各目标月先按time求月均，再对月均等权平均"""
        try:
            if season is None or season == 'annual':
                return data.mean(dim='time', skipna=True)
            months = SEASONS.get(season, [])
            month_means = []
            for m in months:
                sel_m = data.sel(time=data.time.dt.month == m)
                if sel_m.size == 0:
                    continue
                month_means.append(sel_m.mean(dim='time', skipna=True))
            if len(month_means) == 0:
                return None
            return xr.concat(month_means, dim='mm').mean(dim='mm', skipna=True)
        except Exception as e:
            logger.error(f"计算季节气候态失败: {e}")
            return None
    
    def calculate_bias(self, model_clim: xr.DataArray, obs_clim: xr.DataArray) -> Optional[xr.DataArray]:
        """计算偏差（模型 - 观测），并掩膜海洋（观测为NaN的位置）"""
        try:
            obs_interp = obs_clim.interp(lat=model_clim.lat, lon=model_clim.lon, method='linear')
            bias = model_clim - obs_interp
            bias = bias.where(~obs_interp.isnull(), np.nan)
            return bias
        except Exception as e:
            logger.error(f"计算bias失败: {e}")
            return None
    
    def _calculate_spatial_mean(self, data: xr.DataArray) -> float:
        """计算空间平均值（float）"""
        try:
            return float(data.mean(skipna=True).item())
        except Exception:
            return float('nan')
    
    def _calculate_spatial_rmse_from_bias(self, bias: xr.DataArray) -> float:
        """计算空间RMSE（基于bias场，忽略NaN）"""
        try:
            squared = xr.apply_ufunc(lambda x: x * x, bias)
            mean_sq = float(squared.mean(skipna=True).item())
            if not np.isfinite(mean_sq):
                return float('nan')
            return float(np.sqrt(mean_sq))
        except Exception:
            return float('nan')
    
    def _calculate_spatial_corr(self, model_clim: xr.DataArray, obs_clim: xr.DataArray) -> float:
        """计算空间相关系数（Pearson r），只在观测有效区域内计算"""
        try:
            obs_interp = obs_clim.interp(lat=model_clim.lat, lon=model_clim.lon, method='linear')
            valid_mask = np.isfinite(obs_interp.values)
            model_vals = model_clim.values[valid_mask]
            obs_vals = obs_interp.values[valid_mask]
            if model_vals.size < 3:
                return float('nan')
            # 去除NaN
            finite_mask = np.isfinite(model_vals) & np.isfinite(obs_vals)
            model_vals = model_vals[finite_mask].ravel()
            obs_vals = obs_vals[finite_mask].ravel()
            if model_vals.size < 3:
                return float('nan')
            r = np.corrcoef(model_vals, obs_vals)[0, 1]
            return float(r)
        except Exception:
            return float('nan')
    
    def load_climatology_data(self, var_type: str, leadtime: int, season: Optional[str],
                             models: List[str]) -> Tuple[Optional[xr.DataArray], Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
        """
        从NetCDF文件加载气候态和偏差数据
        
        Args:
            var_type: 变量类型
            leadtime: 提前期
            season: 季节名称
            models: 模型列表
        
        Returns:
            (obs_clim, model_clims, model_biases)
        """
        try:
            season_str = season if season else 'annual'
            
            # 加载观测气候态
            obs_file = self.data_dir / var_type / f"climatology_obs_{var_type}_{season_str}.nc"
            if not obs_file.exists():
                logger.error(f"观测气候态文件不存在: {obs_file}")
                return None, {}, {}
            
            obs_clim = xr.open_dataarray(obs_file)
            logger.info(f"观测气候态已加载: {obs_file}")
            
            # 加载模型气候态和偏差
            model_clims = {}
            model_biases = {}
            
            for model in models:
                clim_file = self.data_dir / var_type / f"climatology_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                
                if clim_file.exists() and bias_file.exists():
                    model_clims[model] = xr.open_dataarray(clim_file)
                    model_biases[model] = xr.open_dataarray(bias_file)
                else:
                    logger.warning(f"模型 {model} 数据文件缺失")
            
            logger.info(f"已加载 {len(model_clims)} 个模型的气候态和偏差数据")
            return obs_clim, model_clims, model_biases
            
        except Exception as e:
            logger.error(f"加载气候态数据失败: {e}")
            return None, {}, {}
    
    def calculate_climatology(self, data: xr.DataArray, season: Optional[str] = None) -> xr.DataArray:
        """
        计算气候态
        
        Args:
            data: 输入数据 (time, lat, lon)
            season: 季节名称 (DJF, MAM, JJA, SON) 或 None (年平均)
        
        Returns:
            气候态 (lat, lon)
        """
        try:
            if season is None or season == 'annual':
                # 年平均：所有月份的平均
                climatology = data.mean(dim='time', skipna=True)
            else:
                # 季节平均：筛选特定月份
                if season not in SEASONS:
                    logger.error(f"未知季节: {season}")
                    return None
                
                months = SEASONS[season]
                season_data = data.sel(time=data.time.dt.month.isin(months))
                climatology = season_data.mean(dim='time', skipna=True)
            
            return climatology
            
        except Exception as e:
            logger.error(f"计算气候态失败: {e}")
            return None
    
    def plot_climatology_combined(self, var_type: str, leadtime: int, season: Optional[str],
                                 obs_clim: xr.DataArray, model_clims: Dict[str, xr.DataArray],
                                 model_biases: Dict[str, xr.DataArray],
                                 clim_vmin: Optional[float] = None,
                                 clim_vmax: Optional[float] = None):
        """
        绘制气候态和偏差组合图 (4行×4列)
        
        布局：
        第1行：观测气候态，模型1-3气候态
        第2行：空白，模型1-3偏差
        第3行：模型4-7气候态
        第4行：模型4-7偏差
        
        Args:
            var_type: 变量类型
            leadtime: 提前期
            season: 季节名称
            obs_clim: 观测气候态
            model_clims: 模式气候态字典
            model_biases: 模式偏差字典
        """
        try:
            season_str = season if season else 'annual'
            logger.info(f"绘制气候态和偏差组合图: {var_type} L{leadtime} {season_str}")
            
            # 创建图形：新布局 - 4列，4个grid行（2对空间图+PDF图）
            # 每对的高度比：空间图=2，PDF图=1
            n_grid_rows = 4
            height_ratios = [2, 1, 2, 1]
            # 根据布局计算合适的图像高度：每个空间图约6英寸，PDF图约3英寸
            fig_height = sum(height_ratios) * 3  # 总高度约18英寸
            fig = plt.figure(figsize=(32, fig_height))
            gs = GridSpec(n_grid_rows, 4, figure=fig,  # 4行×4列
                         height_ratios=height_ratios,  # 空间图:PDF = 2:1
                         hspace=0.15, wspace=0.12,
                         left=0.04, right=0.96, top=0.96, bottom=0.04)
            
            # 只收集观测气候态数据用于计算统一颜色范围（移除模型气候态）
            obs_valid = obs_clim.values[np.isfinite(obs_clim.values)]
            
            # 收集所有偏差数据用于计算统一颜色范围
            all_bias_values = []
            for model_bias in model_biases.values():
                # 只收集有限的值（非NaN且非inf）
                bias_valid = model_bias.values[np.isfinite(model_bias.values)]
                all_bias_values.extend(bias_valid)
            
            # 计算观测气候态颜色范围（应用显示单位转换）
            # 如果提供了固定范围，使用固定范围；否则基于观测数据计算
            if clim_vmin is not None and clim_vmax is not None:
                # 使用传入的固定范围（基于观测气候态）
                clim_vmin = self._apply_display_conversion(clim_vmin, var_type)
                clim_vmax = self._apply_display_conversion(clim_vmax, var_type)
                logger.info(f"使用固定气候态colorbar范围: [{clim_vmin:.2f}, {clim_vmax:.2f}]")
            else:
                # 基于观测数据计算范围
                if var_type == 'temp':
                    if len(obs_valid) > 0:
                        obs_mean = np.mean(obs_valid)
                        obs_std = np.std(obs_valid)
                        clim_vmin = self._apply_display_conversion(obs_mean - 2 * obs_std, var_type)
                        clim_vmax = self._apply_display_conversion(obs_mean + 2 * obs_std, var_type)
                    else:
                        logger.warning("没有有效的观测数据，使用默认温度范围")
                        clim_vmin = -10
                        clim_vmax = 30
                else:
                    clim_vmin = 0
                    if len(obs_valid) > 0:
                        clim_vmax = np.percentile(obs_valid, 98)
                    else:
                        logger.warning("没有有效的观测数据，使用默认降水范围")
                        clim_vmax = 10
            
            # 设置colormap - 统一使用 coolwarm
            cmap_clim = 'coolwarm'
            cmap_bias = 'coolwarm'
            if len(all_bias_values) > 0:
                all_bias_values = np.array(all_bias_values)
                # 进一步确保没有无穷值
                all_bias_values = all_bias_values[np.isfinite(all_bias_values)]
                if len(all_bias_values) > 0:
                    bias_abs_max = np.percentile(np.abs(all_bias_values), 99)  # 使用99%分位数，只牺牲1%的极端值
                    # 确保bias_abs_max是有限值
                    if np.isfinite(bias_abs_max) and bias_abs_max > 0:
                        bias_vmin = -bias_abs_max
                        bias_vmax = bias_abs_max
                    else:
                        # 如果计算结果无效，使用默认值
                        logger.warning("偏差范围计算结果无效，使用默认范围")
                        if var_type == 'temp':
                            bias_vmin = -5
                            bias_vmax = 5
                        else:
                            bias_vmin = -2
                            bias_vmax = 2
                else:
                    # 如果过滤后没有有效数据，使用默认值
                    logger.warning("过滤后没有有效的偏差数据，使用默认范围")
                    if var_type == 'temp':
                        bias_vmin = -5
                        bias_vmax = 5
                    else:
                        bias_vmin = -2
                        bias_vmax = 2
            else:
                # 如果没有有效的偏差数据，使用默认值
                logger.warning("没有有效的偏差数据，使用默认范围")
                if var_type == 'temp':
                    bias_vmin = -5
                    bias_vmax = 5
                else:
                    bias_vmin = -2
                    bias_vmax = 2
            
            logger.info(f"气候态范围: [{clim_vmin:.2f}, {clim_vmax:.2f}]")
            logger.info(f"偏差范围: [{bias_vmin:.2f}, {bias_vmax:.2f}]")
            
            # 准备模型列表，按照 MODELS 字典的顺序排列
            model_names = [m for m in MODELS.keys() if m in model_biases]
            
            # 用于保存colorbar的绘图对象
            im_clim = None
            im_bias = None
            
            # ===== 第1行第1列：观测气候态 =====
            ax_obs = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            ax_obs.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
            ax_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax_obs.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax_obs.add_feature(cfeature.LAND, alpha=0.1)
            ax_obs.add_feature(cfeature.OCEAN, alpha=0.1)
            gl = ax_obs.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            # 计算等高线级别（应用显示转换）
            n_levels = 20
            levels = ticker.MaxNLocator(nbins=n_levels, prune=None).tick_values(clim_vmin, clim_vmax)
            obs_clim_display = self._apply_display_conversion(obs_clim, var_type)
            im_clim = ax_obs.contourf(obs_clim.lon, obs_clim.lat, obs_clim_display,
                                     transform=ccrs.PlateCarree(),
                                     cmap=cmap_clim, levels=levels, extend='both')
            
            ax_obs.text(0.02, 0.98, 'Observation',
                       transform=ax_obs.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top', horizontalalignment='left')
            
            # ===== 第2行第1列：图例和colorbar =====
            legend_ax = fig.add_subplot(gs[1, 0])
            legend_ax.axis('off')
            
            # 模型位置映射（按顺序排列7个模型）
            model_positions = [
                (0, 1), (0, 2), (0, 3),  # 第1行，第2-4列
                (2, 0), (2, 1), (2, 2), (2, 3)   # 第3行，全部4列
            ]
            
            # ===== 第一步：计算所有模型的PDF密度范围，用于统一y轴 =====
            all_density_values = []
            model_density_data = {}  # 存储每个模型的密度数据
            seasonal_density_data = {}  # 存储每个模型不同季节的密度数据
            
            # 如果是annual季节，需要加载其他季节的偏差数据
            load_seasonal_data = (season is None or season == 'annual')
            seasonal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
            seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']
            
            for i, model in enumerate(model_names):
                if i >= len(model_positions):
                    break
                
                model_bias = model_biases[model]
                bias_values = model_bias.values[np.isfinite(model_bias.values)]
                
                # 计算当前季节的PDF密度
                if len(bias_values) > 10:
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(bias_values)
                        x_range = np.linspace(bias_vmin, bias_vmax, 200)
                        density = kde(x_range)
                        all_density_values.extend(density)
                        model_density_data[model] = (x_range, density)
                    except Exception as e:
                        logger.debug(f"PDF计算失败 {model}: {e}")
                        model_density_data[model] = None
                else:
                    model_density_data[model] = None
                
                # 如果是annual季节，加载其他季节的偏差数据
                if load_seasonal_data:
                    seasonal_density_data[model] = {}
                    for j, season_name in enumerate(seasonal_names):
                        try:
                            # 加载其他季节的偏差数据
                            seasonal_bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{leadtime}_{var_type}_{season_name}.nc"
                            if seasonal_bias_file.exists():
                                seasonal_bias = xr.open_dataarray(seasonal_bias_file)
                                seasonal_bias_values = seasonal_bias.values[np.isfinite(seasonal_bias.values)]
                                
                                if len(seasonal_bias_values) > 10:
                                    try:
                                        from scipy.stats import gaussian_kde
                                        kde_seasonal = gaussian_kde(seasonal_bias_values)
                                        x_range_seasonal = np.linspace(bias_vmin, bias_vmax, 200)
                                        density_seasonal = kde_seasonal(x_range_seasonal)
                                        seasonal_density_data[model][season_name] = (x_range_seasonal, density_seasonal)
                                        all_density_values.extend(density_seasonal)  # 包含在统一y轴计算中
                                    except Exception as e:
                                        logger.debug(f"季节PDF计算失败 {model} {season_name}: {e}")
                                        seasonal_density_data[model][season_name] = None
                                else:
                                    seasonal_density_data[model][season_name] = None
                            else:
                                seasonal_density_data[model][season_name] = None
                        except Exception as e:
                            logger.debug(f"加载季节数据失败 {model} {season_name}: {e}")
                            seasonal_density_data[model][season_name] = None
            
            # 计算统一的y轴范围
            if all_density_values:
                pdf_ymax = max(all_density_values) * 1.1  # 留10%边距
            else:
                pdf_ymax = 1.0
            
            # ===== 第二步：绘制模型bias空间图和PDF =====
            for i, model in enumerate(model_names):
                if i >= len(model_positions):
                    break
                    
                grid_row, grid_col = model_positions[i]
                display_name = model.replace('-mon', '').replace('mon-', '')
                model_bias = model_biases[model]
                
                # 空间分布图（占用grid_row行）
                ax_spatial = fig.add_subplot(gs[grid_row, grid_col], projection=ccrs.PlateCarree())
                ax_spatial.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                
                # 计算偏差等高线级别
                n_bias_levels = 20
                bias_levels = ticker.MaxNLocator(nbins=n_bias_levels, prune=None).tick_values(bias_vmin, bias_vmax)
                im_bias = ax_spatial.contourf(model_bias.lon, model_bias.lat, model_bias,
                                             transform=ccrs.PlateCarree(),
                                             cmap=cmap_bias, levels=bias_levels, extend='both')
                
                # 模型标签
                label = chr(97 + i)
                ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                               transform=ax_spatial.transAxes, fontsize=18, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                
                # PDF图（占用grid_row+1行）
                ax_pdf = fig.add_subplot(gs[grid_row+1, grid_col])
                
                # 提取bias有效值
                bias_values = model_bias.values[np.isfinite(model_bias.values)]
                
                # 绘制PDF曲线
                if load_seasonal_data and model in seasonal_density_data:
                    # Annual季节：绘制多条季节PDF曲线
                    # 首先绘制annual的PDF（只填充，无线条边框）
                    if model_density_data[model] is not None:
                        x_range, density = model_density_data[model]
                        ax_pdf.fill_between(x_range, density, alpha=0.3, color='gray', label='Annual')
                    
                    # 然后绘制各个季节的PDF（细线，透明）
                    for j, season_name in enumerate(seasonal_names):
                        if (season_name in seasonal_density_data[model] and 
                            seasonal_density_data[model][season_name] is not None):
                            x_range_seasonal, density_seasonal = seasonal_density_data[model][season_name]
                            ax_pdf.plot(x_range_seasonal, density_seasonal, 
                                      linewidth=1.5, color=seasonal_colors[j], 
                                      alpha=0.6, label=season_name)
                else:
                    # 非annual季节：绘制单条PDF曲线
                    if model_density_data[model] is not None:
                        # 使用预计算的密度数据
                        x_range, density = model_density_data[model]
                        ax_pdf.plot(x_range, density, linewidth=2, color='steelblue')
                        ax_pdf.fill_between(x_range, density, alpha=0.3, color='steelblue')
                    elif len(bias_values) > 10:
                        try:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(bias_values)
                            x_range = np.linspace(bias_vmin, bias_vmax, 200)
                            density = kde(x_range)
                            ax_pdf.plot(x_range, density, linewidth=2, color='steelblue')
                            ax_pdf.fill_between(x_range, density, alpha=0.3, color='steelblue')
                        except Exception as e:
                            logger.debug(f"PDF计算失败 {model}: {e}")
                            # 如果核密度估计失败，绘制直方图
                            ax_pdf.hist(bias_values, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
                    else:
                        # 数据点太少，绘制简单直方图
                        ax_pdf.hist(bias_values, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
                
                # 设置PDF图属性（统一y轴范围）
                ax_pdf.set_xlim(bias_vmin, bias_vmax)
                ax_pdf.set_ylim(0, pdf_ymax)
                ax_pdf.set_xlabel(f'Bias ({VAR_CONFIG[var_type]["unit"]})', fontsize=9)
                ax_pdf.set_ylabel('Density', fontsize=9)
                ax_pdf.grid(True, alpha=0.3)
                ax_pdf.tick_params(labelsize=8)
            
            # ===== 第三步：添加colorbar =====
            unit = VAR_CONFIG[var_type]['unit']
            
            # 添加气候态colorbar
            if im_clim is not None:
                cbar_clim_ax = legend_ax.inset_axes([0.15, 0.65, 0.7, 0.08])
                cbar_clim = fig.colorbar(im_clim, cax=cbar_clim_ax, orientation='horizontal')
                cbar_clim.set_label(f'Climatology ({unit})', fontsize=10)
                cbar_clim.ax.tick_params(labelsize=9)
                cbar_clim.ax.xaxis.set_ticks_position('bottom')
                cbar_clim.ax.xaxis.set_label_position('bottom')
            
            # 添加偏差colorbar
            if im_bias is not None:
                cbar_bias_ax = legend_ax.inset_axes([0.15, 0.35, 0.7, 0.08])
                cbar_bias = fig.colorbar(im_bias, cax=cbar_bias_ax, orientation='horizontal')
                cbar_bias.set_label(f'Bias ({unit})', fontsize=10)
                cbar_bias.ax.tick_params(labelsize=9)
                cbar_bias.ax.xaxis.set_ticks_position('bottom')
                cbar_bias.ax.xaxis.set_label_position('bottom')
            
            # 如果是annual季节，添加季节PDF图例（在colorbar下方）
            if load_seasonal_data:
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='gray', alpha=0.3, label='Annual'),
                    Line2D([0], [0], color='#1f77b4', linewidth=2, alpha=0.6, label='DJF'),
                    Line2D([0], [0], color='#ff7f0e', linewidth=2, alpha=0.6, label='MAM'),
                    Line2D([0], [0], color='#2ca02c', linewidth=2, alpha=0.6, label='JJA'),
                    Line2D([0], [0], color='#d62728', linewidth=2, alpha=0.6, label='SON')
                ]
                
                # 在legend_ax下方添加图例
                legend = legend_ax.legend(handles=legend_elements, loc='lower center',
                                        ncol=5, frameon=True, fontsize=9,
                                        bbox_to_anchor=(0.5, 0.05),
                                        framealpha=0.9, edgecolor='gray')
            
            # 保存图像
            output_file_png = self.plots_dir / var_type / f"climatology_{var_type}_L{leadtime}_{season_str}.png"
            output_file_pdf = self.plots_dir / var_type / f"climatology_{var_type}_L{leadtime}_{season_str}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"图像已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制气候态组合图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_leadtimes_combined(self, var_type: str, leadtimes: List[int], season: Optional[str],
                                obs_clim: xr.DataArray, 
                                all_model_biases: Dict[int, Dict[str, xr.DataArray]],
                                clim_vmin: Optional[float] = None,
                                clim_vmax: Optional[float] = None):
        """
        绘制多个leadtime的合并图（只绘制空间分布图，不绘制PDF图）
        
        布局：
        每个leadtime占据2行：
        第1行：观测 + 3个模型
        第2行：4个模型
        观测图只绘制一次（在第一个leadtime的第一行），其他leadtime的观测位置留白
        只绘制Bias空间分布图
        
        Args:
            var_type: 变量类型
            leadtimes: 提前期列表
            season: 季节名称
            obs_clim: 观测气候态
            all_model_biases: 字典 {leadtime: {model: bias_data}}
            clim_vmin: 气候态colorbar最小值
            clim_vmax: 气候态colorbar最大值
        """
        try:
            season_str = season if season else 'annual'
            logger.info(f"绘制leadtime合并图: {var_type} L{leadtimes} {season_str}")
            
            # 准备模型列表，按照 MODELS 字典的顺序排列
            # 使用第一个leadtime的模型列表
            first_leadtime = leadtimes[0]
            if first_leadtime not in all_model_biases:
                logger.error(f"第一个leadtime {first_leadtime} 没有数据")
                return
            
            model_names = [m for m in MODELS.keys() if m in all_model_biases[first_leadtime]]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 收集所有leadtime的所有偏差数据，用于计算统一的colorbar范围
            all_bias_values = []
            for leadtime in leadtimes:
                if leadtime in all_model_biases:
                    for model_bias in all_model_biases[leadtime].values():
                        bias_valid = model_bias.values[np.isfinite(model_bias.values)]
                        all_bias_values.extend(bias_valid)
            
            # 计算偏差颜色范围（对称，保留所有信息）
            # 统一使用 coolwarm
            cmap_bias = 'coolwarm'
            
            if len(all_bias_values) > 0:
                all_bias_values = np.array(all_bias_values)
                all_bias_values = all_bias_values[np.isfinite(all_bias_values)]
                if len(all_bias_values) > 0:
                    bias_abs_max = np.percentile(np.abs(all_bias_values), 99)
                    if np.isfinite(bias_abs_max) and bias_abs_max > 0:
                        bias_vmin = -bias_abs_max
                        bias_vmax = bias_abs_max
                    else:
                        logger.warning("偏差范围计算结果无效，使用默认范围")
                        if var_type == 'temp':
                            bias_vmin = -5
                            bias_vmax = 5
                        else:
                            bias_vmin = -2
                            bias_vmax = 2
                else:
                    logger.warning("过滤后没有有效的偏差数据，使用默认范围")
                    if var_type == 'temp':
                        bias_vmin = -5
                        bias_vmax = 5
                    else:
                        bias_vmin = -2
                        bias_vmax = 2
            else:
                logger.warning("没有有效的偏差数据，使用默认范围")
                if var_type == 'temp':
                    bias_vmin = -5
                    bias_vmax = 5
                else:
                    bias_vmin = -2
                    bias_vmax = 2
            
            logger.info(f"统一偏差范围: [{bias_vmin:.2f}, {bias_vmax:.2f}]")
            
            # 计算布局
            # 每个leadtime占2行：第1行（观测+3模型），第2行（4模型）
            n_leadtimes = len(leadtimes)
            n_cols = 4  # 固定4列：观测/留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 使用固定画布大小（与 combined_pearson_analysis.py 一致）
            fig_width = 20
            fig_height = 12
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 基于观测网格计算经纬度边界，用于等高线填色图
            # 对于半度网格（如40.5代表40-41度），不需要额外扩展边界
            def _compute_edges(center_coords: np.ndarray) -> np.ndarray:
                """
                计算网格边界
                注意：对于半度网格（如40.5代表40-41度），中心坐标本身就代表网格中心
                """
                center_coords = np.asarray(center_coords)
                # 直接使用中心坐标，contourf会自动处理网格边界
                # 不需要额外计算edges，这样可以避免白边问题
                return center_coords
            
            lon_centers = obs_clim.lon.values if hasattr(obs_clim, 'lon') else None
            lat_centers = obs_clim.lat.values if hasattr(obs_clim, 'lat') else None
            if lon_centers is None or lat_centers is None:
                raise RuntimeError("观测气候态缺少经纬度坐标")
            
            # 对于等高线图，直接使用中心坐标，不需要计算edges
            lon_range = (float(lon_centers[0]), float(lon_centers[-1]))
            lat_range = (float(lat_centers[0]), float(lat_centers[-1]))
            
            # GridSpec布局设置
            height_ratios = [1] * n_rows
            width_ratios = [1] * n_cols
            left_margin = 0.05
            top_margin = 0.95
            bottom_margin = 0.08
            
            # 应用GridSpec（为右侧colorbar留出空间）
            gs = GridSpec(n_rows, n_cols, figure=fig,
                          height_ratios=height_ratios,
                          width_ratios=width_ratios,
                          hspace=0.25, wspace=0.15,
                          left=left_margin, right=0.85, top=top_margin, bottom=bottom_margin)
            
            # 预计算经纬度主刻度
            lon_tick_start = int(np.ceil(lon_range[0] / 15.0) * 15)
            lon_tick_end = int(np.floor(lon_range[1] / 15.0) * 15)
            lon_ticks = np.arange(lon_tick_start, lon_tick_end + 1, 15)
            lat_tick_start = int(np.ceil(lat_range[0] / 10.0) * 10)
            lat_tick_end = int(np.floor(lat_range[1] / 10.0) * 10)
            if lat_tick_end < lat_range[1] - 1e-6:
                lat_tick_end += 10
            lat_ticks = np.arange(lat_tick_start, lat_tick_end + 1, 10)
            lon_formatter = LongitudeFormatter(number_format='.0f')
            lat_formatter = LatitudeFormatter(number_format='.0f')
            
            # 用于保存colorbar的绘图对象
            im_obs = None
            im_bias = None
            # 收集内容子图以便统一处理边框与整体外框
            content_axes = []
            
            # 绘制每个leadtime（每个leadtime占2行）
            for lt_idx, leadtime in enumerate(leadtimes):
                if leadtime not in all_model_biases:
                    continue
                
                model_biases = all_model_biases[leadtime]
                
                # 计算该leadtime在GridSpec中的行索引
                row_start = lt_idx * 2  # 每个leadtime从第lt_idx*2行开始
                row_obs = row_start  # 观测行（第1行）
                row_models2 = row_start + 1  # 第2行（4个模型）
                
                # ===== 第1行：观测 + 3个模型 =====
                # 第一列：观测图（只在第一个leadtime绘制，其他留白）
                if lt_idx == 0:
                    ax_obs = fig.add_subplot(gs[row_obs, 0], projection=ccrs.PlateCarree())
                    # 使用数据范围设置显示区域
                    ax_obs.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
                    ax_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_obs.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_obs.add_feature(cfeature.LAND, alpha=0.1)
                    ax_obs.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 所有子图都显示坐标轴标签
                    gl = ax_obs.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xlabel_style = {'size': 12, 'color': 'black'}
                    gl.ylabel_style = {'size': 12, 'color': 'black'}
                    gl.xformatter = lon_formatter
                    gl.yformatter = lat_formatter
                    # 显示底部和左侧标签
                    gl.bottom_labels = True
                    gl.left_labels = True
                    
                    # 计算观测气候态颜色范围（应用显示单位转换）
                    if clim_vmin is not None and clim_vmax is not None:
                        # 应用显示转换
                        clim_vmin = self._apply_display_conversion(clim_vmin, var_type)
                        clim_vmax = self._apply_display_conversion(clim_vmax, var_type)
                    else:
                        obs_valid = obs_clim.values[np.isfinite(obs_clim.values)]
                        if var_type == 'temp':
                            if len(obs_valid) > 0:
                                obs_mean = np.mean(obs_valid)
                                obs_std = np.std(obs_valid)
                                clim_vmin = self._apply_display_conversion(obs_mean - 2 * obs_std, var_type)
                                clim_vmax = self._apply_display_conversion(obs_mean + 2 * obs_std, var_type)
                            else:
                                clim_vmin = -10
                                clim_vmax = 30
                        else:
                            clim_vmin = 0
                            if len(obs_valid) > 0:
                                clim_vmax = np.percentile(obs_valid, 98)
                            else:
                                clim_vmax = 10
                    
                    # 对于温度变量，使气候态colorbar关于0度对称
                    if var_type == 'temp':
                        max_abs = max(abs(clim_vmin), abs(clim_vmax))
                        clim_vmin = -max_abs
                        clim_vmax = max_abs
                        cmap_clim = 'coolwarm'
                    else:
                        cmap_clim = 'coolwarm_r'
                    
                    # 使用等高线填色图（应用显示转换）
                    n_levels = 15
                    levels = ticker.MaxNLocator(nbins=n_levels, prune=None).tick_values(clim_vmin, clim_vmax)
                    obs_clim_display = self._apply_display_conversion(obs_clim, var_type)
                    im_obs = ax_obs.contourf(obs_clim.lon, obs_clim.lat, obs_clim_display,
                                            transform=ccrs.PlateCarree(),
                                            cmap=cmap_clim, levels=levels, extend='both')
                    
                    ax_obs.text(0.02, 0.98, 'Observation',
                               transform=ax_obs.transAxes, fontsize=18, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_obs)
                else:
                    # 其他leadtime的观测位置留白
                    ax_blank = fig.add_subplot(gs[row_obs, 0])
                    ax_blank.axis('off')
                
                # 第1行：绘制前3个模型的Bias图（列1-3）
                for col_idx in range(3):
                    if col_idx >= len(model_names):
                        # 如果模型数量不足，留白
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[col_idx]
                    if model not in model_biases:
                        # 如果该leadtime没有该模型的数据，留白
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        continue
                    
                    model_bias = model_biases[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_obs, col_idx + 1], projection=ccrs.PlateCarree())
                    # 使用数据范围设置显示区域
                    ax_spatial.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 所有子图都显示坐标轴标签
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xlabel_style = {'size': 12, 'color': 'black'}
                    gl.ylabel_style = {'size': 12, 'color': 'black'}
                    gl.xformatter = lon_formatter
                    gl.yformatter = lat_formatter
                    # 显示底部和左侧标签
                    gl.bottom_labels = True
                    gl.left_labels = True
                    
                    # 使用等高线填色图
                    n_bias_levels = 15
                    bias_levels = ticker.MaxNLocator(nbins=n_bias_levels, prune=None).tick_values(bias_vmin, bias_vmax)
                    im_bias = ax_spatial.contourf(model_bias.lon, model_bias.lat, model_bias,
                                                 transform=ccrs.PlateCarree(),
                                                 cmap=cmap_bias, levels=bias_levels, extend='both')
                    
                    # 模型标签
                    label = chr(97 + col_idx)
                    ax_spatial.text(0.02, 0.96, f'({label}) {display_name} Bias',
                                   transform=ax_spatial.transAxes, fontsize=18, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='left')
                    
                    # 添加leadtime标签（在右上角）
                    if col_idx == 0:
                        ax_spatial.text(0.98, 0.96, f'L{leadtime}',
                                       transform=ax_spatial.transAxes, fontsize=18, fontweight='bold',
                                       verticalalignment='top', horizontalalignment='right')
                    
                    content_axes.append(ax_spatial)
                
                # ===== 第2行：4个模型 =====
                for col_idx in range(4):
                    if col_idx + 3 >= len(model_names):
                        # 如果模型数量不足，留白
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[col_idx + 3]  # 从第4个模型开始
                    if model not in model_biases:
                        # 如果该leadtime没有该模型的数据，留白
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model_bias = model_biases[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_models2, col_idx], projection=ccrs.PlateCarree())
                    # 使用数据范围设置显示区域
                    ax_spatial.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 所有子图都显示坐标轴标签
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xlabel_style = {'size': 12, 'color': 'black'}
                    gl.ylabel_style = {'size': 12, 'color': 'black'}
                    gl.xformatter = lon_formatter
                    gl.yformatter = lat_formatter
                    # 显示底部和左侧标签
                    gl.bottom_labels = True
                    gl.left_labels = True
                    
                    # 使用等高线填色图
                    n_bias_levels = 20
                    bias_levels = ticker.MaxNLocator(nbins=n_bias_levels, prune=None).tick_values(bias_vmin, bias_vmax)
                    im_bias = ax_spatial.contourf(model_bias.lon, model_bias.lat, model_bias,
                                                 transform=ccrs.PlateCarree(),
                                                 cmap=cmap_bias, levels=bias_levels, extend='both')
                    
                    # 模型标签
                    label = chr(97 + col_idx + 3)  # 从d开始
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                                   transform=ax_spatial.transAxes, fontsize=18, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_spatial)
            
            
            # 添加colorbar（在图的右侧，竖向排列）
            unit = VAR_CONFIG[var_type]['unit']
            
            # 添加观测colorbar（如果有，上半部分）
            if im_obs is not None:
                cbar_obs_ax = fig.add_axes([0.88, 0.55, 0.015, 0.35])  # 右侧上半部分
                cbar_obs = fig.colorbar(im_obs, cax=cbar_obs_ax, orientation='vertical')
                cbar_obs.set_label(f'Climatology ({unit})', fontsize=14, labelpad=10)
                cbar_obs.ax.tick_params(labelsize=12)
            
            # 添加偏差colorbar（下半部分）
            if im_bias is not None:
                cbar_bias_ax = fig.add_axes([0.88, 0.15, 0.015, 0.35])  # 右侧下半部分
                cbar_bias = fig.colorbar(im_bias, cax=cbar_bias_ax, orientation='vertical')
                cbar_bias.set_label(f'Bias ({unit})', fontsize=14, labelpad=10)
                cbar_bias.ax.tick_params(labelsize=12)
            
            # 保存图像（使用pad_inches=0确保无额外边距）
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.plots_dir / var_type / f"climatology_{var_type}_{leadtimes_str}_{season_str}.png"
            output_file_pdf = self.plots_dir / var_type / f"climatology_{var_type}_{leadtimes_str}_{season_str}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            logger.info(f"图像已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制leadtime合并图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run_rmse_leadtime_analysis(self, var_type: str, leadtimes: List[int],
                                   seasons: List[str], models: List[str]):
        """
        运行RMSE随leadtime变化的分析（等权月季节方案），并将(model, lead)的标量RMSE落盘到metrics文件。
        """
        return
        try:
            # 规范化季节标签：None -> 'annual'
            season_labels: List[str] = []
            for s in seasons:
                if s is None or s == 'annual':
                    season_labels.append('annual')
                else:
                    season_labels.append(s)
            # 为每个季节生成目标月份列表
            seasonal_targets: Dict[str, List[pd.Timestamp]] = {}
            for season in season_labels:
                targets = self.generate_target_months(1993, 2020, season if season != 'annual' else None)
                seasonal_targets[season] = targets
            
            # 逐季节计算RMSE矩阵并落盘
            for season in season_labels:
                obs_clim = self.load_obs_climatology(var_type, season if season != 'annual' else None)
                if obs_clim is None:
                    logger.warning(f"{season}: 观测气候态缺失，跳过RMSE计算")
                    continue
                # 收集各模型的时序（按目标月）以便复用
                model_to_lead_series: Dict[str, Dict[int, xr.DataArray]] = {}
                for model in models:
                    model_to_lead_series[model] = self.load_forecast_data_by_target_month(model, var_type, seasonal_targets[season])
                
                leads_sorted = list(leadtimes)
                model_names = list(models)
                rmse_matrix = np.full((len(model_names), len(leads_sorted)), np.nan, dtype=float)
                
                for mi, model in enumerate(model_names):
                    lead_to_series = model_to_lead_series.get(model, {})
                    for li, lt in enumerate(leads_sorted):
                        try:
                            if lt not in lead_to_series:
                                continue
                            da_time = lead_to_series[lt]
                            model_clim = self.calculate_seasonal_climatology(da_time, season if season != 'annual' else None)
                            if model_clim is None:
                                continue
                            # 计算bias场
                            obs_interp = obs_clim.interp(lat=model_clim.lat, lon=model_clim.lon, method='linear')
                            bias_field = model_clim - obs_interp
                            # 掩膜海洋
                            bias_field = bias_field.where(~obs_interp.isnull(), np.nan)
                            rmse_val = self._calculate_spatial_rmse_from_bias(bias_field)
                            rmse_matrix[mi, li] = rmse_val
                            logger.info(f"{var_type} {season} {model} L{lt}: RMSE={rmse_val:.4f}")
                        except Exception as e:
                            logger.debug(f"RMSE计算失败 {model} {season} L{lt}: {e}")
                            continue
                
                # 转为DataArray并落盘
                try:
                    da_rmse = xr.DataArray(
                        rmse_matrix,
                        coords={'model': model_names, 'lead': leads_sorted},
                        dims=('model', 'lead'),
                        name=f'rmse_{var_type}_{season}'
                    )
                    out_file = self.data_dir / var_type / f"metrics_rmse_{var_type}_{season}.nc"
                    da_rmse.to_netcdf(out_file)
                    logger.info(f"RMSE指标已保存: {out_file}")
                except Exception as e:
                    logger.error(f"保存RMSE指标失败 {season}: {e}")
        except Exception as e:
            logger.error(f"RMSE随leadtime分析失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def plot_rmse_leadtime_timeseries(self, var_type: str, leadtimes: List[int],
                                      seasons: List[str], models: List[str]):
        """
        绘制RMSE随leadtime变化的折线图（年度 + 各季节），横轴固定为传入leadtimes（通常为0..5）。
        直接从metrics文件读取RMSE。
        """
        return
        try:
            season_order = [
                ('annual', 'Annual'),
                ('DJF', 'DJF'),
                ('MAM', 'MAM'),
                ('JJA', 'JJA'),
                ('SON', 'SON')
            ]
            available_seasons = []
            rmse_data_per_season: Dict[str, xr.DataArray] = {}
            for key, label in season_order:
                f = self.data_dir / var_type / f"metrics_rmse_{var_type}_{key}.nc"
                if f.exists():
                    try:
                        with xr.open_dataarray(f) as da:
                            rmse_data_per_season[key] = da.load()
                            available_seasons.append((key, label))
                    except Exception:
                        pass
            if not available_seasons:
                logger.warning("无可用的RMSE指标文件，跳过RMSE折线图绘制")
                return
            
            fig_height = max(2.0 * len(available_seasons) + 1, 3.8)
            fig, axes = plt.subplots(len(available_seasons), 1, sharex=True, figsize=(10, fig_height))
            if len(available_seasons) == 1:
                axes = [axes]
            
            model_order = [m for m in MODELS.keys() if m in models]
            cmap = plt.get_cmap('tab10')
            color_map = {model: cmap(i % cmap.N) for i, model in enumerate(model_order)}
            legend_handles = []
            legend_labels = []
            all_y_vals: List[float] = []
            
            for ax, (season_key, season_label) in zip(axes, available_seasons):
                da_rmse = rmse_data_per_season[season_key]
                for model in model_order:
                    if model not in da_rmse.coords['model'].values:
                        continue
                    y_vals = []
                    x_vals = []
                    for lt in leadtimes:
                        if lt in da_rmse.coords['lead'].values:
                            val = float(da_rmse.sel(model=model, lead=lt).item())
                            if np.isfinite(val):
                                x_vals.append(lt)
                                y_vals.append(val)
                                all_y_vals.append(val)
                    if not x_vals:
                        continue
                    line, = ax.plot(
                        x_vals, y_vals,
                        marker='o', linewidth=1.8, markersize=5,
                        color=color_map[model], label=model
                    )
                    if model not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(model)
                ax.text(0.95, 0.05, season_label, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, fontweight='bold')
                ax.grid(True, axis='y', linestyle=':', alpha=0.4)
                ax.set_ylabel('')
            
            axes[-1].set_xlabel('Lead Time')
            axes[-1].set_xticks(leadtimes)
            axes[-1].set_xlim(leadtimes[0], leadtimes[-1])
            # 统一y轴范围
            if len(all_y_vals) > 0:
                y_min = float(np.min(all_y_vals))
                y_max = float(np.max(all_y_vals))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    if y_min == y_max:
                        delta = 1.0 if y_min == 0 else abs(y_min) * 0.1
                        y_min -= delta
                        y_max += delta
                    margin = 0.05 * (y_max - y_min)
                    y_min = max(0.0, y_min - margin)
                    y_max += margin
                    for ax in axes:
                        ax.set_ylim(y_min, y_max)
            unit = VAR_CONFIG[var_type]['unit']
            fig.text(0.08, 0.5, f'RMSE ({unit})', va='center', rotation='vertical')
            plt.subplots_adjust(left=0.14, right=0.97, top=0.98, bottom=0.18, hspace=0.0)
            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc='lower center',
                           ncol=min(4, len(legend_labels)), frameon=False, bbox_to_anchor=(0.5, 0.06))
            out_png = self.plots_dir / var_type / f"climatology_rmse_leadtime_series_{var_type}.png"
            out_pdf = self.plots_dir / var_type / f"climatology_rmse_leadtime_series_{var_type}.pdf"
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
            plt.close(fig)
            logger.info(f"RMSE随leadtime折线图已保存: {out_png}")
        except Exception as e:
            logger.error(f"绘制RMSE随leadtime折线图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def run_corr_leadtime_analysis(self, var_type: str, leadtimes: List[int],
                                   seasons: List[str], models: List[str]):
        """
        运行空间相关系数r随leadtime变化的分析，等权月季节方案；结果落盘到metrics文件。
        """
        try:
            # 规范化季节标签
            season_labels: List[str] = []
            for s in seasons:
                if s is None or s == 'annual':
                    season_labels.append('annual')
                else:
                    season_labels.append(s)
            # 生成目标月份
            seasonal_targets: Dict[str, List[pd.Timestamp]] = {}
            for season in season_labels:
                targets = self.generate_target_months(1993, 2020, season if season != 'annual' else None)
                seasonal_targets[season] = targets
            
            for season in season_labels:
                obs_clim = self.load_obs_climatology(var_type, season if season != 'annual' else None)
                if obs_clim is None:
                    logger.warning(f"{season}: 观测气候态缺失，跳过相关系数计算")
                    continue
                # 预载每模型的按目标月序列
                model_to_lead_series: Dict[str, Dict[int, xr.DataArray]] = {}
                for model in models:
                    model_to_lead_series[model] = self.load_forecast_data_by_target_month(model, var_type, seasonal_targets[season])
                
                leads_sorted = list(leadtimes)
                model_names = list(models)
                corr_matrix = np.full((len(model_names), len(leads_sorted)), np.nan, dtype=float)
                
                for mi, model in enumerate(model_names):
                    lead_to_series = model_to_lead_series.get(model, {})
                    for li, lt in enumerate(leads_sorted):
                        try:
                            if lt not in lead_to_series:
                                continue
                            da_time = lead_to_series[lt]
                            model_clim = self.calculate_seasonal_climatology(da_time, season if season != 'annual' else None)
                            if model_clim is None:
                                continue
                            r_val = self._calculate_spatial_corr(model_clim, obs_clim)
                            corr_matrix[mi, li] = r_val
                            logger.info(f"{var_type} {season} {model} L{lt}: Corr={r_val:.4f}")
                        except Exception as e:
                            logger.debug(f"相关系数计算失败 {model} {season} L{lt}: {e}")
                            continue
                try:
                    da_corr = xr.DataArray(
                        corr_matrix,
                        coords={'model': model_names, 'lead': leads_sorted},
                        dims=('model', 'lead'),
                        name=f'corr_{var_type}_{season}'
                    )
                    out_file = self.data_dir / var_type / f"metrics_corr_{var_type}_{season}.nc"
                    da_corr.to_netcdf(out_file)
                    logger.info(f"相关系数指标已保存: {out_file}")
                except Exception as e:
                    logger.error(f"保存相关系数指标失败 {season}: {e}")
        except Exception as e:
            logger.error(f"相关系数随leadtime分析失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def plot_corr_leadtime_timeseries(self, var_type: str, leadtimes: List[int],
                                      seasons: List[str], models: List[str]):
        """
        绘制空间相关系数r随leadtime变化折线图（年度+各季节），横轴为传入leadtimes（通常为0..5）。
        """
        return
        try:
            season_order = [
                ('annual', 'Annual'),
                ('DJF', 'DJF'),
                ('MAM', 'MAM'),
                ('JJA', 'JJA'),
                ('SON', 'SON')
            ]
            available_seasons = []
            corr_data_per_season: Dict[str, xr.DataArray] = {}
            for key, label in season_order:
                f = self.data_dir / var_type / f"metrics_corr_{var_type}_{key}.nc"
                if f.exists():
                    try:
                        with xr.open_dataarray(f) as da:
                            corr_data_per_season[key] = da.load()
                            available_seasons.append((key, label))
                    except Exception:
                        pass
            if not available_seasons:
                logger.warning("无可用的相关系数指标文件，跳过相关系数折线图绘制")
                return
            
            fig_height = max(2.0 * len(available_seasons) + 1, 3.8)
            fig, axes = plt.subplots(len(available_seasons), 1, sharex=True, figsize=(10, fig_height))
            if len(available_seasons) == 1:
                axes = [axes]
            
            model_order = [m for m in MODELS.keys() if m in models]
            cmap = plt.get_cmap('tab10')
            color_map = {model: cmap(i % cmap.N) for i, model in enumerate(model_order)}
            legend_handles = []
            legend_labels = []
            all_y_vals: List[float] = []
            
            for ax, (season_key, season_label) in zip(axes, available_seasons):
                da_corr = corr_data_per_season[season_key]
                for model in model_order:
                    if model not in da_corr.coords['model'].values:
                        continue
                    y_vals = []
                    x_vals = []
                    for lt in leadtimes:
                        if lt in da_corr.coords['lead'].values:
                            val = float(da_corr.sel(model=model, lead=lt).item())
                            if np.isfinite(val):
                                x_vals.append(lt)
                                y_vals.append(val)
                                all_y_vals.append(val)
                    if not x_vals:
                        continue
                    line, = ax.plot(
                        x_vals, y_vals,
                        marker='o', linewidth=1.8, markersize=5,
                        color=color_map[model], label=model
                    )
                    if model not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(model)
                ax.text(0.95, 0.05, season_label, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, fontweight='bold')
                ax.grid(True, axis='y', linestyle=':', alpha=0.4)
                ax.set_ylabel('')
            
            axes[-1].set_xlabel('Lead Time')
            axes[-1].set_xticks(leadtimes)
            axes[-1].set_xlim(leadtimes[0], leadtimes[-1])
            # 统一y轴范围
            if len(all_y_vals) > 0:
                y_min = float(np.min(all_y_vals))
                y_max = float(np.max(all_y_vals))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    if y_min == y_max:
                        delta = 0.1
                        y_min -= delta
                        y_max += delta
                    margin = 0.05 * (y_max - y_min)
                    y_min -= margin
                    y_max += margin
                    for ax in axes:
                        ax.set_ylim(y_min, y_max)
            fig.text(0.08, 0.5, 'Spatial Corr (r)', va='center', rotation='vertical')
            plt.subplots_adjust(left=0.14, right=0.97, top=0.98, bottom=0.18, hspace=0.0)
            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc='lower center',
                           ncol=min(4, len(legend_labels)), frameon=False, bbox_to_anchor=(0.5, 0.06))
            out_png = self.plots_dir / var_type / f"climatology_corr_leadtime_series_{var_type}.png"
            out_pdf = self.plots_dir / var_type / f"climatology_corr_leadtime_series_{var_type}.pdf"
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
            plt.close(fig)
            logger.info(f"相关系数随leadtime折线图已保存: {out_png}")
        except Exception as e:
            logger.error(f"绘制相关系数随leadtime折线图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def plot_bias_pdfs_mother(self, var_type: str, leads: List[int], seasons: List[str], models: List[str]):
        """
        绘制单独的PDF母图：上部L0、下部L3；每个lead分两行，第一行[空, 模型1, 模型2, 模型3]，第二行[模型4, 模型5, 模型6, 模型7]。
        每个模型子图叠加annual+DJF+MAM+JJA+SON的Bias PDF曲线；不绘制观测PDF；所有子图共享坐标轴与范围，子图间无缝隙，仅最外侧显示轴标签与脊线。
        """
        try:
            # 总是包含固定季节集合（annual + 四季）
            season_keys = ['annual', 'DJF', 'MAM', 'JJA', 'SON']
            # 仅保留我们需要的leads（通常为[0,3]）
            target_leads = [lt for lt in leads if lt in [0, 3]]
            if not target_leads:
                target_leads = [0, 3]
            # 模型顺序（最多7个）
            model_order = [m for m in MODELS.keys() if m in models]
            if len(model_order) == 0:
                logger.warning("无可用模型，跳过PDF母图绘制")
                return
            # 预扫描：确定PDF统一x范围与y上限
            all_bias_values = []
            for lt in target_leads:
                for model in model_order:
                    for season in season_keys:
                        bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{lt}_{var_type}_{season}.nc"
                        if bias_file.exists():
                            try:
                                with xr.open_dataarray(bias_file) as b:
                                    vals = b.values[np.isfinite(b.values)]
                                    if vals.size > 0:
                                        all_bias_values.extend(vals.tolist())
                            except Exception:
                                pass
            if all_bias_values:
                all_bias_values = np.array(all_bias_values)
                all_bias_values = all_bias_values[np.isfinite(all_bias_values)]
            if all_bias_values is None or len(all_bias_values) == 0:
                logger.warning("没有有效的bias数据用于PDF母图，跳过")
                return
            # 统一x范围（99%分位数对称）
            bias_abs_max = np.percentile(np.abs(all_bias_values), 99)
            if not np.isfinite(bias_abs_max) or bias_abs_max <= 0:
                if var_type == 'temp':
                    bias_abs_max = 5.0
                else:
                    bias_abs_max = 2.0
            x_min, x_max = -float(bias_abs_max), float(bias_abs_max)
            x_range = np.linspace(x_min, x_max, 300)
            
            # 计算所有密度以统一y轴
            from scipy.stats import gaussian_kde
            all_densities_max = 0.0
            density_cache: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}
            for lt in target_leads:
                for model in model_order:
                    for season in season_keys:
                        key = (lt, model, season)
                        bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{lt}_{var_type}_{season}.nc"
                        if not bias_file.exists():
                            continue
                        try:
                            with xr.open_dataarray(bias_file) as b:
                                vals = b.values[np.isfinite(b.values)]
                                if vals.size < 5:
                                    continue
                                kde = gaussian_kde(vals)
                                dens = kde(x_range)
                                density_cache[key] = (x_range, dens)
                                if dens.max() > all_densities_max:
                                    all_densities_max = float(dens.max())
                        except Exception:
                            continue
            if all_densities_max <= 0:
                all_densities_max = 1.0
            
            # 布局：4行×4列（上两行为L0，下两行为L3），共享坐标轴，无缝隙
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(4, 4, figure=fig, hspace=0.0, wspace=0.0,
                          left=0.06, right=0.97, top=0.95, bottom=0.12)
            axes_grid: List[List[Optional[plt.Axes]]] = [[None]*4 for _ in range(4)]
            # 颜色映射：annual+四季
            seasonal_colors = {
                'annual': 'gray',
                'DJF': '#1f77b4',
                'MAM': '#ff7f0e',
                'JJA': '#2ca02c',
                'SON': '#d62728'
            }
            # 绘制每个lead的两行
            for lead_idx, lt in enumerate(target_leads):
                row_start = lead_idx * 2  # 0或2
                # 第一行：空+3模型
                # 空白
                ax_blank = fig.add_subplot(gs[row_start, 0])
                ax_blank.axis('off')
                axes_grid[row_start][0] = ax_blank
                # 三个模型
                for col_idx in range(3):
                    model_idx = col_idx
                    if model_idx >= len(model_order):
                        ax = fig.add_subplot(gs[row_start, col_idx+1])
                        ax.axis('off')
                        axes_grid[row_start][col_idx+1] = ax
                        continue
                    model = model_order[model_idx]
                    ax = fig.add_subplot(gs[row_start, col_idx+1])
                    axes_grid[row_start][col_idx+1] = ax
                    # 绘制该模型的所有季节PDF
                    for season in season_keys:
                        key = (lt, model, season)
                        if key in density_cache:
                            xg, dens = density_cache[key]
                            if season == 'annual':
                                # annual 使用高透明度灰色填充，并绘制细线
                                ax.fill_between(xg, dens, color=seasonal_colors.get(season, 'gray'), alpha=0.25)
                                ax.plot(xg, dens, linewidth=2, color=seasonal_colors.get(season, 'gray'), alpha=0.9)
                            else:
                                ax.plot(xg, dens, linewidth=2, color=seasonal_colors.get(season, 'black'), alpha=0.85)
                    # 标注模型名与lead
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    if col_idx == 0:
                        ax.text(0.98, 0.95, f"L{lt}", transform=ax.transAxes, ha='right', va='top',
                                fontsize=18, fontweight='bold')
                    ax.text(0.02, 0.95, display_name, transform=ax.transAxes, ha='left', va='top',
                            fontsize=18, fontweight='bold')
                # 第二行：四个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(model_order):
                        ax = fig.add_subplot(gs[row_start+1, col_idx])
                        ax.axis('off')
                        axes_grid[row_start+1][col_idx] = ax
                        continue
                    model = model_order[model_idx]
                    ax = fig.add_subplot(gs[row_start+1, col_idx])
                    axes_grid[row_start+1][col_idx] = ax
                    for season in season_keys:
                        key = (lt, model, season)
                        if key in density_cache:
                            xg, dens = density_cache[key]
                            if season == 'annual':
                                ax.fill_between(xg, dens, color=seasonal_colors.get(season, 'gray'), alpha=0.25)
                                ax.plot(xg, dens, linewidth=2, color=seasonal_colors.get(season, 'gray'), alpha=0.9)
                            else:
                                ax.plot(xg, dens, linewidth=2, color=seasonal_colors.get(season, 'black'), alpha=0.85)
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    ax.text(0.02, 0.95, display_name, transform=ax.transAxes, ha='left', va='top',
                            fontsize=18, fontweight='bold')
            
            # 统一坐标范围、去除内侧刻度与脊线，仅最外层显示
            # 收集所有非空axes
            content_axes: List[plt.Axes] = []
            for r in range(4):
                for c in range(4):
                    ax = axes_grid[r][c]
                    if ax is None:
                        continue
                    if len(ax.lines) == 0:
                        continue
                    content_axes.append(ax)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0, all_densities_max * 1.1)
                    # 为每个内容子图添加矩形边框
                    try:
                        ax.add_patch(Rectangle(
                            (0, 0), 1, 1,
                            transform=ax.transAxes,
                            fill=False,
                            edgecolor='black',
                            linewidth=0.6,
                            zorder=1000
                        ))
                    except Exception:
                        pass
            # 仅外侧显示刻度与脊线
            for r in range(4):
                for c in range(4):
                    ax = axes_grid[r][c]
                    if ax is None:
                        continue
                    is_left = c == 0
                    is_bottom = r == 3
                    is_top = r == 0
                    is_right = c == 3
                    # 默认隐藏
                    ax.tick_params(labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    # 外侧打开
                    if is_left:
                        ax.tick_params(labelleft=True)
                        ax.spines['left'].set_visible(True)
                    if is_right:
                        ax.spines['right'].set_visible(True)
                    if is_top:
                        ax.spines['top'].set_visible(True)
                    if is_bottom:
                        ax.tick_params(labelbottom=True)
                        ax.spines['bottom'].set_visible(True)
            # 轴标签与图例
            unit = VAR_CONFIG[var_type]['unit']
            # fig.text(0.03, 0.55, 'Density', va='center', rotation='vertical')
            fig.text(0.5, 0.07, f'Bias ({unit})', ha='center', va='center', fontsize=18)
            # 图例（底部集中）
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', linewidth=4, label='Annual'),
                Line2D([0], [0], color='#1f77b4', linewidth=4, label='DJF'),
                Line2D([0], [0], color='#ff7f0e', linewidth=4, label='MAM'),
                Line2D([0], [0], color='#2ca02c', linewidth=4, label='JJA'),
                Line2D([0], [0], color='#d62728', linewidth=4, label='SON'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=False,
                       bbox_to_anchor=(0.5, 0.01), fontsize=16)
            # 保存
            out_png = self.plots_dir / var_type / f"climatology_bias_pdfs_{var_type}_L0_L3.png"
            out_pdf = self.plots_dir / var_type / f"climatology_bias_pdfs_{var_type}_L0_L3.pdf"
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
            plt.close(fig)
            logger.info(f"PDF母图已保存: {out_png}")
        except Exception as e:
            logger.error(f"绘制PDF母图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    def run_bias_leadtime_analysis(self, var_type: str, leadtimes: List[int],
                                   seasons: List[str], models: List[str]):
        """
        运行bias随leadtime变化的分析（等权月季节方案），并绘制折线图。
        保存每个 (model, season, lead) 的季节气候态与偏差到 NetCDF。
        """
        try:
            # 规范化季节标签：None -> 'annual'
            season_labels: List[str] = []
            for s in seasons:
                if s is None or s == 'annual':
                    season_labels.append('annual')
                else:
                    season_labels.append(s)
            # 生成每个季节的目标月份
            seasonal_targets: Dict[str, List[pd.Timestamp]] = {}
            for season in season_labels:
                targets = self.generate_target_months(1993, 2020, season if season != 'annual' else None)
                logger.info(f"{season}: 目标月份 {len(targets)} 个")
                seasonal_targets[season] = targets
            # seasonal bias means: season -> model -> lead -> mean
            seasonal_bias_means: Dict[str, Dict[str, Dict[int, float]]] = {}
            for season in season_labels:
                seasonal_bias_means[season] = {}
                obs_clim = self.load_obs_climatology(var_type, season if season != 'annual' else None)
                if obs_clim is None:
                    logger.warning(f"{season}: 观测气候态缺失，跳过")
                    continue
                for model in models:
                    seasonal_bias_means[season][model] = {}
                    lead_to_series = self.load_forecast_data_by_target_month(model, var_type, seasonal_targets[season])
                    if not lead_to_series:
                        logger.warning(f"{season} {model}: 无可用预报数据")
                        for lt in leadtimes:
                            seasonal_bias_means[season][model][lt] = float('nan')
                        continue
                    for lt in leadtimes:
                        if lt not in lead_to_series:
                            seasonal_bias_means[season][model][lt] = float('nan')
                            continue
                        da_time = lead_to_series[lt]
                        model_clim = self.calculate_seasonal_climatology(da_time, season if season != 'annual' else None)
                        if model_clim is None:
                            seasonal_bias_means[season][model][lt] = float('nan')
                            continue
                        bias_field = self.calculate_bias(model_clim, obs_clim)
                        if bias_field is None:
                            seasonal_bias_means[season][model][lt] = float('nan')
                            continue
                        # 保存季节气候态与偏差
                        try:
                            clim_file = self.data_dir / var_type / f"climatology_{model}_L{lt}_{var_type}_{season}.nc"
                            bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{lt}_{var_type}_{season}.nc"
                            model_clim.to_netcdf(clim_file)
                            bias_field.to_netcdf(bias_file)
                        except Exception:
                            pass
                        mean_val = self._calculate_spatial_mean(bias_field)
                        seasonal_bias_means[season][model][lt] = mean_val
                        logger.info(f"{var_type} {season} {model} L{lt}: bias={mean_val:.4f}")
            # 绘制折线图
        except Exception as e:
            logger.error(f"bias随leadtime分析失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def plot_bias_leadtime_timeseries(self, var_type: str, leadtimes: List[int],
                                      seasonal_bias_means: Dict[str, Dict[str, Dict[int, float]]]):
        """
        绘制bias随leadtime变化的折线图（年度 + 各季节），确保lead定义对应目标月份。

        Args:
            var_type: 变量类型（如'temp', 'prec'）
            leadtimes: leadtime的列表（提前期）
            seasonal_bias_means: 预存储的bias均值字典，格式为[season][model][leadtime] = val
        """
        logger.info("按用户要求：已禁用 bias 随 leadtime 的折线图绘制（不再输出该图）。")
        return
        try:
            # 定义plot中季节的顺序和对应的英文标记
            season_order = [
                ('annual', 'Annual'),
                ('DJF', 'DJF'),
                ('MAM', 'MAM'),
                ('JJA', 'JJA'),
                ('SON', 'SON')
            ]
            # 需要展示的leadtimes，一般为一组固定的leadtime，如0~5
            desired_leadtimes = BIAS_TIMESERIES_LEADS
            # 设定所有模式的顺序
            model_order = list(MODELS.keys())
            
            # 三个缓存字典
            obs_cache: Dict[str, Optional[xr.DataArray]] = {}         # 季节->观测气候态
            fcst_cache: Dict[Tuple[str, int], Optional[xr.DataArray]] = {}  # (模式, lead) -> 预报时序
            bias_cache: Dict[Tuple[str, str, int], float] = {}        # (模式, 季节, lead) -> bias均值
            
            # 读取观测气候态（按季节），带缓存，加速插值等操作
            def get_obs_clim(season_key: str) -> Optional[xr.DataArray]:
                if season_key in obs_cache:
                    return obs_cache[season_key]
                obs_file = self.data_dir / var_type / f"climatology_obs_{var_type}_{season_key}.nc"
                if obs_file.exists():
                    try:
                        with xr.open_dataarray(obs_file) as obs_da:
                            obs_cache[season_key] = obs_da.load()
                    except Exception:
                        obs_cache[season_key] = None
                else:
                    obs_cache[season_key] = None
                return obs_cache[season_key]
            
            # 季节名到月份列表的映射
            season_months_map = {
                'annual': list(range(1, 13)),
                'DJF': SEASONS['DJF'],
                'MAM': SEASONS['MAM'],
                'JJA': SEASONS['JJA'],
                'SON': SEASONS['SON']
            }
            
            # 智能提取bias均值，支持优先用已有NetCDF bias文件，否则现场计算
            def resolve_bias_mean(model: str, season_key: str, lt: int) -> float:
                cache_key = (model, season_key, lt)
                if cache_key in bias_cache:
                    return bias_cache[cache_key]
                
                # 优先直接读取已保存的bias均值NetCDF（如果有）
                bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{lt}_{var_type}_{season_key}.nc"
                if bias_file.exists():
                    try:
                        with xr.open_dataarray(bias_file) as bias_da:
                            val = float(bias_da.mean(skipna=True).item())
                        bias_cache[cache_key] = val
                        return val
                    except Exception:
                        pass
                
                # 若无，读取气候态场，插值观测场，计算空间均值bias
                model_file = self.data_dir / var_type / f"climatology_{model}_L{lt}_{var_type}_{season_key}.nc"
                obs_clim = get_obs_clim(season_key)
                if model_file.exists() and obs_clim is not None:
                    try:
                        with xr.open_dataarray(model_file) as model_da:
                            model_clim = model_da.load()
                        # 插值观测场到模式格点
                        obs_interp = obs_clim.interp(lat=model_clim.lat, lon=model_clim.lon, method='linear')
                        bias_field = model_clim - obs_interp
                        val = float(bias_field.mean(skipna=True).item())
                        bias_cache[cache_key] = val
                        return val
                    except Exception:
                        pass
                
                # 若无保存，则现场计算bias
                obs_clim = get_obs_clim(season_key)
                if obs_clim is None:
                    bias_cache[cache_key] = float(np.nan)
                    return bias_cache[cache_key]
                
                fcst_key = (model, lt)
                # 缓存模式预报时序数据
                if fcst_key not in fcst_cache:
                    fcst_cache[fcst_key] = self.load_forecast_data(model, var_type, lt)
                fcst_data = fcst_cache[fcst_key]
                # 数据无效直接返回nan
                if fcst_data is None or 'time' not in fcst_data.coords:
                    bias_cache[cache_key] = float(np.nan)
                    return bias_cache[cache_key]
                
                try:
                    # 等权月方案：季内各目标月分别月平均，再等权平均
                    season_months = season_months_map.get(season_key, list(range(1, 13)))
                    monthly_means = []
                    for m in season_months:
                        sel_m = fcst_data.sel(time=fcst_data.time.dt.month == m)
                        if sel_m.size == 0:
                            continue
                        monthly_means.append(sel_m.mean(dim='time', skipna=True))
                except Exception:
                    bias_cache[cache_key] = float(np.nan)
                    return bias_cache[cache_key]
                
                if len(monthly_means) == 0:
                    bias_cache[cache_key] = float(np.nan)
                    return bias_cache[cache_key]
                
                try:
                    fcst_clim = xr.concat(monthly_means, dim='mm').mean(dim='mm', skipna=True)
                    obs_interp = obs_clim.interp(lat=fcst_clim.lat, lon=fcst_clim.lon, method='linear')
                    bias_field = fcst_clim - obs_interp
                    val = float(bias_field.mean(skipna=True).item())
                    # 按需落盘该模型/季节/lead 的季节气候态与偏差
                    try:
                        season_out = season_key
                        clim_file = self.data_dir / var_type / f"climatology_{model}_L{lt}_{var_type}_{season_out}.nc"
                        bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{lt}_{var_type}_{season_out}.nc"
                        fcst_clim.to_netcdf(clim_file)
                        bias_field.to_netcdf(bias_file)
                    except Exception:
                        pass
                except Exception:
                    val = float(np.nan)
                bias_cache[cache_key] = val
                return val
            
            # 得到[season][model][lead] = bias均值的大字典，优先用已给定值，否则补算
            resolved_bias_means: Dict[str, Dict[str, Dict[int, float]]] = {}
            for season_key, _ in season_order:
                season_precomputed = seasonal_bias_means.get(season_key, {})
                season_model_means: Dict[str, Dict[int, float]] = {}
                for model in model_order:
                    model_leads: Dict[int, float] = {}
                    precomputed = season_precomputed.get(model, {})
                    for lt in desired_leadtimes:
                        val = precomputed.get(lt, np.nan)
                        if not np.isfinite(val):
                            val = resolve_bias_mean(model, season_key, lt)
                        model_leads[lt] = val
                    season_model_means[model] = model_leads
                resolved_bias_means[season_key] = season_model_means
            
            # 检查哪些季节有效（有可用的bias数据才画），构造有效季节列表
            available_seasons = []
            for season_key, season_label in season_order:
                season_data = resolved_bias_means.get(season_key, {})
                has_data = any(
                    np.isfinite(val)
                    for model_bias in season_data.values()
                    for val in model_bias.values()
                )
                if has_data:
                    available_seasons.append((season_key, season_label))
            
            # 如果都没有可用数据，则跳过绘图
            if not available_seasons:
                logger.warning("无可用的季节bias数据，跳过leadtime折线图绘制")
                return
            
            # 自动设置画布高，适应不同季节数量，保证美观
            fig_height = max(2.0 * len(available_seasons) + 1, 3.8)
            fig, axes = plt.subplots(len(available_seasons), 1, sharex=True, figsize=(10, fig_height))
            # 保证axes永远为list型，兼容单子图和多子图的写法
            if len(available_seasons) == 1:
                axes = [axes]
            
            # 设置模式与颜色映射，便于在legend中区分
            cmap = plt.get_cmap('tab10')
            color_map = {model: cmap(i % cmap.N) for i, model in enumerate(model_order)}
            legend_handles = []
            legend_labels = []
            
            # 依次对每个季节画一张子图
            all_y_vals: List[float] = []
            for ax, (season_key, season_label) in zip(axes, available_seasons):
                # 取出季节所有模式的数据
                data_for_season = resolved_bias_means.get(season_key, {})
                
                # 在该季节下，对每个模式做一条曲线
                for model in model_order:
                    lead_to_bias = data_for_season.get(model, {})
                    x_vals: List[int] = []
                    y_vals: List[float] = []
                    for lt in desired_leadtimes:
                        bias_val = lead_to_bias.get(lt)
                        if bias_val is not None and np.isfinite(bias_val):
                            x_vals.append(lt)
                            y_vals.append(bias_val)
                            all_y_vals.append(bias_val)
                    if not x_vals:
                        continue
                    line, = ax.plot(
                        x_vals,
                        y_vals,
                        marker='o',
                        linewidth=1.8,
                        markersize=5,
                        color=color_map[model],
                        label=model
                    )
                    # 只记录一次legend
                    if model not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(model)
                
                # 在子图右下角添加季节大标签
                ax.text(
                    0.95,
                    0.05,
                    season_label,
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )
                # 添加y方向网格
                ax.grid(True, axis='y', linestyle=':', alpha=0.4)
                ax.set_ylabel('')
            
            # 底部x轴&范围设置
            axes[-1].set_xlabel('Lead Time')
            axes[-1].set_xticks(desired_leadtimes)
            axes[-1].set_xlim(desired_leadtimes[0], desired_leadtimes[-1])
            # 统一y轴范围（所有子图共享）
            if len(all_y_vals) > 0:
                y_min = float(np.min(all_y_vals))
                y_max = float(np.max(all_y_vals))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    if y_min == y_max:
                        delta = 1.0 if y_min == 0 else abs(y_min) * 0.1
                        y_min -= delta
                        y_max += delta
                    margin = 0.05 * (y_max - y_min)
                    y_min -= margin
                    y_max += margin
                    for ax in axes:
                        ax.set_ylim(y_min, y_max)
            unit = VAR_CONFIG[var_type]['unit']
            # 左y轴大标签
            fig.text(0.08, 0.5, f'Bias ({unit})', va='center', rotation='vertical')
            # 紧致布局
            plt.subplots_adjust(left=0.14, right=0.97, top=0.98, bottom=0.18, hspace=0.0)
            
            # 下方联合legend
            if legend_handles:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc='lower center',
                    ncol=min(4, len(legend_labels)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.06)
                )
            
            # 输出保存图片（PNG/PDF）到plots目录
            output_file_png = self.plots_dir / var_type / f"climatology_bias_leadtime_series_{var_type}.png"
            output_file_pdf = self.plots_dir / var_type / f"climatology_bias_leadtime_series_{var_type}.pdf"
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close(fig)
            logger.info(f"bias随leadtime折线图已保存: {output_file_png}")
        except Exception as e:
            logger.error(f"绘制bias随leadtime折线图失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    def run_analysis(self, var_types: List[str], leadtimes: List[int], seasons: List[str],
                    models: List[str], parallel: bool = True, n_jobs: int = 32,
                    plot_only: bool = False, save_data: bool = True,
                    run_bias_timeseries: bool = True):
        """
        运行气候态分析
        
        Args:
            var_types: 变量类型列表
            leadtimes: 提前期列表
            seasons: 季节列表 (包括'annual')
            models: 模型列表
            parallel: 是否使用并行处理
            n_jobs: 并行作业数
            plot_only: 仅绘图模式，从已保存的数据绘图
            save_data: 是否保存数据到NetCDF
        """
        logger.info(f"开始气候态分析")
        logger.info(f"变量: {var_types}")
        logger.info(f"提前期: {leadtimes}")
        logger.info(f"季节: {seasons}")
        logger.info(f"模型数量: {len(models)}")
        logger.info(f"并行处理: {parallel}, 作业数: {n_jobs}")
        logger.info(f"仅绘图模式: {plot_only}, 保存数据: {save_data}")
        
        for var_type in var_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理变量: {var_type.upper()}")
            logger.info(f"{'='*60}")
            seasonal_bias_means: Dict[str, Dict[str, Dict[int, float]]] = {}
            
            # 如果不是仅绘图模式，需要加载原始观测数据
            obs_data = None
            if not plot_only:
                obs_data = self.load_obs_data(var_type)
                if obs_data is None:
                    logger.error(f"观测数据加载失败，跳过 {var_type}")
                    continue
            
            # 按季节循环（外层），确保观测气候态对每个季节只计算一次
            for season in seasons:
                season_str = season if season else 'annual'
                seasonal_bias_means.setdefault(season_str, {})
                
                # ===== 计算或加载观测气候态（与leadtime无关）=====
                obs_clim = None
                obs_clim_file = self.data_dir / var_type / f"climatology_obs_{var_type}_{season_str}.nc"
                
                if plot_only:
                    # 仅绘图模式：从文件加载观测气候态
                    if obs_clim_file.exists():
                        obs_clim = xr.open_dataarray(obs_clim_file)
                        logger.info(f"加载观测气候态: {season_str}")
                    else:
                        logger.error(f"观测气候态文件不存在: {obs_clim_file}")
                        continue
                else:
                    # 计算模式：计算观测气候态
                    logger.info(f"计算观测气候态: {var_type} {season_str}")
                    obs_clim = self.calculate_climatology(obs_data, season)
                    if obs_clim is None:
                        logger.error(f"观测气候态计算失败")
                        continue
                    
                    # 保存观测气候态（只保存一次）
                    if save_data:
                        obs_clim.to_netcdf(obs_clim_file)
                        logger.info(f"观测气候态已保存: {obs_clim_file}")
                
                # 输出观测气候态的数据范围
                valid_obs = obs_clim.values[np.isfinite(obs_clim.values)]
                if len(valid_obs) > 0:
                    logger.info(f"观测气候态范围: [{np.min(valid_obs):.2f}, {np.max(valid_obs):.2f}] {VAR_CONFIG[var_type]['unit']}")
                
                # ===== 计算固定的气候态colorbar范围（基于观测，所有leadtime使用相同范围）=====
                if len(valid_obs) > 0:
                    if var_type == 'temp':
                        obs_mean = np.mean(valid_obs)
                        obs_std = np.std(valid_obs)
                        fixed_clim_vmin = obs_mean - 2 * obs_std
                        fixed_clim_vmax = obs_mean + 2 * obs_std
                    else:  # prec
                        fixed_clim_vmin = 0
                        fixed_clim_vmax = np.percentile(valid_obs, 98)
                    logger.info(f"固定气候态colorbar范围（用于所有leadtime）: [{fixed_clim_vmin:.2f}, {fixed_clim_vmax:.2f}]")
                else:
                    # 如果没有有效观测数据，使用None，让绘图函数使用默认值
                    fixed_clim_vmin = None
                    fixed_clim_vmax = None
                    logger.warning("没有有效的观测数据，将使用默认colorbar范围")
                
                # ===== 按leadtime循环处理模型数据，收集所有leadtime的数据 =====
                all_model_biases = {}  # {leadtime: {model: bias_data}}
                
                for leadtime in leadtimes:
                    logger.info(f"\n处理: {var_type} L{leadtime} {season_str}")
                    
                    if plot_only:
                        # ===== 仅绘图模式：从文件加载模型数据 =====
                        logger.info("仅绘图模式：加载已保存的模型数据...")
                        _, model_clims, model_biases = self.load_climatology_data(
                            var_type, leadtime, season, models
                        )
                        
                        if len(model_clims) == 0:
                            logger.warning(f"模型数据加载失败，跳过 L{leadtime}")
                            continue
                        
                        logger.info(f"成功加载 {len(model_clims)} 个模型的数据")
                        
                    else:
                        # ===== 计算模式：计算模型气候态和偏差 =====
                        model_clims = {}
                        model_biases = {}
                        
                        if parallel and len(models) > 1:
                            # 并行处理
                            max_workers = min(n_jobs, cpu_count(), len(models))
                            logger.info(f"使用并行处理: {max_workers} 个进程")
                            
                            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                                future_to_model = {
                                    executor.submit(_process_single_model_climatology,
                                                  (model, var_type, leadtime, season, obs_clim)): model
                                    for model in models
                                }
                                
                                completed = 0
                                for future in as_completed(future_to_model):
                                    model = future_to_model[future]
                                    try:
                                        result = future.result(timeout=900)
                                        if result is not None:
                                            model_name, clim_data, bias_data = result
                                            model_clims[model_name] = clim_data
                                            model_biases[model_name] = bias_data
                                            completed += 1
                                            logger.info(f"完成 {model} L{leadtime} {season_str} ({completed}/{len(models)})")
                                        else:
                                            logger.warning(f"模型 {model} 处理失败")
                                    except Exception as e:
                                        logger.error(f"模型 {model} 处理出错: {e}")
                        else:
                            # 串行处理
                            logger.info("使用串行处理")
                            for i, model in enumerate(models):
                                try:
                                    result = _process_single_model_climatology(
                                        (model, var_type, leadtime, season, obs_clim)
                                    )
                                    if result is not None:
                                        model_name, clim_data, bias_data = result
                                        model_clims[model_name] = clim_data
                                        model_biases[model_name] = bias_data
                                        logger.info(f"完成 {model} L{leadtime} {season_str} ({i+1}/{len(models)})")
                                    else:
                                        logger.warning(f"模型 {model} 处理失败")
                                except Exception as e:
                                    logger.error(f"模型 {model} 处理出错: {e}")
                        
                        # 检查是否有足够的模型数据
                        if len(model_clims) == 0:
                            logger.warning(f"没有可用的模型数据，跳过 L{leadtime}")
                            continue
                        
                        logger.info(f"成功计算 {len(model_clims)} 个模型的气候态和偏差")
                        
                        # 保存模型数据（不包含观测气候态，因为已经保存过了）
                        if save_data:
                            logger.info("保存模型气候态和偏差数据到NetCDF...")
                            for model in model_clims.keys():
                                # 气候态
                                clim_file = self.data_dir / var_type / f"climatology_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                                model_clims[model].to_netcdf(clim_file)
                                
                                # 偏差
                                bias_file = self.data_dir / var_type / f"climatology_bias_{model}_L{leadtime}_{var_type}_{season_str}.nc"
                                model_biases[model].to_netcdf(bias_file)
                            
                            logger.info(f"已保存 {len(model_clims)} 个模型的气候态和偏差数据")
                    
                    # 保存该leadtime的偏差数据
                    if len(model_biases) > 0:
                        all_model_biases[leadtime] = model_biases
                        for model_name, bias_da in model_biases.items():
                            try:
                                mean_val = float(bias_da.mean(skipna=True).item())
                            except Exception:
                                mean_val = float(np.nan)
                            if not np.isfinite(mean_val):
                                mean_val = float(np.nan)
                            seasonal_bias_means[season_str].setdefault(model_name, {})[leadtime] = mean_val
                
                # ===== 绘制合并图（所有leadtime合并在一起）=====
                if len(all_model_biases) > 0:
                    self.plot_leadtimes_combined(
                        var_type, leadtimes, season, obs_clim, all_model_biases,
                        clim_vmin=fixed_clim_vmin, clim_vmax=fixed_clim_vmax
                    )
                else:
                    logger.warning(f"没有可用的模型偏差数据，跳过绘图")
            
            # ===== 运行 RMSE/Corr 折线分析与PDF母图（仅在请求时；横轴固定0..5）=====
            if run_bias_timeseries:
                lead_for_lines = BIAS_TIMESERIES_LEADS
                # 固定按年度+四季进行分析与绘图（不受CLI --seasons影响）
                season_labels: List[str] = ['annual', 'DJF', 'MAM', 'JJA', 'SON']
                # RMSE（仅非plot_only时执行计算与落盘；plot_only下只绘图）
                if not plot_only:
                    self.run_rmse_leadtime_analysis(var_type, lead_for_lines, season_labels, models)
                else:
                    logger.info("仅绘图模式：跳过RMSE分析计算，仅从metrics绘图")
                # Corr（仅非plot_only时执行计算与落盘；plot_only下只绘图）
                if not plot_only:
                    self.run_corr_leadtime_analysis(var_type, lead_for_lines, season_labels, models)
                else:
                    logger.info("仅绘图模式：跳过相关系数分析计算，仅从metrics绘图")
                # PDF母图（L0/L3）
                self.plot_bias_pdfs_mother(var_type, leads=[0, 3], seasons=season_labels, models=models)
        
        logger.info(f"\n{'='*60}")
        logger.info("气候态分析完成！")
        logger.info(f"数据保存在: {self.data_dir}")
        logger.info(f"图像保存在: {self.plots_dir}")
        logger.info(f"{'='*60}")


def _process_single_model_climatology(args):
    """
    处理单个模型的气候态和偏差计算（用于并行处理）
    
    Args:
        args: (model, var_type, leadtime, season, obs_clim)
    
    Returns:
        (model_name, climatology_data, bias_data) 或 None
    """
    model, var_type, leadtime, season, obs_clim = args
    
    try:
        # 创建临时分析器实例
        analyzer = ClimatologyAnalyzer()
        
        # 加载预报数据
        fcst_data = analyzer.load_forecast_data(model, var_type, leadtime)
        if fcst_data is None:
            return None
        
        # 计算气候态
        clim_data = analyzer.calculate_climatology(fcst_data, season)
        if clim_data is None:
            return None
        
        # 输出模型气候态的数据范围
        valid_clim = clim_data.values[np.isfinite(clim_data.values)]
        if len(valid_clim) > 0:
            logger.info(f"{model} 气候态范围: [{np.min(valid_clim):.2f}, {np.max(valid_clim):.2f}]")
        
        # 计算偏差（模型 - 观测）
        # 插值观测数据到模型网格，然后逐格点计算偏差
        try:
            # 插值观测数据到模型网格
            obs_interp = obs_clim.interp(lat=clim_data.lat, lon=clim_data.lon, method='linear')
            
            # 直接逐格点计算偏差
            bias_data = clim_data - obs_interp
            
            # 排除海洋区域：将观测为NaN的位置（海洋）的bias也设为NaN
            bias_data = bias_data.where(~obs_interp.isnull(), np.nan)
            
            # 输出偏差范围
            valid_bias = bias_data.values[np.isfinite(bias_data.values)]
            if len(valid_bias) > 0:
                logger.info(f"{model} 偏差范围: [{np.min(valid_bias):.2f}, {np.max(valid_bias):.2f}]")
            
        except Exception as e:
            logger.error(f"计算偏差失败 {model}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果插值失败，返回全NaN的偏差数据
            bias_data = xr.full_like(clim_data, np.nan)
        
        # 验证bias数据有效性
        n_valid_bias = np.sum(np.isfinite(bias_data.values))
        total_bias = bias_data.size
        logger.info(f"{model} bias数据有效性: {n_valid_bias}/{total_bias} ({n_valid_bias/total_bias*100:.1f}%)")
        if n_valid_bias == 0:
            logger.warning(f"{model} bias数据全为NaN！")
        
        return (model, clim_data, bias_data)
        
    except Exception as e:
        logger.error(f"处理模型 {model} 失败: {e}")
        return None


def parse_args():
    """解析命令行参数"""
    parser = create_parser(
        description='气候态分析和绘图',
        include_seasons=True,
        include_data=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # climatology_analysis 特有的默认值
    parser.set_defaults(parallel=True, n_jobs=32)
    
    # 处理seasons参数的特殊逻辑
    parser.set_defaults(seasons=['all'])
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 解析参数
    models = parse_models(args.models, list(MODELS.keys())) if args.models else list(MODELS.keys())
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_types = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    # 处理季节列表
    if 'all' in args.seasons:
        seasons = [None]  # 默认只绘制annual（None表示annual）
    else:
        seasons = []
        for s in args.seasons:
            if s.lower() == 'annual':
                seasons.append(None)
            else:
                seasons.append(s)
    
    # 处理参数逻辑
    save_data = not args.no_save
    
    logger.info(f"配置：")
    logger.info(f"  变量: {args.var}")
    logger.info(f"  提前期: {args.leadtimes}")
    logger.info(f"  季节: {[s if s else 'annual' for s in seasons]}")
    logger.info(f"  模型: {models}")
    logger.info(f"  并行处理: {args.parallel}")
    logger.info(f"  并行作业数: {args.n_jobs}")
    logger.info(f"  仅绘图模式: {args.plot_only}")
    logger.info(f"  保存数据: {save_data}")
    
    # 创建分析器并运行分析
    analyzer = ClimatologyAnalyzer()
    analyzer.run_analysis(
        var_types=var_types,
        leadtimes=leadtimes,
        seasons=seasons,
        models=models,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        plot_only=args.plot_only,
        save_data=save_data
    )


if __name__ == "__main__":
    main()


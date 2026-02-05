#!/usr/bin/env python3
"""
Index Analysis

功能：
1. 计算和分析 EAWM (East Asian Winter Monsoon) 指数
2. 计算和分析 Nino3.4 指数
3. 绘制多模式平均 (MMM) 时间序列图 (参考 nino34_eawm_index_calculation.ipynb 的绘图逻辑)

使用方法：
python index_analysis.py --models all --leadtimes 0 3
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加工具包路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    CLIMATOLOGY_PERIOD,
    SPATIAL_BOUNDS as COMMON_SPATIAL_BOUNDS,
    COLORS,
)

from src.utils.data_loader import DataLoader
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, normalize_parallel_args

warnings.filterwarnings('ignore')

# 配置绘图参数 (参考 notebook)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# 配置日志
logger = setup_logging(
    log_file='index_analysis.log',
    module_name=__name__
)

# 全局配置
MODELS = MODEL_LIST
SPATIAL_BOUNDS = COMMON_SPATIAL_BOUNDS

# ================= 新增的多进程 Worker 函数 =================
def _worker_task(model, leadtime):
    """
    独立函数用于多进程执行。
    必须在类外部定义，以便被 pickle 序列化。
    """
    # 在子进程中重新初始化 Analyzer，避免共享状态
    # 这里的开销很小，因为只有计算逻辑，没有预加载大文件
    analyzer = IndexAnalyzer() 
    
    # 调用原有的处理逻辑
    return (model, leadtime), analyzer._process_single_model_leadtime(model, leadtime)
# ==========================================================

class IndexAnalyzer:
    """
    指数分析器：负责EAWM和Nino3.4指数的计算与绘图
    """
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader or DataLoader()
        logger.info(f"初始化指数分析器")

    def area_weighted_mean(self, da: xr.DataArray, lat_name: str = 'lat') -> xr.DataArray:
        """
        计算面积加权平均（使用 cos(latitude) 权重）
        """
        try:
            # 计算权重：cos(latitude in radians)
            weights = np.cos(np.deg2rad(da[lat_name]))
            
            # 对空间维度进行加权平均
            if 'lon' in da.dims:
                da_lon_mean = da.mean(dim='lon')
            else:
                da_lon_mean = da
            
            weighted_mean = (da_lon_mean * weights).sum(dim=lat_name) / weights.sum()
            return weighted_mean
        except Exception as e:
            logger.error(f"面积加权平均失败: {e}")
            raise

    def compute_monthly_anomaly(self, series: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """
        计算逐月气候态异常
        """
        try:
            start_year, end_year = baseline.split('-')
            clim_data = series.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            climatology = clim_data.groupby('time.month').mean(dim='time')
            anomaly = series.groupby('time.month') - climatology
            return anomaly
        except Exception as e:
            logger.error(f"计算月异常失败: {e}")
            return None

    # ========================== 数据加载方法 ==========================

    def load_obs_pressure_level_data(self, var_name: str, pressure_level: int = 500,
                                     year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        从MonthlyPressureLevel目录加载ERA5观测气压层数据
        参考 circulation_analysis.py 中的实现
        
        Args:
            var_name: 变量名（u, v, q, z）
            pressure_level: 气压层（hPa），默认500
            year_range: 年份范围，默认(1993, 2020)
        
        Returns:
            观测数据 (time, lat, lon) 或 None
        """
        try:
            obs_dir = Path("/sas12t1/ffyan/MonthlyPressureLevel")
            
            if not obs_dir.exists():
                logger.error(f"观测数据目录不存在: {obs_dir}")
                return None
            
            logger.info(f"从MonthlyPressureLevel加载观测数据: {var_name} @ {pressure_level}hPa")
            
            monthly_da_list = []
            start_year, end_year = year_range
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = obs_dir / f"era5_pressure_levels_{year}{month:02d}.nc"
                    
                    if not file_path.exists():
                        logger.debug(f"文件不存在: {file_path}")
                        continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            # 查找变量（支持大小写变体）
                            var_candidates = [var_name, var_name.upper(), var_name.lower()]
                            actual_var = None
                            for candidate in var_candidates:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            
                            if actual_var is None:
                                logger.debug(f"变量 {var_name} 不在文件 {file_path.name} 中")
                                continue
                            
                            da = ds[actual_var]
                            
                            # 检查维度
                            if 'pressure_level' not in da.dims and 'level' not in da.dims:
                                logger.warning(f"变量 {var_name} 没有气压层维度，跳过: {file_path.name}")
                                continue
                            
                            # 选择气压层
                            level_coord = 'pressure_level' if 'pressure_level' in da.dims else 'level'
                            level_values = ds[level_coord].values
                            
                            if pressure_level not in level_values:
                                logger.warning(f"气压层 {pressure_level} hPa 不在数据中，可用层: {level_values}")
                                continue
                            
                            da = da.sel({level_coord: pressure_level}).drop_vars(level_coord, errors='ignore')
                            
                            # 处理时间维度（ERA5文件有valid_time维度）
                            time_coord = None
                            if 'valid_time' in da.dims:
                                time_coord = 'valid_time'
                            elif 'time' in da.dims:
                                time_coord = 'time'
                            
                            if time_coord:
                                # 取第一个时间点（月平均值）
                                if da[time_coord].size > 0:
                                    da = da.isel({time_coord: 0})
                            
                            # 重命名坐标为标准名称
                            if 'latitude' in da.coords:
                                da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords:
                                da = da.rename({'longitude': 'lon'})
                            
                            # 添加时间坐标
                            time_stamp = pd.Timestamp(year, month, 1)
                            da = da.expand_dims(time=[time_stamp])
                            
                            # 加载数据到内存（确保文件关闭后数据仍然可用）
                            monthly_da_list.append(da.load())
                        
                    except Exception as e:
                        logger.warning(f"处理文件 {file_path.name} 时出错: {e}")
                        continue
            
            if not monthly_da_list:
                logger.warning(f"未找到观测数据: {var_name} @ {pressure_level}hPa")
                return None
            
            # 拼接所有月份数据
            data = xr.concat(monthly_da_list, dim='time')
            data = data.sortby('time')
            data = data.load()
            
            logger.info(f"观测数据加载完成: {var_name} @ {pressure_level}hPa, shape={data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载观测气压层数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def load_era5_sst_daily(self, year_range: Tuple[int, int] = (1993, 2020),
                             era5_root: str = '/sas12t1/ffyan/ERA5/daily-nc/single-level/') -> Optional[xr.DataArray]:
        """
        从本地加载 ERA5 SST 日值数据
        参考 circulation_analysis.py 和 notebook 中的实现
        
        Args:
            year_range: 年份范围，默认(1993, 2020)
            era5_root: ERA5 single-level 数据根目录
        
        Returns:
            日值数据 (time, lat, lon) 或 None
        """
        try:
            var_name = 'sst'
            var_dir = Path(era5_root) / var_name
            
            if not var_dir.exists():
                logger.error(f"变量目录不存在: {var_dir}")
                return None
            
            logger.info(f"从 ERA5 single-level 加载数据: {var_name}, 年份范围: {year_range}")
            
            start_year, end_year = year_range
            
            # 准备文件列表
            monthly_da_list = []
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = var_dir / f"era5_daily_{var_name}_{year}{month:02d}.nc"
                    
                    if not file_path.exists():
                        continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            # 查找SST变量（可能的名称）
                            sst_vars = ['sst', 'tos', 'sea_surface_temperature', 'SST', 'TOS']
                            actual_var = None
                            for candidate in sst_vars:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            
                            if actual_var is None:
                                continue
                            
                            da = ds[actual_var]
                            
                            # 标准化坐标名（ERA5 使用 latitude/longitude）
                            if 'latitude' in da.coords and 'lat' not in da.coords:
                                da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords and 'lon' not in da.coords:
                                da = da.rename({'longitude': 'lon'})
                            
                            # 确保数据已加载到内存（文件关闭后数据仍然可用）
                            monthly_da_list.append(da.load())
                        
                    except Exception as e:
                        logger.debug(f"加载文件失败 {year}{month:02d}: {e}")
                        continue
            
            if not monthly_da_list:
                logger.warning(f"未找到数据: {var_name}, 年份范围: {year_range}")
                return None
            
            # 拼接所有月份数据
            data = xr.concat(monthly_da_list, dim='time')
            data = data.sortby('time')
            
            # 确保纬度是升序的（方便后续处理）
            if 'lat' in data.coords and data.lat.values[0] > data.lat.values[-1]:
                data = data.sortby('lat')
                logger.debug("纬度已排序为升序")
            
            logger.info(f"ERA5 single-level 数据加载完成: {var_name}, shape={data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载 ERA5 single-level 数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def daily_to_monthly(self, da: xr.DataArray, var_name: str = 'sst') -> Optional[xr.DataArray]:
        """
        将日值数据聚合为月平均
        参考 circulation_analysis.py 中的实现
        
        Args:
            da: 日值数据 (time, lat, lon)
            var_name: 变量名（用于选择聚合方法）
        
        Returns:
            月平均数据 (time, lat, lon)
        """
        try:
            logger.info(f"将日值数据聚合为月平均: {var_name}")
            
            # 对于 SST、T2M 等温度变量，使用平均值
            monthly = da.resample(time='1MS').mean(dim='time')
            
            logger.info(f"月聚合完成: shape={monthly.shape}")
            return monthly
            
        except Exception as e:
            logger.error(f"月聚合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def load_model_sst_monthly(self, model: str, leadtime: int,
                                year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        加载模式SST月平均数据
        参考 circulation_analysis.py 和 notebook 中的实现
        
        Args:
            model: 模式名称
            leadtime: 提前期
            year_range: 年份范围
        
        Returns:
            SST月平均数据 (time, number, lat, lon) 或 (time, lat, lon)
        """
        try:
            forecast_dir = Path("/raid62/EC-C3S/month")
            model_dir = forecast_dir / model
            
            if not model_dir.exists():
                logger.warning(f"模型目录不存在: {model_dir}")
                return None
            
            # 获取文件后缀（SST通常在sfc文件中）
            if model not in self.data_loader.models:
                logger.error(f"不支持的模型: {model}")
                return None
            
            suffix = self.data_loader.models[model].get('sfc', None)
            if suffix is None:
                logger.error(f"模型 {model} 没有sfc文件配置")
                return None
            
            logger.info(f"加载模式 SST 数据: {model} L{leadtime}")
            
            monthly_da_list = []
            start_year, end_year = year_range
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = model_dir / f"{year}{month:02d}.{suffix}.nc"
                    
                    if not file_path.exists():
                        logger.debug(f"文件不存在: {file_path}")
                        continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            # 查找SST变量（可能的名称）
                            sst_vars = ['sst', 'tos', 'sea_surface_temperature', 'SST', 'TOS']
                            actual_var = None
                            for candidate in sst_vars:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            
                            if actual_var is None:
                                logger.debug(f"SST变量不在文件 {file_path.name} 中")
                                continue
                            
                            da = ds[actual_var]
                            
                            # 选择 leadtime
                            if 'time' in da.dims and da.time.size > leadtime:
                                da = da.isel(time=leadtime)
                            elif 'time' in da.dims:
                                # 尝试按时间选择
                                init_time = pd.Timestamp(year, month, 1)
                                try:
                                    da = da.sel(time=init_time, method='nearest', tolerance='15D')
                                except:
                                    logger.debug(f"无法选择 leadtime {leadtime} 的数据")
                                    continue
                            
                            # 确保有 number 维度（如果没有则创建）
                            if 'number' not in da.dims:
                                da = da.expand_dims('number')
                            
                            # 标准化坐标名
                            if 'latitude' in da.coords and 'lat' not in da.coords:
                                da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords and 'lon' not in da.coords:
                                da = da.rename({'longitude': 'lon'})
                            
                            # 创建时间坐标
                            forecast_time = pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                            da = da.expand_dims(time=[forecast_time])
                            
                            # 加载数据到内存（文件关闭后数据仍然可用）
                            monthly_da_list.append(da.load())
                        
                    except Exception as e:
                        logger.debug(f"处理文件 {file_path.name} 时出错: {e}")
                        continue
            
            if not monthly_da_list:
                logger.warning(f"未找到数据: {model} L{leadtime} SST")
                return None
            
            # 拼接数据
            data = xr.concat(monthly_da_list, dim='time')
            data = data.sortby('time')
            
            logger.info(f"模式 SST 数据加载完成: {model} L{leadtime}, shape={data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载模式 SST 数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # ========================== EAWM 指数相关 ==========================

    def compute_eawm_index(self, u_500: xr.DataArray) -> Optional[xr.DataArray]:
        """
        计算东亚冬季季风指数（I_EAWM）
        基于 500hPa 纬向风 (u_south - u_north)，仅DJF季节，标准化。
        """
        try:
            logger.info("开始计算EAWM指数...")
            
            # 如果有number维度，先计算ensemble mean
            if 'number' in u_500.dims:
                u_500_mean = u_500.mean(dim='number')
            else:
                u_500_mean = u_500
            
            # 定义两个区域
            south_region = {'lat': slice(25, 35), 'lon': slice(80, 120)}
            north_region = {'lat': slice(45, 55), 'lon': slice(80, 120)}
            
            # 计算两个区域的空间平均（使用面积加权）
            u_south = self.area_weighted_mean(u_500_mean.sel(**south_region), lat_name='lat')
            u_north = self.area_weighted_mean(u_500_mean.sel(**north_region), lat_name='lat')
            
            # 计算指数原型
            index_raw = u_south - u_north
            
            # 仅选择DJF季节（12、1、2月）
            djf_months = [12, 1, 2]
            index_djf = index_raw.sel(time=index_raw.time.dt.month.isin(djf_months))
            
            if len(index_djf.time) < 3:
                return None
            
            # z-score标准化
            index_values = index_djf.values
            index_mean = np.nanmean(index_values)
            index_std = np.nanstd(index_values)
            
            if index_std < 1e-10:
                return None
            
            index_normalized = (index_values - index_mean) / index_std
            
            # 创建结果DataArray
            index_result = xr.DataArray(
                index_normalized,
                coords={'time': index_djf.time},
                dims=['time'],
                name='eawm_index'
            )
            index_result.attrs = {'long_name': 'EAWM Index', 'season': 'DJF'}
            
            return index_result
        except Exception as e:
            logger.error(f"计算EAWM指数失败: {e}")
            return None

    # ========================== Nino3.4 指数相关 ==========================

    def compute_nino34_index_optimized(self, baseline: str = CLIMATOLOGY_PERIOD,
                                       era5_root: str = '/sas12t1/ffyan/ERA5/daily-nc/single-level/',
                                       year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        内存优化版：逐月处理并即时释放内存
        参考 notebook 中的实现
        
        Args:
            baseline: 气候态基准期
            era5_root: ERA5数据根目录
            year_range: 年份范围
        
        Returns:
            Nino3.4指数时间序列 (time,)
        """
        try:
            logger.info("启动内存优化模式计算 Nino3.4...")
            var_name = 'sst'
            var_dir = Path(era5_root) / var_name
            
            if not var_dir.exists():
                logger.error(f"变量目录不存在: {var_dir}")
                return None
            
            start_year, end_year = year_range
            nino34_timeseries = []  # 只存储最终的时间序列值，不存网格数据
            
            # 逐月读取处理
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = var_dir / f"era5_daily_{var_name}_{year}{month:02d}.nc"
                    if not file_path.exists():
                        continue
                    
                    try:
                        with xr.open_dataset(file_path) as ds:
                            # 查找SST变量
                            sst_vars = ['sst', 'tos', 'sea_surface_temperature', 'SST', 'TOS']
                            actual_var = None
                            for candidate in sst_vars:
                                if candidate in ds:
                                    actual_var = candidate
                                    break
                            
                            if actual_var is None:
                                continue
                            
                            da = ds[actual_var]
                            
                            # 1. 先裁剪区域 (Nino3.4: 5S-5N, 190E-240E)
                            if 'latitude' in da.coords:
                                da = da.rename({'latitude': 'lat'})
                            if 'longitude' in da.coords:
                                da = da.rename({'longitude': 'lon'})
                            da = da.sortby('lat')
                            da_sub = da.sel(lat=slice(-5, 5), lon=slice(190, 240))
                            
                            if da_sub.size == 0:
                                logger.debug(f"区域裁剪后无数据: {year}-{month:02d}")
                                continue
                            
                            # 2. 计算月平均 (将每天的数据压缩为一个值)
                            da_mon = da_sub.resample(time='1MS').mean('time')
                            
                            # 3. 计算区域加权平均
                            weights = np.cos(np.deg2rad(da_mon.lat))
                            da_weighted = da_mon.weighted(weights)
                            mean_val = da_weighted.mean(dim=['lat', 'lon']).load()  # load() 将极小的数据读入内存
                            
                            nino34_timeseries.append(mean_val)
                            
                    except Exception as e:
                        logger.warning(f"处理文件失败 {year}-{month:02d}: {e}")
                        continue
            
            if not nino34_timeseries:
                logger.warning("未找到有效的Nino3.4数据")
                return None
            
            # 合并时间序列
            nino34_da = xr.concat(nino34_timeseries, dim='time').sortby('time')
            
            # 计算距平 (Anomaly)
            start_year_str, end_year_str = baseline.split('-')
            clim_data = nino34_da.sel(time=slice(f"{start_year_str}-01-01", f"{end_year_str}-12-31"))
            clim = clim_data.groupby('time.month').mean('time')
            nino34_anom = nino34_da.groupby('time.month') - clim
            
            nino34_anom.name = 'nino34_index'
            nino34_anom.attrs = {
                'long_name': 'Nino3.4 Index',
                'description': 'SST anomaly in Nino3.4 region (5N-5S, 190-240E)',
                'units': 'K'
            }
            
            logger.info("内存优化模式计算 Nino3.4 完成")
            return nino34_anom
            
        except Exception as e:
            logger.error(f"内存优化模式计算 Nino3.4 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def compute_nino34_index(self, sst_monthly: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """
        计算 Nino3.4 指数 (SST anomaly in 5N-5S, 190E-240E)
        """
        try:
            logger.info("计算 Nino3.4 指数...")
            
            # 裁剪到 Nino3.4 区域
            nino34_region = sst_monthly.sel(lat=slice(-5, 5), lon=slice(190, 240))
            if nino34_region.size == 0:
                # 尝试反向切片
                nino34_region = sst_monthly.sel(lat=slice(5, -5), lon=slice(190, 240))
            
            if nino34_region.size == 0:
                logger.warning("Nino3.4 区域裁剪后无数据")
                return None

            # 计算面积加权空间平均
            nino34_spatial_mean = self.area_weighted_mean(nino34_region, lat_name='lat')
            
            # 计算逐月异常
            nino34_anomaly = self.compute_monthly_anomaly(nino34_spatial_mean, baseline)
            
            if nino34_anomaly is not None:
                nino34_anomaly.name = 'nino34_index'
                nino34_anomaly.attrs = {
                    'long_name': 'Nino3.4 Index',
                    'description': 'SST anomaly in Nino3.4 region (5N-5S, 190-240E)',
                    'units': 'K'
                }
            
            return nino34_anomaly
        except Exception as e:
            logger.error(f"计算 Nino3.4 指数失败: {e}")
            return None

    # ========================== 绘图逻辑 (来自 Notebook) ==========================

    def plot_nino34_mmm(self, nino34_indices: Dict[str, xr.DataArray], output_file: Path, time_resolution: str = 'annual'):
        """
        绘制Nino3.4指数多模式平均时间序列图
        逻辑源自 nino34_eawm_index_calculation.ipynb
        
        Args:
            nino34_indices: Nino3.4指数字典 {'ERA5': DataArray, 'model_L0': DataArray, ...}
            output_file: 输出文件路径
            time_resolution: 时间分辨率，可选 'monthly'（逐月）、'annual'（年平均）、'seasonal'（季节平均）
        """
        try:
            logger.info(f"绘制 Nino3.4 MMM 图: {time_resolution}")
            
            # 辅助函数：根据时间分辨率聚合数据
            def aggregate_data(times, values, resolution, season=None):
                df = pd.DataFrame({'time': pd.DatetimeIndex(times), 'value': values})
                
                if resolution == 'monthly':
                    return df['time'].values, df['value'].values
                elif resolution == 'annual':
                    df['year'] = df['time'].dt.year
                    yearly = df.groupby('year')['value'].mean()
                    plot_times = pd.DatetimeIndex([pd.Timestamp(f'{y}-01-01') for y in yearly.index])
                    return plot_times.values, yearly.values
                elif resolution == 'seasonal':
                    df['year'] = df['time'].dt.year
                    df['month'] = df['time'].dt.month
                    def get_season(month):
                        if month in [12, 1, 2]: return 'DJF'
                        elif month in [3, 4, 5]: return 'MAM'
                        elif month in [6, 7, 8]: return 'JJA'
                        else: return 'SON'
                    df['season'] = df['month'].apply(get_season)
                    df['season_year'] = df['year']
                    df.loc[df['month'] == 12, 'season_year'] = df.loc[df['month'] == 12, 'year'] + 1
                    
                    if season is not None:
                        df = df[df['season'] == season]
                    
                    df['season_key'] = df['season_year'].astype(str) + '-' + df['season']
                    seasonal = df.groupby('season_key')['value'].mean()
                    plot_times = []
                    for key in seasonal.index:
                        year, s = key.split('-')
                        month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
                        plot_times.append(pd.Timestamp(f'{year}-{month_map[s]:02d}-15'))
                    return pd.DatetimeIndex(plot_times).values, seasonal.values
                else:
                    raise ValueError(f"不支持的时间分辨率: {resolution}")
            
            # 分离ERA5和各模式的数据
            era5_data = nino34_indices.get('ERA5')
            if era5_data is None:
                logger.warning("绘图缺少 ERA5 数据")
                return
            
            # 按leadtime分组
            model_data_l0 = {k.replace('_L0', ''): v for k, v in nino34_indices.items() if '_L0' in k}
            model_data_l3 = {k.replace('_L3', ''): v for k, v in nino34_indices.items() if '_L3' in k}
            
            # 如果是季节平均，为每个季节分别生成一幅图
            if time_resolution == 'seasonal':
                seasons = ['DJF', 'MAM', 'JJA', 'SON']
                for season in seasons:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    # 聚合ERA5
                    era5_t, era5_v = aggregate_data(era5_data.time.values, era5_data.values, 'seasonal', season)
                    era5_t = pd.DatetimeIndex(era5_t)
                    ax.plot(era5_t, era5_v, color='black', linewidth=2.5, marker='o', markersize=4, label='ERA5', zorder=10)
                    
                    # 处理 MMM
                    for leadtime, model_data, color, name in [(0, model_data_l0, 'red', 'L0'), (3, model_data_l3, 'blue', 'L3')]:
                        if len(model_data) > 1:
                            # 聚合所有模型数据
                            model_aggs = {}
                            for m, d in model_data.items():
                                t, v = aggregate_data(d.time.values, d.values, 'seasonal', season)
                                model_aggs[m] = pd.Series(v, index=pd.DatetimeIndex(t))
                            
                            # 计算 MMM
                            df_models = pd.DataFrame(model_aggs)
                            mmm = df_models.mean(axis=1)
                            std = df_models.std(axis=1)
                            common_t = mmm.index
                            
                            # 计算相关系数
                            corr_str = ''
                            # 简单对齐计算相关
                            common_idx = era5_t.intersection(common_t)
                            if len(common_idx) > 2:
                                e_vals = pd.Series(era5_v, index=era5_t).loc[common_idx]
                                m_vals = mmm.loc[common_idx]
                                try:
                                    r, _ = pearsonr(e_vals, m_vals)
                                    corr_str = f' [r={r:.2f}]'
                                except: pass
                            
                            ax.plot(common_t, mmm, color=color, linewidth=2, linestyle='-' if leadtime==0 else '--', 
                                    marker='o' if leadtime==0 else 's', markersize=4, label=f'MMM ({name}){corr_str}', zorder=5)
                            ax.fill_between(common_t, mmm-std, mmm+std, color=color, alpha=0.2, label=f'±1σ Spread ({name})')

                    # 样式设置
                    ax.tick_params(axis='both', labelsize=20)
                    ax.set_ylabel('Nino3.4 Index (K)', fontsize=20, fontweight='bold')
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18, framealpha=0.9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.2)
                    
                    s_out = output_file.parent / f"{output_file.stem}_{season.lower()}{output_file.suffix}"
                    plt.savefig(s_out, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"已保存季节图: {s_out}")
                return

            # Monthly / Annual
            fig, ax = plt.subplots(figsize=(14, 7))
            era5_t, era5_v = aggregate_data(era5_data.time.values, era5_data.values, time_resolution)
            era5_t = pd.DatetimeIndex(era5_t)
            ax.plot(era5_t, era5_v, color='black', linewidth=2.5, marker='o', markersize=4, label='ERA5', zorder=10)
            
            for leadtime, model_data, color, name in [(0, model_data_l0, 'red', 'L0'), (3, model_data_l3, 'blue', 'L3')]:
                if len(model_data) > 1:
                    model_aggs = {}
                    for m, d in model_data.items():
                        t, v = aggregate_data(d.time.values, d.values, time_resolution)
                        model_aggs[m] = pd.Series(v, index=pd.DatetimeIndex(t))
                    
                    df_models = pd.DataFrame(model_aggs)
                    mmm = df_models.mean(axis=1)
                    std = df_models.std(axis=1)
                    common_t = mmm.index
                    
                    corr_str = ''
                    common_idx = era5_t.intersection(common_t)
                    if len(common_idx) > 2:
                        e_vals = pd.Series(era5_v, index=era5_t).loc[common_idx]
                        m_vals = mmm.loc[common_idx]
                        try:
                            r, _ = pearsonr(e_vals, m_vals)
                            corr_str = f' [r={r:.2f}]'
                        except: pass
                    
                    ax.plot(common_t, mmm, color=color, linewidth=2, linestyle='-' if leadtime==0 else '--', 
                            marker='o' if leadtime==0 else 's', markersize=4, label=f'MMM ({name}){corr_str}', zorder=5)
                    ax.fill_between(common_t, mmm-std, mmm+std, color=color, alpha=0.2, label=f'±1σ Spread ({name})')

            ax.tick_params(axis='both', labelsize=20)
            ax.set_ylabel('Nino3.4 Index (K)', fontsize=20, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"已保存图: {output_file}")

        except Exception as e:
            logger.error(f"绘制Nino3.4图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def plot_eawm_index_timeseries(self, index_dict: Dict[str, xr.DataArray], output_file: Path):
        """
        绘制EAWM指数时间序列图 (MMM风格)
        """
        try:
            logger.info("绘制 EAWM 指数时间序列...")
            
            # 将所有数据转换为DataFrame以便处理 (Annual/DJF year based)
            # EAWM 只有 DJF，这里假设数据已经是年/季度的
            # 数据预处理：转换为 DataFrame，index 为 DJF Year
            data_frames = {}
            for name, da in index_dict.items():
                # 假设 da.time 指向 DJF 的中间月份 (Jan) 或者 起始 (Dec)
                # 简单起见，提取年份。注意：DJF 1993 通常指 1993/12, 1994/01, 1994/02 -> 归属 1993 或 1994
                # 这里的逻辑应与 calculate 一致。假设 da.time 已经是 DJF 均值的时间点
                years = [t.year if t.month > 6 else t.year - 1 for t in pd.DatetimeIndex(da.time.values)]
                df = pd.Series(da.values, index=years, name=name)
                data_frames[name] = df
            
            era5_s = data_frames.get('ERA5')
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if era5_s is not None:
                ax.plot(era5_s.index, era5_s.values, color='black', linewidth=2.5, marker='o', markersize=4, label='ERA5', zorder=10)
            
            # 分组 L0 和 L3
            l0_models = [k for k in data_frames.keys() if '_L0' in k]
            l3_models = [k for k in data_frames.keys() if '_L3' in k]
            
            for leadtime, models, color, name in [(0, l0_models, 'red', 'L0'), (3, l3_models, 'blue', 'L3')]:
                if models:
                    df_m = pd.DataFrame({m: data_frames[m] for m in models})
                    mmm = df_m.mean(axis=1)
                    std = df_m.std(axis=1)
                    
                    corr_str = ''
                    if era5_s is not None:
                        common = era5_s.index.intersection(mmm.index)
                        if len(common) > 2:
                            try:
                                r, _ = pearsonr(era5_s.loc[common], mmm.loc[common])
                                corr_str = f' [r={r:.2f}]'
                            except: pass
                    
                    ax.plot(mmm.index, mmm.values, color=color, linewidth=2, linestyle='-' if leadtime==0 else '--',
                            marker='o' if leadtime==0 else 's', markersize=4, label=f'MMM ({name}){corr_str}', zorder=5)
                    ax.fill_between(mmm.index, mmm-std, mmm+std, color=color, alpha=0.2, label=f'±1σ Spread ({name})')

            ax.tick_params(axis='both', labelsize=20)
            ax.set_ylabel('EAWM Index (Standardized)', fontsize=20, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"EAWM 绘图完成: {output_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制 EAWM 图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _process_single_model_leadtime(self, model: str, leadtime: int) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
        """
        处理单个模型和提前期的EAWM和Nino3.4指数计算
        用于并行处理的包装函数
        
        Args:
            model: 模型名称
            leadtime: 提前期
        
        Returns:
            (eawm_index, nino34_index) 元组，如果计算失败则返回 (None, None)
        """
        try:
            logger.info(f"处理 {model} L{leadtime}...")
            eawm_index = None
            nino34_index = None
            
            # EAWM (U500)
            u500_model = self.data_loader.load_pressure_level_data(
                model=model,
                leadtime=leadtime,
                var_name='u',
                pressure_level=500
            )
            if u500_model is not None:
                eawm_index = self.compute_eawm_index(u500_model)
                if eawm_index is None:
                    logger.warning(f"无法计算 {model} L{leadtime} EAWM指数")
            else:
                logger.warning(f"无法加载 {model} L{leadtime} U500 数据")
            
            # Nino3.4 (SST)
            sst_model = self.load_model_sst_monthly(model, leadtime)
            if sst_model is not None:
                # 如果有number维度，先计算ensemble mean
                if 'number' in sst_model.dims:
                    sst_model = sst_model.mean(dim='number')
                nino34_index = self.compute_nino34_index(sst_model)
                if nino34_index is None:
                    logger.warning(f"无法计算 {model} L{leadtime} Nino3.4指数")
            else:
                logger.warning(f"无法加载 {model} L{leadtime} SST 数据")
            
            return eawm_index, nino34_index
            
        except Exception as e:
            logger.error(f"处理 {model} L{leadtime} 出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    def run_analysis(self, models: List[str], leadtimes: List[int], 
                     parallel: bool = False, n_jobs: Optional[int] = None):
        """
        执行完整分析流程
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
            parallel: 是否使用并行处理
            n_jobs: 并行作业数（如果为None，则使用默认值）
        """
        results_dir = Path("/sas12t1/ffyan/output/index_analysis/results")
        plots_dir = Path("/sas12t1/ffyan/output/index_analysis/plots")
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载/计算 ERA5 EAWM 指数 (需要 U500)
        logger.info("Step 1: 处理 ERA5 EAWM 指数")
        eawm_era5 = None
        try:
            # 首先尝试从已保存的文件加载
            eawm_file = results_dir / "circulation_obs_eawm_index.nc"
            if eawm_file.exists():
                try:
                    eawm_era5 = xr.open_dataarray(eawm_file)
                    logger.info("已加载 ERA5 EAWM 指数文件")
                except Exception as e:
                    logger.warning(f"加载 EAWM 指数文件失败: {e}")
            
            # 如果文件不存在，则从原始数据计算
            if eawm_era5 is None:
                logger.info("开始从ERA5数据计算EAWM指数...")
                u500_era5 = self.load_obs_pressure_level_data('u', 500)
                if u500_era5 is not None:
                    eawm_era5 = self.compute_eawm_index(u500_era5)
                    if eawm_era5 is not None:
                        # 保存计算结果
                        eawm_file.parent.mkdir(parents=True, exist_ok=True)
                        eawm_era5.to_netcdf(eawm_file)
                        logger.info(f"EAWM指数已保存到: {eawm_file}")
                else:
                    logger.warning("无法加载 ERA5 U500 数据")
        except Exception as e:
            logger.warning(f"ERA5 EAWM 计算受阻: {e}")
            import traceback
            logger.error(traceback.format_exc())
            eawm_era5 = None

        # 2. 加载/计算 ERA5 Nino3.4 指数 (需要 SST)
        logger.info("Step 2: 处理 ERA5 Nino3.4 指数")
        nino34_era5 = None
        try:
            # 首先尝试从已保存的文件加载
            nino34_file = results_dir / "circulation_nino34_era5.nc"
            if nino34_file.exists():
                try:
                    nino34_era5 = xr.open_dataarray(nino34_file)
                    logger.info("已加载 ERA5 Nino3.4 指数文件")
                except Exception as e:
                    logger.warning(f"加载 Nino3.4 指数文件失败: {e}")
            
            # 如果文件不存在，则从原始数据计算
            if nino34_era5 is None:
                logger.info("开始从ERA5数据计算Nino3.4指数（使用内存优化模式）...")
                # 使用内存优化版本：逐月处理，不一次性加载所有数据
                nino34_era5 = self.compute_nino34_index_optimized()
                if nino34_era5 is not None:
                    # 保存计算结果
                    nino34_file.parent.mkdir(parents=True, exist_ok=True)
                    nino34_era5.to_netcdf(nino34_file)
                    logger.info(f"Nino3.4指数已保存到: {nino34_file}")
                else:
                    logger.warning("无法计算 ERA5 Nino3.4 指数")
        except Exception as e:
            logger.warning(f"ERA5 Nino3.4 计算受阻: {e}")
            import traceback
            logger.error(traceback.format_exc())
            nino34_era5 = None
        
        # 3. 处理各模式
        eawm_indices = {}
        nino34_indices = {}
        
        if eawm_era5 is not None: eawm_indices['ERA5'] = eawm_era5
        if nino34_era5 is not None: nino34_indices['ERA5'] = nino34_era5
        
        # 准备任务列表
        model_tasks = [(model, leadtime) for leadtime in leadtimes for model in models]
        
        if parallel and n_jobs and n_jobs > 1 and len(model_tasks) > 1:
            # === 修改开始：使用 ProcessPoolExecutor (多进程) ===
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # 注意：不需再限制为 16 或更小，进程池通常受限于 CPU 核数或内存
            # 如果内存吃紧，请适当调小 n_jobs
            max_workers = n_jobs 
            
            logger.info(f"使用并行处理 (多进程): {max_workers} Workers")
            
            parallel_failed = False
            try:
                # 使用 ProcessPoolExecutor 替代 ThreadPoolExecutor
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 提交顶层函数 _worker_task，而不是类方法
                    future_to_task = {
                        executor.submit(_worker_task, model, leadtime): (model, leadtime)
                        for model, leadtime in model_tasks
                    }
                    
                    completed = 0
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            # 获取返回值
                            (model, leadtime), (eawm_index, nino34_index) = future.result(timeout=1800)
                            
                            if eawm_index is not None:
                                eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                            if nino34_index is not None:
                                nino34_indices[f"{model}_L{leadtime}"] = nino34_index
                            
                            completed += 1
                            logger.info(f"完成 {completed}/{len(model_tasks)}: {model} L{leadtime}")
                        except Exception as e:
                            logger.error(f"任务失败 {task}: {e}")
                            # import traceback
                            # logger.error(traceback.format_exc())
                
                logger.info(f"并行处理完成: {completed}/{len(model_tasks)} 成功")
            except (Exception, SystemError, OSError) as e:
                logger.error(f"并行处理失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("回退到串行处理...")
                parallel_failed = True
            # === 修改结束 ===
            
            # 如果并行失败，使用串行处理
            if parallel_failed:
                for model, leadtime in model_tasks:
                    eawm_index, nino34_index = self._process_single_model_leadtime(model, leadtime)
                    if eawm_index is not None:
                        eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                    if nino34_index is not None:
                        nino34_indices[f"{model}_L{leadtime}"] = nino34_index
        else:
            # 串行处理
            logger.info("使用串行处理模式")
            for model, leadtime in model_tasks:
                eawm_index, nino34_index = self._process_single_model_leadtime(model, leadtime)
                if eawm_index is not None:
                    eawm_indices[f"{model}_L{leadtime}"] = eawm_index
                if nino34_index is not None:
                    nino34_indices[f"{model}_L{leadtime}"] = nino34_index

        # 4. 绘图
        if len(eawm_indices) > 0:
            self.plot_eawm_index_timeseries(eawm_indices, plots_dir / "circulation_eawm_index_mmm.png")
        
        if len(nino34_indices) > 0:
            # 绘制不同分辨率的图
            self.plot_nino34_mmm(nino34_indices, plots_dir / "circulation_nino34_index_mmm_monthly.png", 'monthly')
            self.plot_nino34_mmm(nino34_indices, plots_dir / "circulation_nino34_index_mmm_annual.png", 'annual')
            self.plot_nino34_mmm(nino34_indices, plots_dir / "circulation_nino34_index_mmm_seasonal.png", 'seasonal')

def main():
    parser = create_parser(description="指数分析：EAWM 和 Nino3.4", var_required=False)
    args = parser.parse_args()
    
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    
    # 解析并行参数
    parallel = normalize_parallel_args(args)
    
    logger.info(f"模型列表: {models}")
    logger.info(f"提前期列表: {leadtimes}")
    logger.info(f"并行处理: {parallel}")
    logger.info(f"并行作业数: {args.n_jobs}")
    
    analyzer = IndexAnalyzer()
    analyzer.run_analysis(
        models=models,
        leadtimes=leadtimes,
        parallel=parallel,
        n_jobs=args.n_jobs
    )
    
    logger.info("所有任务完成！")

if __name__ == "__main__":
    main()
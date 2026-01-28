#!/usr/bin/env python3
"""
Circulation Analysis

实现对系统u、v风场进行分析，绘制：
1、850hPa的u、v风场和GHT场
2、500hPa的u、v风场和GHT场
皆需要计算Annual、DJF、MAM、JJA、SON的气候平均态

基于ERA5观测数据（从MonthlyPressureLevel目录读取），计算模式与观测的偏差。

使用方法：
# 完整分析（计算+绘图）
python circulation_analysis.py --models all --leadtimes 0 3
python circulation_analysis.py --models CMCC-35 ECMWF-51 --leadtimes 0

# 仅绘图（基于已有NetCDF结果文件）
python circulation_analysis.py --models all --leadtimes 0 3 --plot-only
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
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加工具包路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    CLIMATOLOGY_PERIOD,
    SEASONS,
    SPATIAL_BOUNDS as COMMON_SPATIAL_BOUNDS,
    REMOVE_OUTLIERS,
    OUTLIER_METHOD,
    OUTLIER_THRESHOLD,
    COLORS,
    DEFAULT_TIME_CHUNK,
)

from src.utils.data_loader import DataLoader
from src.utils.alignment import align_time_to_monthly, align_spatial_to_obs
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, normalize_parallel_args

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logging(
    log_file='circulation_analysis.log',
    module_name=__name__
)

# 全局配置（从 common_config 导入，但保留特殊的并行配置）
MODELS = MODEL_LIST
# 注意：此脚本使用更高的并行度配置
MAX_WORKERS_TEMP = 32     # 温度最大并发进程数（高于默认值）
MAX_WORKERS_PREC = 24     # 降水最大并发进程数（高于默认值）
HARD_WORKER_CAP = 32      # 强制上限（高于默认值）

# 空间配置
SPATIAL_BOUNDS = COMMON_SPATIAL_BOUNDS

class CirculationAnalyzer:
    """
    风场分析器
    """
    def __init__(self, var_type: str = None, data_loader: DataLoader = None):
        """
        初始化风场分析器
        
        Args:
            var_type: 变量类型（可选，用于兼容性）
            data_loader: 数据加载器
        """
        self.var_type = var_type
        self.data_loader = data_loader or DataLoader()
        
        # 环流分析直接加载特定变量（u, v, q, z），不需要var_config
        logger.info(f"初始化风场分析器")

    def align_time_data(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        统一的时间对齐方法
        """
        return align_time_to_monthly(obs_data, fcst_data, min_common_months=12)

    def align_spatial_grid(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        统一的空间网格对齐方法
        """
        return align_spatial_to_obs(obs_data, fcst_data)

    def load_obs_data(self) -> xr.DataArray:
        """
        加载观测数据（环流分析中未使用，保留用于兼容性）
        """
        # 环流分析不需要观测数据，此方法保留用于兼容性
        raise NotImplementedError("环流分析不需要观测数据")
    
    def load_obs_pressure_level_data(self, var_name: str, pressure_level: int = 850,
                                     year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        从MonthlyPressureLevel目录加载ERA5观测气压层数据
        
        Args:
            var_name: 变量名（u, v, q, z）
            pressure_level: 气压层（hPa），默认850
            year_range: 年份范围，默认(1993, 2020)
        
        Returns:
            观测数据 (time, lat, lon) 或 None
        """
        try:
            from src.utils.data_utils import load_netcdf_data, find_variable, dynamic_coord_sel
            
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
                        # 加载数据
                        ds = xr.open_dataset(file_path)
                        
                        # 查找变量（支持大小写变体）
                        var_candidates = [var_name, var_name.upper(), var_name.lower()]
                        actual_var = None
                        for candidate in var_candidates:
                            if candidate in ds:
                                actual_var = candidate
                                break
                        
                        if actual_var is None:
                            logger.debug(f"变量 {var_name} 不在文件 {file_path.name} 中")
                            ds.close()
                            continue
                        
                        da = ds[actual_var]
                        
                        # 检查维度
                        if 'pressure_level' not in da.dims and 'level' not in da.dims:
                            logger.warning(f"变量 {var_name} 没有气压层维度，跳过: {file_path.name}")
                            ds.close()
                            continue
                        
                        # 选择气压层
                        level_coord = 'pressure_level' if 'pressure_level' in da.dims else 'level'
                        level_values = ds[level_coord].values
                        
                        if pressure_level not in level_values:
                            logger.warning(f"气压层 {pressure_level} hPa 不在数据中，可用层: {level_values}")
                            ds.close()
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
                        
                        # 空间裁剪
                        if 'lat' in da.coords and 'lon' in da.coords:
                            # 确保坐标递增
                            if da.lat.values[0] > da.lat.values[-1]:
                                da = da.sortby('lat')
                            
                            # 空间裁剪到指定范围
                            da = da.sel(
                                lat=slice(SPATIAL_BOUNDS["lat"][0], SPATIAL_BOUNDS["lat"][1]),
                                lon=slice(SPATIAL_BOUNDS["lon"][0], SPATIAL_BOUNDS["lon"][1])
                            )
                        
                        # 添加时间坐标
                        time_stamp = pd.Timestamp(year, month, 1)
                        da = da.expand_dims(time=[time_stamp])
                        
                        monthly_da_list.append(da)
                        
                        ds.close()
                        
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

    def load_fcst_data(self, model: str, leadtime: int) -> Optional[xr.DataArray]:
        """
        加载预报数据（sfc文件）
        
        Args:
            model: 模型名称
            leadtime: 提前期
        
        Returns:
            ensemble数据 (time, number, lat, lon) 或 None
        """
        try:
            forecast_dir = Path("/raid62/EC-C3S/month")
            model_dir = forecast_dir / model

            if not model_dir.exists():
                logger.warning(f"模型目录不存在: {model_dir}")
                return None

            # 环流分析不使用sfc文件，此方法保留用于兼容性
            # 如果需要使用，需要提供var_type和相应的配置
            if self.var_type is None:
                logger.warning(f"load_fcst_data需要var_type参数，但环流分析使用load_fcst_data_at_level")
                return None
            try:
                config = self.data_loader.var_config[self.var_type]
                suffix = self.data_loader.models[model][config['file_type']]
            except:
                logger.warning(f"无法获取模型 {model} 的文件后缀")
                return None
            monthly_da_list = []

            for year in range(1993, 2021):
                for month in range(1, 13):
                    fp = model_dir / f"{year}{month:02d}.{suffix}.nc"
                    if not fp.exists():
                        continue
                    try:
                        with xr.open_dataset(fp) as ds:
                            var_name = self.data_loader.find_variable(ds, config['fcst_names'])
                            if var_name is None:
                                continue

                            da = ds[var_name]

                            if 'time' in da.dims and 'number' in da.dims:  # 有time和number维度的情况
                                if da.time.size <= leadtime:
                                    continue
                                da = da.isel(time=leadtime)
                                # 保留number维度
                            elif 'time' in da.dims:
                                init = pd.Timestamp(year, month, 1)
                                da = da.sel(time=init, method='nearest', tolerance='15D')
                                # 如果没有number维度，创建单成员维度
                                if 'number' not in da.dims:
                                    da = da.expand_dims('number')
                            else:  # 没有time维度的情况
                                # 没有time维度的情况，创建单成员维度
                                if 'number' not in da.dims:
                                    da = da.expand_dims('number')
                            
                            # 空间裁剪
                            da = self.data_loader.dynamic_coord_sel(da, {'lat': (15, 55), 'lon': (70, 140)})

                            if 'number' in da.dims and 'lat' in da.dims and 'lon' in da.dims:
                                # 使用实际的预报时间（从文件内time坐标获取）
                                forecast_time = pd.Timestamp(da.time.values) if hasattr(da, 'time') and 'time' in da.coords else pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                                monthly_da_list.append((forecast_time, da))
                    except Exception as e:
                        logger.error(f"处理文件 {fp} 时出错: {e}")
                        continue
            if not monthly_da_list:
                logger.warning(f"无ensemble数据 {model} L{leadtime}")
                return None

            # 拼接数据
            data = xr.concat(
                [da for t, da in monthly_da_list],
                dim=xr.DataArray([t for t, _ in monthly_da_list], dims='time', name='time')
            )
            data = data.sortby('time')
            logger.info(f"Ensemble数据加载成功 {model} L{leadtime}: {data.shape}, 成员数={data.number.size}")
            return data
            
        except Exception as e:
            logger.error(f"加载预报数据失败: {e}")
            return None
    
    def compute_monthly_climatology(self, data: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """
        计算逐月气候态
        
        Args:
            data: 输入数据 (time, lat, lon) 或 (time, number, lat, lon)
            baseline: 气候态基期，如"1993-2020"
        
        Returns:
            clim: 气候态 (month, lat, lon) 或 (month, number, lat, lon)
        """
        try:
            # 解析基期
            start_year, end_year = baseline.split('-')
            start_year, end_year = int(start_year), int(end_year)
            
            # 选择基期数据
            clim_data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            
            # 按月份计算气候态
            clim = clim_data.groupby('time.month').mean(dim='time')
            
            logger.info(f"计算气候态完成: {clim.shape}, 基期: {baseline}")
            return clim
            
        except Exception as e:
            logger.error(f"计算气候态失败: {e}")
            return None
    
    def compute_seasonal_climatology(self, data: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Dict[str, xr.DataArray]:
        """
        计算季节气候态
        
        Args:
            data: 输入数据 (time, lat, lon) 或 (time, number, lat, lon)
            baseline: 气候态基期，如"1993-2020"
        
        Returns:
            dict: {
                'Annual': 全年平均 (lat, lon) 或 (number, lat, lon),
                'DJF': 12、1、2月平均,
                'MAM': 3、4、5月平均,
                'JJA': 6、7、8月平均,
                'SON': 9、10、11月平均
            }
        """
        try:
            # 解析基期
            start_year, end_year = baseline.split('-')
            start_year, end_year = int(start_year), int(end_year)
            
            # 选择基期数据
            clim_data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            
            results = {}
            
            # Annual: 全年平均
            results['Annual'] = clim_data.mean(dim='time')
            
            # DJF: 12、1、2月
            djf_months = [12, 1, 2]
            djf_data = clim_data.sel(time=clim_data.time.dt.month.isin(djf_months))
            if djf_data.size > 0:
                results['DJF'] = djf_data.mean(dim='time')
            else:
                results['DJF'] = None
            
            # MAM: 3、4、5月
            mam_months = [3, 4, 5]
            mam_data = clim_data.sel(time=clim_data.time.dt.month.isin(mam_months))
            if mam_data.size > 0:
                results['MAM'] = mam_data.mean(dim='time')
            else:
                results['MAM'] = None
            
            # JJA: 6、7、8月
            jja_months = [6, 7, 8]
            jja_data = clim_data.sel(time=clim_data.time.dt.month.isin(jja_months))
            if jja_data.size > 0:
                results['JJA'] = jja_data.mean(dim='time')
            else:
                results['JJA'] = None
            
            # SON: 9、10、11月
            son_months = [9, 10, 11]
            son_data = clim_data.sel(time=clim_data.time.dt.month.isin(son_months))
            if son_data.size > 0:
                results['SON'] = son_data.mean(dim='time')
            else:
                results['SON'] = None
            
            logger.info(f"计算季节气候态完成: 基期: {baseline}")
            return results
            
        except Exception as e:
            logger.error(f"计算季节气候态失败: {e}")
            return {}
    
    def load_fcst_data_at_level(self, model: str, leadtime: int, var_name: str, pressure_level: int = 850) -> Optional[xr.DataArray]:
        """
        加载指定气压层的预报数据。现在直接复用toolkit的DataLoader实现，
        保持两处逻辑一致。
        """
        try:
            return self.data_loader.load_pressure_level_data(
                model=model,
                leadtime=leadtime,
                var_name=var_name,
                pressure_level=pressure_level
            )
        except AttributeError:
            logger.error("DataLoader缺少load_pressure_level_data方法，请更新climate_analysis_toolkit")
            return None
        except Exception as e:
            logger.error(f"加载预报数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load_fcst_surface_data(self, model: str, leadtime: int, var_name: str,
                               year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        加载模式 surface 数据（温度或降水）
        
        Args:
            model: 模型名称
            leadtime: 提前期
            var_name: 变量名（'temp' 或 'prec'）
            year_range: 年份范围
        
        Returns:
            模式数据 (time, number, lat, lon) 或 (time, lat, lon)
        """
        try:
            forecast_dir = Path("/raid62/EC-C3S/month")
            model_dir = forecast_dir / model
            
            if not model_dir.exists():
                logger.warning(f"模型目录不存在: {model_dir}")
                return None
            
            # 获取变量配置
            if var_name not in self.data_loader.var_config:
                logger.error(f"不支持的变量: {var_name}")
                return None
            
            var_config = self.data_loader.var_config[var_name]
            file_type = var_config['file_type']  # 应该是 'sfc'
            
            # 获取模型文件后缀
            if model not in self.data_loader.models:
                logger.error(f"不支持的模型: {model}")
                return None
            
            suffix = self.data_loader.models[model][file_type]
            
            logger.info(f"加载模式 surface 数据: {model} L{leadtime} {var_name}")
            
            monthly_da_list = []
            start_year, end_year = year_range
            
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = model_dir / f"{year}{month:02d}.{suffix}.nc"
                    
                    if not file_path.exists():
                        logger.debug(f"文件不存在: {file_path}")
                        continue
                    
                    try:
                        ds = xr.open_dataset(file_path)
                        
                        # 查找变量
                        actual_var = None
                        for var_candidate in var_config['fcst_names']:
                            if var_candidate in ds:
                                actual_var = var_candidate
                                break
                        
                        if actual_var is None:
                            logger.debug(f"变量 {var_name} 不在文件 {file_path.name} 中")
                            ds.close()
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
                                ds.close()
                                continue
                        
                        # 确保有 number 维度（如果没有则创建）
                        if 'number' not in da.dims:
                            da = da.expand_dims('number')
                        
                        # 标准化坐标名
                        if 'latitude' in da.coords and 'lat' not in da.coords:
                            da = da.rename({'latitude': 'lat'})
                        if 'longitude' in da.coords and 'lon' not in da.coords:
                            da = da.rename({'longitude': 'lon'})
                        
                        # 空间裁剪到中国区域
                        # 检查纬度是否降序，如果是则使用反向 slice
                        if 'lat' in da.coords and 'lon' in da.coords:
                            is_lat_desc = da.lat.values[0] > da.lat.values[-1]
                            if is_lat_desc:
                                # 降序纬度：使用反向 slice
                                da = da.sel(
                                    lat=slice(SPATIAL_BOUNDS['lat'][1], SPATIAL_BOUNDS['lat'][0]),
                                    lon=slice(SPATIAL_BOUNDS['lon'][0], SPATIAL_BOUNDS['lon'][1])
                                )
                            else:
                                # 升序纬度：使用正向 slice
                                da = da.sel(
                                    lat=slice(SPATIAL_BOUNDS['lat'][0], SPATIAL_BOUNDS['lat'][1]),
                                    lon=slice(SPATIAL_BOUNDS['lon'][0], SPATIAL_BOUNDS['lon'][1])
                                )
                        
                        # 创建时间坐标
                        forecast_time = pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                        da = da.expand_dims(time=[forecast_time])
                        
                        da = da.load()
                        monthly_da_list.append(da)
                        
                        ds.close()
                        
                    except Exception as e:
                        logger.debug(f"处理文件 {file_path.name} 时出错: {e}")
                        continue
            
            if not monthly_da_list:
                logger.warning(f"未找到数据: {model} L{leadtime} {var_name}")
                return None
            
            # 拼接数据
            data = xr.concat(monthly_da_list, dim='time')
            data = data.sortby('time')
            
            logger.info(f"模式 surface 数据加载完成: {model} L{leadtime} {var_name}, shape={data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载模式 surface 数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def calculate_bias(self, model_clim: xr.DataArray, obs_clim: xr.DataArray) -> Optional[xr.DataArray]:
        """
        计算偏差（模式 - 观测）
        
        Args:
            model_clim: 模式气候态
            obs_clim: 观测气候态
        
        Returns:
            偏差场 (lat, lon)
        """
        try:
            # 插值观测数据到模式网格
            obs_interp = obs_clim.interp(lat=model_clim.lat, lon=model_clim.lon, method='linear')
            
            # 计算偏差
            bias = model_clim - obs_interp
            
            # 掩膜海洋（观测为NaN的位置）
            bias = bias.where(~obs_interp.isnull(), np.nan)
            
            logger.info(f"偏差计算完成，形状: {bias.shape}")
            return bias
            
        except Exception as e:
            logger.error(f"计算偏差失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _load_single_era5_file(self, file_path: Path, var_name: str) -> Optional[xr.DataArray]:
        """
        加载单个 ERA5 文件的辅助函数（用于并行处理）
        
        Args:
            file_path: 文件路径
            var_name: 变量名
        
        Returns:
            数据数组或 None
        """
        try:
            if not file_path.exists():
                return None
            
            ds = xr.open_dataset(file_path)
            
            # 查找变量
            if var_name not in ds:
                ds.close()
                return None
            
            da = ds[var_name]
            
            # 标准化坐标名（ERA5 使用 latitude/longitude）
            if 'latitude' in da.coords and 'lat' not in da.coords:
                da = da.rename({'latitude': 'lat'})
            if 'longitude' in da.coords and 'lon' not in da.coords:
                da = da.rename({'longitude': 'lon'})
            
            # 确保数据已加载
            da = da.load()
            ds.close()
            
            return da
            
        except Exception as e:
            logger.debug(f"处理文件 {file_path.name} 时出错: {e}")
            return None
    
    def load_era5_single_level_daily(self, var_name: str, year_range: Tuple[int, int] = (1993, 2020),
                                     root_dir: str = '/sas12t1/ffyan/ERA5/daily-nc/single-level/',
                                     parallel: bool = False, n_jobs: int = None) -> Optional[xr.DataArray]:
        """
        从本地加载 ERA5 single-level 日值数据
        
        Args:
            var_name: 变量名（如 'sst'）
            year_range: 年份范围，默认(1993, 2020)
            root_dir: ERA5 single-level 数据根目录
            parallel: 是否并行加载文件
            n_jobs: 并行作业数（仅用于并行模式）
        
        Returns:
            日值数据 (time, latitude, longitude) 或 None
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from multiprocessing import cpu_count
            
            var_dir = Path(root_dir) / var_name
            
            if not var_dir.exists():
                logger.error(f"变量目录不存在: {var_dir}")
                return None
            
            logger.info(f"从 ERA5 single-level 加载数据: {var_name}, 年份范围: {year_range}, 并行: {parallel}")
            
            start_year, end_year = year_range
            
            # 准备文件列表
            file_list = []
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    file_path = var_dir / f"era5_daily_{var_name}_{year}{month:02d}.nc"
                    file_list.append((file_path, year, month))
            
            monthly_da_list = []
            
            if parallel and n_jobs and n_jobs > 1:
                # 并行加载文件（使用 ThreadPoolExecutor 因为主要是I/O操作）
                # 限制并行数以避免 netCDF 库的线程安全问题
                max_workers = min(n_jobs, len(file_list), min(cpu_count() * 2, 16))
                logger.info(f"并行加载 ERA5 文件: {max_workers} 线程（限制以避免 netCDF 冲突）")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._load_single_era5_file, file_path, var_name): (year, month)
                        for file_path, year, month in file_list
                    }
                    
                    completed = 0
                    for future in as_completed(future_to_file):
                        year, month = future_to_file[future]
                        try:
                            da = future.result()
                            if da is not None:
                                monthly_da_list.append((year, month, da))
                            completed += 1
                            if completed % 12 == 0:
                                logger.debug(f"已加载 {completed}/{len(file_list)} 个文件")
                        except Exception as e:
                            logger.debug(f"加载文件失败 {year}{month:02d}: {e}")
                
                # 按时间排序
                monthly_da_list.sort(key=lambda x: (x[0], x[1]))
                monthly_da_list = [da for _, _, da in monthly_da_list]
            else:
                # 串行加载
                for file_path, year, month in file_list:
                    da = self._load_single_era5_file(file_path, var_name)
                    if da is not None:
                        monthly_da_list.append(da)
            
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
    
    def daily_to_monthly(self, da: xr.DataArray, var_name: str) -> Optional[xr.DataArray]:
        """
        将日值数据聚合为月平均
        
        Args:
            da: 日值数据 (time, lat, lon)
            var_name: 变量名（用于选择聚合方法）
        
        Returns:
            月平均数据 (time, lat, lon)
        """
        try:
            logger.info(f"将日值数据聚合为月平均: {var_name}")
            
            # 对于 SST、T2M 等温度变量，使用平均值
            # 对于降水等累积变量，使用求和（但这里主要处理 SST）
            monthly = da.resample(time='1MS').mean(dim='time')
            
            logger.info(f"月聚合完成: shape={monthly.shape}")
            return monthly
            
        except Exception as e:
            logger.error(f"月聚合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def area_weighted_mean(self, da: xr.DataArray, lat_name: str = 'lat') -> xr.DataArray:
        """
        计算面积加权平均（使用 cos(latitude) 权重）
        
        Args:
            da: 输入数据（需包含纬度维度）
            lat_name: 纬度坐标名
        
        Returns:
            面积加权空间平均后的数据
        """
        try:
            # 计算权重：cos(latitude in radians)
            weights = np.cos(np.deg2rad(da[lat_name]))
            
            # 对空间维度进行加权平均
            # 首先对经度求平均，然后对纬度加权平均
            if 'lon' in da.dims:
                # 先对经度求平均
                da_lon_mean = da.mean(dim='lon')
            else:
                da_lon_mean = da
            
            # 对纬度加权平均
            weighted_mean = (da_lon_mean * weights).sum(dim=lat_name) / weights.sum()
            
            return weighted_mean
            
        except Exception as e:
            logger.error(f"面积加权平均失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def compute_monthly_anomaly(self, series: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """
        计算逐月气候态异常
        
        Args:
            series: 时间序列数据 (time, ...)
            baseline: 气候态基期（如 "1993-2020"）
        
        Returns:
            月异常时间序列
        """
        try:
            # 解析基期
            start_year, end_year = baseline.split('-')
            start_year, end_year = int(start_year), int(end_year)
            
            logger.info(f"计算逐月气候态异常，基期: {baseline}")
            
            # 选择基期数据
            clim_data = series.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            
            # 按月份分组计算气候态（12个月，每月的多年平均）
            climatology = clim_data.groupby('time.month').mean(dim='time')
            
            # 计算异常：原始值 - 对应月份的气候态
            anomaly = series.groupby('time.month') - climatology
            
            logger.info(f"月异常计算完成: shape={anomaly.shape}")
            return anomaly
            
        except Exception as e:
            logger.error(f"计算月异常失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compute_seasonal_anomaly_mean(self, anomaly_series: xr.DataArray, 
                                      season: str) -> Optional[xr.DataArray]:
        """
        计算季节平均异常
        
        Args:
            anomaly_series: 月异常时间序列 (time,)
            season: 季节名称 ('Annual', 'DJF', 'MAM', 'JJA', 'SON')
        
        Returns:
            季平均异常时间序列，年份作为索引 (每年一个值)
        """
        try:
            times = pd.DatetimeIndex(anomaly_series.time.values)
            
            if season == 'Annual':
                # 按年分组，计算每年12个月的平均
                annual_means = anomaly_series.groupby('time.year').mean('time')
                # 重命名坐标为Year
                annual_means = annual_means.rename({'year': 'Year'})
                logger.info(f"季节平均计算完成 ({season}): {len(annual_means.Year)} 年")
                return annual_means
            
            elif season == 'DJF':
                # DJF需要跨年处理：被报日期为上一年的December与下一年的January和February
                # 例如：1993年12月 + 1994年1-2月 → 1994年DJF
                # 注意：anomaly_series的时间坐标代表实际被报日期（起报日期+leadtime）
                djf_data = []
                djf_years = []
                for year in range(times.year.min(), times.year.max()):
                    # 选择该年的12月和下一年的1-2月（基于实际被报日期）
                    mask = ((times.year == year) & (times.month == 12)) | \
                           ((times.year == year + 1) & (times.month.isin([1, 2])))
                    if mask.sum() == 3:  # 确保有3个月的数据
                        djf_data.append(float(anomaly_series.isel(time=mask).mean('time').values))
                        djf_years.append(year + 1)  # DJF年份标记为下一年的年份（January/February所在的年份）
                
                if djf_data:
                    result = xr.DataArray(
                        djf_data, 
                        dims=['time'], 
                        coords={'time': djf_years}
                    )
                    logger.info(f"季节平均计算完成 ({season}): {len(djf_years)} 年")
                    return result
                return None
            
            else:  # MAM, JJA, SON
                month_map = {'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
                if season not in month_map:
                    logger.error(f"不支持的季节: {season}")
                    return None
                    
                months = month_map[season]
                # 筛选对应月份
                season_data = anomaly_series.sel(time=anomaly_series.time.dt.month.isin(months))
                if season_data.size == 0:
                    logger.warning(f"季节 {season} 无数据")
                    return None
                # 按年分组计算平均
                seasonal_means = season_data.groupby('time.year').mean('time')
                seasonal_means = seasonal_means.rename({'year': 'Year'})
                logger.info(f"季节平均计算完成 ({season}): {len(seasonal_means.Year)} 年")
                return seasonal_means
                
        except Exception as e:
            logger.error(f"计算季节平均异常失败 ({season}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load_cmfd_data(self, var_name: str) -> Optional[xr.DataArray]:
        """
        加载 CMFD 观测数据（温度或降水）
        
        Args:
            var_name: 变量名（'temp' 或 'prec'）
        
        Returns:
            CMFD 数据 (time, lat, lon) 或 None
        """
        try:
            if var_name == 'temp':
                file_path = Path("/sas12t1/ffyan/obs/temp_1deg_199301-202012.nc")
            elif var_name == 'prec':
                file_path = Path("/sas12t1/ffyan/obs/prec_1deg_199301-202012.nc")
            else:
                logger.error(f"不支持的变量名: {var_name}")
                return None
            
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return None
            
            logger.info(f"加载 CMFD 数据: {var_name} from {file_path}")
            
            ds = xr.open_dataset(file_path)
            
            # 查找变量
            if var_name not in ds:
                logger.error(f"变量 {var_name} 不在文件中")
                ds.close()
                return None
            
            da = ds[var_name]
            da = da.load()
            ds.close()
            
            logger.info(f"CMFD 数据加载完成: {var_name}, shape={da.shape}")
            return da
            
        except Exception as e:
            logger.error(f"加载 CMFD 数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compute_china_region_mean(self, data: xr.DataArray, 
                                  spatial_bounds: Dict = None,
                                  skip_spatial_select: bool = False) -> Optional[xr.DataArray]:
        """
        计算中国区域的面积加权空间平均
        
        Args:
            data: 输入数据 (time, lat, lon) 或已裁剪的数据
            spatial_bounds: 空间范围，默认使用 SPATIAL_BOUNDS
            skip_spatial_select: 是否跳过空间裁剪（如果数据已经裁剪过）
        
        Returns:
            区域平均时间序列 (time,)
        """
        try:
            if spatial_bounds is None:
                spatial_bounds = SPATIAL_BOUNDS
            
            logger.info(f"计算中国区域平均，范围: {spatial_bounds}")
            
            # 如果未跳过空间裁剪，则执行裁剪
            if not skip_spatial_select:
                # 检查纬度顺序
                if 'lat' in data.coords and len(data.lat) > 1:
                    is_lat_desc = data.lat.values[0] > data.lat.values[-1]
                    if is_lat_desc:
                        # 降序纬度
                        lat_slice = slice(spatial_bounds['lat'][1], spatial_bounds['lat'][0])
                    else:
                        # 升序纬度
                        lat_slice = slice(spatial_bounds['lat'][0], spatial_bounds['lat'][1])
                    
                    lon_slice = slice(spatial_bounds['lon'][0], spatial_bounds['lon'][1])
                    region_data = data.sel(lat=lat_slice, lon=lon_slice)
                else:
                    region_data = data
            else:
                region_data = data
            
            if region_data.size == 0:
                logger.error("区域数据为空")
                return None
            
            # 计算面积权重（cos(latitude)）
            weights = np.cos(np.deg2rad(region_data.lat))
            weights = weights / weights.sum()  # 归一化
            
            # 对每个时间点计算加权空间平均
            # 先对经度求平均，再对纬度加权平均
            region_mean_lon = region_data.mean(dim='lon')
            region_mean = (region_mean_lon * weights).sum(dim='lat')
            
            logger.info(f"区域平均计算完成: shape={region_mean.shape}")
            return region_mean
            
        except Exception as e:
            logger.error(f"计算区域平均失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compute_cmfd_china_anomaly(self, var_name: str, 
                                   baseline: str = CLIMATOLOGY_PERIOD) -> Optional[xr.DataArray]:
        """
        计算 CMFD 中国区域温度/降水异常
        
        Args:
            var_name: 变量名（'temp' 或 'prec'）
            baseline: 气候态基期
        
        Returns:
            异常时间序列 (time,)
        """
        try:
            logger.info(f"计算 CMFD {var_name} 中国区域异常...")
            
            # 加载 CMFD 数据
            cmfd_data = self.load_cmfd_data(var_name)
            if cmfd_data is None:
                return None
            
            # 计算区域平均
            region_mean = self.compute_china_region_mean(cmfd_data)
            if region_mean is None:
                return None
            
            # 计算逐月异常
            anomaly = self.compute_monthly_anomaly(region_mean, baseline)
            if anomaly is None:
                return None
            
            # 单位转换：如果是降水，将CMFD从 kg m⁻² s⁻¹ 转换为 m s⁻¹（与模式一致）
            if var_name == 'prec':
                original_units = cmfd_data.attrs.get('units', '')
                if 'kg m-2 s-1' in original_units.lower() or 'kg m⁻² s⁻¹' in original_units:
                    # 转换：1 kg m⁻² s⁻¹ = 0.001 m s⁻¹（假设水的密度为 1000 kg/m³）
                    # 因为：kg m⁻² s⁻¹ = (kg/m³) × (m/s) = ρ × (m/s)，ρ_water ≈ 1000 kg/m³
                    anomaly = anomaly * 0.001
                    logger.info(f"CMFD 降水单位已转换：{original_units} → m s⁻¹")
                    target_units = 'm s⁻¹'
                else:
                    target_units = original_units
            else:
                target_units = cmfd_data.attrs.get('units', 'unknown')
            
            # 添加属性
            anomaly.attrs = {
                'long_name': f'CMFD {var_name} anomaly over China',
                'description': f'Area-weighted spatial mean of {var_name} anomaly over China region',
                'units': target_units,
                'baseline_period': baseline,
                'data_source': 'CMFD',
                'spatial_bounds': str(SPATIAL_BOUNDS)
            }
            
            logger.info(f"CMFD {var_name} 异常计算完成: {len(anomaly.time)} 个月")
            return anomaly
            
        except Exception as e:
            logger.error(f"计算 CMFD {var_name} 异常失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_china_temp_prec_anomaly(self, temp_anom: xr.DataArray, 
                                     prec_anom: xr.DataArray,
                                     source: str = 'cmfd') -> bool:
        """
        保存中国区域温度/降水异常到 NetCDF 文件
        
        Args:
            temp_anom: 温度异常时间序列
            prec_anom: 降水异常时间序列
            source: 数据源（'cmfd', 'era5', 或模型名）
        
        Returns:
            是否保存成功
        """
        try:
            output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"保存 {source} 中国区域温度/降水异常...")
            
            # 创建 Dataset
            ds = xr.Dataset({
                'temp_anom': temp_anom,
                'prec_anom': prec_anom
            })
            
            ds.attrs.update({
                'title': f'China region temperature and precipitation anomaly - {source}',
                'description': 'Area-weighted spatial mean anomaly over China region',
                'baseline_period': CLIMATOLOGY_PERIOD,
                'spatial_bounds': str(SPATIAL_BOUNDS),
                'data_source': source,
                'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 保存 NetCDF
            output_file = output_dir / f"circulation_china_temp_prec_anom_{source}.nc"
            ds.to_netcdf(output_file)
            logger.info(f"NetCDF 文件已保存: {output_file}")
            
            # 也保存为 CSV（方便查看）
            csv_file = output_dir / f"circulation_china_temp_prec_anom_{source}.csv"
            df = ds.to_dataframe()
            df.to_csv(csv_file, float_format='%.6f')
            logger.info(f"CSV 文件已保存: {csv_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存中国区域温度/降水异常失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def save_seasonal_anomaly(self, season: str,
                             cmfd_temp_seasonal: xr.DataArray = None,
                             cmfd_prec_seasonal: xr.DataArray = None,
                             model_temp_seasonal: Dict[Tuple[str, int], xr.DataArray] = None,
                             model_prec_seasonal: Dict[Tuple[str, int], xr.DataArray] = None) -> bool:
        """
        保存季节平均异常数据
        
        Args:
            season: 季节名称 ('Annual', 'DJF', 'MAM', 'JJA', 'SON')
            cmfd_temp_seasonal: CMFD 温度季平均异常
            cmfd_prec_seasonal: CMFD 降水季平均异常
            model_temp_seasonal: 模式温度季平均异常字典 {(model, leadtime): DataArray}
            model_prec_seasonal: 模式降水季平均异常字典 {(model, leadtime): DataArray}
        
        Returns:
            是否保存成功
        """
        try:
            output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"保存季节平均异常数据 ({season})...")
            
            # 保存 CMFD 季节平均
            if cmfd_temp_seasonal is not None and cmfd_prec_seasonal is not None:
                ds = xr.Dataset({
                    'temp_anom': cmfd_temp_seasonal,
                    'prec_anom': cmfd_prec_seasonal
                })
                ds.attrs.update({
                    'title': f'China region temperature and precipitation anomaly - CMFD - {season}',
                    'season': season,
                    'baseline_period': CLIMATOLOGY_PERIOD,
                    'spatial_bounds': str(SPATIAL_BOUNDS)
                })
                output_file = output_dir / f"circulation_china_temp_prec_anom_cmfd_{season}.nc"
                ds.to_netcdf(output_file)
                csv_file = output_dir / f"circulation_china_temp_prec_anom_cmfd_{season}.csv"
                ds.to_dataframe().to_csv(csv_file, float_format='%.6f')
                logger.info(f"CMFD {season} 数据已保存")
            
            # 保存模式季节平均
            if model_temp_seasonal and model_prec_seasonal:
                for (model, leadtime), temp_anom in model_temp_seasonal.items():
                    if (model, leadtime) in model_prec_seasonal:
                        prec_anom = model_prec_seasonal[(model, leadtime)]
                        ds = xr.Dataset({
                            'temp_anom': temp_anom,
                            'prec_anom': prec_anom
                        })
                        ds.attrs.update({
                            'title': f'China region temperature and precipitation anomaly - {model} L{leadtime} - {season}',
                            'season': season,
                            'baseline_period': CLIMATOLOGY_PERIOD,
                            'spatial_bounds': str(SPATIAL_BOUNDS)
                        })
                        model_key = f"{model}_L{leadtime}".lower().replace('-', '_')
                        output_file = output_dir / f"circulation_china_temp_prec_anom_{model_key}_{season}.nc"
                        ds.to_netcdf(output_file)
                        csv_file = output_dir / f"circulation_china_temp_prec_anom_{model_key}_{season}.csv"
                        ds.to_dataframe().to_csv(csv_file, float_format='%.6f')
                        logger.debug(f"{model} L{leadtime} {season} 数据已保存")
                
                logger.info(f"所有模式 {season} 数据已保存")
            
            return True
            
        except Exception as e:
            logger.error(f"保存季节平均异常数据失败 ({season}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def compute_model_china_anomaly(self, model: str, leadtime: int, var_name: str,
                                    baseline: str = CLIMATOLOGY_PERIOD,
                                    year_range: Tuple[int, int] = (1993, 2020)) -> Optional[xr.DataArray]:
        """
        计算模式中国区域温度/降水异常
        
        Args:
            model: 模型名称
            leadtime: 提前期
            var_name: 变量名（'temp' 或 'prec'）
            baseline: 气候态基期
            year_range: 年份范围
        
        Returns:
            异常时间序列 (time,)
        """
        try:
            logger.info(f"计算模式 {model} L{leadtime} {var_name} 中国区域异常...")
            
            # 加载模式数据
            model_data = self.load_fcst_surface_data(model, leadtime, var_name, year_range)
            if model_data is None:
                return None
            
            # 如果有 number 维度，先计算 ensemble mean
            if 'number' in model_data.dims:
                model_data_mean = model_data.mean(dim='number')
            else:
                model_data_mean = model_data
            
            # 计算区域平均（数据已经在加载时裁剪到中国区域，跳过二次裁剪）
            region_mean = self.compute_china_region_mean(model_data_mean, skip_spatial_select=True)
            if region_mean is None:
                return None
            
            # 计算逐月异常
            anomaly = self.compute_monthly_anomaly(region_mean, baseline)
            if anomaly is None:
                return None
            
            # 添加属性
            anomaly.attrs = {
                'long_name': f'{model} {var_name} anomaly over China',
                'description': f'Area-weighted spatial mean of {var_name} anomaly over China region',
                'units': model_data.attrs.get('units', 'unknown'),
                'baseline_period': baseline,
                'data_source': f'{model} L{leadtime}',
                'spatial_bounds': str(SPATIAL_BOUNDS)
            }
            
            logger.info(f"模式 {model} L{leadtime} {var_name} 异常计算完成: {len(anomaly.time)} 个月")
            return anomaly
            
        except Exception as e:
            logger.error(f"计算模式 {model} L{leadtime} {var_name} 异常失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_obs_climatology(self, obs_clim_dict: Dict) -> bool:
        """
        保存观测气候态到NetCDF文件
        
        Args:
            obs_clim_dict: 观测气候态字典，格式为{var_name: {season: DataArray}}
        
        Returns:
            是否保存成功
        """
        try:
            output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 各变量的单位/说明配置
            attr_map = {
                'u_850': {'units': 'm s^-1', 'long_name': 'Zonal wind at 850 hPa'},
                'v_850': {'units': 'm s^-1', 'long_name': 'Meridional wind at 850 hPa'},
                'ght_850': {'units': 'm', 'long_name': 'Geopotential height at 850 hPa'},
                'u_500': {'units': 'm s^-1', 'long_name': 'Zonal wind at 500 hPa'},
                'v_500': {'units': 'm s^-1', 'long_name': 'Meridional wind at 500 hPa'},
                'ght_500': {'units': 'm', 'long_name': 'Geopotential height at 500 hPa'}
            }
            
            # 为每个变量-季节组合保存单独的文件
            for var_name, seasonal_data in obs_clim_dict.items():
                
                if not seasonal_data:
                    continue
                
                for season, data in seasonal_data.items():
                    if data is None:
                        continue
                    
                    try:
                        # 添加属性
                        data_with_attrs = data.copy()
                        if var_name in attr_map:
                            data_with_attrs.attrs.update(attr_map[var_name])
                        data_with_attrs.attrs.update({
                            'source': 'ERA5',
                            'climatology_period': CLIMATOLOGY_PERIOD,
                            'season': season,
                            'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        # 保存文件
                        output_file = output_dir / f"circulation_obs_{var_name}_{season}.nc"
                        data_with_attrs.to_netcdf(output_file)
                        logger.info(f"观测气候态已保存: {output_file}")
                        
                    except Exception as e:
                        logger.error(f"保存观测数据失败 {var_name} {season}: {e}")
                        continue
            
            logger.info("观测气候态保存完成")
            return True
            
        except Exception as e:
            logger.error(f"保存观测气候态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    





    def compute_moisture_flux_divergence(self, u: xr.DataArray, v: xr.DataArray, q: xr.DataArray) -> Optional[xr.DataArray]:
        """
        计算水汽通量散度
        
        水汽通量 Q = q * (u, v)
        散度 = ∇·Q = ∂(q*u)/∂x + ∂(q*v)/∂y
        
        Args:
            u: U风场分量 (time, number, lat, lon) 或 (time, lat, lon)
            v: V风场分量 (time, number, lat, lon) 或 (time, lat, lon)
            q: 比湿 (time, number, lat, lon) 或 (time, lat, lon)
        
        Returns:
            水汽通量散度 (time, number, lat, lon) 或 (time, lat, lon)
        """
        try:
            # 确保u、v、q的维度一致
            if u.shape != v.shape or u.shape != q.shape:
                logger.error(f"u、v、q的形状不一致: u={u.shape}, v={v.shape}, q={q.shape}")
                return None
            
            # 计算水汽通量分量
            qu = q * u  # 纬向水汽通量 (kg/kg * m/s)
            qv = q * v  # 经向水汽通量 (kg/kg * m/s)
            
            # 地球半径（米）
            R = 6371000.0
            
            # 使用xarray的diff方法计算梯度
            # 对经度求导：∂(q*u)/∂lon
            qu_diff_lon = qu.diff('lon') / qu.lon.diff('lon')
            # 转换为实际距离（米）
            # 需要为每个纬度计算经度间距
            lat_rad = np.deg2rad(qu.lat)
            lon_diff_rad = np.deg2rad(qu.lon.diff('lon'))
            # 扩展维度以匹配qu_diff_lon
            cos_lat = np.cos(lat_rad)
            lon_diff_m = (R * cos_lat.values[:, np.newaxis] * lon_diff_rad.values[np.newaxis, :])
            # 创建xarray DataArray
            lon_diff_m_da = xr.DataArray(
                lon_diff_m,
                coords={'lat': qu.lat, 'lon': qu.lon.isel(lon=slice(1, None))},
                dims=['lat', 'lon']
            )
            # 对齐坐标
            qu_diff_lon_aligned = qu_diff_lon / lon_diff_m_da
            
            # 对纬度求导：∂(q*v)/∂lat
            qv_diff_lat = qv.diff('lat') / qv.lat.diff('lat')
            # 转换为实际距离（米）
            lat_diff_rad = np.deg2rad(qv.lat.diff('lat'))
            lat_diff_m = R * lat_diff_rad.values
            # 创建xarray DataArray
            lat_diff_m_da = xr.DataArray(
                lat_diff_m,
                coords={'lat': qv.lat.isel(lat=slice(1, None))},
                dims=['lat']
            )
            # 对齐坐标
            qv_diff_lat_aligned = qv_diff_lat / lat_diff_m_da
            
            # 插值到原始网格（因为diff会减少一个点）
            # 使用前向填充或插值
            qu_diff_lon_interp = qu_diff_lon_aligned.reindex_like(qu, method='nearest', fill_value=np.nan)
            qv_diff_lat_interp = qv_diff_lat_aligned.reindex_like(qv, method='nearest', fill_value=np.nan)
            
            # 散度 = ∂(q*u)/∂x + ∂(q*v)/∂y
            divergence = qu_diff_lon_interp + qv_diff_lat_interp
            
            # 设置属性
            # 注意：如果q是比湿(kg/kg)，u和v是m/s，则q*u和q*v的单位是m/s
            # 散度的单位是 1/s (s^-1)
            divergence.attrs = {
                'long_name': 'Moisture flux divergence',
                'description': 'Divergence of moisture flux: ∇·(q*(u,v))',
                'units': 's^-1'
            }
            
            logger.info(f"水汽通量散度计算完成: {divergence.shape}")
            return divergence
            
        except Exception as e:
            logger.error(f"计算水汽通量散度失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_obs_circulation_data(self) -> Optional[Dict]:
        """
        处理观测数据的环流分析计算
        
        Returns:
            包含观测气候态的字典或None
        """
        try:
            logger.info("处理观测数据")
            
            results = {}
            
            # 1. 加载850hPa的观测数据：u、v、z（位势）
            logger.info("加载850hPa观测数据...")
            u_850_obs = self.load_obs_pressure_level_data('u', 850)
            v_850_obs = self.load_obs_pressure_level_data('v', 850)
            z_850_obs = self.load_obs_pressure_level_data('z', 850)
            
            if u_850_obs is None or v_850_obs is None or z_850_obs is None:
                logger.warning("850hPa u、v、z观测数据加载失败")
                return None
            
            # 2. 加载500hPa的观测数据：u、v、z（位势）
            logger.info("加载500hPa观测数据...")
            u_500_obs = self.load_obs_pressure_level_data('u', 500)
            v_500_obs = self.load_obs_pressure_level_data('v', 500)
            z_500_obs = self.load_obs_pressure_level_data('z', 500)
            
            if u_500_obs is None or v_500_obs is None or z_500_obs is None:
                logger.warning("500hPa观测数据加载失败")
                return None
            
            # 3. 将位势（z）转换为位势高度（GHT）
            g = 9.80665  # 标准重力加速度
            
            # 850hPa GHT
            ght_850_obs = z_850_obs / g
            ght_850_obs.attrs = {
                'long_name': 'Geopotential Height',
                'units': 'm',
                'description': 'Geopotential height at 850 hPa from ERA5'
            }
            
            # 500hPa GHT
            ght_500_obs = z_500_obs / g
            ght_500_obs.attrs = {
                'long_name': 'Geopotential Height',
                'units': 'm',
                'description': 'Geopotential height at 500 hPa from ERA5'
            }
            
            # 4. 计算各季节的气候平均态
            logger.info("计算观测气候平均态...")
            
            # 850hPa的u、v风场和GHT
            u_850_seasonal = self.compute_seasonal_climatology(u_850_obs)
            v_850_seasonal = self.compute_seasonal_climatology(v_850_obs)
            ght_850_seasonal = self.compute_seasonal_climatology(ght_850_obs)
            
            # 500hPa的u、v风场和GHT
            u_500_seasonal = self.compute_seasonal_climatology(u_500_obs)
            v_500_seasonal = self.compute_seasonal_climatology(v_500_obs)
            ght_500_seasonal = self.compute_seasonal_climatology(ght_500_obs)
            
            # 5. 保存观测结果
            results['u_850'] = u_850_seasonal
            results['v_850'] = v_850_seasonal
            results['ght_850'] = ght_850_seasonal
            results['u_500'] = u_500_seasonal
            results['v_500'] = v_500_seasonal
            results['ght_500'] = ght_500_seasonal
            
            logger.info("观测数据处理完成")
            return results
            
        except Exception as e:
            logger.error(f"处理观测数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_circulation_analysis(self, model: str, leadtime: int, 
                                    obs_climatology: Optional[Dict] = None) -> Optional[Dict]:
        """
        处理单个模型和提前期的环流分析计算
        
        Args:
            model: 模型名称
            leadtime: 提前期
            obs_climatology: 观测气候态（如果提供，则计算偏差）
        
        Returns:
            包含输出文件路径的字典或None
        """
        try:
            logger.info(f"处理 {model} L{leadtime}")
            
            results = {}
            
            # 1. 加载850hPa的模式数据：u、v、z（位势）
            logger.info("加载850hPa模式数据...")
            u_850 = self.load_fcst_data_at_level(model, leadtime, 'u', 850)
            v_850 = self.load_fcst_data_at_level(model, leadtime, 'v', 850)
            z_850 = self.load_fcst_data_at_level(model, leadtime, 'z', 850)
            
            if u_850 is None or v_850 is None or z_850 is None:
                logger.warning(f"850hPa u、v、z数据加载失败: {model} L{leadtime}")
                return None
            
            # 2. 加载500hPa的模式数据：u、v、z（位势高度）
            logger.info("加载500hPa模式数据...")
            u_500 = self.load_fcst_data_at_level(model, leadtime, 'u', 500)
            v_500 = self.load_fcst_data_at_level(model, leadtime, 'v', 500)
            z_500 = self.load_fcst_data_at_level(model, leadtime, 'z', 500)
            
            if u_500 is None or v_500 is None or z_500 is None:
                logger.warning(f"500hPa数据加载失败: {model} L{leadtime}")
                return None
            
            # 3. 计算ensemble mean（如果有多成员）
            if 'number' in u_850.dims:
                u_850_mean = u_850.mean(dim='number')
                v_850_mean = v_850.mean(dim='number')
                z_850_mean = z_850.mean(dim='number')
            else:
                u_850_mean = u_850
                v_850_mean = v_850
                z_850_mean = z_850
            
            if 'number' in u_500.dims:
                u_500_mean = u_500.mean(dim='number')
                v_500_mean = v_500.mean(dim='number')
                z_500_mean = z_500.mean(dim='number')
            else:
                u_500_mean = u_500
                v_500_mean = v_500
                z_500_mean = z_500
            
            # 4. 将位势（z）转换为位势高度（GHT）
            # GHT = z / g, 其中 g = 9.80665 m/s^2
            g = 9.80665  # 标准重力加速度
            
            ght_850_mean = z_850_mean / g
            ght_850_mean.attrs = {
                'long_name': 'Geopotential Height',
                'units': 'm',
                'description': 'Geopotential height at 850 hPa'
            }
            
            ght_500_mean = z_500_mean / g
            ght_500_mean.attrs = {
                'long_name': 'Geopotential Height',
                'units': 'm',
                'description': 'Geopotential height at 500 hPa'
            }
            
            # 5. 计算各季节的气候平均态
            logger.info("计算模式气候平均态...")
            
            # 850hPa的u、v风场和GHT
            u_850_seasonal = self.compute_seasonal_climatology(u_850_mean)
            v_850_seasonal = self.compute_seasonal_climatology(v_850_mean)
            ght_850_seasonal = self.compute_seasonal_climatology(ght_850_mean)
            
            # 500hPa的u、v风场和GHT
            u_500_seasonal = self.compute_seasonal_climatology(u_500_mean)
            v_500_seasonal = self.compute_seasonal_climatology(v_500_mean)
            ght_500_seasonal = self.compute_seasonal_climatology(ght_500_mean)
            
            # 7. 保存模式气候态和偏差结果
            try:
                output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建输出数据集
                output_ds_dict = {}
                bias_ds_dict = {}
                
                # 添加850hPa数据（移除level坐标以避免冲突）
                def _with_attrs(da: xr.DataArray, attrs: Dict[str, str]) -> xr.DataArray:
                    """复制并附加属性，避免原对象被污染。"""
                    da = da.copy()
                    da.attrs.update(attrs)
                    return da
                
                # 各变量的单位/说明配置
                attr_map = {
                    'u_850': {'units': 'm s^-1', 'long_name': 'Zonal wind at 850 hPa'},
                    'v_850': {'units': 'm s^-1', 'long_name': 'Meridional wind at 850 hPa'},
                    'ght_850': {'units': 'm', 'long_name': 'Geopotential height at 850 hPa'},
                    'u_500': {'units': 'm s^-1', 'long_name': 'Zonal wind at 500 hPa'},
                    'v_500': {'units': 'm s^-1', 'long_name': 'Meridional wind at 500 hPa'},
                    'ght_500': {'units': 'm', 'long_name': 'Geopotential height at 500 hPa'}
                }
                
                # 保存模式气候态和偏差（如果有观测）
                for season in ['Annual', 'DJF', 'MAM', 'JJA', 'SON']:
                    # 850hPa u
                    if season in u_850_seasonal and u_850_seasonal[season] is not None:
                        da = u_850_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'u_850_{season}'] = _with_attrs(da, attr_map['u_850'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'u_850' in obs_climatology:
                            if season in obs_climatology['u_850'] and obs_climatology['u_850'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['u_850'][season])
                                if bias is not None:
                                    bias_ds_dict[f'u_850_{season}'] = _with_attrs(bias, attr_map['u_850'])
                    
                    # 850hPa v
                    if season in v_850_seasonal and v_850_seasonal[season] is not None:
                        da = v_850_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'v_850_{season}'] = _with_attrs(da, attr_map['v_850'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'v_850' in obs_climatology:
                            if season in obs_climatology['v_850'] and obs_climatology['v_850'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['v_850'][season])
                                if bias is not None:
                                    bias_ds_dict[f'v_850_{season}'] = _with_attrs(bias, attr_map['v_850'])
                    
                    # 850hPa GHT
                    if season in ght_850_seasonal and ght_850_seasonal[season] is not None:
                        da = ght_850_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'ght_850_{season}'] = _with_attrs(da, attr_map['ght_850'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'ght_850' in obs_climatology:
                            if season in obs_climatology['ght_850'] and obs_climatology['ght_850'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['ght_850'][season])
                                if bias is not None:
                                    bias_ds_dict[f'ght_850_{season}'] = _with_attrs(bias, attr_map['ght_850'])
                    
                    # 500hPa u
                    if season in u_500_seasonal and u_500_seasonal[season] is not None:
                        da = u_500_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'u_500_{season}'] = _with_attrs(da, attr_map['u_500'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'u_500' in obs_climatology:
                            if season in obs_climatology['u_500'] and obs_climatology['u_500'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['u_500'][season])
                                if bias is not None:
                                    bias_ds_dict[f'u_500_{season}'] = _with_attrs(bias, attr_map['u_500'])
                    
                    # 500hPa v
                    if season in v_500_seasonal and v_500_seasonal[season] is not None:
                        da = v_500_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'v_500_{season}'] = _with_attrs(da, attr_map['v_500'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'v_500' in obs_climatology:
                            if season in obs_climatology['v_500'] and obs_climatology['v_500'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['v_500'][season])
                                if bias is not None:
                                    bias_ds_dict[f'v_500_{season}'] = _with_attrs(bias, attr_map['v_500'])
                    
                    # 500hPa GHT
                    if season in ght_500_seasonal and ght_500_seasonal[season] is not None:
                        da = ght_500_seasonal[season].drop_vars('level', errors='ignore')
                        output_ds_dict[f'ght_500_{season}'] = _with_attrs(da, attr_map['ght_500'])
                        
                        # 计算偏差（如果有观测）
                        if obs_climatology is not None and 'ght_500' in obs_climatology:
                            if season in obs_climatology['ght_500'] and obs_climatology['ght_500'][season] is not None:
                                bias = self.calculate_bias(da, obs_climatology['ght_500'][season])
                                if bias is not None:
                                    bias_ds_dict[f'ght_500_{season}'] = _with_attrs(bias, attr_map['ght_500'])
                
                if not output_ds_dict:
                    logger.warning(f"没有有效数据可保存: {model} L{leadtime}")
                    return None
                
                # 保存模式气候态
                output_ds = xr.Dataset(output_ds_dict)
                output_ds.attrs.update({
                    'model_name': model,
                    'leadtime': leadtime,
                    'climatology_period': CLIMATOLOGY_PERIOD,
                    'description': 'Circulation analysis: u, v wind fields and geopotential height at 850hPa and 500hPa. Seasonal climatology: Annual, DJF, MAM, JJA, SON.',
                    'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                output_file = output_dir / f"circulation_{model}_L{leadtime}.nc"
                output_ds.to_netcdf(output_file)
                results['output_file'] = output_file
                logger.info(f"模式气候态已保存: {output_file}")
                
                # 保存偏差（如果有）
                if bias_ds_dict:
                    bias_ds = xr.Dataset(bias_ds_dict)
                    bias_ds.attrs.update({
                        'model_name': model,
                        'leadtime': leadtime,
                        'climatology_period': CLIMATOLOGY_PERIOD,
                        'description': 'Circulation bias (model - observation): u, v wind fields and geopotential height at 850hPa and 500hPa.',
                        'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    bias_file = output_dir / f"circulation_bias_{model}_L{leadtime}.nc"
                    bias_ds.to_netcdf(bias_file)
                    results['bias_file'] = bias_file
                    logger.info(f"偏差已保存: {bias_file}")
                
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            return results
            
        except Exception as e:
            logger.error(f"处理 {model} L{leadtime} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_circulation_climatology(self, var_name: str, var_title: str, season: str,
                                    obs_clim: xr.DataArray, 
                                    all_leadtime_biases: Dict[int, Dict[str, xr.DataArray]],
                                    unit: str,
                                    unified_clim_vmin: Optional[float] = None,
                                    unified_clim_vmax: Optional[float] = None,
                                    obs_clim_component: Optional[xr.DataArray] = None,
                                    all_leadtime_biases_component: Optional[Dict[int, Dict[str, xr.DataArray]]] = None):
        """
        绘制环流气候态和偏差组合图（lead0和lead3）
        
        布局（4行4列）：
        - 第一行第一个：观测气候态
        - 第一行后三个：三个模式的lead0偏差
        - 第二行：四个模式的lead0偏差
        - 第三行第一个：留空
        - 第三行后三个：三个模式的lead3偏差
        - 第四行：四个模式的lead3偏差
        
        Args:
            var_name: 变量名（如u_850）
            var_title: 变量标题（如"U Wind at 850hPa"）
            season: 季节名称
            obs_clim: 观测气候态
            all_leadtime_biases: 偏差字典 {leadtime: {model: DataArray}}
            unit: 单位
        """
        try:
            from matplotlib.gridspec import GridSpec
            from matplotlib.patches import Rectangle
            from matplotlib.ticker import FixedLocator
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
            
            logger.info(f"绘制环流气候态图: {var_name} {season}")
            
            # 准备模型列表（从第一个leadtime获取）
            leadtime_keys = sorted(all_leadtime_biases.keys())
            if not leadtime_keys:
                logger.warning("没有可用的leadtime数据")
                return
            
            first_leadtime = leadtime_keys[0]
            model_names = list(all_leadtime_biases[first_leadtime].keys())
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 判断是否为风场（wind变量或u、v变量）
            is_wind_field = var_name.startswith('wind_') or var_name.startswith('u_') or var_name.startswith('v_')
            
            # 收集所有leadtime的所有偏差数据，计算统一colorbar范围
            # 对于wind场，应该计算风速偏差大小（合成后的矢量偏差大小）
            all_bias_values = []
            if is_wind_field and all_leadtime_biases_component is not None:
                # 对于wind场，计算风速偏差大小
                for leadtime in all_leadtime_biases.keys():
                    leadtime_biases = all_leadtime_biases[leadtime]
                    if leadtime in all_leadtime_biases_component:
                        leadtime_biases_comp = all_leadtime_biases_component[leadtime]
                        for model in leadtime_biases.keys():
                            if model in leadtime_biases_comp:
                                u_bias = leadtime_biases[model].values
                                v_bias = leadtime_biases_comp[model].values
                                wind_bias_magnitude = np.sqrt(u_bias**2 + v_bias**2)
                                bias_valid = wind_bias_magnitude[np.isfinite(wind_bias_magnitude)]
                                all_bias_values.extend(bias_valid)
            else:
                # 对于非wind场，使用原始偏差值
                for leadtime_biases in all_leadtime_biases.values():
                    for model_bias in leadtime_biases.values():
                        bias_valid = model_bias.values[np.isfinite(model_bias.values)]
                        all_bias_values.extend(bias_valid)
            
            # 计算偏差颜色范围
            # 根据变量类型选择不同的bias colormap
            if 'ght' in var_name.lower():
                cmap_bias = 'coolwarm'  # ght使用coolwarm配色
            else:
                cmap_bias = 'RdBu_r'  # u、v使用红蓝配色
            
            if len(all_bias_values) > 0:
                if is_wind_field:
                    # 对于wind场，偏差大小不能为负，使用0到最大值
                    bias_abs_max = np.percentile(all_bias_values, 99)
                    if np.isfinite(bias_abs_max) and bias_abs_max > 0:
                        bias_vmin = 0
                        bias_vmax = bias_abs_max
                    else:
                        bias_vmin = 0
                        bias_vmax = 1
                else:
                    # 对于非wind场，使用对称范围
                    bias_abs_max = np.percentile(np.abs(all_bias_values), 99)
                    if np.isfinite(bias_abs_max) and bias_abs_max > 0:
                        bias_vmin = -bias_abs_max
                        bias_vmax = bias_abs_max
                    else:
                        bias_vmin = -1
                        bias_vmax = 1
            else:
                if is_wind_field:
                    bias_vmin = 0
                    bias_vmax = 1
                else:
                    bias_vmin = -1
                    bias_vmax = 1
            
            # 计算观测气候态颜色范围
            # 对于wind场，应该计算风速大小（合成后的矢量大小）
            if is_wind_field and obs_clim_component is not None:
                # 计算风速大小
                wind_speed = np.sqrt(obs_clim.values**2 + obs_clim_component.values**2)
                obs_valid = wind_speed[np.isfinite(wind_speed)]
                if len(obs_valid) > 0:
                    obs_mean = np.mean(obs_valid)
                    obs_std = np.std(obs_valid)
                    clim_vmin = max(0, obs_mean - 2 * obs_std)  # 风速大小不能为负
                    clim_vmax = obs_mean + 2 * obs_std
                else:
                    clim_vmin = 0
                    clim_vmax = np.nanmax(wind_speed)
            else:
                # 优先使用统一的范围（如果提供）
                if unified_clim_vmin is not None and unified_clim_vmax is not None:
                    clim_vmin = unified_clim_vmin
                    clim_vmax = unified_clim_vmax
                    logger.info(f"使用统一气候态范围: [{clim_vmin:.2f}, {clim_vmax:.2f}]")
                else:
                    # 否则单独计算
                    obs_valid = obs_clim.values[np.isfinite(obs_clim.values)]
                    if len(obs_valid) > 0:
                        obs_mean = np.mean(obs_valid)
                        obs_std = np.std(obs_valid)
                        clim_vmin = obs_mean - 2 * obs_std
                        clim_vmax = obs_mean + 2 * obs_std
                    else:
                        clim_vmin = np.nanmin(obs_clim.values)
                        clim_vmax = np.nanmax(obs_clim.values)
            
            # 选择气候态colormap（与bias区分）
            if 'ght' in var_name.lower():
                cmap_clim = 'coolwarm'  # ght使用冷暖配色
            else:
                cmap_clim = 'seismic'  # u、v使用地震配色（与RdBu_r不同）
            
            logger.info(f"观测范围: [{clim_vmin:.2f}, {clim_vmax:.2f}]")
            logger.info(f"偏差范围: [{bias_vmin:.2f}, {bias_vmax:.2f}]")
            
            # 辅助函数：计算等高线级别数量
            def _compute_n_levels(data, vmin, vmax, default=12):
                """根据数据分布自动决定等高线数量"""
                data_range = vmax - vmin
                if data_range == 0:
                    return default
                valid_data = data[np.isfinite(data)]
                if len(valid_data) == 0:
                    return default
                
                data_std = np.std(valid_data)
                if data_range < 3 * data_std:
                    return 15  # 密集
                elif data_range < 10 * data_std:
                    return 12  # 中等
                else:
                    return 10  # 稀疏
            
            # 辅助函数：绘制偏差（支持矢量箭头）
            def _plot_bias(ax, model_bias, model_lon_centers, model_lat_centers, 
                          model_lon_edges, model_lat_edges, leadtime, model):
                """绘制偏差，对于u、v风场使用矢量箭头"""
                if is_wind_field and all_leadtime_biases_component is not None:
                    # 获取对应的分量偏差
                    model_bias_comp = None
                    if leadtime in all_leadtime_biases_component and model in all_leadtime_biases_component[leadtime]:
                        model_bias_comp = all_leadtime_biases_component[leadtime][model]
                    
                    if model_bias_comp is not None:
                        # 计算偏差大小（用于箭头大小和颜色）
                        bias_magnitude = np.sqrt(model_bias.values**2 + model_bias_comp.values**2)
                        
                        # 确定u和v分量偏差
                        if var_name.startswith('wind_'):
                            # wind变量：model_bias是u偏差，model_bias_comp是v偏差
                            u_bias = model_bias.values
                            v_bias = model_bias_comp.values
                        elif var_name.startswith('u_'):
                            u_bias = model_bias.values
                            v_bias = model_bias_comp.values
                        else:  # v_
                            u_bias = model_bias_comp.values
                            v_bias = model_bias.values
                        
                        # 每个点都绘制箭头，箭头长度统一，颜色表示偏差大小
                        lon_sub = model_lon_centers
                        lat_sub = model_lat_centers
                        u_sub = u_bias
                        v_sub = v_bias
                        mag_sub = bias_magnitude
                        
                        # 归一化u和v分量，使所有箭头长度统一（单位向量）
                        # 计算每个点的偏差大小
                        magnitude = np.sqrt(u_sub**2 + v_sub**2)
                        # 避免除零：对于magnitude为0或NaN的点，u和v保持为0
                        magnitude_safe = np.where(magnitude > 1e-10, magnitude, 1.0)
                        u_normalized = np.where(magnitude > 1e-10, u_sub / magnitude_safe, 0.0)
                        v_normalized = np.where(magnitude > 1e-10, v_sub / magnitude_safe, 0.0)
                        
                        # 使用固定的scale值，使箭头长度统一
                        # scale值越大，箭头越小。这里使用一个固定值，使箭头长度适中
                        scale = 50.0  # 固定scale值，所有箭头长度相同
                        
                        # 仅调整箭头头部尺寸（线段部分由 scale/width 控制，保持不变）
                        # 用户诉求：箭头头部略微增大
                        headwidth = 2.8
                        headlength = 3.6
                        headaxislength = 2.8
                        
                        # 绘制矢量箭头，箭头长度统一，颜色表示偏差大小
                        im = ax.quiver(lon_sub, lat_sub, u_normalized, v_normalized, mag_sub,
                                      transform=ccrs.PlateCarree(),
                                      scale=scale, width=0.002, cmap='coolwarm',
                                      clim=(bias_vmin, bias_vmax),
                                      headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)
                        return im
                    else:
                        # 如果没有分量数据，回退到填色等高线图
                        # 使用 MaxNLocator 自动选择等间距且“美观”的刻度（如 0.1, 0.5, 1 等）
                        levels = ticker.MaxNLocator(nbins=11, prune=None).tick_values(bias_vmin, bias_vmax)
                        im = ax.contourf(model_lon_centers, model_lat_centers, model_bias.values,
                                        levels=levels, transform=ccrs.PlateCarree(),
                                        cmap=cmap_bias, extend='both', alpha=0.8)
                        return im
                else:
                    # 非风场使用填色等高线图
                    # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                    levels = ticker.MaxNLocator(nbins=11, prune=None).tick_values(bias_vmin, bias_vmax)
                    im = ax.contourf(model_lon_centers, model_lat_centers, model_bias.values,
                                    levels=levels, transform=ccrs.PlateCarree(),
                                    cmap=cmap_bias, extend='both', alpha=0.8)
                    return im
            
            # 计算边界
            def _compute_edges(center_coords: np.ndarray) -> np.ndarray:
                center_coords = np.asarray(center_coords)
                diffs = np.diff(center_coords)
                first_edge = center_coords[0] - diffs[0] / 2.0 if diffs.size > 0 else center_coords[0] - 0.5
                last_edge = center_coords[-1] + diffs[-1] / 2.0 if diffs.size > 0 else center_coords[-1] + 0.5
                mid_edges = center_coords[:-1] + diffs / 2.0 if diffs.size > 0 else np.array([])
                return np.concatenate([[first_edge], mid_edges, [last_edge]])
            
            # 判断是否为风场（wind变量或u、v变量）
            is_wind_field = var_name.startswith('wind_') or var_name.startswith('u_') or var_name.startswith('v_')
            # 判断是否为GHT（位势高度）
            is_ght = var_name.startswith('ght_')
            
            # 对于风场和GHT，创建1度网格进行插值
            if (is_wind_field and obs_clim_component is not None) or is_ght:
                # 获取观测数据的空间范围
                lon_min = float(obs_clim.lon.min())
                lon_max = float(obs_clim.lon.max())
                lat_min = float(obs_clim.lat.min())
                lat_max = float(obs_clim.lat.max())
                
                # 创建1度网格
                lon_1deg = np.arange(np.floor(lon_min), np.ceil(lon_max) + 1, 1.0)
                lat_1deg = np.arange(np.floor(lat_min), np.ceil(lat_max) + 1, 1.0)
                
                # 将观测数据插值到1度网格
                obs_clim_1deg = obs_clim.interp(lon=lon_1deg, lat=lat_1deg, method='linear')
                
                # 对于风场，还需要插值分量
                if is_wind_field and obs_clim_component is not None:
                    obs_clim_component_1deg = obs_clim_component.interp(lon=lon_1deg, lat=lat_1deg, method='linear')
                    obs_clim_component = obs_clim_component_1deg
                    logger.info(f"观测风场已插值到1度网格: {len(lat_1deg)} x {len(lon_1deg)}")
                else:
                    logger.info(f"观测GHT已插值到1度网格: {len(lat_1deg)} x {len(lon_1deg)}")
                
                # 使用1度网格的坐标
                lon_centers = lon_1deg
                lat_centers = lat_1deg
                # 更新obs_clim为插值后的数据
                obs_clim = obs_clim_1deg
            else:
                lon_centers = obs_clim.lon.values
                lat_centers = obs_clim.lat.values
            
            lon_edges = _compute_edges(lon_centers)
            lat_edges = _compute_edges(lat_centers)
            
            # 创建图形（4行4列）
            n_rows = 4
            n_cols = 4
            fig_width = n_cols * 4.5
            
            # 计算物理纵横比
            lon_span = float(lon_edges[-1] - lon_edges[0])
            lat_span = float(lat_edges[-1] - lat_edges[0])
            mid_lat = float((lat_edges[0] + lat_edges[-1]) / 2.0)
            cos_mid = np.cos(np.deg2rad(mid_lat)) if lon_span != 0 else 1.0
            phys_aspect = (lat_span / max(lon_span * max(cos_mid, 1e-6), 1e-6))
            
            left_margin = 0.05
            right_margin = 0.95
            top_margin = 0.98
            bottom_margin = 0.06
            inner_width_frac = right_margin - left_margin
            inner_height_frac = top_margin - bottom_margin
            
            fig_height = fig_width * (inner_width_frac / inner_height_frac) * (n_rows / n_cols) * phys_aspect
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # GridSpec：无间距
            gs = GridSpec(n_rows, n_cols, figure=fig,
                         height_ratios=[1] * n_rows,
                         width_ratios=[1] * n_cols,
                         hspace=-0.45, wspace=0,
                         left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)
            
            # 预计算经纬度主刻度
            lon_tick_start = int(np.ceil(lon_edges[0] / 15.0) * 15)
            lon_tick_end = int(np.floor(lon_edges[-1] / 15.0) * 15)
            lon_ticks = np.arange(lon_tick_start, lon_tick_end + 1, 15)
            lat_tick_start = int(np.ceil(lat_edges[0] / 10.0) * 10)
            lat_tick_end = int(np.floor(lat_edges[-1] / 10.0) * 10)
            if lat_tick_end < lat_edges[-1] - 1e-6:
                lat_tick_end += 10
            lat_ticks = np.arange(lat_tick_start, lat_tick_end + 1, 10)
            lon_formatter = LongitudeFormatter(number_format='.0f')
            lat_formatter = LatitudeFormatter(number_format='.0f')
            
            # 小幅扩展每个Axes的位置
            def _expand_axes_vertically(ax, is_first_row: bool, is_last_row: bool, expand_frac: float = 0.001):
                pos = ax.get_position()
                new_y0 = pos.y0 - (0 if is_first_row else expand_frac)
                new_y1 = pos.y1 + (0 if is_last_row else expand_frac)
                new_y0 = max(new_y0, bottom_margin)
                new_y1 = min(new_y1, top_margin)
                ax.set_position([pos.x0, new_y0, pos.width, new_y1 - new_y0])
            
            im_obs = None
            im_bias = None
            im_clim = None
            content_axes = []
            
            # ===== 第一行第一列：观测气候态 =====
            ax_obs = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            ax_obs.set_extent([lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]], crs=ccrs.PlateCarree())
            ax_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax_obs.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax_obs.add_feature(cfeature.LAND, alpha=0.1)
            ax_obs.add_feature(cfeature.OCEAN, alpha=0.1)
            
            gl = ax_obs.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = FixedLocator(lon_ticks)
            gl.ylocator = FixedLocator(lat_ticks)
            gl.xformatter = lon_formatter
            gl.yformatter = lat_formatter
            gl.bottom_labels = False
            gl.left_labels = True
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            
            if is_wind_field and obs_clim_component is not None:
                # 对于风场，使用矢量箭头图
                # 计算风速大小（用于箭头大小）
                wind_speed = np.sqrt(obs_clim.values**2 + obs_clim_component.values**2)
                
                # 确定u和v分量
                if var_name.startswith('wind_'):
                    # wind变量：obs_clim是u，obs_clim_component是v
                    u_comp = obs_clim.values
                    v_comp = obs_clim_component.values
                elif var_name.startswith('u_'):
                    u_comp = obs_clim.values
                    v_comp = obs_clim_component.values
                else:  # v_
                    u_comp = obs_clim_component.values
                    v_comp = obs_clim.values
                
                # 绘制矢量箭头，箭头长度统一，颜色表示风速大小
                # 每个点都绘制箭头，不进行稀疏化
                lon_sub = lon_centers
                lat_sub = lat_centers
                u_sub = u_comp
                v_sub = v_comp
                speed_sub = wind_speed
                
                # 归一化u和v分量，使所有箭头长度统一（单位向量）
                # 计算每个点的风速大小
                magnitude = np.sqrt(u_sub**2 + v_sub**2)
                # 避免除零：对于magnitude为0或NaN的点，u和v保持为0
                magnitude_safe = np.where(magnitude > 1e-10, magnitude, 1.0)
                u_normalized = np.where(magnitude > 1e-10, u_sub / magnitude_safe, 0.0)
                v_normalized = np.where(magnitude > 1e-10, v_sub / magnitude_safe, 0.0)
                
                # 使用固定的scale值，使箭头长度统一
                # scale值越大，箭头越小。这里使用一个固定值，使箭头长度适中
                scale = 50.0  # 固定scale值，所有箭头长度相同
                
                # 仅调整箭头头部尺寸（线段部分由 scale/width 控制，保持不变）
                headwidth = 2.8
                headlength = 3.6
                headaxislength = 2.8
                
                speed_max = np.nanmax(speed_sub)
                # 使用白色到黑色渐变，高值为黑色
                im_obs = ax_obs.quiver(lon_sub, lat_sub, u_normalized, v_normalized, speed_sub,
                                      transform=ccrs.PlateCarree(),
                                      scale=scale, width=0.002, cmap='gray_r',
                                      clim=(0, speed_max),
                                      headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)
            else:
                # 对于非风场（GHT），使用填色等高线图
                n_levels = _compute_n_levels(obs_clim.values, clim_vmin, clim_vmax)
                # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                levels = ticker.MaxNLocator(nbins=n_levels-1, prune=None).tick_values(clim_vmin, clim_vmax)
                
                im_obs = ax_obs.contourf(lon_centers, lat_centers, obs_clim.values,
                                        levels=levels, transform=ccrs.PlateCarree(),
                                        cmap=cmap_clim, extend='both', alpha=0.8)
            
            ax_obs.text(0.02, 0.98, 'Observation',
                       transform=ax_obs.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top', horizontalalignment='left')
            
            _expand_axes_vertically(ax_obs, True, False)
            content_axes.append(ax_obs)
            
            # 获取lead0和lead3的数据（如果可用）
            lead0_biases = all_leadtime_biases.get(0, {})
            lead3_biases = all_leadtime_biases.get(3, {})
            
            # ===== 第一行2-4列：前三个模式的lead0偏差 =====
            for col_idx in range(3):
                if col_idx >= len(model_names):
                    ax_blank = fig.add_subplot(gs[0, col_idx + 1])
                    ax_blank.axis('off')
                    continue
                
                model = model_names[col_idx]
                if model not in lead0_biases:
                    ax_blank = fig.add_subplot(gs[0, col_idx + 1])
                    ax_blank.axis('off')
                    ax_blank.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                                fontsize=11, fontweight='bold', color='red',
                                transform=ax_blank.transAxes)
                    continue
                
                model_bias = lead0_biases[model]
                display_name = model.replace('-mon', '').replace('mon-', '')
                
                ax_spatial = fig.add_subplot(gs[0, col_idx + 1], projection=ccrs.PlateCarree())
                
                # 基于模型自身网格计算边界
                try:
                    model_lon_centers = model_bias.lon.values
                    model_lat_centers = model_bias.lat.values
                except Exception:
                    model_lon_centers = lon_centers
                    model_lat_centers = lat_centers
                model_lon_edges = _compute_edges(model_lon_centers)
                model_lat_edges = _compute_edges(model_lat_centers)
                
                ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1], 
                                     model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = FixedLocator(lon_ticks)
                gl.ylocator = FixedLocator(lat_ticks)
                gl.xformatter = lon_formatter
                gl.yformatter = lat_formatter
                gl.bottom_labels = False
                gl.left_labels = False
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                
                # 使用辅助函数绘制偏差
                im_bias = _plot_bias(ax_spatial, model_bias, model_lon_centers, model_lat_centers,
                                    model_lon_edges, model_lat_edges, 0, model)
                
                # 模型标签
                label = chr(97 + col_idx)
                ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                               transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                
                # 添加L0标签（在第一个模式图上）
                if col_idx == 0:
                    ax_spatial.text(0.98, 0.98, 'L0',
                                   transform=ax_spatial.transAxes, fontsize=12, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                _expand_axes_vertically(ax_spatial, True, False)
                content_axes.append(ax_spatial)
            
            # ===== 第二行：4个模式的lead0偏差 =====
            for col_idx in range(4):
                model_idx = col_idx + 3
                if model_idx >= len(model_names):
                    ax_blank = fig.add_subplot(gs[1, col_idx])
                    ax_blank.axis('off')
                    continue
                
                model = model_names[model_idx]
                if model not in lead0_biases:
                    ax_blank = fig.add_subplot(gs[1, col_idx])
                    ax_blank.axis('off')
                    ax_blank.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                                fontsize=11, fontweight='bold', color='red',
                                transform=ax_blank.transAxes)
                    continue
                
                model_bias = lead0_biases[model]
                display_name = model.replace('-mon', '').replace('mon-', '')
                
                ax_spatial = fig.add_subplot(gs[1, col_idx], projection=ccrs.PlateCarree())
                
                try:
                    model_lon_centers = model_bias.lon.values
                    model_lat_centers = model_bias.lat.values
                except Exception:
                    model_lon_centers = lon_centers
                    model_lat_centers = lat_centers
                model_lon_edges = _compute_edges(model_lon_centers)
                model_lat_edges = _compute_edges(model_lat_centers)
                
                ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1],
                                     model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = FixedLocator(lon_ticks)
                gl.ylocator = FixedLocator(lat_ticks)
                gl.xformatter = lon_formatter
                gl.yformatter = lat_formatter
                gl.bottom_labels = False
                if col_idx == 0:
                    gl.left_labels = True
                else:
                    gl.left_labels = False
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                
                # 使用辅助函数绘制偏差（L0）
                _plot_bias(ax_spatial, model_bias, model_lon_centers, model_lat_centers,
                          model_lon_edges, model_lat_edges, 0, model)
                
                label = chr(97 + model_idx)
                ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                               transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                
                _expand_axes_vertically(ax_spatial, False, False)
                content_axes.append(ax_spatial)
            
            # ===== 第三行第一列：留空 =====
            ax_blank = fig.add_subplot(gs[2, 0])
            ax_blank.axis('off')
            
            # ===== 第三行2-4列：三个模式的lead3偏差 =====
            for col_idx in range(3):
                if col_idx >= len(model_names):
                    ax_blank = fig.add_subplot(gs[2, col_idx + 1])
                    ax_blank.axis('off')
                    continue
                
                model = model_names[col_idx]
                if model not in lead3_biases:
                    ax_blank = fig.add_subplot(gs[2, col_idx + 1])
                    ax_blank.axis('off')
                    ax_blank.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                                fontsize=11, fontweight='bold', color='red',
                                transform=ax_blank.transAxes)
                    continue
                
                model_bias = lead3_biases[model]
                display_name = model.replace('-mon', '').replace('mon-', '')
                
                ax_spatial = fig.add_subplot(gs[2, col_idx + 1], projection=ccrs.PlateCarree())
                
                try:
                    model_lon_centers = model_bias.lon.values
                    model_lat_centers = model_bias.lat.values
                except Exception:
                    model_lon_centers = lon_centers
                    model_lat_centers = lat_centers
                model_lon_edges = _compute_edges(model_lon_centers)
                model_lat_edges = _compute_edges(model_lat_centers)
                
                ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1],
                                     model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = FixedLocator(lon_ticks)
                gl.ylocator = FixedLocator(lat_ticks)
                gl.xformatter = lon_formatter
                gl.yformatter = lat_formatter
                gl.bottom_labels = False
                gl.left_labels = False
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                
                # 使用辅助函数绘制偏差（L3）
                _plot_bias(ax_spatial, model_bias, model_lon_centers, model_lat_centers,
                          model_lon_edges, model_lat_edges, 3, model)
                
                label = chr(97 + col_idx)
                ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                               transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                
                # 添加L3标签（在第一个模式图上）
                if col_idx == 0:
                    ax_spatial.text(0.98, 0.98, 'L3',
                                   transform=ax_spatial.transAxes, fontsize=12, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                _expand_axes_vertically(ax_spatial, False, False)
                content_axes.append(ax_spatial)
            
            # ===== 第四行：4个模式的lead3偏差 =====
            for col_idx in range(4):
                model_idx = col_idx + 3
                if model_idx >= len(model_names):
                    ax_blank = fig.add_subplot(gs[3, col_idx])
                    ax_blank.axis('off')
                    continue
                
                model = model_names[model_idx]
                if model not in lead3_biases:
                    ax_blank = fig.add_subplot(gs[3, col_idx])
                    ax_blank.axis('off')
                    ax_blank.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                                fontsize=11, fontweight='bold', color='red',
                                transform=ax_blank.transAxes)
                    continue
                
                model_bias = lead3_biases[model]
                display_name = model.replace('-mon', '').replace('mon-', '')
                
                ax_spatial = fig.add_subplot(gs[3, col_idx], projection=ccrs.PlateCarree())
                
                try:
                    model_lon_centers = model_bias.lon.values
                    model_lat_centers = model_bias.lat.values
                except Exception:
                    model_lon_centers = lon_centers
                    model_lat_centers = lat_centers
                model_lon_edges = _compute_edges(model_lon_centers)
                model_lat_edges = _compute_edges(model_lat_centers)
                
                ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1],
                                     model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = FixedLocator(lon_ticks)
                gl.ylocator = FixedLocator(lat_ticks)
                gl.xformatter = lon_formatter
                gl.yformatter = lat_formatter
                gl.bottom_labels = True
                if col_idx == 0:
                    gl.left_labels = True
                else:
                    gl.left_labels = False
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                
                # 使用辅助函数绘制偏差（L3）
                _plot_bias(ax_spatial, model_bias, model_lon_centers, model_lat_centers,
                          model_lon_edges, model_lat_edges, 3, model)
                
                label = chr(97 + model_idx)
                ax_spatial.text(0.02, 0.98, f'({label}) {display_name} Bias',
                               transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                               verticalalignment='top', horizontalalignment='left')
                
                _expand_axes_vertically(ax_spatial, False, True)
                content_axes.append(ax_spatial)
            
            # 去除Cartopy不规则外框，改为每个子图绘制规则矩形边框
            for ax in content_axes:
                try:
                    ax.spines['geo'].set_visible(False)
                    ax.set_frame_on(False)
                    ax.add_patch(Rectangle((0, 0), 1, 1,
                                         transform=ax.transAxes,
                                         fill=False,
                                         edgecolor='black',
                                         linewidth=0.6,
                                         zorder=1000))
                except Exception:
                    pass
            
            # 添加colorbar（在图的底部，横向排列）
            if im_obs is not None:
                cbar_obs_ax = fig.add_axes([0.02, 0.04, 0.48, 0.02])
                cbar_obs = fig.colorbar(im_obs, cax=cbar_obs_ax, orientation='horizontal')
                # 对于风场，显示Wind Speed；对于其他变量，显示原标题
                if is_wind_field:
                    cbar_obs.set_label(f'Wind Speed ({unit})', fontsize=18, labelpad=5)
                else:
                    cbar_obs.set_label(f'{var_title} ({unit})', fontsize=18, labelpad=5)
                cbar_obs.ax.tick_params(labelsize=14)
            
            if im_bias is not None:
                cbar_bias_ax = fig.add_axes([0.5, 0.04, 0.48, 0.02])
                cbar_bias = fig.colorbar(im_bias, cax=cbar_bias_ax, orientation='horizontal')
                # 对于风场，显示Wind Bias Magnitude；对于其他变量，显示原标题
                if is_wind_field:
                    cbar_bias.set_label(f'Wind Bias Magnitude ({unit})', fontsize=18, labelpad=5)
                else:
                    cbar_bias.set_label(f'{var_title} Bias ({unit})', fontsize=18, labelpad=5)
                cbar_bias.ax.tick_params(labelsize=14)
            
            # 保存图像（文件名包含L0_L3表示两个leadtime）
            plots_dir = Path("/sas12t1/ffyan/output/circulation_analysis/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # 确定leadtime字符串（包含实际可用的leadtime）
            leadtime_str = '_'.join([f'L{lt}' for lt in sorted(all_leadtime_biases.keys())])
            
            output_file_png = plots_dir / f"circulation_{var_name}_{leadtime_str}_{season}.png"
            output_file_pdf = plots_dir / f"circulation_{var_name}_{leadtime_str}_{season}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            logger.info(f"图像已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制环流气候态图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run_analysis(self, models: List[str] = None, leadtimes: List[int] = None, 
                     parallel: bool = False, n_jobs: int = None,
                     plot_only: bool = False, save_obs: bool = True,
                     plot_figures: bool = True,
                     era5_single_level_root: str = '/sas12t1/ffyan/ERA5/daily-nc/single-level/'):
        """
        运行环流分析
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
            parallel: 是否并行处理
            n_jobs: 并行作业数
            plot_only: 仅绘图模式
            save_obs: 是否保存观测数据
            plot_figures: 是否绘制图像            era5_single_level_root: ERA5 single-level 数据根目录        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import cpu_count
        
        models = models or MODELS
        leadtimes = leadtimes or LEADTIMES
        
        logger.info(f"开始环流分析: {len(models)} 模型, {len(leadtimes)} 提前期")
        logger.info(f"仅绘图模式: {plot_only}, 保存观测: {save_obs}, 绘制图像: {plot_figures}")
        
        # 步骤B: 处理 CMFD 数据（不依赖环流分析）
        logger.info("\n" + "="*60)
        logger.info("步骤B: 计算/加载 CMFD 异常（与环流分析并行）")
        logger.info("="*60)
        
        if not plot_only:
            # 计算 CMFD 温度异常
            logger.info("计算 CMFD 温度异常...")
            cmfd_temp_anom = self.compute_cmfd_china_anomaly('temp', CLIMATOLOGY_PERIOD)
            
            # 计算 CMFD 降水异常
            logger.info("计算 CMFD 降水异常...")
            cmfd_prec_anom = self.compute_cmfd_china_anomaly('prec', CLIMATOLOGY_PERIOD)
            
            # 保存 CMFD 异常
            if cmfd_temp_anom is not None and cmfd_prec_anom is not None:
                if save_obs:
                    if not self.save_china_temp_prec_anomaly(cmfd_temp_anom, cmfd_prec_anom, 'cmfd'):
                        logger.warning("CMFD 异常数据保存失败")
            else:
                logger.error("CMFD 异常计算失败")
        
        elif plot_only:
            # 仅绘图模式：从文件加载
            logger.info("仅绘图模式：从文件加载 CMFD 异常...")
            try:
                output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
                cmfd_file = output_dir / "circulation_china_temp_prec_anom_cmfd.nc"
                if cmfd_file.exists():
                    ds = xr.open_dataset(cmfd_file)
                    cmfd_temp_anom = ds['temp_anom'].load()
                    cmfd_prec_anom = ds['prec_anom'].load()
                    ds.close()
                    logger.info("CMFD 异常数据加载成功")
                else:
                    logger.warning(f"CMFD 异常文件不存在: {cmfd_file}")
            except Exception as e:
                logger.error(f"加载 CMFD 异常失败: {e}")
        
        obs_climatology = None
        
        # 如果不是仅绘图模式，需要处理观测数据
        if not plot_only:
            # 1. 检查是否已有保存的观测气候态文件
            logger.info("\n" + "="*60)
            logger.info("步骤1: 处理观测数据")
            logger.info("="*60)
            
            # 先尝试从文件加载
            obs_climatology = self._load_obs_climatology_from_files()
            
            # 检查所有必需的文件是否存在
            required_vars = ['u_850', 'v_850', 'ght_850', 'u_500', 'v_500', 'ght_500']
            required_seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
            output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
            
            missing_files = []
            for var_name in required_vars:
                for season in required_seasons:
                    file_path = output_dir / f"circulation_obs_{var_name}_{season}.nc"
                    if not file_path.exists():
                        missing_files.append(f"{var_name}_{season}")
            
            if missing_files:
                # 如果有文件缺失，需要重新计算
                logger.info(f"观测气候态文件缺失 ({len(missing_files)}/{len(required_vars)*len(required_seasons)} 个文件)，开始计算...")
                obs_climatology = self.process_obs_circulation_data()
                
                if obs_climatology is None:
                    logger.error("观测数据处理失败")
                    return
                
                # 保存观测气候态
                if save_obs:
                    logger.info("\n" + "="*60)
                    logger.info("步骤2: 保存观测气候态")
                    logger.info("="*60)
                    
                    if not self.save_obs_climatology(obs_climatology):
                        logger.warning("观测气候态保存失败")
            else:
                # 所有文件都存在，使用已保存的数据
                logger.info("所有观测气候态文件已存在，使用已保存的数据")
            
            # 3. 处理模式数据
            logger.info("\n" + "="*60)
            logger.info("步骤3: 处理模式数据")
            logger.info("="*60)
            
            # 准备任务列表
            tasks = []
            for model in models:
                for leadtime in leadtimes:
                    tasks.append((model, leadtime))
            
            if parallel and n_jobs != 1:
                # 并行处理
                max_workers = min(max(1, cpu_count() // 4), MAX_WORKERS_TEMP, HARD_WORKER_CAP, len(tasks))
                n_jobs = min(n_jobs or max_workers, HARD_WORKER_CAP)
                logger.info(f"使用并行处理: {n_jobs} 进程")
                
                try:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        future_to_task = {
                            executor.submit(self.process_circulation_analysis, model, leadtime, obs_climatology): (model, leadtime)
                            for model, leadtime in tasks
                        }
                        
                        completed = 0
                        failed = 0
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            try:
                                result = future.result(timeout=1800)  # 30分钟超时
                                completed += 1
                                logger.info(f"完成 {completed}/{len(tasks)}: {task}")
                            except Exception as e:
                                failed += 1
                                logger.error(f"任务失败 {task}: {e}")
                        
                        logger.info(f"并行处理完成: {completed} 成功, {failed} 失败")
                        
                except Exception as e:
                    logger.error(f"并行处理失败: {e}")
                    logger.info("回退到串行处理...")
                    parallel = False
            
            if n_jobs == 1 or not parallel:
                # 串行处理
                logger.info("使用串行处理")
                completed = 0
                failed = 0
                for i, (model, leadtime) in enumerate(tasks):
                    try:
                        result = self.process_circulation_analysis(model, leadtime, obs_climatology)
                        if result:
                            completed += 1
                        else:
                            failed += 1
                        logger.info(f"完成 {i+1}/{len(tasks)}: {model} L{leadtime}")
                    except Exception as e:
                        failed += 1
                        logger.error(f"任务失败 {model} L{leadtime}: {e}")
                
                logger.info(f"串行处理完成: {completed} 成功, {failed} 失败")
        
        # 5. 绘制环流图像
        if plot_figures:
            logger.info("\n" + "="*60)
            logger.info("步骤5: 绘制环流图像")
            logger.info("="*60)
            
            # 如果是仅绘图模式，需要从文件加载观测数据
            if plot_only:
                logger.info("仅绘图模式：从文件加载观测气候态...")
                obs_climatology = self._load_obs_climatology_from_files()
                
                # 检查所有必需的文件是否存在
                required_vars = ['u_850', 'v_850', 'ght_850', 'u_500', 'v_500', 'ght_500']
                required_seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
                output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
                
                missing_files = []
                for var_name in required_vars:
                    for season in required_seasons:
                        file_path = output_dir / f"circulation_obs_{var_name}_{season}.nc"
                        if not file_path.exists():
                            missing_files.append(f"{var_name}_{season}")
                
                if missing_files:
                    logger.warning(f"观测气候态文件缺失 ({len(missing_files)}/{len(required_vars)*len(required_seasons)} 个文件): {', '.join(missing_files[:5])}{'...' if len(missing_files) > 5 else ''}")
                    logger.warning("仅绘图模式下，将使用已存在的文件，缺失的数据将无法绘制")
                    # 仅绘图模式下，不自动计算，只使用已存在的文件
                else:
                    logger.info("所有观测气候态文件已存在，使用已保存的数据")
            
            # 为每个变量和季节绘图
            variables = [
                ('wind_850', 'Wind at 850hPa', 'm/s'),
                ('ght_850', 'Geopotential Height at 850hPa', 'm'),
                ('wind_500', 'Wind at 500hPa', 'm/s'),
                ('ght_500', 'Geopotential Height at 500hPa', 'm')
            ]
            
            seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
            
            # 只绘制lead0和lead3的组合图
            target_leadtimes = [0, 3]
            # 检查哪些leadtime可用
            available_leadtimes = [lt for lt in target_leadtimes if lt in leadtimes]
            
            if len(available_leadtimes) < 2:
                logger.warning(f"需要lead0和lead3才能绘制组合图，当前可用: {available_leadtimes}")
                if available_leadtimes:
                    logger.warning(f"将使用单leadtime模式绘图")
            
            logger.info(f"绘图leadtimes: {available_leadtimes}")
            
            # 预先计算每个变量所有季节的统一气候态范围（跳过风场）
            unified_clim_ranges = {}  # {var_name: (vmin, vmax)}
            for var_name, var_title, unit in variables:
                # 跳过风场（wind、u、v），它们不需要统一范围
                if var_name.startswith('wind_') or var_name.startswith('u_') or var_name.startswith('v_'):
                    unified_clim_ranges[var_name] = (None, None)
                    continue
                
                all_obs_clim_values = []
                for season in seasons:
                    if var_name in obs_climatology:
                        if season in obs_climatology[var_name] and obs_climatology[var_name][season] is not None:
                            obs_data = obs_climatology[var_name][season]
                            valid_values = obs_data.values[np.isfinite(obs_data.values)]
                            all_obs_clim_values.extend(valid_values)
                
                # 计算该变量的统一气候态颜色范围
                if len(all_obs_clim_values) > 0:
                    obs_mean = np.mean(all_obs_clim_values)
                    obs_std = np.std(all_obs_clim_values)
                    unified_vmin = obs_mean - 2 * obs_std
                    unified_vmax = obs_mean + 2 * obs_std
                    unified_clim_ranges[var_name] = (unified_vmin, unified_vmax)
                    logger.info(f"{var_name} 统一气候态范围: [{unified_vmin:.2f}, {unified_vmax:.2f}]")
                else:
                    unified_clim_ranges[var_name] = (None, None)
            
            # 绘制各变量各季节的图
            for var_name, var_title, unit in variables:
                # 获取该变量的统一范围
                unified_vmin, unified_vmax = unified_clim_ranges.get(var_name, (None, None))
                
                for season in seasons:
                    try:
                        # 对于wind变量，需要从u和v合成
                        obs_clim = None
                        obs_clim_component = None
                        
                        if var_name.startswith('wind_'):
                            # wind_850需要u_850和v_850，wind_500需要u_500和v_500
                            level = var_name.replace('wind_', '')
                            u_var_name = f'u_{level}'
                            v_var_name = f'v_{level}'
                            
                            # 检查u和v数据是否存在
                            if u_var_name not in obs_climatology or v_var_name not in obs_climatology:
                                logger.warning(f"跳过 {var_name}（u或v数据不存在）")
                                continue
                            if season not in obs_climatology[u_var_name] or obs_climatology[u_var_name][season] is None:
                                logger.warning(f"跳过 {var_name} {season}（u数据不存在）")
                                continue
                            if season not in obs_climatology[v_var_name] or obs_climatology[v_var_name][season] is None:
                                logger.warning(f"跳过 {var_name} {season}（v数据不存在）")
                                continue
                            
                            # 加载u和v数据
                            obs_clim = obs_climatology[u_var_name][season]  # 使用u作为主分量
                            obs_clim_component = obs_climatology[v_var_name][season]  # v作为分量
                        else:
                            # 对于非wind变量（GHT），正常加载
                            if var_name not in obs_climatology:
                                logger.warning(f"跳过 {var_name}（观测数据不存在）")
                                continue
                            if season not in obs_climatology[var_name] or obs_climatology[var_name][season] is None:
                                logger.warning(f"跳过 {var_name} {season}（观测数据不存在）")
                                continue
                            
                            obs_clim = obs_climatology[var_name][season]
                        
                        # 加载两个leadtime的模式偏差
                        all_leadtime_biases = {}  # {leadtime: {model: bias}}
                        all_leadtime_biases_component = {}  # {leadtime: {model: bias_component}}
                        
                        for leadtime in available_leadtimes:
                            model_biases = {}
                            model_biases_component = {}
                            
                            for model in models:
                                # 加载偏差
                                bias_file = Path("/sas12t1/ffyan/output/circulation_analysis/results") / f"circulation_bias_{model}_L{leadtime}.nc"
                                if bias_file.exists():
                                    try:
                                        ds = xr.open_dataset(bias_file)
                                        
                                        if var_name.startswith('wind_'):
                                            # 对于wind变量，加载u和v的偏差
                                            level = var_name.replace('wind_', '')
                                            u_var_key = f"u_{level}_{season}"
                                            v_var_key = f"v_{level}_{season}"
                                            
                                            if u_var_key in ds and v_var_key in ds:
                                                model_biases[model] = ds[u_var_key]  # 使用u作为主分量
                                                model_biases_component[model] = ds[v_var_key]  # v作为分量
                                        else:
                                            # 对于非wind变量（GHT），正常加载
                                            var_key = f"{var_name}_{season}"
                                            if var_key in ds:
                                                model_biases[model] = ds[var_key]
                                        
                                        ds.close()
                                    except Exception as e:
                                        logger.debug(f"加载偏差失败 {model} L{leadtime}: {e}")
                            
                            if model_biases:
                                all_leadtime_biases[leadtime] = model_biases
                            if model_biases_component:
                                all_leadtime_biases_component[leadtime] = model_biases_component
                        
                        if not all_leadtime_biases:
                            logger.warning(f"跳过 {var_name} {season}（无偏差数据）")
                            continue
                        
                        # 绘制组合图（上半部分lead0，下半部分lead3，使用统一的气候态范围）
                        self.plot_circulation_climatology(
                            var_name, var_title, season,
                            obs_clim, all_leadtime_biases, unit,
                            unified_clim_vmin=unified_vmin,
                            unified_clim_vmax=unified_vmax,
                            obs_clim_component=obs_clim_component,
                            all_leadtime_biases_component=all_leadtime_biases_component if all_leadtime_biases_component else None
                        )
                        
                    except Exception as e:
                        logger.error(f"绘图失败 {var_name} {season}: {e}")
                        continue
        
            
            # 准备模式任务列表
            model_tasks = [(m, lt) for m in models for lt in leadtimes]
            model_results = {}
            
            # 处理每个模式的异常（支持并行）
            parallel_failed = False
            if not plot_only:
                if parallel and n_jobs and n_jobs > 1 and len(model_tasks) > 1:
                    # 并行处理模式异常（使用 ThreadPoolExecutor，因为主要是I/O操作）
                    # 限制并行数以避免 netCDF 库的线程安全问题
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from multiprocessing import cpu_count
                    # 对于 netCDF I/O，使用较少的线程数以避免冲突
                    # 限制为 CPU 核心数或 8，取较小值
                    max_workers = min(n_jobs, len(model_tasks), min(cpu_count(), 8))
                    logger.info(f"并行处理模式异常: {max_workers} 线程（限制以避免 netCDF 冲突）")
                    
                    def _compute_model_anomalies_wrapper(args):
                        """包装函数用于并行处理"""
                        model, leadtime = args
                        try:
                            temp_anom = self.compute_model_china_anomaly(
                                model, leadtime, 'temp', CLIMATOLOGY_PERIOD
                            )
                            prec_anom = self.compute_model_china_anomaly(
                                model, leadtime, 'prec', CLIMATOLOGY_PERIOD
                            )
                            return (model, leadtime), (temp_anom, prec_anom)
                        except Exception as e:
                            logger.error(f"处理 {model} L{leadtime} 失败: {e}")
                            return (model, leadtime), (None, None)
                    
                    try:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_task = {
                                executor.submit(_compute_model_anomalies_wrapper, task): task
                                for task in model_tasks
                            }
                            
                            completed = 0
                            for future in as_completed(future_to_task):
                                task = future_to_task[future]
                                try:
                                    key, (temp_anom, prec_anom) = future.result(timeout=1800)
                                    if temp_anom is not None and prec_anom is not None:
                                        model_results[key] = (temp_anom, prec_anom)
                                        
                                        # 保存模式异常
                                        if save_obs:
                                            model_key = f"{key[0]}_L{key[1]}".lower().replace('-', '_')
                                            if not self.save_china_temp_prec_anomaly(
                                                temp_anom, prec_anom, model_key
                                            ):
                                                logger.warning(f"{key[0]} L{key[1]} 异常数据保存失败")
                                    
                                    completed += 1
                                    logger.info(f"完成 {completed}/{len(model_tasks)}: {task}")
                                except Exception as e:
                                    logger.error(f"任务失败 {task}: {e}")
                        
                        logger.info(f"并行处理完成: {completed} 成功")
                    except (Exception, SystemError, OSError) as e:
                        logger.error(f"并行处理失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.info("回退到串行处理...")
                        # 标记为不使用并行，继续使用串行处理
                        parallel_failed = True
                
                # 如果并行失败或未启用并行，使用串行处理
                if parallel_failed or not (parallel and n_jobs and n_jobs > 1 and len(model_tasks) > 1):
                    # 串行处理
                    for model, leadtime in model_tasks:
                        logger.info(f"\n处理 {model} L{leadtime}...")
                        
                        # 计算模式温度异常
                        logger.info(f"计算 {model} L{leadtime} 温度异常...")
                        model_temp_anom = self.compute_model_china_anomaly(
                            model, leadtime, 'temp', CLIMATOLOGY_PERIOD
                        )
                        
                        # 计算模式降水异常
                        logger.info(f"计算 {model} L{leadtime} 降水异常...")
                        model_prec_anom = self.compute_model_china_anomaly(
                            model, leadtime, 'prec', CLIMATOLOGY_PERIOD
                        )
                        
                        # 保存模式异常
                        if model_temp_anom is not None and model_prec_anom is not None:
                            if save_obs:
                                model_key = f"{model}_L{leadtime}".lower().replace('-', '_')
                                if not self.save_china_temp_prec_anomaly(
                                    model_temp_anom, model_prec_anom, model_key
                                ):
                                    logger.warning(f"{model} L{leadtime} 异常数据保存失败")
                            
                            model_results[(model, leadtime)] = (model_temp_anom, model_prec_anom)
                
                elif plot_only:
                    # 仅绘图模式：从文件加载
                    logger.info(f"仅绘图模式：从文件加载 {model} L{leadtime} 异常...")
                    try:
                        output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
                        model_key = f"{model}_L{leadtime}".lower().replace('-', '_')
                        model_file = output_dir / f"circulation_china_temp_prec_anom_{model_key}.nc"
                        if model_file.exists():
                            ds = xr.open_dataset(model_file)
                            model_temp_anom = ds['temp_anom'].load()
                            model_prec_anom = ds['prec_anom'].load()
                            ds.close()
                            logger.info(f"{model} L{leadtime} 异常数据加载成功")
                            model_results[(model, leadtime)] = (model_temp_anom, model_prec_anom)
                        else:
                            logger.debug(f"{model} L{leadtime} 异常文件不存在: {model_file}")
                    except Exception as e:
                        logger.debug(f"加载 {model} L{leadtime} 异常失败: {e}")
            
            # 步骤C2: 计算并保存季节平均异常
            if save_obs and not plot_only:
                logger.info("\n计算季节平均异常...")
                seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
                
                for season in seasons:
                    logger.info(f"处理季节: {season}")
                    
                    # 计算 CMFD 季节平均
                    cmfd_temp_seasonal = None
                    cmfd_prec_seasonal = None
                    if cmfd_temp_anom is not None:
                        cmfd_temp_seasonal = self.compute_seasonal_anomaly_mean(cmfd_temp_anom, season)
                    if cmfd_prec_anom is not None:
                        cmfd_prec_seasonal = self.compute_seasonal_anomaly_mean(cmfd_prec_anom, season)
                    
                    # 计算模式季节平均
                    model_temp_seasonal = {}
                    model_prec_seasonal = {}
                    for (model, leadtime), (temp_anom, prec_anom) in model_results.items():
                        if temp_anom is not None:
                            temp_seas = self.compute_seasonal_anomaly_mean(temp_anom, season)
                            if temp_seas is not None:
                                model_temp_seasonal[(model, leadtime)] = temp_seas
                        if prec_anom is not None:
                            prec_seas = self.compute_seasonal_anomaly_mean(prec_anom, season)
                            if prec_seas is not None:
                                model_prec_seasonal[(model, leadtime)] = prec_seas
                    
                    # 保存季节平均数据
                    self.save_seasonal_anomaly(
                        season=season,
                        cmfd_temp_seasonal=cmfd_temp_seasonal,
                        cmfd_prec_seasonal=cmfd_prec_seasonal,
                        model_temp_seasonal=model_temp_seasonal if model_temp_seasonal else None,
                        model_prec_seasonal=model_prec_seasonal if model_prec_seasonal else None
                    )
            
            # 步骤D: 绘制所有关系图（汇总在一张图中）
            if plot_figures:
                logger.info("\n" + "="*60)
                
                
                # D2: 绘制季节平均关系图
                logger.info("\nD2: 绘制季节平均关系图...")
                seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
                
                for season in seasons:
                    logger.info(f"绘制季节 {season} 关系图...")
                    
                    # 加载或计算季节平均数据
                    cmfd_temp_seasonal = None
                    cmfd_prec_seasonal = None
                    model_temp_seasonal_dict = {}
                    model_prec_seasonal_dict = {}
                    
                    output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
                    
                    # 加载 CMFD 季节平均
                    cmfd_file = output_dir / f"circulation_china_temp_prec_anom_cmfd_{season}.nc"
                    if cmfd_file.exists():
                        try:
                            ds = xr.open_dataset(cmfd_file)
                            cmfd_temp_seasonal = ds['temp_anom'].load()
                            cmfd_prec_seasonal = ds['prec_anom'].load()
                            ds.close()
                        except Exception as e:
                            logger.debug(f"加载 CMFD {season} 失败: {e}")
                    elif cmfd_temp_anom is not None and cmfd_prec_anom is not None and not plot_only:
                        # 临时计算
                        cmfd_temp_seasonal = self.compute_seasonal_anomaly_mean(cmfd_temp_anom, season)
                        cmfd_prec_seasonal = self.compute_seasonal_anomaly_mean(cmfd_prec_anom, season)
                    
                    # 加载模式季节平均
                    for (model, leadtime) in model_results.keys():
                        model_key = f"{model}_L{leadtime}".lower().replace('-', '_')
                        model_file = output_dir / f"circulation_china_temp_prec_anom_{model_key}_{season}.nc"
                        if model_file.exists():
                            try:
                                ds = xr.open_dataset(model_file)
                                if 'temp_anom' in ds:
                                    model_temp_seasonal_dict[(model, leadtime)] = ds['temp_anom'].load()
                                if 'prec_anom' in ds:
                                    model_prec_seasonal_dict[(model, leadtime)] = ds['prec_anom'].load()
                                ds.close()
                            except Exception as e:
                                logger.debug(f"加载 {model} L{leadtime} {season} 失败: {e}")
                        elif not plot_only:
                            # 临时计算
                            if (model, leadtime) in model_results:
                                temp_anom, prec_anom = model_results[(model, leadtime)]
                                if temp_anom is not None:
                                    temp_seas = self.compute_seasonal_anomaly_mean(temp_anom, season)
                                    if temp_seas is not None:
                                        model_temp_seasonal_dict[(model, leadtime)] = temp_seas
                                if prec_anom is not None:
                                    prec_seas = self.compute_seasonal_anomaly_mean(prec_anom, season)
                                    if prec_seas is not None:
                                        model_prec_seasonal_dict[(model, leadtime)] = prec_seas
                    
                    # 绘制季节关系图（如果有足够数据）
                    if cmfd_temp_seasonal is not None or model_temp_seasonal_dict:
                        # 如果有温度数据，可以绘制温度关系图
                        logger.debug(f"季节 {season} 温度数据可用")
                    if cmfd_prec_seasonal is not None or model_prec_seasonal_dict:
                        # 如果有降水数据，可以绘制降水关系图
                        logger.debug(f"季节 {season} 降水数据可用")
                    if not (cmfd_temp_seasonal is not None or model_temp_seasonal_dict or 
                            cmfd_prec_seasonal is not None or model_prec_seasonal_dict):
                        logger.warning(f"季节 {season} 数据不足，跳过绘图")
        
        logger.info("\n" + "="*60)
        logger.info("环流分析完成")
        logger.info("="*60)
    
    def _load_obs_climatology_from_files(self) -> Optional[Dict]:
        """
        从文件加载观测气候态
        
        Returns:
            观测气候态字典 {var_name: {season: DataArray}}
        """
        try:
            output_dir = Path("/sas12t1/ffyan/output/circulation_analysis/results")
            
            obs_climatology = {}
            variables = ['u_850', 'v_850', 'ght_850', 'u_500', 'v_500', 'ght_500']
            seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
            
            for var_name in variables:
                obs_climatology[var_name] = {}
                for season in seasons:
                    file_path = output_dir / f"circulation_obs_{var_name}_{season}.nc"
                    if file_path.exists():
                        try:
                            da = xr.open_dataarray(file_path)
                            obs_climatology[var_name][season] = da.load()
                            logger.debug(f"加载观测数据: {var_name} {season}")
                        except Exception as e:
                            logger.debug(f"加载观测数据失败 {var_name} {season}: {e}")
                            obs_climatology[var_name][season] = None
                    else:
                        obs_climatology[var_name][season] = None
            
            logger.info("观测气候态加载完成")
            return obs_climatology
            
        except Exception as e:
            logger.error(f"加载观测气候态失败: {e}")
            return None


def main():
    """主函数"""
    # 环流分析不需要变量参数，使用统一的参数解析系统
    parser = create_parser(
        description="环流分析：计算u、v风场、水汽通量散度和GHT场的气候平均态（基于ERA5观测）",
        var_default=None,  # 环流分析不需要变量参数
        var_required=False
    )
    
    # 添加环流分析特有的参数
    parser.add_argument('--no-save-obs', action='store_true',
                       help='不保存观测气候态到NetCDF文件（默认总是保存）')
    
    # 添加 Nino3.4 相关参数
    
    parser.add_argument('--era5-single-level-root', type=str,
                       default='/sas12t1/ffyan/ERA5/daily-nc/single-level/',
                       help='ERA5 single-level 数据根目录（默认: /sas12t1/ffyan/ERA5/daily-nc/single-level/）')
    
    args = parser.parse_args()
    
    # 解析参数
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    
    # 标准化并行参数
    parallel = normalize_parallel_args(args)
    
    # 确定绘图模式（使用normalize_plot_args处理plot参数）
    from src.utils.cli_args import normalize_plot_args
    normalize_plot_args(args)
    plot_only = args.plot_only if hasattr(args, 'plot_only') else False
    plot_figures = not args.no_plot if hasattr(args, 'no_plot') else True
    save_obs = not args.no_save_obs if hasattr(args, 'no_save_obs') else True  # 默认保存
    
    logger.info("="*60)
    logger.info("开始环流分析（基于ERA5观测）")
    logger.info("="*60)
    logger.info(f"模型: {models}")
    logger.info(f"提前期: {leadtimes}")
    logger.info(f"并行处理: {parallel}")
    logger.info(f"并行作业数: {args.n_jobs}")
    logger.info(f"仅绘图模式: {plot_only}")
    logger.info(f"保存观测: {save_obs}")
    logger.info(f"绘制图像: {plot_figures}")
    
    # 创建分析器（环流分析不需要var_type参数）
    analyzer = CirculationAnalyzer()
    
    # 运行分析
    analyzer.run_analysis(
        models=models,
        leadtimes=leadtimes,
        parallel=parallel,
        n_jobs=args.n_jobs,
        plot_only=plot_only,
        save_obs=save_obs,
        plot_figures=plot_figures,
    )
    
    logger.info("="*60)
    logger.info("所有任务完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()

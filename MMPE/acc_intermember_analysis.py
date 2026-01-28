#!/usr/bin/env python3
"""
ACC与Inter-member Correlation分析脚本

实现Anomaly Correlation Coefficient (ACC)和ensemble成员间相关性的计算与可视化
参考rmse_spread_analysis.py的架构，实现并行计算和组合图可视化

主要功能：
1. 计算ACC：ensemble mean与观测的异常相关系数
2. 计算inter-member correlation：成员间相关性（两种方法）
3. 去除季节气候态后再计算
4. 并行处理多个模型和提前期
5. 生成空间分布+散点组合图
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats

# 统一导入toolkit路径
sys.path.insert(0, str(Path(__file__).parent.parent / "climate_analysis_toolkit"))
from src.utils.data_loader import DataLoader
from src.utils.alignment import align_time_to_monthly, align_spatial_to_obs
from src.utils.data_utils import remove_outliers_iqr, find_valid_data_bounds
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args, normalize_plot_args

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    CLIMATOLOGY_PERIOD,
    SEASONS,
    MAX_WORKERS_TEMP,
    MAX_WORKERS_PREC,
    HARD_WORKER_CAP,
    REMOVE_OUTLIERS,
    OUTLIER_METHOD,
    OUTLIER_THRESHOLD,
    COLORS,
    DEFAULT_TIME_CHUNK,
)

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logging(
    log_file='acc_intermember_analysis.log',
    module_name=__name__
)

# 全局配置（从 common_config 导入）
MODELS = MODEL_LIST



def pearson_r_along_time(a: np.ndarray, b: np.ndarray, remove_outliers: bool = None) -> np.ndarray:
    """
    沿时间维度计算Pearson相关系数，处理NaN值和异常值
    
    Args:
        a, b: 形状为(time, ...)的数组，可以是多维
        remove_outliers: 是否去除异常值，None时使用全局配置
    
    Returns:
        r: 形状为(...)的相关系数数组
    """
    if remove_outliers is None:
        remove_outliers = REMOVE_OUTLIERS
    
    # 如果启用异常值去除
    if remove_outliers:
        # 使用IQR方法去除异常值
        a = remove_outliers_iqr(a, axis=0, threshold=OUTLIER_THRESHOLD)
        b = remove_outliers_iqr(b, axis=0, threshold=OUTLIER_THRESHOLD)
    
    # 检查有效值
    valid = ~(np.isnan(a) | np.isnan(b))
    valid_count = valid.sum(axis=0)
    
    # 至少需要5个有效点才计算相关系数（提高阈值以确保统计可靠性）
    mask = valid_count >= 5
    
    # 计算均值
    a_mean = np.nanmean(a, axis=0)
    b_mean = np.nanmean(b, axis=0)
    
    # 计算偏差
    a_dev = a - a_mean
    b_dev = b - b_mean
    
    # 计算相关系数
    numerator = np.nansum(a_dev * b_dev, axis=0)
    denominator = np.sqrt(
        np.nansum(a_dev**2, axis=0) * np.nansum(b_dev**2, axis=0)
    )
    
    # 避免除零，设置无效值为NaN
    r = np.where(mask & (denominator > 0), numerator / denominator, np.nan)
    return r


def diagnose_nan_causes(a: np.ndarray, b: np.ndarray, threshold: int = 3) -> Dict:
    """
    诊断相关系数计算中NaN产生的原因
    
    Args:
        a, b: 输入数组
        threshold: 最小样本数阈值
    
    Returns:
        诊断结果字典
    """
    valid = ~(np.isnan(a) | np.isnan(b))
    valid_count = valid.sum(axis=0)
    
    # 统计各种情况
    total_points = valid_count.size
    insufficient_samples = np.sum(valid_count < threshold)
    
    # 计算标准差
    a_std = np.nanstd(a, axis=0)
    b_std = np.nanstd(b, axis=0)
    
    zero_variance = np.sum((a_std == 0) | (b_std == 0))
    
    # 计算相关系数
    r = pearson_r_along_time(a, b)
    nan_count = np.sum(np.isnan(r))
    finite_count = np.sum(np.isfinite(r))
    
    return {
        'total_grid_points': int(total_points),
        'insufficient_samples': int(insufficient_samples),
        'zero_variance_points': int(zero_variance),
        'final_nan_count': int(nan_count),
        'final_finite_count': int(finite_count),
        'valid_sample_range': [int(valid_count.min()), int(valid_count.max())] if valid_count.size > 0 else [0, 0]
    }



class ACCAnalyzer:
    """ACC与Inter-member Correlation分析器"""
    
    def __init__(self, var_type: str, data_loader: DataLoader = None):
        """
        初始化ACC分析器
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
            data_loader: 数据加载器，如果为None则自动创建
        """
        self.var_type = var_type
        self.data_loader = data_loader or DataLoader()
        
        # 直接观测数据路径（用于统计计算，避免插值）
        self.obs_direct_path = f"/sas12t1/ffyan/obs/{var_type}_1deg_199301-202012.nc"
        
        logger.info(f"初始化ACC分析器: {var_type}")
    
    def align_time_data(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """统一的时间对齐方法"""
        return align_time_to_monthly(obs_data, fcst_data, min_common_months=12)

    def align_spatial_grid(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """统一的空间网格对齐方法"""
        return align_spatial_to_obs(obs_data, fcst_data)
    
    def load_obs_data_direct(self) -> xr.DataArray:
        """直接加载obs目录中的观测数据，避免DataLoader的额外处理"""
        try:
            import xarray as xr
            ds = xr.open_dataset(self.obs_direct_path)
            # 根据变量类型选择合适的变量名
            var_candidates = ['temp', 'prec', 't2m', 'tprate', 't', 'tp']
            var_name = None
            for candidate in var_candidates:
                if candidate in ds:
                    var_name = candidate
                    break
            
            if var_name is None:
                raise ValueError(f"在 {self.obs_direct_path} 中未找到合适的变量")
            
            obs_data = ds[var_name]
            ds.close()
            
            # 确保时间坐标为datetime格式
            if 'time' in obs_data.coords:
                obs_data = obs_data.resample(time='1MS').mean()
            
            logger.info(f"直接加载观测数据成功: {obs_data.shape}")
            return obs_data
            
        except Exception as e:
            logger.warning(f"直接加载观测数据失败，回退到DataLoader: {e}")
            return self.data_loader.load_obs_data(self.var_type)
    
    
    def compute_monthly_climatology(self, data: xr.DataArray, baseline: str = CLIMATOLOGY_PERIOD) -> xr.DataArray:
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
    
    def remove_climatology(self, data: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
        """
        去除季节气候态得到anomaly，并去除异常值
        
        Args:
            data: 输入数据 (time, lat, lon) 或 (time, number, lat, lon)
            clim: 气候态 (month, lat, lon) 或 (month, number, lat, lon)
        
        Returns:
            anom: 异常值 (time, lat, lon) 或 (time, number, lat, lon)
        """
        try:
            # 按月份减去气候态
            anom = data.groupby('time.month') - clim
            
            # 去除异常值
            logger.info("去除异常值...")
            if 'number' in anom.dims:
                # ensemble数据：对每个成员分别去除异常值
                anom_clean = anom.copy()
                for i in range(anom.number.size):
                    member_data = anom.isel(number=i)
                    member_clean = remove_outliers_iqr(member_data.values, axis=0, threshold=OUTLIER_THRESHOLD)
                    anom_clean[dict(number=i)] = member_clean
                anom = anom_clean
            else:
                # 单变量数据：直接去除异常值
                anom_clean = remove_outliers_iqr(anom.values, axis=0, threshold=OUTLIER_THRESHOLD)
                anom = xr.DataArray(anom_clean, coords=anom.coords, dims=anom.dims)
            
            logger.info(f"去除气候态和异常值完成: {anom.shape}")
            return anom
            
        except Exception as e:
            logger.error(f"去除气候态失败: {e}")
            return None
    
    def calculate_acc(self, ensemble_data: xr.DataArray, obs_data: xr.DataArray) -> xr.DataArray:
        """
        计算Anomaly Correlation Coefficient
        
        Args:
            ensemble_data: ensemble数据 (time, number, lat, lon) - 已去气候态
            obs_data: 观测数据 (time, lat, lon) - 已去气候态
        
        Returns:
            acc: ACC (lat, lon) - 每格点的异常相关系数
        """
        try:
            # 1. 计算ensemble mean
            ens_mean = ensemble_data.mean(dim='number')  # (time, lat, lon)
            
            # 2. 对每个格点计算时间序列相关系数
            logger.info("开始计算ACC...")
            acc = xr.apply_ufunc(
                pearson_r_along_time,
                ens_mean, obs_data,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[[]],
                vectorize=True,
                dask='parallelized'
            )
            
            # 设置属性
            acc.attrs = {
                'long_name': 'Anomaly Correlation Coefficient',
                'description': 'ACC between ensemble mean and observations (after removing climatology)',
                'units': 'dimensionless'
            }
            
            logger.info(f"ACC计算完成，范围: [{float(acc.min()):.3f}, {float(acc.max()):.3f}]")
            return acc
            
        except Exception as e:
            logger.error(f"计算ACC失败: {e}")
            return None
    
    def calculate_annual_anomaly_spatial_correlation(self, ensemble_data: xr.DataArray, obs_data: xr.DataArray) -> Dict[str, Dict[int, float]]:
        """
        计算每个模型每一年的距平空间相关系数（两种方法）
        
        Args:
            ensemble_data: ensemble异常数据 (time, number, lat, lon) - 已去气候态
            obs_data: 观测异常数据 (time, lat, lon) - 已去气候态
        
        Returns:
            Dict[str, Dict[int, float]]: {
                'spatial_mean': {year: spatial_corr},  # 方法1：先时间平均再空间相关
                'temporal_spatial': {year: temporal_spatial_corr}  # 方法2：时间-空间综合
            }
        """
        try:
            logger.info("开始计算年度距平空间相关系数（两种方法）...")
            
            # 1. 计算ensemble mean
            ens_mean = ensemble_data.mean(dim='number')  # (time, lat, lon)
            
            # 2. 按年份分组计算
            annual_corrs_spatial_mean = {}  # 方法1：先时间平均再空间相关
            annual_corrs_temporal_spatial = {}  # 方法2：时间-空间综合
            
            # 获取所有年份
            years = np.unique(ens_mean['time'].dt.year.values)
            
            for year in years:
                try:
                    year_int = int(year)
                    
                    # 提取该年的数据
                    ens_year = ens_mean.sel(time=ens_mean.time.dt.year == year_int)
                    obs_year = obs_data.sel(time=obs_data.time.dt.year == year_int)
                    
                    # 检查数据有效性
                    if ens_year.size == 0 or obs_year.size == 0:
                        logger.debug(f"年份 {year_int}: 数据为空，跳过")
                        annual_corrs_spatial_mean[year_int] = np.nan
                        annual_corrs_temporal_spatial[year_int] = np.nan
                        continue
                    
                    # ===== 方法1：先时间平均再空间相关 =====
                    try:
                        # 对该年的所有时间点取平均，得到年平均场
                        ens_year_mean = ens_year.mean(dim='time', skipna=True)  # (lat, lon)
                        obs_year_mean = obs_year.mean(dim='time', skipna=True)  # (lat, lon)
                        
                        # 展平为一维向量
                        ens_flat = ens_year_mean.values.flatten()
                        obs_flat = obs_year_mean.values.flatten()
                        
                        # 移除NaN值
                        valid_mask = ~(np.isnan(ens_flat) | np.isnan(obs_flat))
                        
                        if np.sum(valid_mask) >= 10:
                            ens_valid = ens_flat[valid_mask]
                            obs_valid = obs_flat[valid_mask]
                            
                            if len(ens_valid) >= 3:
                                corr = np.corrcoef(ens_valid, obs_valid)[0, 1]
                                annual_corrs_spatial_mean[year_int] = float(corr)
                            else:
                                annual_corrs_spatial_mean[year_int] = np.nan
                        else:
                            annual_corrs_spatial_mean[year_int] = np.nan
                    except Exception as e:
                        logger.debug(f"方法1计算失败 {year_int}: {e}")
                        annual_corrs_spatial_mean[year_int] = np.nan
                    
                    # ===== 方法2：时间-空间综合 =====
                    try:
                        # 将该年的所有时间点×所有空间点展平为一维向量
                        ens_year_flat = ens_year.values.flatten()  # 所有时间-空间点
                        obs_year_flat = obs_year.values.flatten()  # 所有时间-空间点
                        
                        # 移除NaN值
                        valid_mask = ~(np.isnan(ens_year_flat) | np.isnan(obs_year_flat))
                        
                        if np.sum(valid_mask) >= 10:
                            ens_valid = ens_year_flat[valid_mask]
                            obs_valid = obs_year_flat[valid_mask]
                            
                            if len(ens_valid) >= 3:
                                corr = np.corrcoef(ens_valid, obs_valid)[0, 1]
                                annual_corrs_temporal_spatial[year_int] = float(corr)
                            else:
                                annual_corrs_temporal_spatial[year_int] = np.nan
                        else:
                            annual_corrs_temporal_spatial[year_int] = np.nan
                    except Exception as e:
                        logger.debug(f"方法2计算失败 {year_int}: {e}")
                        annual_corrs_temporal_spatial[year_int] = np.nan
                    
                    # 输出日志
                    corr1 = annual_corrs_spatial_mean.get(year_int, np.nan)
                    corr2 = annual_corrs_temporal_spatial.get(year_int, np.nan)
                    if np.isfinite(corr1) or np.isfinite(corr2):
                        logger.info(f"年份 {year_int}: 方法1(先平均)={corr1:.4f}, 方法2(时间-空间)={corr2:.4f}")
                        
                except Exception as e:
                    logger.debug(f"计算年份 {year} 的距平空间相关系数失败: {e}")
                    annual_corrs_spatial_mean[int(year)] = np.nan
                    annual_corrs_temporal_spatial[int(year)] = np.nan
            
            logger.info(f"年度距平空间相关系数计算完成，共 {len(years)} 年")
            return {
                'spatial_mean': annual_corrs_spatial_mean,  # 方法1
                'temporal_spatial': annual_corrs_temporal_spatial  # 方法2
            }
            
        except Exception as e:
            logger.error(f"计算年度距平空间相关系数失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'spatial_mean': {}, 'temporal_spatial': {}}
    
    def calculate_monthly_annual_anomaly_spatial_correlation(self, ensemble_data: xr.DataArray, obs_data: xr.DataArray) -> Dict[int, Dict[int, float]]:
        """
        计算每个模型每个（年份，月份）组合的距平空间相关系数（方法1：先时间平均再空间相关）
        
        对于每个（年份，月份）组合：
        1. 提取该年该月的数据（单个月份，时间维度为1）
        2. 得到空间场 (lat, lon)
        3. 将空间场展平为一维向量
        4. 计算ensemble mean和观测的空间相关系数
        
        Args:
            ensemble_data: ensemble异常数据 (time, number, lat, lon) - 已去气候态
            obs_data: 观测异常数据 (time, lat, lon) - 已去气候态
        
        Returns:
            Dict[int, Dict[int, float]]: {year: {month: acc_value}}
        """
        try:
            logger.info("开始计算逐月+逐年距平空间相关系数（方法1：先时间平均再空间相关）...")
            
            # 1. 计算ensemble mean
            ens_mean = ensemble_data.mean(dim='number')  # (time, lat, lon)
            
            # 2. 按（年份，月份）组合计算
            monthly_annual_corrs = {}  # {year: {month: corr}}
            
            # 获取所有年份
            years = np.unique(ens_mean['time'].dt.year.values)
            
            for year in years:
                try:
                    year_int = int(year)
                    monthly_annual_corrs[year_int] = {}
                    
                    # 提取该年的数据
                    ens_year = ens_mean.sel(time=ens_mean.time.dt.year == year_int)
                    obs_year = obs_data.sel(time=obs_data.time.dt.year == year_int)
                    
                    # 检查数据有效性
                    if ens_year.size == 0 or obs_year.size == 0:
                        logger.debug(f"年份 {year_int}: 数据为空，跳过")
                        for month in range(1, 13):
                            monthly_annual_corrs[year_int][month] = np.nan
                        continue
                    
                    # 按月份分组计算
                    for month in range(1, 13):
                        try:
                            # 提取该年该月的数据
                            ens_month = ens_year.sel(time=ens_year.time.dt.month == month)
                            obs_month = obs_year.sel(time=obs_year.time.dt.month == month)
                            
                            # 检查数据有效性
                            if ens_month.size == 0 or obs_month.size == 0:
                                monthly_annual_corrs[year_int][month] = np.nan
                                continue
                            
                            # 方法1：先时间平均再空间相关
                            # 对于单个月份，时间维度只有1个点，直接压缩时间维度得到空间场
                            if 'time' in ens_month.dims:
                                # 压缩时间维度，得到空间场
                                ens_month_spatial = ens_month.isel(time=0)  # (lat, lon)
                                obs_month_spatial = obs_month.isel(time=0)  # (lat, lon)
                            else:
                                # 没有时间维度，直接使用
                                ens_month_spatial = ens_month  # (lat, lon)
                                obs_month_spatial = obs_month  # (lat, lon)
                            
                            # 将空间场展平为一维向量
                            ens_flat = ens_month_spatial.values.flatten()
                            obs_flat = obs_month_spatial.values.flatten()
                            
                            # 移除NaN值
                            valid_mask = ~(np.isnan(ens_flat) | np.isnan(obs_flat))
                            
                            if np.sum(valid_mask) >= 10:
                                ens_valid = ens_flat[valid_mask]
                                obs_valid = obs_flat[valid_mask]
                                
                                if len(ens_valid) >= 3:
                                    corr = np.corrcoef(ens_valid, obs_valid)[0, 1]
                                    monthly_annual_corrs[year_int][month] = float(corr)
                                else:
                                    monthly_annual_corrs[year_int][month] = np.nan
                            else:
                                monthly_annual_corrs[year_int][month] = np.nan
                                
                        except Exception as e:
                            logger.debug(f"计算 {year_int}年{month}月 失败: {e}")
                            monthly_annual_corrs[year_int][month] = np.nan
                    
                    # 输出日志（仅输出有有效值的月份）
                    valid_months = [m for m in range(1, 13) if np.isfinite(monthly_annual_corrs[year_int].get(m, np.nan))]
                    if valid_months:
                        logger.debug(f"年份 {year_int}: 计算了 {len(valid_months)} 个月的有效数据")
                        
                except Exception as e:
                    logger.debug(f"计算年份 {year} 的逐月距平空间相关系数失败: {e}")
                    monthly_annual_corrs[int(year)] = {m: np.nan for m in range(1, 13)}
            
            logger.info(f"逐月+逐年距平空间相关系数计算完成，共 {len(years)} 年")
            return monthly_annual_corrs
            
        except Exception as e:
            logger.error(f"计算逐月+逐年距平空间相关系数失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def calculate_inter_member_correlation(self, ensemble_data: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        计算inter-member correlation（两种方法）
        
        Args:
            ensemble_data: ensemble数据 (time, number, lat, lon) - 已去气候态
        
        Returns:
            dict with:
            - 'pairwise_mean': 所有成员两两相关的平均值 (lat, lon)
            - 'member_to_mean': 每个成员与ensemble_mean的相关平均值 (lat, lon)
        """
        try:
            logger.info("开始计算inter-member correlation...")
            
            # 检查ensemble数据有效性
            n_valid = np.sum(~np.isnan(ensemble_data.values))
            total_elements = ensemble_data.size
            logger.info(f"Ensemble数据有效性: {n_valid}/{total_elements} ({n_valid/total_elements*100:.1f}%)")
            
            if n_valid == 0:
                logger.error("Ensemble数据全为NaN，无法计算inter-member correlation")
                return None
            
            # 计算 ensemble mean
            ens_mean = ensemble_data.mean(dim='number')  # (time, lat, lon)
            
            # Method A: Mean pairwise correlation
            # 计算所有成员两两之间的相关系数，然后取平均
            members_vals = ensemble_data.values  # (time, number, lat, lon)
            n_members = members_vals.shape[1]  # number维度是第2个维度
            
            logger.info(f"计算{self.var_type} {n_members}个成员的pairwise correlation...")
            
            # 初始化结果数组
            lat_shape = members_vals.shape[2]
            lon_shape = members_vals.shape[3]
            pairwise_sum = np.zeros((lat_shape, lon_shape))
            pair_count = 0
            
            # 计算所有两两组合
            for i in range(n_members):
                for j in range(i+1, n_members):
                    # 提取第i和第j个成员的数据 (time, lat, lon)
                    member_i = members_vals[:, i, :, :]  # (time, lat, lon)
                    member_j = members_vals[:, j, :, :]  # (time, lat, lon)
                    
                    # 检查形状匹配
                    if member_i.shape != member_j.shape:
                        logger.warning(f"Pairwise形状不匹配: member[{i}] {member_i.shape} vs member[{j}] {member_j.shape}")
                        continue
                    
                    r_ij = pearson_r_along_time(member_i, member_j)
                    # 直接累加，保留NaN值用于后续处理
                    pairwise_sum += r_ij
                    pair_count += 1
            
            # 计算平均值，正确处理NaN值
            if pair_count > 0:
                # 对于每个格点，只考虑非NaN的值进行平均
                pairwise_mean_r = np.full((lat_shape, lon_shape), np.nan)
                for i in range(lat_shape):
                    for j in range(lon_shape):
                        # 这里需要重新计算每个格点的平均值，考虑NaN值
                        valid_corrs = []
                        for member_i in range(n_members):
                            for member_j in range(member_i + 1, n_members):
                                member_i_data = members_vals[:, member_i, i, j]
                                member_j_data = members_vals[:, member_j, i, j]
                                if member_i_data.shape == member_j_data.shape:
                                    r_ij = pearson_r_along_time(member_i_data, member_j_data)
                                    if np.isfinite(r_ij):
                                        valid_corrs.append(r_ij)
                        if len(valid_corrs) > 0:
                            pairwise_mean_r[i, j] = np.mean(valid_corrs)
            else:
                pairwise_mean_r = np.full((lat_shape, lon_shape), np.nan)
            pairwise_da = xr.DataArray(
                pairwise_mean_r, 
                coords=[ensemble_data.lat, ensemble_data.lon], 
                dims=['lat', 'lon']
            )
            pairwise_da.attrs = {
                'long_name': 'Inter-member Pairwise Mean Correlation',
                'description': 'Mean of all pairwise correlations between ensemble members',
                'units': 'dimensionless'
            }
            
            # Method B: Member-to-ensemble-mean correlation
            # 计算每个成员与ensemble_mean的相关，然后取平均
            logger.info(f"计算{self.var_type} member-to-ensemble-mean correlation...")
            
            # 确保ens_mean和members_vals的时间维度一致
            logger.info(f"ens_mean shape: {ens_mean.shape}")
            logger.info(f"members_vals shape: {members_vals.shape}")
            
            r_sum = np.zeros((lat_shape, lon_shape))
            for i in range(n_members):
                # 提取第i个成员的数据 (time, lat, lon)
                member_data = members_vals[:, i, :, :]  # (time, lat, lon)
                ens_mean_data = ens_mean.values  # (time, lat, lon)
                
                # 检查形状是否匹配
                if member_data.shape != ens_mean_data.shape:
                    logger.warning(f"形状不匹配: member[{i}] {member_data.shape} vs ens_mean {ens_mean_data.shape}")
                    # 尝试重新对齐
                    min_time = min(member_data.shape[0], ens_mean_data.shape[0])
                    member_data = member_data[:min_time]
                    ens_mean_data = ens_mean_data[:min_time]
                
                r_i = pearson_r_along_time(member_data, ens_mean_data)
                # 直接累加，保留NaN值用于后续处理
                r_sum += r_i
            
            # 计算平均值，正确处理NaN值
            if n_members > 0:
                r_mean_to_ens = np.full((lat_shape, lon_shape), np.nan)
                for i in range(lat_shape):
                    for j in range(lon_shape):
                        valid_corrs = []
                        for member_idx in range(n_members):
                            member_data = members_vals[:, member_idx, i, j]
                            ens_mean_data = ens_mean.values[:, i, j]
                            min_time = min(len(member_data), len(ens_mean_data))
                            if min_time > 0:
                                r_i = pearson_r_along_time(
                                    member_data[:min_time], 
                                    ens_mean_data[:min_time]
                                )
                                if np.isfinite(r_i):
                                    valid_corrs.append(r_i)
                        if len(valid_corrs) > 0:
                            r_mean_to_ens[i, j] = np.mean(valid_corrs)
            else:
                r_mean_to_ens = np.full((lat_shape, lon_shape), np.nan)
            r_mean_da = xr.DataArray(
                r_mean_to_ens,
                coords=[ensemble_data.lat, ensemble_data.lon],
                dims=['lat', 'lon']
            )
            r_mean_da.attrs = {
                'long_name': 'Inter-member Mean Correlation to Ensemble Mean',
                'description': 'Mean correlation of each member with ensemble mean',
                'units': 'dimensionless'
            }
            
            results = {
                'pairwise_mean': pairwise_da,
                'member_to_mean': r_mean_da
            }
            
            logger.info(f"Inter-member correlation计算完成")
            logger.info(f"  Pairwise mean范围: [{float(pairwise_da.min()):.3f}, {float(pairwise_da.max()):.3f}]")
            logger.info(f"  Member-to-mean范围: [{float(r_mean_da.min()):.3f}, {float(r_mean_da.max()):.3f}]")
            
            # 诊断NaN产生原因
            if n_members >= 2:
                try:
                    # 选择第一个和第二个成员进行诊断
                    member_1 = members_vals[:, 0, :, :]
                    member_2 = members_vals[:, 1, :, :]
                    diag_result = diagnose_nan_causes(member_1, member_2)
                    logger.info(f"  NaN诊断结果:")
                    logger.info(f"    总格点数: {diag_result['total_grid_points']}")
                    logger.info(f"    样本不足(<3): {diag_result['insufficient_samples']}")
                    logger.info(f"    零方差格点: {diag_result['zero_variance_points']}")
                    logger.info(f"    最终NaN数: {diag_result['final_nan_count']}")
                    logger.info(f"    最终有效数: {diag_result['final_finite_count']}")
                    logger.info(f"    有效样本范围: {diag_result['valid_sample_range']}")
                except Exception as e:
                    logger.debug(f"NaN诊断失败: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"计算inter-member correlation失败: {e}")
            return None
    
    def process_model_leadtime(self, model: str, leadtime: int, calc_ic: bool = True) -> Dict:
        """
        处理单个模型和提前期的ACC计算
        
        Args:
            model: 模型名称
            leadtime: 提前期
            calc_ic: 是否计算IC（Inter-member Correlation），默认True
                    如果为False，将跳过IC计算，空间分布图、散点图和柱状图将无法绘制
        """
        try:
            logger.info(f"处理 {model} L{leadtime}")
            
            # 1. 加载观测数据
            obs_data = self.load_obs_data_direct()
            
            # 2. 加载ensemble数据
            ensemble_data = self.data_loader.load_forecast_data_ensemble(
                model,
                self.var_type,
                leadtime
            )
            
            if obs_data is None or ensemble_data is None:
                logger.warning(f"数据加载失败: {model} L{leadtime}")
                return None
            
            # 3. 时间对齐
            obs_aligned, ensemble_aligned = self.align_time_data(obs_data, ensemble_data)
            if obs_aligned is None:
                logger.warning(f"时间对齐失败: {model} L{leadtime}")
                return None
            
            # 4. 空间对齐
            obs_aligned, ensemble_aligned = self.align_spatial_grid(obs_aligned, ensemble_aligned)
            
            # 5. 计算并去除气候态
            logger.info("计算气候态...")
            obs_clim = self.compute_monthly_climatology(obs_aligned)
            ensemble_clim = self.compute_monthly_climatology(ensemble_aligned)
            
            if obs_clim is None or ensemble_clim is None:
                logger.warning(f"气候态计算失败: {model} L{leadtime}")
                return None
            
            logger.info("去除气候态...")
            obs_anom = self.remove_climatology(obs_aligned, obs_clim)
            ensemble_anom = self.remove_climatology(ensemble_aligned, ensemble_clim)
            
            if obs_anom is None or ensemble_anom is None:
                logger.warning(f"去除气候态失败: {model} L{leadtime}")
                return None
            
            # 6. 计算ACC
            acc = self.calculate_acc(ensemble_anom, obs_anom)
            if acc is None:
                logger.warning(f"ACC计算失败: {model} L{leadtime}")
                return None
            
            # 7. 计算inter-member correlation（可选）
            inter_corr_results = None
            if calc_ic:
                inter_corr_results = self.calculate_inter_member_correlation(ensemble_anom)
                if inter_corr_results is None:
                    logger.warning(f"Inter-member correlation计算失败: {model} L{leadtime}")
                    return None
            
            # 8. 计算年度距平空间相关系数（两种方法）
            annual_spatial_corr = self.calculate_annual_anomaly_spatial_correlation(ensemble_anom, obs_anom)
            if not annual_spatial_corr or not annual_spatial_corr.get('spatial_mean'):
                logger.warning(f"年度距平空间相关系数计算失败: {model} L{leadtime}")
                # 继续执行，不返回None
            
            # 8.5. 计算逐月+逐年距平空间相关系数（方法1：先时间平均再空间相关）
            monthly_annual_corr = self.calculate_monthly_annual_anomaly_spatial_correlation(ensemble_anom, obs_anom)
            if not monthly_annual_corr:
                logger.warning(f"逐月+逐年距平空间相关系数计算失败或返回空: {model} L{leadtime}")
                # 继续执行，不返回None
            else:
                logger.info(f"逐月+逐年距平空间相关系数计算成功: {model} L{leadtime}, 共 {len(monthly_annual_corr)} 年")
            
            # 9. 保存NetCDF
            results = {}
            try:
                output_dir = Path("/sas12t1/ffyan/output/acc_analysis/results")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建输出数据集
                output_ds_dict = {'ACC': acc}
                if inter_corr_results is not None:
                    output_ds_dict['inter_member_pairwise'] = inter_corr_results['pairwise_mean']
                    output_ds_dict['inter_member_to_mean'] = inter_corr_results['member_to_mean']
                
                output_ds = xr.Dataset(output_ds_dict)
                
                # 添加年度距平空间相关系数（两种方法）
                if annual_spatial_corr:
                    # 方法1：先时间平均再空间相关（默认用于绘图）
                    if annual_spatial_corr.get('spatial_mean'):
                        corr_spatial_mean = annual_spatial_corr['spatial_mean']
                        years = sorted([y for y in corr_spatial_mean.keys() if np.isfinite(corr_spatial_mean[y])])
                        if years:
                            corr_values = [corr_spatial_mean[y] for y in years]
                            annual_corr_da = xr.DataArray(
                                corr_values,
                                coords={'year': years},
                                dims=['year'],
                                attrs={
                                    'long_name': 'Annual Anomaly Spatial Correlation (Method 1: Spatial Mean)',
                                    'description': 'Spatial correlation coefficient between ensemble mean and observations for each year. Method: time-averaged annual fields first, then spatial correlation.',
                                    'units': 'dimensionless',
                                    'method': 'spatial_mean'
                                }
                            )
                            output_ds['annual_anomaly_spatial_corr'] = annual_corr_da
                            logger.info(f"年度距平空间相关系数（方法1）已添加到输出数据集，共 {len(years)} 年")
                    
                    # 方法2：时间-空间综合
                    if annual_spatial_corr.get('temporal_spatial'):
                        corr_temporal_spatial = annual_spatial_corr['temporal_spatial']
                        years2 = sorted([y for y in corr_temporal_spatial.keys() if np.isfinite(corr_temporal_spatial[y])])
                        if years2:
                            corr_values2 = [corr_temporal_spatial[y] for y in years2]
                            annual_corr_da2 = xr.DataArray(
                                corr_values2,
                                coords={'year': years2},
                                dims=['year'],
                                attrs={
                                    'long_name': 'Annual Anomaly Spatial Correlation (Method 2: Temporal-Spatial)',
                                    'description': 'Temporal-spatial correlation coefficient between ensemble mean and observations for each year. Method: all time-space points flattened, then correlation.',
                                    'units': 'dimensionless',
                                    'method': 'temporal_spatial'
                                }
                            )
                            output_ds['annual_anomaly_spatial_corr_temporal'] = annual_corr_da2
                            logger.info(f"年度距平空间相关系数（方法2）已添加到输出数据集，共 {len(years2)} 年")
                
                # 添加逐月+逐年距平空间相关系数（方法2）
                if monthly_annual_corr:
                    # 收集所有年份和月份
                    all_years = sorted([y for y in monthly_annual_corr.keys()])
                    all_months = list(range(1, 13))
                    
                    # 创建矩阵 (year, month)
                    corr_matrix = np.full((len(all_years), len(all_months)), np.nan)
                    for yi, year in enumerate(all_years):
                        for mi, month in enumerate(all_months):
                            if year in monthly_annual_corr and month in monthly_annual_corr[year]:
                                corr_matrix[yi, mi] = monthly_annual_corr[year][month]
                    
                    # 创建DataArray
                    monthly_annual_corr_da = xr.DataArray(
                        corr_matrix,
                        coords={
                            'year': all_years,
                            'month': all_months
                        },
                        dims=['year', 'month'],
                        attrs={
                            'long_name': 'Monthly-Annual Anomaly Spatial Correlation (Method 1: Spatial Mean)',
                            'description': 'Spatial correlation coefficient between ensemble mean and observations for each (year, month) combination. Method: spatial field flattened, then correlation.',
                            'units': 'dimensionless',
                            'method': 'spatial_mean'
                        }
                    )
                    output_ds['monthly_annual_anomaly_spatial_corr'] = monthly_annual_corr_da
                    logger.info(f"逐月+逐年距平空间相关系数（方法1）已添加到输出数据集，共 {len(all_years)} 年 × 12 月")
                
                # 添加全局属性
                output_ds.attrs.update({
                    'model_name': model,
                    'leadtime': leadtime,
                    'variable': self.var_type,
                    'climatology_period': CLIMATOLOGY_PERIOD,
                    'description': 'ACC and inter-member correlation analysis (anomalies computed by monthly climatology removed)',
                    'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # 保存文件
                output_file = output_dir / f"acc_{model}_L{leadtime}_{self.var_type}.nc"
                output_ds.to_netcdf(output_file)
                
                results['output_file'] = output_file
                logger.info(f"结果已保存: {output_file}")
                
                # 额外保存逐月+逐年数据到单独文件（用于等高线图）
                if monthly_annual_corr:
                    try:
                        monthly_output_file = output_dir / f"acc_monthly_{model}_L{leadtime}_{self.var_type}.nc"
                        monthly_ds = xr.Dataset({'monthly_annual_anomaly_spatial_corr': monthly_annual_corr_da})
                        monthly_ds.attrs.update({
                            'model_name': model,
                            'leadtime': leadtime,
                            'variable': self.var_type,
                            'climatology_period': CLIMATOLOGY_PERIOD,
                            'description': 'Monthly-annual anomaly spatial correlation (Method 1: Spatial Mean)',
                            'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        monthly_ds.to_netcdf(monthly_output_file)
                        logger.info(f"逐月+逐年ACC数据已保存: {monthly_output_file}")
                    except Exception as e:
                        logger.warning(f"保存逐月+逐年ACC数据失败: {e}")
                
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
                return None
            
            return results
            
        except Exception as e:
            logger.error(f"处理 {model} L{leadtime} 失败: {e}")
            return None
    
    def run_analysis(self, models: List[str] = None, leadtimes: List[int] = None, 
                     parallel: bool = False, n_jobs: int = None, calc_ic: bool = True):
        """
        运行ACC分析
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
            parallel: 是否并行处理
            n_jobs: 并行作业数
            calc_ic: 是否计算IC（Inter-member Correlation），默认True
                    如果为False，将跳过IC计算，空间分布图、散点图和柱状图将无法绘制
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        ic_status = "启用" if calc_ic else "禁用"
        logger.info(f"开始ACC分析: {len(models)} 模型, {len(leadtimes)} 提前期, IC计算: {ic_status}")
        
        # 准备任务列表
        tasks = []
        for model in models:
            for leadtime in leadtimes:
                tasks.append((model, leadtime))
        
        if parallel and n_jobs != 1:
            # 并行处理，针对prec数据减少并行度以避免内存问题
            if self.var_type == 'prec':
                # 降水数据使用更少的并行进程
                max_workers = min(max(1, cpu_count() // 8), MAX_WORKERS_PREC, HARD_WORKER_CAP, len(tasks))
            else:
                max_workers = min(max(1, cpu_count() // 4), MAX_WORKERS_TEMP, HARD_WORKER_CAP, len(tasks))
            
            n_jobs = min(n_jobs or max_workers, HARD_WORKER_CAP)
            logger.info(f"使用并行处理: {n_jobs} 进程 (数据类型: {self.var_type})")
            
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_task = {
                        executor.submit(self.process_model_leadtime, model, leadtime, calc_ic): (model, leadtime)
                        for model, leadtime in tasks
                    }
                    
                    completed = 0
                    failed = 0
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result(timeout=900)  # 15分钟超时
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
                    result = self.process_model_leadtime(model, leadtime, calc_ic)
                    if result:
                        completed += 1
                    else:
                        failed += 1
                    logger.info(f"完成 {i+1}/{len(tasks)}: {model} L{leadtime}")
                except Exception as e:
                    failed += 1
                    logger.error(f"任务失败 {model} L{leadtime}: {e}")
            
            logger.info(f"串行处理完成: {completed} 成功, {failed} 失败")
        
        logger.info("ACC分析完成")
    
    def run_acc_leadtime_analysis(self, models: List[str] = None, leadtimes: List[int] = None):
        """
        聚合所有模型和leadtime的年度距平空间相关系数（两种方法），保存为metrics文件
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        logger.info(f"开始聚合年度距平空间相关系数（两种方法）: {len(models)} 模型, {len(leadtimes)} leadtime")
        
        try:
            output_dir = Path("/sas12t1/ffyan/output/acc_analysis/results")
            
            # 收集所有数据（两种方法）
            all_years = set()
            data_dict_spatial_mean = {}  # 方法1：{(model, leadtime): {year: corr}}
            data_dict_temporal_spatial = {}  # 方法2：{(model, leadtime): {year: corr}}
            
            for model in models:
                for leadtime in leadtimes:
                    acc_file = output_dir / f"acc_{model}_L{leadtime}_{self.var_type}.nc"
                    
                    if not acc_file.exists():
                        logger.debug(f"文件不存在: {acc_file}")
                        continue
                    
                    try:
                        with xr.open_dataset(acc_file) as ds:
                            # 方法1：先时间平均再空间相关
                            if 'annual_anomaly_spatial_corr' in ds:
                                annual_corr = ds['annual_anomaly_spatial_corr']
                                years = annual_corr['year'].values
                                corr_values = annual_corr.values
                                
                                data_dict_spatial_mean[(model, leadtime)] = {
                                    int(y): float(c) for y, c in zip(years, corr_values)
                                }
                                all_years.update(years)
                                
                                logger.debug(f"加载 {model} L{leadtime} 方法1: {len(years)} 年")
                            
                            # 方法2：时间-空间综合
                            if 'annual_anomaly_spatial_corr_temporal' in ds:
                                annual_corr = ds['annual_anomaly_spatial_corr_temporal']
                                years = annual_corr['year'].values
                                corr_values = annual_corr.values
                                
                                data_dict_temporal_spatial[(model, leadtime)] = {
                                    int(y): float(c) for y, c in zip(years, corr_values)
                                }
                                all_years.update(years)
                                
                                logger.debug(f"加载 {model} L{leadtime} 方法2: {len(years)} 年")
                    except Exception as e:
                        logger.warning(f"读取文件失败 {acc_file}: {e}")
                        continue
            
            if not data_dict_spatial_mean and not data_dict_temporal_spatial:
                logger.warning("没有找到任何年度距平空间相关系数数据")
                return
            
            # 创建矩阵: (model, lead, year)
            all_years_sorted = sorted(list(all_years))
            n_models = len(models)
            n_leads = len(leadtimes)
            n_years = len(all_years_sorted)
            
            # ===== 方法1：先时间平均再空间相关 =====
            if data_dict_spatial_mean:
                corr_matrix_spatial_mean = np.full((n_models, n_leads, n_years), np.nan)
                
                for mi, model in enumerate(models):
                    for li, leadtime in enumerate(leadtimes):
                        key = (model, leadtime)
                        if key in data_dict_spatial_mean:
                            for yi, year in enumerate(all_years_sorted):
                                if year in data_dict_spatial_mean[key]:
                                    corr_matrix_spatial_mean[mi, li, yi] = data_dict_spatial_mean[key][year]
                
                # 创建DataArray
                acc_annual_da_spatial_mean = xr.DataArray(
                    corr_matrix_spatial_mean,
                    coords={
                        'model': models,
                        'lead': leadtimes,
                        'year': all_years_sorted
                    },
                    dims=['model', 'lead', 'year'],
                    attrs={
                        'long_name': 'Annual Anomaly Spatial Correlation (Method 1: Spatial Mean)',
                        'description': 'Spatial correlation coefficient between ensemble mean and observations for each year. Method: time-averaged annual fields first, then spatial correlation.',
                        'units': 'dimensionless',
                        'method': 'spatial_mean'
                    }
                )
                
                # 保存到文件
                metrics_file = output_dir / f"metrics_acc_annual_spatial_corr_{self.var_type}.nc"
                acc_annual_da_spatial_mean.to_netcdf(metrics_file)
                logger.info(f"年度距平空间相关系数metrics已保存（方法1）: {metrics_file}")
                logger.info(f"数据维度: {acc_annual_da_spatial_mean.shape} (model={n_models}, lead={n_leads}, year={n_years})")
                
                # 计算并保存年度平均（跨年份）
                acc_mean_da_spatial_mean = acc_annual_da_spatial_mean.mean(dim='year', skipna=True)
                acc_mean_da_spatial_mean.attrs = {
                    'long_name': 'Mean Annual Anomaly Spatial Correlation (Method 1: Spatial Mean)',
                    'description': 'Mean spatial correlation coefficient across all years. Method: time-averaged annual fields first, then spatial correlation.',
                    'units': 'dimensionless',
                    'method': 'spatial_mean'
                }
                
                metrics_mean_file = output_dir / f"metrics_acc_mean_spatial_corr_{self.var_type}.nc"
                acc_mean_da_spatial_mean.to_netcdf(metrics_mean_file)
                logger.info(f"年度平均距平空间相关系数已保存（方法1）: {metrics_mean_file}")
            
            # ===== 方法2：时间-空间综合 =====
            if data_dict_temporal_spatial:
                corr_matrix_temporal_spatial = np.full((n_models, n_leads, n_years), np.nan)
                
                for mi, model in enumerate(models):
                    for li, leadtime in enumerate(leadtimes):
                        key = (model, leadtime)
                        if key in data_dict_temporal_spatial:
                            for yi, year in enumerate(all_years_sorted):
                                if year in data_dict_temporal_spatial[key]:
                                    corr_matrix_temporal_spatial[mi, li, yi] = data_dict_temporal_spatial[key][year]
                
                # 创建DataArray
                acc_annual_da_temporal_spatial = xr.DataArray(
                    corr_matrix_temporal_spatial,
                    coords={
                        'model': models,
                        'lead': leadtimes,
                        'year': all_years_sorted
                    },
                    dims=['model', 'lead', 'year'],
                    attrs={
                        'long_name': 'Annual Anomaly Spatial Correlation (Method 2: Temporal-Spatial)',
                        'description': 'Temporal-spatial correlation coefficient between ensemble mean and observations for each year. Method: all time-space points flattened, then correlation.',
                        'units': 'dimensionless',
                        'method': 'temporal_spatial'
                    }
                )
                
                # 保存到文件
                metrics_file_temporal = output_dir / f"metrics_acc_annual_spatial_corr_temporal_{self.var_type}.nc"
                acc_annual_da_temporal_spatial.to_netcdf(metrics_file_temporal)
                logger.info(f"年度距平空间相关系数metrics已保存（方法2）: {metrics_file_temporal}")
                logger.info(f"数据维度: {acc_annual_da_temporal_spatial.shape} (model={n_models}, lead={n_leads}, year={n_years})")
                
                # 计算并保存年度平均（跨年份）
                acc_mean_da_temporal_spatial = acc_annual_da_temporal_spatial.mean(dim='year', skipna=True)
                acc_mean_da_temporal_spatial.attrs = {
                    'long_name': 'Mean Annual Anomaly Spatial Correlation (Method 2: Temporal-Spatial)',
                    'description': 'Mean temporal-spatial correlation coefficient across all years. Method: all time-space points flattened, then correlation.',
                    'units': 'dimensionless',
                    'method': 'temporal_spatial'
                }
                
                metrics_mean_file_temporal = output_dir / f"metrics_acc_mean_spatial_corr_temporal_{self.var_type}.nc"
                acc_mean_da_temporal_spatial.to_netcdf(metrics_mean_file_temporal)
                logger.info(f"年度平均距平空间相关系数已保存（方法2）: {metrics_mean_file_temporal}")
            
        except Exception as e:
            logger.error(f"聚合年度距平空间相关系数失败: {e}")
            import traceback
            logger.error(traceback.format_exc())


class ACCPlotter:
    """ACC可视化类"""
    
    def __init__(self, var_type: str):
        """
        初始化ACCPlotter
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
        """
        self.var_type = var_type
        self.acc_data_dir = Path("/sas12t1/ffyan/output/acc_analysis/results")
        self.output_dir = Path("/sas12t1/ffyan/output/acc_analysis/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化ACCPlotter: {var_type}")
    
    # ===== 以下函数已注释，被拆分为三个独立函数：plot_acc_spatial_distribution, plot_acc_scatter, plot_acc_ic_bar =====
    def plot_combined_spatial_scatter(self, leadtime: int, models: List[str]):
        """
        绘制空间分布+散点组合图
        每个模型包含上方的ACC空间分布图和下方的散点图
        空白位置显示图例和统计表格
        
        Args:
            leadtime: 提前期
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制ACC空间分布+散点组合图: L{leadtime} {self.var_type}")
            
            # 收集所有模型的数据
            all_models_data = {}
            
            for model in models:
                acc_file = self.acc_data_dir / f"acc_{model}_L{leadtime}_{self.var_type}.nc"
                
                if not acc_file.exists():
                    logger.debug(f"{model}: ACC文件不存在，跳过")
                    continue
                
                try:
                    # 加载ACC数据
                    acc_ds = xr.open_dataset(acc_file)
                    
                    # 获取变量（可能有不同的变量名）
                    var_candidates = ['ACC', 'acc', '__xarray_dataarray_variable__']
                    acc_data = None
                    for var in var_candidates:
                        if var in acc_ds:
                            acc_data = acc_ds[var]
                            break
                    if acc_data is None:
                        data_vars = [v for v in acc_ds.data_vars]
                        if data_vars:
                            acc_data = acc_ds[data_vars[0]]
                        else:
                            acc_ds.close()
                            continue
                    
                    # 获取inter-member pairwise数据
                    ic_candidates = ['inter_member_pairwise', 'inter_member_pairwise_mean', '__xarray_dataarray_variable__']
                    ic_data = None
                    for var in ic_candidates:
                        if var in acc_ds:
                            ic_data = acc_ds[var]
                            break
                    if ic_data is None:
                        # 尝试其他可能的变量名
                        for var in acc_ds.data_vars:
                            if 'pairwise' in var.lower() or 'inter' in var.lower():
                                ic_data = acc_ds[var]
                                break
                        if ic_data is None:
                            acc_ds.close()
                            continue
                    
                    # 验证网格一致性
                    if acc_data.shape != ic_data.shape:
                        logger.warning(f"{model}: 网格不匹配，跳过")
                        acc_ds.close()
                        continue
                    
                    all_models_data[model] = {
                        'ACC': acc_data,
                        'inter_member': ic_data
                    }
                    
                    acc_ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model}数据失败: {e}")
                    continue
            
            if not all_models_data:
                logger.warning(f"没有有效数据用于组合图 L{leadtime}")
                return
            
            n_models = len(all_models_data)
            logger.info(f"准备绘制 {n_models} 个模型的组合图")
            
            # 计算布局
            cols = 3
            rows_of_models = (n_models + 2) // 3
            
            # 创建图形，画布为正方形
            fig_width = 6 * cols
            fig_height = fig_width  # 正方形，1:1的纵横比
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 创建GridSpec：每个模型占2行（空间图3份高度，散点图1份高度）
            n_grid_rows = rows_of_models * 2
            height_ratios = [3, 1] * rows_of_models
            gs = GridSpec(n_grid_rows, cols, figure=fig,
                         height_ratios=height_ratios,
                         hspace=0.18, wspace=0.13,
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # 计算统一范围 - ACC/inter-member correlation ratio
            all_ratio_values = []
            for model, data in all_models_data.items():
                # 计算ratio并保留所有有限值
                ratio = data['ACC'] / data['inter_member']
                ratio = ratio.where(np.isfinite(ratio))
                valid_values = ratio.values[~np.isnan(ratio.values)]
                all_ratio_values.extend(valid_values)
            
            if all_ratio_values:
                # 使用更保守的方法：只去除真正的异常值（使用IQR方法）
                q1 = np.percentile(all_ratio_values, 25)
                q3 = np.percentile(all_ratio_values, 75)
                iqr = q3 - q1
                
                # 使用1.5倍IQR作为异常值阈值（比3倍IQR更保守）
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # 过滤异常值，但保留大部分数据
                all_ratio_array = np.array(all_ratio_values)
                filtered_values = all_ratio_array[(all_ratio_array >= lower_bound) & (all_ratio_array <= upper_bound)]
                
                if len(filtered_values) > 0:
                    data_min = np.min(filtered_values)
                    data_max = np.max(filtered_values)
                else:
                    # 如果过滤后没有数据，使用原始范围
                    data_min = np.min(all_ratio_values)
                    data_max = np.max(all_ratio_values)
                
                # 使用实际数据的取整范围
                ratio_min = np.floor(data_min * 10) / 10
                ratio_max = np.ceil(data_max * 10) / 10
                
                # 确保对称范围用于RdBu colormap
                max_abs = max(abs(ratio_min), abs(ratio_max))
                acc_min, acc_max = -max_abs, max_abs
                
                logger.info(f"原始数据范围: [{np.min(all_ratio_values):.2f}, {np.max(all_ratio_values):.2f}]")
                logger.info(f"IQR异常值阈值: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logger.info(f"过滤后数据范围: [{data_min:.2f}, {data_max:.2f}]")
                logger.info(f"保留数据点: {len(filtered_values)}/{len(all_ratio_values)} ({len(filtered_values)/len(all_ratio_values)*100:.1f}%)")
            else:
                acc_min, acc_max = -1.0, 1.0
            
            # 散点图坐标范围
            all_acc_vals = []
            all_ic_vals = []
            for model, data in all_models_data.items():
                acc_flat = data['ACC'].values.flatten()
                ic_flat = data['inter_member'].values.flatten()
                valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                all_acc_vals.extend(acc_flat[valid_mask])
                all_ic_vals.extend(ic_flat[valid_mask])
            
            if all_acc_vals and all_ic_vals:
                # Y轴固定为0-1范围
                scatter_acc_min = 0.0
                scatter_acc_max = 1.0
                
                # X轴使用IQR方法去除真正的异常值
                ic_q1 = np.percentile(all_ic_vals, 25)
                ic_q3 = np.percentile(all_ic_vals, 75)
                ic_iqr = ic_q3 - ic_q1
                
                # 使用1.5倍IQR作为异常值阈值
                ic_lower_bound = ic_q1 - 1.5 * ic_iqr
                ic_upper_bound = ic_q3 + 1.5 * ic_iqr
                
                # 过滤异常值
                all_ic_array = np.array(all_ic_vals)
                ic_filtered = all_ic_array[(all_ic_array >= ic_lower_bound) & (all_ic_array <= ic_upper_bound)]
                
                if len(ic_filtered) > 0:
                    scatter_ic_min = np.min(ic_filtered)
                    scatter_ic_max = np.max(ic_filtered)
                else:
                    # 如果过滤后没有数据，使用原始范围
                    scatter_ic_min = np.min(all_ic_vals)
                    scatter_ic_max = np.max(all_ic_vals)
                
                # 添加适当边距
                ic_range = scatter_ic_max - scatter_ic_min
                scatter_ic_min = scatter_ic_min - ic_range * 0.05
                scatter_ic_max = scatter_ic_max + ic_range * 0.05
                
                logger.info(f"散点图IC原始范围: [{np.min(all_ic_vals):.3f}, {np.max(all_ic_vals):.3f}]")
                logger.info(f"散点图IC IQR异常值阈值: [{ic_lower_bound:.3f}, {ic_upper_bound:.3f}]")
                logger.info(f"散点图IC过滤后范围: [{scatter_ic_min:.3f}, {scatter_ic_max:.3f}]")
                logger.info(f"散点图IC保留数据点: {len(ic_filtered)}/{len(all_ic_vals)} ({len(ic_filtered)/len(all_ic_vals)*100:.1f}%)")
            else:
                scatter_acc_min, scatter_acc_max = 0.0, 1.0
                scatter_ic_min, scatter_ic_max = 0.0, 1.0
            
            logger.info(f"ACC/IC Ratio范围: [{acc_min:.1f}, {acc_max:.1f}]")
            logger.info(f"散点ACC范围: [{scatter_acc_min:.2f}, {scatter_acc_max:.2f}]")
            logger.info(f"散点IC范围: [{scatter_ic_min:.2f}, {scatter_ic_max:.2f}]")
            
            cmap = 'RdBu_r'  # 红色到蓝色渐变（高值红色，低值蓝色）
            
            # 用于统一colorbar
            im_for_cbar = None
            scatter_for_cbar = None
            
            # 用于统计表格
            stats_data = []
            
            # 预计算所有模型的散点密度范围（用于统一colorbar）
            all_densities = []
            for model, data in all_models_data.items():
                acc_flat = data['ACC'].values.flatten()
                ic_flat = data['inter_member'].values.flatten()
                valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                acc_valid = acc_flat[valid_mask]
                ic_valid = ic_flat[valid_mask]
                
                if len(acc_valid) > 100:  # 足够的点才计算密度
                    try:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([ic_valid, acc_valid])
                        kde = gaussian_kde(xy)
                        z = kde(xy)
                        all_densities.extend(z)
                    except:
                        pass
            
            # 计算统一的密度范围，归一化到[0,1]
            if all_densities:
                raw_density_min = np.percentile(all_densities, 5)
                raw_density_max = np.percentile(all_densities, 95)
                density_min = 0.0
                density_max = 1.0
                
                logger.info(f"原始密度范围: [{raw_density_min:.2e}, {raw_density_max:.2e}]")
                logger.info(f"归一化密度范围: [0.0, 1.0]")
            else:
                density_min, density_max = 0.0, 1.0
                raw_density_min, raw_density_max = 0, 1
            
            # 绘制每个模型的组合子图
            for i, (model, data) in enumerate(all_models_data.items()):
                row_pair = i // cols
                col = i % cols
                
                # ===== 上半部分：空间分布图 =====
                ax_spatial = fig.add_subplot(gs[row_pair*2, col], projection=ccrs.PlateCarree())
                
                # 找到有效数据边界
                lat_min, lat_max, lon_min, lon_max = find_valid_data_bounds(
                    data['ACC'].values, data['ACC'].lat.values, data['ACC'].lon.values
                )
                
                # 设置地图范围（使用有效数据边界）
                ax_spatial.set_extent([lon_min-1, lon_max+1, lat_min-1, lat_max+1], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                
                # 绘制经纬度网格
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = True
                gl.bottom_labels = True
                
                # 计算ACC/inter-member correlation ratio
                acc_inter_ratio = data['ACC'] / data['inter_member']
                # 只过滤无穷大值，保留所有有限数值（包括极端值）
                acc_inter_ratio = acc_inter_ratio.where(np.isfinite(acc_inter_ratio))
                
                # 空间图保留极端值，让它们用超出范围的颜色显示
                
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
                
                # 绘制ACC/inter-member correlation ratio
                n_levels = _compute_n_levels(acc_inter_ratio.values, acc_min, acc_max)
                levels = np.linspace(acc_min, acc_max, n_levels)
                
                im = ax_spatial.contour(data['ACC'].lon, data['ACC'].lat, acc_inter_ratio,
                                       levels=levels, transform=ccrs.PlateCarree(),
                                       cmap=cmap, linewidths=1.2, alpha=0.8)
                
                if im_for_cbar is None:
                    im_for_cbar = im
                
                # 模型名称（在子图左上角显示，带序号）
                display_name = model.replace('-mon', '').replace('mon-', '')
                subplot_label = chr(97 + i)  # a, b, c, d...
                ax_spatial.text(0.02, 0.98, f'({subplot_label}) {display_name}', 
                               transform=ax_spatial.transAxes,
                               fontsize=11, fontweight='bold', color='black',
                               verticalalignment='top', horizontalalignment='left')
                
                # ===== 下半部分：散点图 =====
                ax_scatter = fig.add_subplot(gs[row_pair*2+1, col])
                
                # 准备散点数据
                acc_flat = data['ACC'].values.flatten()
                ic_flat = data['inter_member'].values.flatten()
                valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                acc_valid = acc_flat[valid_mask]
                ic_valid = ic_flat[valid_mask]
                
                if len(acc_valid) > 0:
                    # 散点图：过滤极端值，不参与绘制和线性拟合
                    if len(acc_valid) > 10:  # 有足够的数据点才进行异常值检测
                        # 使用IQR方法检测散点图中的异常值
                        ic_q1, ic_q3 = np.percentile(ic_valid, [25, 75])
                        ic_iqr = ic_q3 - ic_q1
                        ic_lower = ic_q1 - 1.5 * ic_iqr
                        ic_upper = ic_q3 + 1.5 * ic_iqr
                        
                        acc_q1, acc_q3 = np.percentile(acc_valid, [25, 75])
                        acc_iqr = acc_q3 - acc_q1
                        acc_lower = acc_q1 - 1.5 * acc_iqr
                        acc_upper = acc_q3 + 1.5 * acc_iqr
                        
                        # 过滤异常值（散点图不显示极端值）
                        outlier_mask = (
                            (ic_valid >= ic_lower) & (ic_valid <= ic_upper) &
                            (acc_valid >= acc_lower) & (acc_valid <= acc_upper)
                        )
                        
                        if outlier_mask.sum() > 5:  # 确保过滤后仍有足够的数据点
                            ic_valid = ic_valid[outlier_mask]
                            acc_valid = acc_valid[outlier_mask]
                            logger.debug(f"{model}: 散点图过滤极端值，保留 {len(acc_valid)}/{len(outlier_mask)} 个点")
                    
                    # 绘制散点（使用密度着色）
                    if len(acc_valid) > 100:
                        try:
                            from scipy.stats import gaussian_kde
                            xy = np.vstack([ic_valid, acc_valid])
                            kde = gaussian_kde(xy)
                            z_raw = kde(xy)
                            
                            # 归一化密度值到[0,1]范围
                            z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                            z = np.clip(z, 0, 1)
                            
                            # 绘制密度着色的散点
                            scatter = ax_scatter.scatter(ic_valid, acc_valid, c=z, 
                                                        s=8, alpha=0.6,
                                                        cmap='viridis', 
                                                        vmin=density_min, vmax=density_max,
                                                        edgecolors='none')
                            
                            if scatter_for_cbar is None:
                                scatter_for_cbar = scatter
                        except:
                            # 如果密度计算失败，使用简单散点
                            ax_scatter.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                              color='steelblue', edgecolors='none')
                    else:
                        # 点太少，使用简单散点
                        ax_scatter.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                    
                    # 移除y=0参考线，因为Y轴范围固定为0-1
                    
                    # ACC=Inter-member correlation理想线（y=x）
                    ideal_x = np.array([scatter_ic_min, scatter_ic_max])
                    ideal_y = ideal_x  # y = x
                    ax_scatter.plot(ideal_x, ideal_y, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='ACC=IC')
                    
                    # 拟合线
                    if len(ic_valid) > 2:
                        # 1. 全部数据的线性拟合
                        slope_all, intercept_all, _, _, _ = stats.linregress(ic_valid, acc_valid)
                        r_all = np.corrcoef(ic_valid, acc_valid)[0, 1]
                        
                        x_fit = np.array([scatter_ic_min, scatter_ic_max])
                        y_fit_all = slope_all * x_fit + intercept_all
                        ax_scatter.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7, label='All data fit')
                        
                        # 2. 去除离群值的线性拟合
                        # 计算残差
                        residuals = acc_valid - (slope_all * ic_valid + intercept_all)
                        # 使用IQR方法去除离群值
                        q1, q3 = np.percentile(residuals, [25, 75])
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # 过滤离群值
                        outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
                        ic_robust = ic_valid[outlier_mask]
                        acc_robust = acc_valid[outlier_mask]
                        
                        if len(ic_robust) > 2:
                            slope_robust, intercept_robust, _, _, _ = stats.linregress(ic_robust, acc_robust)
                            r_robust = np.corrcoef(ic_robust, acc_robust)[0, 1]
                            
                            y_fit_robust = slope_robust * x_fit + intercept_robust
                            ax_scatter.plot(x_fit, y_fit_robust, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='Robust fit')
                            
                            # 收集统计数据
                            stats_data.append({
                                'Model': display_name,
                                'N_all': len(acc_valid),
                                'N_robust': len(acc_robust),
                                'R²_all': f'{r_all**2:.3f}',
                                'R²_robust': f'{r_robust**2:.3f}',
                                'Slope_all': f'{slope_all:.3f}',
                                'Slope_robust': f'{slope_robust:.3f}',
                                'ACC_mean': f'{np.mean(acc_valid):.3f}',
                                'IC_mean': f'{np.mean(ic_valid):.3f}'
                            })
                        else:
                            # 如果没有足够的robust数据，只记录全部数据统计
                            stats_data.append({
                                'Model': display_name,
                                'N_all': len(acc_valid),
                                'N_robust': 0,
                                'R²_all': f'{r_all**2:.3f}',
                                'R²_robust': 'N/A',
                                'Slope_all': f'{slope_all:.3f}',
                                'Slope_robust': 'N/A',
                                'ACC_mean': f'{np.mean(acc_valid):.3f}',
                                'IC_mean': f'{np.mean(ic_valid):.3f}'
                            })
                
                # 设置坐标范围
                ax_scatter.set_xlim(scatter_ic_min, scatter_ic_max)
                ax_scatter.set_ylim(scatter_acc_min, scatter_acc_max)
                
                # 轴标签
                ax_scatter.set_xlabel('Inter-member Correlation', fontsize=9)
                ax_scatter.set_ylabel('ACC', fontsize=9)
                
                ax_scatter.grid(True, alpha=0.3, linestyle='--')
                ax_scatter.set_axisbelow(True)
            
            # ===== 保存统计数据为CSV =====
            if stats_data:
                import pandas as pd
                df = pd.DataFrame(stats_data)
                csv_file = self.output_dir / f"acc_statistics_L{leadtime}_{self.var_type}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"统计数据已保存到: {csv_file}")
            
            # ===== 空白位置：ACC/IC Ratio柱状图 =====
            if n_models % 3 != 0:  # 有空白位置
                legend_row = rows_of_models - 1
                bar_ax = fig.add_subplot(gs[legend_row*2:legend_row*2+2, 1])
                
                # 计算每个模型的平均ACC/IC ratio
                model_labels = []
                model_ratios = []
                for idx, (model, data) in enumerate(all_models_data.items()):
                    acc_values = data['ACC'].values
                    ic_values = data['inter_member'].values
                    
                    # 计算ACC/IC ratio，保留所有有限值
                    ratio = acc_values / ic_values
                    # 只过滤NaN和无穷大值，保留所有有限数值
                    valid_ratios = ratio[np.isfinite(ratio)]
                    
                    if len(valid_ratios) > 0:
                        avg_ratio = np.mean(valid_ratios)
                        subplot_label = chr(97 + idx)
                        model_labels.append(subplot_label)
                        model_ratios.append(avg_ratio)
                
                # 绘制柱状图
                if model_labels:  # 确保有数据才绘制
                    x_pos = np.arange(len(model_labels))
                    bars = bar_ax.bar(x_pos, model_ratios, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    bar_ax.set_xlabel('Model', fontsize=10)
                    bar_ax.set_ylabel('Mean ACC/IC Ratio', fontsize=10)
                    bar_ax.set_xticks(x_pos)
                    bar_ax.set_xticklabels(model_labels, fontsize=9)
                    bar_ax.tick_params(axis='y', labelsize=9)
                    
                    # 添加y=1参考线（理想情况：ACC=IC）
                    bar_ax.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='ACC=IC')
                    
                    bar_ax.grid(True, axis='y', alpha=0.3, linestyle='--')
                    bar_ax.set_axisbelow(True)
                    
                    # 添加图例
                    bar_ax.legend(fontsize=8, loc='upper right')
            
            # ===== 空白位置：Colorbar+图例 =====
            if n_models % 3 != 0:  # 有空白位置
                legend_row = rows_of_models - 1
                legend_ax = fig.add_subplot(gs[legend_row*2:legend_row*2+2, 2])
                legend_ax.axis('off')
                
                # 添加ACC/IC Ratio Colorbar（上部）
                if im_for_cbar is not None:
                    cbar_acc_ax = legend_ax.inset_axes([0.2, 0.62, 0.6, 0.06])
                    cbar_acc = fig.colorbar(im_for_cbar, cax=cbar_acc_ax, orientation='horizontal')
                    cbar_acc.set_label('ACC/IC Ratio', fontsize=9)
                    cbar_acc.ax.tick_params(labelsize=8)
                
                # 添加Density Colorbar（中部）
                if scatter_for_cbar is not None:
                    cbar_density_ax = legend_ax.inset_axes([0.2, 0.42, 0.6, 0.06])
                    cbar_density = fig.colorbar(scatter_for_cbar, cax=cbar_density_ax, orientation='horizontal')
                    cbar_density.set_label('Point Density', fontsize=9)
                    cbar_density.ax.tick_params(labelsize=8)
                
                # 图例（下部）
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='orange', linestyle=':', linewidth=4, label='ACC=IC'),
                    Line2D([0], [0], color='#ff7f0e', linestyle='-', linewidth=4, label='All data fit'),
                    Line2D([0], [0], color='green', linestyle='--', linewidth=4, label='Robust fit'),
                ]
                
                legend = legend_ax.legend(handles=legend_elements, loc='center',
                                        ncol=1, frameon=True, fontsize=11,
                                        bbox_to_anchor=(0.5, 0.12),
                                        framealpha=0.9, edgecolor='gray')
            
            # 保存图像
            output_file_png = self.output_dir / f"acc_combined_spatial_scatter_L{leadtime}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_combined_spatial_scatter_L{leadtime}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"ACC组合图已保存到: {output_file_png}")
            logger.info(f"ACC矢量图已保存到: {output_file_pdf}")
            
        except Exception as e:
            logger.error(f"绘制ACC组合图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    # ===== 注释结束 =====
    
    def _load_models_data(self, leadtimes: List[int], models: List[str]) -> Dict[int, Dict[str, Dict]]:
        """
        加载多个leadtime的模型数据
        
        Args:
            leadtimes: 提前期列表
            models: 模型列表
            
        Returns:
            {leadtime: {model: {'ACC': acc_data, 'inter_member': ic_data}}}
        """
        all_leadtimes_data = {}
        
        for leadtime in leadtimes:
            leadtime_data = {}
            for model in models:
                acc_file = self.acc_data_dir / f"acc_{model}_L{leadtime}_{self.var_type}.nc"
                
                if not acc_file.exists():
                    logger.debug(f"{model} L{leadtime}: ACC文件不存在，跳过")
                    continue
                
                try:
                    acc_ds = xr.open_dataset(acc_file)
                    
                    # 获取ACC变量
                    var_candidates = ['ACC', 'acc', '__xarray_dataarray_variable__']
                    acc_data = None
                    for var in var_candidates:
                        if var in acc_ds:
                            acc_data = acc_ds[var]
                            break
                    if acc_data is None:
                        data_vars = [v for v in acc_ds.data_vars]
                        if data_vars:
                            acc_data = acc_ds[data_vars[0]]
                        else:
                            acc_ds.close()
                            continue
                    
                    # 获取inter-member pairwise数据
                    ic_candidates = ['inter_member_pairwise', 'inter_member_pairwise_mean', '__xarray_dataarray_variable__']
                    ic_data = None
                    for var in ic_candidates:
                        if var in acc_ds:
                            ic_data = acc_ds[var]
                            break
                    if ic_data is None:
                        for var in acc_ds.data_vars:
                            if 'pairwise' in var.lower() or 'inter' in var.lower():
                                ic_data = acc_ds[var]
                                break
                        if ic_data is None:
                            acc_ds.close()
                            continue
                    
                    # 验证网格一致性
                    if acc_data.shape != ic_data.shape:
                        logger.warning(f"{model} L{leadtime}: 网格不匹配，跳过")
                        acc_ds.close()
                        continue
                    
                    leadtime_data[model] = {
                        'ACC': acc_data,
                        'inter_member': ic_data
                    }
                    acc_ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model} L{leadtime}数据失败: {e}")
                    continue
            
            if leadtime_data:
                all_leadtimes_data[leadtime] = leadtime_data
        
        return all_leadtimes_data
    
    def plot_acc_spatial_distribution(self, leadtimes: List[int], models: List[str]):
        """
        绘制ACC/IC Ratio空间分布图
        分为上下两半（L0和L3），每个lead占2行，第1行留空+3模型，第2行4模型
        子图之间不留空隙，仅在最外围绘制经纬度标签和脊线，最下方绘制colorbar
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            models: 模型列表
        """
        try:
            logger.info(f"绘制ACC空间分布图: L{leadtimes} {self.var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_models_data(leadtimes, models)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于空间分布图")
                return
            
            # 准备模型列表，按顺序排列
            first_leadtime = leadtimes[0]
            if first_leadtime not in all_leadtimes_data:
                logger.error(f"第一个leadtime {first_leadtime} 没有数据")
                return
            
            model_names = [m for m in models if m in all_leadtimes_data[first_leadtime]]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 收集所有leadtime的所有ratio数据，用于计算统一的colorbar范围
            all_ratio_values = []
            for leadtime in leadtimes:
                if leadtime in all_leadtimes_data:
                    for model_data in all_leadtimes_data[leadtime].values():
                        ratio = model_data['ACC'] / model_data['inter_member']
                        ratio = ratio.where(np.isfinite(ratio))
                        valid_values = ratio.values[~np.isnan(ratio.values)]
                        all_ratio_values.extend(valid_values)
            
            # 计算统一范围
            if all_ratio_values:
                q1 = np.percentile(all_ratio_values, 25)
                q3 = np.percentile(all_ratio_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                all_ratio_array = np.array(all_ratio_values)
                filtered_values = all_ratio_array[(all_ratio_array >= lower_bound) & (all_ratio_array <= upper_bound)]
                
                if len(filtered_values) > 0:
                    data_min = np.min(filtered_values)
                    data_max = np.max(filtered_values)
                else:
                    data_min = np.min(all_ratio_values)
                    data_max = np.max(all_ratio_values)
                
                ratio_min = np.floor(data_min * 10) / 10
                ratio_max = np.ceil(data_max * 10) / 10
                max_abs = max(abs(ratio_min), abs(ratio_max))
                acc_min, acc_max = -max_abs, max_abs
            else:
                acc_min, acc_max = -1.0, 1.0
            
            logger.info(f"ACC/IC Ratio范围: [{acc_min:.1f}, {acc_max:.1f}]")
            
            cmap = 'RdBu_r'
            
            # 计算布局
            n_leadtimes = len(leadtimes)
            n_cols = 4  # 固定4列：留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 基于第一个模型的数据计算经纬度边界
            first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
            sample_acc = first_model_data['ACC']
            lon_centers = sample_acc.lon.values if hasattr(sample_acc, 'lon') else None
            lat_centers = sample_acc.lat.values if hasattr(sample_acc, 'lat') else None
            
            if lon_centers is None or lat_centers is None:
                logger.error("数据缺少经纬度坐标")
                return
            
            # 计算边界
            def _compute_edges(center_coords: np.ndarray) -> np.ndarray:
                center_coords = np.asarray(center_coords)
                diffs = np.diff(center_coords)
                first_edge = center_coords[0] - diffs[0] / 2.0 if diffs.size > 0 else center_coords[0] - 0.5
                last_edge = center_coords[-1] + diffs[-1] / 2.0 if diffs.size > 0 else center_coords[-1] + 0.5
                mid_edges = center_coords[:-1] + diffs / 2.0 if diffs.size > 0 else np.array([])
                return np.concatenate([[first_edge], mid_edges, [last_edge]])
            
            lon_edges = _compute_edges(lon_centers)
            lat_edges = _compute_edges(lat_centers)
            
            # 计算画布大小（与散点图边界一致）
            fig_width = n_cols * 4.5
            left_margin = 0.06
            right_margin = 0.97
            top_margin = 1
            bottom_margin = 0.07
            inner_width_frac = right_margin - left_margin
            inner_height_frac = top_margin - bottom_margin
            
            lon_span = float(lon_edges[-1] - lon_edges[0])
            lat_span = float(lat_edges[-1] - lat_edges[0])
            mid_lat = float((lat_edges[0] + lat_edges[-1]) / 2.0)
            cos_mid = np.cos(np.deg2rad(mid_lat)) if lon_span != 0 else 1.0
            phys_aspect = (lat_span / max(lon_span * max(cos_mid, 1e-6), 1e-6))
            fig_height = fig_width * (inner_width_frac / inner_height_frac) * (n_rows / n_cols) * phys_aspect
            
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # GridSpec：无间距
            height_ratios = [1] * n_rows
            width_ratios = [1] * n_cols
            gs = GridSpec(n_rows, n_cols, figure=fig,
                          height_ratios=height_ratios,
                          width_ratios=width_ratios,
                          hspace=-0.45, wspace=0,
                          left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)
            
            # 预计算经纬度主刻度
            lon_tick_start = int(np.ceil((lon_edges[0] - 5.0) / 10.0) * 10 + 5)
            lon_tick_end = int(np.floor((lon_edges[-1] - 5.0) / 10.0) * 10 + 5)
            lon_ticks = np.arange(lon_tick_start, lon_tick_end + 1, 10)
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
            
            # 用于保存colorbar的绘图对象
            im_for_cbar = None
            content_axes = []
            
            # 绘制每个leadtime
            for lt_idx, leadtime in enumerate(leadtimes):
                if leadtime not in all_leadtimes_data:
                    continue
                
                model_data_dict = all_leadtimes_data[leadtime]
                row_start = lt_idx * 2
                row_obs = row_start
                row_models2 = row_start + 1
                
                # 第1行：留空 + 3个模型
                # 留空
                ax_blank = fig.add_subplot(gs[row_obs, 0])
                ax_blank.axis('off')
                
                # 三个模型
                for col_idx in range(3):
                    if col_idx >= len(model_names):
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[col_idx]
                    if model not in model_data_dict:
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        continue
                    
                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_obs, col_idx + 1], projection=ccrs.PlateCarree())
                    
                    # 基于模型自身网格计算边界
                    try:
                        model_lon_centers = model_data['ACC'].lon.values
                        model_lat_centers = model_data['ACC'].lat.values
                    except Exception:
                        model_lon_centers = lon_centers
                        model_lat_centers = lat_centers
                    model_lon_edges = _compute_edges(model_lon_centers)
                    model_lat_edges = _compute_edges(model_lat_centers)
                    ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1], model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 只在外围显示坐标轴标签
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlocator = FixedLocator(lon_ticks)
                    gl.ylocator = FixedLocator(lat_ticks)
                    gl.xformatter = lon_formatter
                    gl.yformatter = lat_formatter
                    gl.bottom_labels = False
                    gl.left_labels = False
                    
                    # 计算并绘制ACC/IC ratio
                    acc_inter_ratio = model_data['ACC'] / model_data['inter_member']
                    acc_inter_ratio = acc_inter_ratio.where(np.isfinite(acc_inter_ratio))
                    
                    im = ax_spatial.pcolormesh(model_lon_edges, model_lat_edges, acc_inter_ratio.values,
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmap, vmin=acc_min, vmax=acc_max, shading='flat')
                    
                    if im_for_cbar is None:
                        im_for_cbar = im
                    
                    # 模型标签
                    label = chr(97 + col_idx)
                    ax_spatial.text(0.02, 0.96, f'({label}) {display_name}',
                                   transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='left')
                    
                    # 添加leadtime标签
                    if col_idx == 0:
                        ax_spatial.text(0.98, 0.96, f'L{leadtime}',
                                       transform=ax_spatial.transAxes, fontsize=12, fontweight='bold',
                                       verticalalignment='top', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_spatial.set_position(gs[row_obs, col_idx + 1].get_position(fig))
                    _expand_axes_vertically(ax_spatial, is_first_row=(row_obs == 0), is_last_row=False)
                    content_axes.append(ax_spatial)
                
                # 第2行：4个模型
                for col_idx in range(4):
                    if col_idx + 3 >= len(model_names):
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[col_idx + 3]
                    if model not in model_data_dict:
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_models2, col_idx], projection=ccrs.PlateCarree())
                    
                    try:
                        model_lon_centers = model_data['ACC'].lon.values
                        model_lat_centers = model_data['ACC'].lat.values
                    except Exception:
                        model_lon_centers = lon_centers
                        model_lat_centers = lat_centers
                    model_lon_edges = _compute_edges(model_lon_centers)
                    model_lat_edges = _compute_edges(model_lat_centers)
                    ax_spatial.set_extent([model_lon_edges[0], model_lon_edges[-1], model_lat_edges[0], model_lat_edges[-1]], crs=ccrs.PlateCarree())
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
                    if row_models2 == n_rows - 1:
                        gl.bottom_labels = True
                    else:
                        gl.bottom_labels = False
                    if col_idx == 0:
                        gl.left_labels = True
                    else:
                        gl.left_labels = False
                    
                    acc_inter_ratio = model_data['ACC'] / model_data['inter_member']
                    acc_inter_ratio = acc_inter_ratio.where(np.isfinite(acc_inter_ratio))
                    
                    im = ax_spatial.pcolormesh(model_lon_edges, model_lat_edges, acc_inter_ratio.values,
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmap, vmin=acc_min, vmax=acc_max, shading='flat')
                    
                    # 模型标签
                    label = chr(97 + col_idx + 3)
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name}',
                                   transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                   verticalalignment='top', horizontalalignment='left')
                    
                    # 添加leadtime标签（第2行的第一个子图）
                    if col_idx == 0:
                        ax_spatial.text(0.98, 0.98, f'L{leadtime}',
                                       transform=ax_spatial.transAxes, fontsize=12, fontweight='bold',
                                       verticalalignment='top', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_spatial.set_position(gs[row_models2, col_idx].get_position(fig))
                    _expand_axes_vertically(ax_spatial, is_first_row=False, is_last_row=(row_models2 == n_rows - 1))
                    content_axes.append(ax_spatial)
            
            # 去除Cartopy不规则外框，改为每个子图绘制规则矩形边框
            for ax in content_axes:
                try:
                    ax.spines['geo'].set_visible(False)
                except Exception:
                    pass
                try:
                    ax.set_frame_on(False)
                except Exception:
                    pass
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
            
            # 添加colorbar（在图的底部）
            if im_for_cbar is not None:
                cbar_ax = fig.add_axes([0.3, 0.085, 0.4, 0.02])
                cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='horizontal')
                cbar.set_label('ACC/IC Ratio', fontsize=11, labelpad=5)
                cbar.ax.tick_params(labelsize=9)
            
            # 保存图像
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.output_dir / f"acc_spatial_distribution_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_spatial_distribution_{leadtimes_str}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            logger.info(f"ACC空间分布图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制ACC空间分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_acc_scatter(self, leadtimes: List[int], models: List[str]):
        """
        绘制ACC与Inter-member Correlation的散点图
        分为上下两半（L0和L3），模式排列与空间分布图一致
        子图之间不留空隙，仅在最外围绘制横纵坐标标签和脊线
        所有子图同步横纵坐标，内部不绘制网格
        最下方左右排列绘制图例和密度colorbar

        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            models: 模型列表
        """
        try:
            from scipy.stats import gaussian_kde

            logger.info(f"绘制ACC散点图: L{leadtimes} {self.var_type}")

            # 加载数据
            all_leadtimes_data = self._load_models_data(leadtimes, models)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于散点图")
                return

            # 准备模型列表
            first_leadtime = leadtimes[0]
            if first_leadtime not in all_leadtimes_data:
                logger.error(f"第一个leadtime {first_leadtime} 没有数据")
                return

            model_names = [m for m in models if m in all_leadtimes_data[first_leadtime]]
            n_models = len(model_names)

            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return

            # 收集所有数据，用于计算统一的坐标范围和密度范围
            all_acc_vals = []
            all_ic_vals = []
            all_densities = []

            for leadtime in leadtimes:
                if leadtime in all_leadtimes_data:
                    for model_data in all_leadtimes_data[leadtime].values():
                        acc_flat = model_data['ACC'].values.flatten()
                        ic_flat = model_data['inter_member'].values.flatten()
                        valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                        acc_valid = acc_flat[valid_mask]
                        ic_valid = ic_flat[valid_mask]
                        all_acc_vals.extend(acc_valid)
                        all_ic_vals.extend(ic_valid)

                        if len(acc_valid) > 100:
                            try:
                                xy = np.vstack([ic_valid, acc_valid])
                                kde = gaussian_kde(xy)
                                z = kde(xy)
                                all_densities.extend(z)
                            except:
                                pass

            # 计算统一的坐标范围
            if all_acc_vals and all_ic_vals:
                scatter_acc_min = -0.2
                scatter_acc_max = 0.8

                ic_q1 = np.percentile(all_ic_vals, 25)
                ic_q3 = np.percentile(all_ic_vals, 75)
                ic_iqr = ic_q3 - ic_q1
                ic_lower_bound = ic_q1 - 1.5 * ic_iqr
                ic_upper_bound = ic_q3 + 1.5 * ic_iqr

                all_ic_array = np.array(all_ic_vals)
                ic_filtered = all_ic_array[(all_ic_array >= ic_lower_bound) & (all_ic_array <= ic_upper_bound)]

                if len(ic_filtered) > 0:
                    scatter_ic_min = max(np.min(ic_filtered), -0.2)
                    scatter_ic_max = min(np.max(ic_filtered), 1)
                else:
                    scatter_ic_min = max(np.min(all_ic_vals), -0.2)
                    scatter_ic_max = min(np.max(all_ic_vals), 1)

                ic_range = scatter_ic_max - scatter_ic_min
                scatter_ic_min = scatter_ic_min - ic_range * 0.05
                scatter_ic_max = scatter_ic_max + ic_range * 0.05
            else:
                scatter_acc_min, scatter_acc_max = -0.2, 0.8
                scatter_ic_min, scatter_ic_max = -0.2, 0.8

            # 计算统一的密度范围
            if all_densities:
                raw_density_min = np.percentile(all_densities, 5)
                raw_density_max = np.percentile(all_densities, 95)
                density_min = 0.0
                density_max = 1.0
            else:
                density_min, density_max = 0.0, 1.0
                raw_density_min, raw_density_max = 0, 1

            logger.info(f"散点图IC范围: [{scatter_ic_min:.2f}, {scatter_ic_max:.2f}]")
            logger.info(f"散点图ACC范围: [{scatter_acc_min:.2f}, {scatter_acc_max:.2f}]")

            # 布局：4行×4列（上两行为L0，下两行为L3）
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(4, 4, figure=fig, hspace=0.0, wspace=0.0,
                          left=0.06, right=0.97, top=0.95, bottom=0.12)
            axes_grid = [[None]*4 for _ in range(4)]

            # 用于保存colorbar的绘图对象
            scatter_for_cbar = None
            content_axes = []

            # 绘制每个lead的两行
            for lead_idx, leadtime in enumerate(leadtimes):
                if leadtime not in all_leadtimes_data:
                    continue

                row_start = lead_idx * 2
                model_data_dict = all_leadtimes_data[leadtime]

                # 第一行：留空 + 3个模型
                ax_blank = fig.add_subplot(gs[row_start, 0])
                ax_blank.axis('off')
                axes_grid[row_start][0] = ax_blank

                for col_idx in range(3):
                    model_idx = col_idx
                    if model_idx >= len(model_names):
                        ax = fig.add_subplot(gs[row_start, col_idx+1])
                        ax.axis('off')
                        axes_grid[row_start][col_idx+1] = ax
                        continue

                    model = model_names[model_idx]
                    if model not in model_data_dict:
                        ax = fig.add_subplot(gs[row_start, col_idx+1])
                        ax.axis('off')
                        axes_grid[row_start][col_idx+1] = ax
                        continue

                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')

                    ax = fig.add_subplot(gs[row_start, col_idx+1])
                    axes_grid[row_start][col_idx+1] = ax
                    content_axes.append(ax)

                    # 准备散点数据
                    acc_flat = model_data['ACC'].values.flatten()
                    ic_flat = model_data['inter_member'].values.flatten()
                    valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                    acc_all = acc_flat[valid_mask]  # 保存所有有效数据用于标记超出范围的点
                    ic_all = ic_flat[valid_mask]
                    acc_valid = acc_all.copy()
                    ic_valid = ic_all.copy()

                    if len(acc_valid) > 0:
                        # 识别超出绘制范围的点（在过滤极端值之前）
                        out_x_low = ic_all < scatter_ic_min
                        out_x_high = ic_all > scatter_ic_max
                        out_y_low = acc_all < scatter_acc_min
                        out_y_high = acc_all > scatter_acc_max
                        out_x = out_x_low | out_x_high
                        out_y = out_y_low | out_y_high
                        out_any = out_x | out_y
                        
                        # 标记超出范围的点在坐标轴上
                        if np.any(out_any):
                            # 超出X轴范围的点：在X轴边界上标记，Y坐标保持原值（限制在Y轴范围内）
                            if np.any(out_x_low):
                                x_mark_low = np.full(np.sum(out_x_low), scatter_ic_min)
                                y_mark_low = np.clip(acc_all[out_x_low], scatter_acc_min, scatter_acc_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_x_high):
                                x_mark_high = np.full(np.sum(out_x_high), scatter_ic_max)
                                y_mark_high = np.clip(acc_all[out_x_high], scatter_acc_min, scatter_acc_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            # 超出Y轴范围的点：在Y轴边界上标记，X坐标保持原值（限制在X轴范围内）
                            if np.any(out_y_low):
                                y_mark_low = np.full(np.sum(out_y_low), scatter_acc_min)
                                x_mark_low = np.clip(ic_all[out_y_low], scatter_ic_min, scatter_ic_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_high):
                                y_mark_high = np.full(np.sum(out_y_high), scatter_acc_max)
                                x_mark_high = np.clip(ic_all[out_y_high], scatter_ic_min, scatter_ic_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                        
                        # 过滤极端值
                        if len(acc_valid) > 10:
                            ic_q1, ic_q3 = np.percentile(ic_valid, [25, 75])
                            ic_iqr = ic_q3 - ic_q1
                            ic_lower = ic_q1 - 1.5 * ic_iqr
                            ic_upper = ic_q3 + 1.5 * ic_iqr

                            acc_q1, acc_q3 = np.percentile(acc_valid, [25, 75])
                            acc_iqr = acc_q3 - acc_q1
                            acc_lower = acc_q1 - 1.5 * acc_iqr
                            acc_upper = acc_q3 + 1.5 * acc_iqr

                            outlier_mask = (
                                (ic_valid >= ic_lower) & (ic_valid <= ic_upper) &
                                (acc_valid >= acc_lower) & (acc_valid <= acc_upper)
                            )

                            if outlier_mask.sum() > 5:
                                ic_valid = ic_valid[outlier_mask]
                                acc_valid = acc_valid[outlier_mask]

                        # 绘制散点（使用密度着色）
                        if len(acc_valid) > 100:
                            try:
                                xy = np.vstack([ic_valid, acc_valid])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)

                                scatter = ax.scatter(ic_valid, acc_valid, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')

                                if scatter_for_cbar is None:
                                    scatter_for_cbar = scatter
                            except:
                                ax.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')

                        # ACC=IC参考线（y=x）
                        ideal_x = np.array([scatter_ic_min, scatter_ic_max])
                        ideal_y = ideal_x
                        ax.plot(ideal_x, ideal_y, color='orange', linestyle=':', linewidth=2, alpha=0.8)

                        # 拟合线
                        if len(ic_valid) > 2:
                            slope_all, intercept_all, _, _, _ = stats.linregress(ic_valid, acc_valid)
                            x_fit = np.array([scatter_ic_min, scatter_ic_max])
                            y_fit_all = slope_all * x_fit + intercept_all
                            ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)

                            # Robust fit
                            residuals = acc_valid - (slope_all * ic_valid + intercept_all)
                            q1, q3 = np.percentile(residuals, [25, 75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
                            ic_robust = ic_valid[outlier_mask]
                            acc_robust = acc_valid[outlier_mask]

                            if len(ic_robust) > 2:
                                slope_robust, intercept_robust, _, _, _ = stats.linregress(ic_robust, acc_robust)
                                y_fit_robust = slope_robust * x_fit + intercept_robust
                                ax.plot(x_fit, y_fit_robust, color='green', linestyle='--', linewidth=1.2, alpha=0.7)

                    # 设置坐标范围（所有子图统一）
                    ax.set_xlim(scatter_ic_min, scatter_ic_max)
                    ax.set_ylim(scatter_acc_min, scatter_acc_max)

                    # 标注模型名与lead（左侧模型，右侧leadtime）
                    label = chr(97 + col_idx)
                    ax.text(0.02, 0.95, f"({label}) {display_name}", transform=ax.transAxes, ha='left', va='top',
                            fontsize=10, fontweight='bold')
                    if col_idx == 0:
                        ax.text(0.98, 0.95, f"L{leadtime}", transform=ax.transAxes, ha='right', va='top',
                                fontsize=11, fontweight='bold')

                # 第二行：四个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(model_names):
                        ax = fig.add_subplot(gs[row_start+1, col_idx])
                        ax.axis('off')
                        axes_grid[row_start+1][col_idx] = ax
                        continue

                    model = model_names[model_idx]
                    if model not in model_data_dict:
                        ax = fig.add_subplot(gs[row_start+1, col_idx])
                        ax.axis('off')
                        axes_grid[row_start+1][col_idx] = ax
                        continue

                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')

                    ax = fig.add_subplot(gs[row_start+1, col_idx])
                    axes_grid[row_start+1][col_idx] = ax
                    content_axes.append(ax)

                    # 准备散点数据
                    acc_flat = model_data['ACC'].values.flatten()
                    ic_flat = model_data['inter_member'].values.flatten()
                    valid_mask = ~(np.isnan(acc_flat) | np.isnan(ic_flat))
                    acc_all = acc_flat[valid_mask]  # 保存所有有效数据用于标记超出范围的点
                    ic_all = ic_flat[valid_mask]
                    acc_valid = acc_all.copy()
                    ic_valid = ic_all.copy()

                    if len(acc_valid) > 0:
                        # 识别超出绘制范围的点（在过滤极端值之前）
                        out_x_low = ic_all < scatter_ic_min
                        out_x_high = ic_all > scatter_ic_max
                        out_y_low = acc_all < scatter_acc_min
                        out_y_high = acc_all > scatter_acc_max
                        out_x = out_x_low | out_x_high
                        out_y = out_y_low | out_y_high
                        out_any = out_x | out_y
                        
                        # 标记超出范围的点在坐标轴上
                        if np.any(out_any):
                            # 超出X轴范围的点：在X轴边界上标记，Y坐标保持原值（限制在Y轴范围内）
                            if np.any(out_x_low):
                                x_mark_low = np.full(np.sum(out_x_low), scatter_ic_min)
                                y_mark_low = np.clip(acc_all[out_x_low], scatter_acc_min, scatter_acc_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_x_high):
                                x_mark_high = np.full(np.sum(out_x_high), scatter_ic_max)
                                y_mark_high = np.clip(acc_all[out_x_high], scatter_acc_min, scatter_acc_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            # 超出Y轴范围的点：在Y轴边界上标记，X坐标保持原值（限制在X轴范围内）
                            if np.any(out_y_low):
                                y_mark_low = np.full(np.sum(out_y_low), scatter_acc_min)
                                x_mark_low = np.clip(ic_all[out_y_low], scatter_ic_min, scatter_ic_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_high):
                                y_mark_high = np.full(np.sum(out_y_high), scatter_acc_max)
                                x_mark_high = np.clip(ic_all[out_y_high], scatter_ic_min, scatter_ic_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                        
                        # 过滤极端值
                        if len(acc_valid) > 10:
                            ic_q1, ic_q3 = np.percentile(ic_valid, [25, 75])
                            ic_iqr = ic_q3 - ic_q1
                            ic_lower = ic_q1 - 1.5 * ic_iqr
                            ic_upper = ic_q3 + 1.5 * ic_iqr

                            acc_q1, acc_q3 = np.percentile(acc_valid, [25, 75])
                            acc_iqr = acc_q3 - acc_q1
                            acc_lower = acc_q1 - 1.5 * acc_iqr
                            acc_upper = acc_q3 + 1.5 * acc_iqr

                            outlier_mask = (
                                (ic_valid >= ic_lower) & (ic_valid <= ic_upper) &
                                (acc_valid >= acc_lower) & (acc_valid <= acc_upper)
                            )

                            if outlier_mask.sum() > 5:
                                ic_valid = ic_valid[outlier_mask]
                                acc_valid = acc_valid[outlier_mask]

                        # 绘制散点
                        if len(acc_valid) > 100:
                            try:
                                xy = np.vstack([ic_valid, acc_valid])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)

                                scatter = ax.scatter(ic_valid, acc_valid, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')
                            except:
                                ax.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(ic_valid, acc_valid, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')

                        # ACC=IC参考线
                        ideal_x = np.array([scatter_ic_min, scatter_ic_max])
                        ideal_y = ideal_x
                        ax.plot(ideal_x, ideal_y, color='orange', linestyle=':', linewidth=2, alpha=0.8)

                        # 拟合线
                        if len(ic_valid) > 2:
                            slope_all, intercept_all, _, _, _ = stats.linregress(ic_valid, acc_valid)
                            x_fit = np.array([scatter_ic_min, scatter_ic_max])
                            y_fit_all = slope_all * x_fit + intercept_all
                            ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)

                            # Robust fit
                            residuals = acc_valid - (slope_all * ic_valid + intercept_all)
                            q1, q3 = np.percentile(residuals, [25, 75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
                            ic_robust = ic_valid[outlier_mask]
                            acc_robust = acc_valid[outlier_mask]

                            if len(ic_robust) > 2:
                                slope_robust, intercept_robust, _, _, _ = stats.linregress(ic_robust, acc_robust)
                                y_fit_robust = slope_robust * x_fit + intercept_robust
                                ax.plot(x_fit, y_fit_robust, color='green', linestyle='--', linewidth=1.2, alpha=0.7)

                    # 设置坐标范围（所有子图统一）
                    ax.set_xlim(scatter_ic_min, scatter_ic_max)
                    ax.set_ylim(scatter_acc_min, scatter_acc_max)

                    # 标注模型名与lead（左侧模型，右侧leadtime）
                    label = chr(97 + col_idx + 3)
                    ax.text(0.02, 0.95, f"({label}) {display_name}", transform=ax.transAxes, ha='left', va='top',
                            fontsize=10, fontweight='bold')
                    if col_idx == 0:
                        ax.text(0.98, 0.95, f"L{leadtime}", transform=ax.transAxes, ha='right', va='top',
                                fontsize=11, fontweight='bold')

            # 统一坐标范围、去除内侧刻度与脊线，仅最外层显示
            for r in range(4):
                for c in range(4):
                    ax = axes_grid[r][c]
                    if ax is None:
                        continue
                    if ax not in content_axes:
                        continue

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

                    is_left = c == 0
                    is_bottom = r == 3
                    is_top = r == 0
                    is_right = c == 3

                    # 默认隐藏
                    ax.tick_params(labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    # 外侧打开
                    # 修正：L0的第一张内容图（在row=0,col=1）要补全下边框
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
                    if r == 1 and c == 0:
                        ax.spines['bottom'].set_visible(True)

            # 轴标签与图例
            fig.text(0.025, 0.63, 'ACC', va='center', rotation='vertical', fontsize=11)
            fig.text(0.5, 0.07, 'Inter-member Correlation', ha='center', va='center', fontsize=11)

            # 图例（左侧）和密度colorbar（右侧）
            legend_elements = [
                Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label='ACC=IC'),
                Line2D([0], [0], color='#ff7f0e', linestyle='-', linewidth=2, label='All data fit'),
                Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Robust fit'),
            ]
            fig.legend(handles=legend_elements, loc='lower left', ncol=3, frameon=False,
                       bbox_to_anchor=(0.15, 0.05))

            # 密度colorbar
            if scatter_for_cbar is not None:
                cbar_density_ax = fig.add_axes([0.65, 0.05, 0.25, 0.02])
                cbar_density = fig.colorbar(scatter_for_cbar, cax=cbar_density_ax, orientation='horizontal')
                cbar_density.set_label('Point Density', fontsize=10)
                cbar_density.ax.tick_params(labelsize=9)

            # 保存图像
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.output_dir / f"acc_scatter_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_scatter_{leadtimes_str}_{self.var_type}.pdf"

            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()

            logger.info(f"ACC散点图已保存: {output_file_png}")

        except Exception as e:
            logger.error(f"绘制ACC散点图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def plot_acc_ic_bar(self, leadtimes: List[int], models: List[str]):
        """
        绘制ACC/IC Ratio柱状图
        L0和L3绘制为上下两张子图，横轴为Model，子图之间不留间隙，共用横轴，纵轴为指标
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制ACC/IC柱状图: L{leadtimes} {self.var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_models_data(leadtimes, models)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于柱状图")
                return
            
            # 准备模型列表
            first_leadtime = leadtimes[0]
            if first_leadtime not in all_leadtimes_data:
                logger.error(f"第一个leadtime {first_leadtime} 没有数据")
                return
            
            model_names = [m for m in models if m in all_leadtimes_data[first_leadtime]]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 为每个leadtime计算每个模型的平均ACC/IC ratio
            leadtime_ratios = {}  # {leadtime: {model: ratio}}
            
            for leadtime in leadtimes:
                if leadtime not in all_leadtimes_data:
                    continue
                leadtime_ratios[leadtime] = {}
                for idx, model in enumerate(model_names):
                    if model in all_leadtimes_data[leadtime]:
                        model_data = all_leadtimes_data[leadtime][model]
                        acc_values = model_data['ACC'].values
                        ic_values = model_data['inter_member'].values
                        ratio = acc_values / ic_values
                        valid_ratios = ratio[np.isfinite(ratio)]
                        if len(valid_ratios) > 0:
                            avg_ratio = np.mean(valid_ratios)
                            leadtime_ratios[leadtime][model] = avg_ratio
            
            if not leadtime_ratios:
                logger.warning("没有可用的ratio数据")
                return
            
            # 创建图形：上下两个子图
            fig_width = max(10.0, n_models * 0.8)
            fig_height = 8.0
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # GridSpec：无间距，上下两个子图
            gs = GridSpec(2, 1, figure=fig, hspace=0.0,
                         left=0.1, right=0.95, top=0.95, bottom=0.1)
            
            # 准备模型标签（带(a)(b)(c)标签）
            model_labels = []
            for idx, model in enumerate(model_names):
                subplot_label = chr(97 + idx)
                display_name = model.replace('-mon', '').replace('mon-', '')
                model_labels.append(f"({subplot_label}) {display_name}")
            
            x_pos = np.arange(len(model_labels))
            
            # 计算统一的y轴范围
            all_ratios = []
            for leadtime in leadtimes:
                if leadtime in leadtime_ratios:
                    for model in model_names:
                        if model in leadtime_ratios[leadtime]:
                            all_ratios.append(leadtime_ratios[leadtime][model])
            
            if all_ratios:
                y_min = min(0.0, np.min(all_ratios) * 1.1)
                y_max = max(1.0, np.max(all_ratios) * 1.1)
            else:
                y_min, y_max = 0.0, 1.0
            
            # 绘制每个leadtime的子图
            for lt_idx, leadtime in enumerate(leadtimes):
                if leadtime not in leadtime_ratios:
                    continue
                
                ax = fig.add_subplot(gs[lt_idx, 0])
                
                # 准备该leadtime的数据
                ratios = []
                for model in model_names:
                    if model in leadtime_ratios[leadtime]:
                        ratios.append(leadtime_ratios[leadtime][model])
                    else:
                        ratios.append(np.nan)
                
                # 绘制柱状图
                bars = ax.bar(x_pos, ratios, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # 设置坐标轴
                ax.set_xticks(x_pos)
                if lt_idx == len(leadtimes) - 1:  # 最后一个子图显示x轴标签
                    ax.set_xticklabels(model_labels, fontsize=9, rotation=45, ha='right')
                else:
                    ax.set_xticklabels([])  # 其他子图不显示x轴标签
                
                ax.set_ylabel('ACC/IC Ratio', fontsize=11)
                ax.set_ylim(y_min, y_max)
                ax.tick_params(axis='y', labelsize=9)
                
                # 添加y=1参考线（理想情况：ACC=IC）
                ax.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
                
                # 移除y=1的刻度标签，避免与参考线重叠
                yticks = ax.get_yticks()
                yticklabels = ax.get_yticklabels()
                # 如果y=1在刻度中，移除其标签
                for i, tick in enumerate(yticks):
                    if abs(tick - 1.0) < 1e-6:  # 检查是否接近1.0
                        yticklabels[i].set_text('')
                ax.set_yticklabels(yticklabels)
                
                # 添加leadtime标签
                ax.text(0.98, 0.95, f'L{leadtime}', transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 网格
                ax.grid(True, axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
                
                # 隐藏上边框（除了第一个子图）
                if lt_idx > 0:
                    ax.spines['top'].set_visible(False)
                # 隐藏下边框（除了最后一个子图）
                if lt_idx < len(leadtimes) - 1:
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(axis='x', bottom=False)
            
            # 保存图像
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.output_dir / f"acc_ic_bar_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_ic_bar_{leadtimes_str}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"ACC/IC柱状图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制ACC/IC柱状图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_acc_monthly_contour(self, models: List[str] = None, leadtimes: List[int] = None):
        """
        绘制ACC逐月等高线图
        横轴为月份（1-12），纵轴为leadtime
        布局：第一行留空+3个模型，第二行4个模型
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        try:
            logger.info(f"开始绘制ACC逐月等高线图: {self.var_type}")
            
            # 加载所有模型的逐月+逐年ACC数据
            all_models_data = {}  # {model: {leadtime: DataArray}}
            
            for model in models:
                model_data = {}
                for leadtime in leadtimes:
                    monthly_file = self.acc_data_dir / f"acc_monthly_{model}_L{leadtime}_{self.var_type}.nc"
                    
                    if not monthly_file.exists():
                        logger.debug(f"{model} L{leadtime}: 逐月ACC文件不存在，跳过")
                        continue
                    
                    try:
                        with xr.open_dataset(monthly_file) as ds:
                            if 'monthly_annual_anomaly_spatial_corr' in ds:
                                data = ds['monthly_annual_anomaly_spatial_corr'].load()
                                model_data[leadtime] = data
                            else:
                                logger.warning(f"{model} L{leadtime}: 文件中没有找到逐月ACC数据")
                    except Exception as e:
                        logger.error(f"加载{model} L{leadtime}数据失败: {e}")
                        continue
                
                if model_data:
                    all_models_data[model] = model_data
            
            if not all_models_data:
                logger.warning("没有找到任何逐月ACC数据")
                return
            
            # 准备模型列表（按顺序）
            model_names = [m for m in models if m in all_models_data]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 对每个模型，按月份和leadtime组织数据（跨年份平均）
            # 数据结构：{model: np.array(shape=(n_leadtimes, 12))} - 转置后，leadtime为行，month为列
            model_contour_data = {}
            all_values = []
            
            for model in model_names:
                # 创建矩阵 (month, leadtime)，然后转置为 (leadtime, month)
                months = list(range(1, 13))
                contour_matrix = np.full((len(months), len(leadtimes)), np.nan)
                
                for mi, month in enumerate(months):
                    for li, leadtime in enumerate(leadtimes):
                        if leadtime in all_models_data[model]:
                            data = all_models_data[model][leadtime]
                            # 选择该月份的所有年份数据，然后取平均
                            month_data = data.sel(month=month)
                            if month_data.size > 0:
                                # 跨年份平均
                                mean_val = float(month_data.mean(skipna=True).item())
                                if np.isfinite(mean_val):
                                    contour_matrix[mi, li] = mean_val
                                    all_values.append(mean_val)
                
                # 转置矩阵：从 (month, leadtime) 转为 (leadtime, month)
                model_contour_data[model] = contour_matrix.T
            
            if not all_values:
                logger.warning("没有有效的ACC数据用于绘制等高线图")
                return
            
            # 计算统一的colorbar范围
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
            
            logger.info(f"ACC等高线图数据范围: [{vmin:.3f}, {vmax:.3f}]")
            
            # 创建图形：2行4列布局
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
                         left=0.08, right=0.95, top=0.95, bottom=0.1)
            
            # 第一行：留空 + 3个模型
            for col_idx in range(4):
                if col_idx == 0:
                    # 留空
                    ax_blank = fig.add_subplot(gs[0, col_idx])
                    ax_blank.axis('off')
                elif col_idx - 1 < len(model_names[:3]):
                    model = model_names[col_idx - 1]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax = fig.add_subplot(gs[0, col_idx])
                    
                    # 准备数据（已转置为 (leadtime, month)）
                    contour_data = model_contour_data[model]
                    
                    # 绘制等高线图（不填充颜色）
                    # X轴为月份，Y轴为leadtime
                    months = list(range(1, 13))
                    X, Y = np.meshgrid(months, leadtimes)
                    
                    # 根据数据范围自动计算合适的等高线数量（6条，避免过于密集）
                    valid_data = contour_data[~np.isnan(contour_data)]
                    if len(valid_data) > 0:
                        data_min = np.nanmin(contour_data)
                        data_max = np.nanmax(contour_data)
                        data_range = data_max - data_min
                        # 根据数据范围自动确定等高线间隔
                        if data_range > 1e-6:  # 避免除零错误
                            # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                            levels = ticker.MaxNLocator(nbins=7, prune=None).tick_values(data_min, data_max)
                        else:
                            # 数据范围太小，使用单一值
                            levels = [data_min] if not np.isnan(data_min) else 6
                    else:
                        # 没有有效数据，使用默认值
                        levels = 6
                    
                    contours = ax.contour(X, Y, contour_data, levels=levels, colors='black', 
                                         linewidths=1.2, alpha=0.8)
                    # 只标注部分等高线，避免过于密集（自动选择标注位置）
                    ax.clabel(contours, inline=True, fontsize=14, fmt='%.2f', 
                             manual=False, colors='black')
                    
                    # 设置坐标轴（横轴为月份，纵轴为leadtime）
                    ax.set_xticks(months)
                    ax.set_yticks(leadtimes)
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_xlabel('Month', fontsize=16)
                    if col_idx == 1:  # 第一列显示y轴标签
                        ax.set_ylabel('Lead Time', fontsize=16)
                    
                    # 模型标签
                    label = chr(97 + col_idx - 1)  # a, b, c
                    ax.text(0.02, 0.98, f'({label}) {display_name}', 
                           transform=ax.transAxes, fontsize=16, fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
                    
                    ax.grid(True, alpha=0.3, linestyle='--')
            
            # 第二行：4个模型
            for col_idx in range(4):
                model_idx = col_idx + 3
                if model_idx < len(model_names):
                    model = model_names[model_idx]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax = fig.add_subplot(gs[1, col_idx])
                    
                    # 准备数据（已转置为 (leadtime, month)）
                    contour_data = model_contour_data[model]
                    
                    # 绘制等高线图（不填充颜色）
                    # X轴为月份，Y轴为leadtime
                    months = list(range(1, 13))
                    X, Y = np.meshgrid(months, leadtimes)
                    
                    # 根据数据范围自动计算合适的等高线数量（6条，避免过于密集）
                    valid_data = contour_data[~np.isnan(contour_data)]
                    if len(valid_data) > 0:
                        data_min = np.nanmin(contour_data)
                        data_max = np.nanmax(contour_data)
                        data_range = data_max - data_min
                        # 根据数据范围自动确定等高线间隔
                        if data_range > 1e-6:  # 避免除零错误
                            # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                            levels = ticker.MaxNLocator(nbins=7, prune=None).tick_values(data_min, data_max)
                        else:
                            # 数据范围太小，使用单一值
                            levels = [data_min] if not np.isnan(data_min) else 6
                    else:
                        # 没有有效数据，使用默认值
                        levels = 6
                    
                    contours = ax.contour(X, Y, contour_data, levels=levels, colors='black', 
                                         linewidths=1.2, alpha=0.8)
                    # 只标注部分等高线，避免过于密集（自动选择标注位置）
                    ax.clabel(contours, inline=True, fontsize=14, fmt='%.2f', 
                             manual=False, colors='black')
                    
                    # 设置坐标轴（横轴为月份，纵轴为leadtime）
                    ax.set_xticks(months)
                    ax.set_yticks(leadtimes)
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_xlabel('Month', fontsize=16)
                    ax.set_ylabel('Lead Time', fontsize=16)
                    
                    # 模型标签
                    label = chr(97 + model_idx)  # d, e, f, g
                    ax.text(0.02, 0.98, f'({label}) {display_name}', 
                           transform=ax.transAxes, fontsize=16, fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
                    
                    ax.grid(True, alpha=0.3, linestyle='--')
                else:
                    # 空白
                    ax_blank = fig.add_subplot(gs[1, col_idx])
                    ax_blank.axis('off')
            
            # 保存图像
            output_file_png = self.output_dir / f"acc_monthly_contour_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_monthly_contour_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"ACC逐月等高线图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制ACC逐月等高线图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_acc_leadtime_timeseries(self, models: List[str] = None, leadtimes: List[int] = None, 
                                     use_temporal_spatial: bool = True):
        """
        绘制ACC（年度距平空间相关系数）随leadtime变化的折线图
        参考climatology_analysis.py中的plot_corr_leadtime_timeseries方法
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
            use_temporal_spatial: 是否使用方法2（时间-空间综合），默认True使用方法2
                                 设置为False可使用方法1（先时间平均再空间相关，已注释）
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        try:
            method_name = "时间-空间综合" if use_temporal_spatial else "先时间平均再空间相关"
            logger.info(f"开始绘制ACC随leadtime变化的折线图（方法: {method_name}）...")
            
            # 选择对应的变量名
            var_suffix = "temporal" if use_temporal_spatial else ""
            var_name = f"acc_mean_spatial_corr{'_temporal' if use_temporal_spatial else ''}_{self.var_type}"
            
            # 读取年度平均数据
            metrics_file = Path("/sas12t1/ffyan/output/acc_analysis/results") / f"metrics_{var_name}.nc"
            
            if not metrics_file.exists():
                logger.warning(f"Metrics文件不存在: {metrics_file}")
                return
            
            # 加载数据
            with xr.open_dataarray(metrics_file) as da:
                acc_mean_da = da.load()
            
            logger.info(f"加载数据成功: {acc_mean_da.shape}")
            
            # 创建图形
            fig_height = 6.0
            fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
            
            # 设置颜色映射
            model_order = [m for m in MODELS if m in models]
            cmap = plt.get_cmap('tab10')
            color_map = {model: cmap(i % cmap.N) for i, model in enumerate(model_order)}
            
            legend_handles = []
            legend_labels = []
            all_y_vals = []
            
            # 绘制每个模型的曲线
            for model in model_order:
                if model not in acc_mean_da.coords['model'].values:
                    continue
                
                y_vals = []
                x_vals = []
                
                for lt in leadtimes:
                    if lt in acc_mean_da.coords['lead'].values:
                        val = float(acc_mean_da.sel(model=model, lead=lt).item())
                        if np.isfinite(val):
                            x_vals.append(lt)
                            y_vals.append(val)
                            all_y_vals.append(val)
                
                if not x_vals:
                    continue
                
                line, = ax.plot(
                    x_vals, y_vals,
                    marker='o', linewidth=2.0, markersize=6,
                    color=color_map[model], label=model
                )
                
                if model not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(model)
            
            # 设置坐标轴
            ax.set_xlabel('Lead Time', fontsize=12)
            ax.set_ylabel('ACC (Annual Anomaly Spatial Correlation)', fontsize=12)
            ax.set_xticks(leadtimes)
            ax.set_xlim(leadtimes[0] - 0.2, leadtimes[-1] + 0.2)
            
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
                    y_max = min(1.0, y_max + margin)  # ACC最大值为1
                    ax.set_ylim(y_min, y_max)
            
            # 网格
            ax.grid(True, axis='y', linestyle=':', alpha=0.4)
            ax.set_axisbelow(True)
            
            # 图例设置：放在图像外下方，横向、分两行
            if legend_handles:
                ncol = (len(legend_handles) + 1) // 2  # 每行约一半
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.27),
                    frameon=True,
                    fontsize=10,
                    ncol=ncol,
                    columnspacing=1.5,
                    handlelength=2,
                    handletextpad=0.6,
                    borderaxespad=0.5,
                    borderpad=0.8,
                    fancybox=True,
                )
            
            # 保存图像
            plt.tight_layout()
            output_file_png = self.output_dir / f"acc_leadtime_timeseries_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"acc_leadtime_timeseries_{self.var_type}.pdf"
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"ACC随leadtime折线图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制ACC随leadtime折线图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    def plot_acc_results(self, models: List[str] = None, leadtimes: List[int] = None,
                         plot_spatial: bool = True, plot_scatter: bool = True, 
                         plot_bar: bool = True, plot_timeseries: bool = True,
                         plot_monthly_contour: bool = True, use_temporal_spatial: bool = True):
        """
        绘制ACC相关图表
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
            plot_spatial: 是否绘制空间分布图
            plot_scatter: 是否绘制散点图
            plot_bar: 是否绘制柱状图
            plot_timeseries: 是否绘制时间序列图
            use_temporal_spatial: 是否使用方法2（时间-空间综合）绘制时间序列图，默认True使用方法2
                                 设置为False可使用方法1（先时间平均再空间相关，已注释）
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        logger.info("="*60)
        logger.info(f"开始绘制ACC图表: {self.var_type}")
        logger.info("="*60)
        
        # 绘制空间分布图、散点图和柱状图（使用L0和L3）
        target_leadtimes = [0, 3] if 0 in leadtimes and 3 in leadtimes else leadtimes[:2] if len(leadtimes) >= 2 else leadtimes
        
        if plot_spatial:
            logger.info("\n绘制ACC空间分布图...")
            self.plot_acc_spatial_distribution(target_leadtimes, models)
        
        if plot_scatter:
            logger.info("\n绘制ACC散点图...")
            self.plot_acc_scatter(target_leadtimes, models)
        
        if plot_bar:
            logger.info("\n绘制ACC/IC柱状图...")
            self.plot_acc_ic_bar(target_leadtimes, models)
        
        # 绘制ACC随leadtime变化的折线图
        if plot_timeseries:
            method_name = "时间-空间综合" if use_temporal_spatial else "先时间平均再空间相关"
            logger.info(f"\n绘制ACC随leadtime变化的折线图（方法: {method_name}）...")
            self.plot_acc_leadtime_timeseries(models, leadtimes, use_temporal_spatial=use_temporal_spatial)
        
        # 绘制ACC逐月等高线图
        if plot_monthly_contour:
            logger.info("\n绘制ACC逐月等高线图...")
            self.plot_acc_monthly_contour(models, leadtimes)
        
        logger.info("\n" + "="*60)
        logger.info("ACC图表绘制完成")
        logger.info("="*60)


def main():
    """主函数"""
    parser = create_parser(
        description="ACC与Inter-member Correlation分析",
        include_outliers=True,
        include_acc_specific=True,
        var_default=None  # 允许不指定，默认处理temp和prec
    )
    args = parser.parse_args()
    
    # 解析参数
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_types = parse_vars(args.var) if args.var else ["temp", "prec"]
    
    # 更新全局配置
    global REMOVE_OUTLIERS, OUTLIER_THRESHOLD
    REMOVE_OUTLIERS = not args.no_outliers
    OUTLIER_THRESHOLD = args.outlier_threshold
    
    # 标准化绘图参数
    normalize_plot_args(args)
    
    logger.info(f"将处理变量: {var_types}")
    logger.info(f"将处理模型: {models}")
    logger.info(f"将处理提前期: {leadtimes}")
    logger.info(f"绘图模式: no_plot={args.no_plot}, plot_only={args.plot_only}")
    logger.info(f"异常值处理: 启用={REMOVE_OUTLIERS}, 阈值={OUTLIER_THRESHOLD}")
    
    # 为每个变量运行分析
    for var_type in var_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始处理变量: {var_type.upper()}")
        logger.info(f"{'='*50}")
        
        # 创建分析器
        analyzer = ACCAnalyzer(var_type)
        
        # 运行ACC分析计算（除非是仅绘图模式）
        # 注意：空间分布图、散点图和柱状图统称为ratio图像，使用相同的ACC和IC数据，统一控制
        # 时间序列图使用年度距平空间相关系数，不需要IC
        if not args.plot_only:
            # 确定需要计算/绘制的图表类型
            calc_ratio = not args.no_calc_ratio  # ratio图像（空间分布图、散点图、柱状图）
            calc_timeseries = not args.no_calc_timeseries  # 时间序列图
            
            # 如果至少有一个需要计算，就运行分析
            if calc_ratio or calc_timeseries:
                # 标准化并行参数
                parallel = normalize_parallel_args(args)
                
                analyzer.run_analysis(
                    models=models,
                    leadtimes=leadtimes,
                    parallel=parallel,
                    n_jobs=args.n_jobs,
                    calc_ic=calc_ratio  # 只有需要ratio图像时才计算IC
                )
                logger.info(f"{var_type.upper()} 计算完成 (IC计算: {'是' if calc_ratio else '否'})")
                
                # 聚合年度距平空间相关系数（如果时间序列图需要）
                if calc_timeseries:
                    logger.info(f"\n聚合{var_type.upper()}年度距平空间相关系数...")
                    analyzer.run_acc_leadtime_analysis(
                        models=models,
                        leadtimes=leadtimes
                    )
        
        # 绘制ACC图表（默认绘图，除非指定了--no-plot）
        if not args.no_plot:
            try:
                # 绘图时总是尝试绘制所有图表类型，只要数据文件存在
                # --no-calc-ratio 和 --no-calc-timeseries 只控制是否计算，不影响绘图
                
                # 创建绘图器并绘制图表
                plotter = ACCPlotter(var_type)
                plotter.plot_acc_results(
                    models=models,
                    leadtimes=leadtimes,
                    plot_spatial=True,  # 总是尝试绘制，如果数据不存在会自动跳过
                    plot_scatter=True,
                    plot_bar=True,
                    plot_timeseries=True,
                    use_temporal_spatial=not args.use_spatial_mean  # 默认True（方法2），如果指定--use-spatial-mean则为False（方法1）
                )
            except Exception as e:
                logger.error(f"绘制ACC图表时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*50}")
    logger.info("所有任务完成！")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

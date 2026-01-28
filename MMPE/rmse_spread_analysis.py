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
import fcntl  # 用于文件锁
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
from src.utils.data_utils import find_valid_data_bounds
from src.utils.logging_config import setup_logging
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_parallel_args, normalize_plot_args
from src.config.output_config import get_rmse_spatial_data_path, get_rmse_temporal_data_path, get_spatial_rmse_ts_path, get_aggregated_csv_path
from src.utils.aggregation import compute_aggregates

from common_config import (
    MODEL_LIST,
    LEADTIMES,
    SEASONS,
    MAX_WORKERS_TEMP,
    MAX_WORKERS_PREC,
    HARD_WORKER_CAP,
    DEFAULT_TIME_CHUNK,
)

# 尝试导入climpred（可选依赖）
try:
    import climpred
    from climpred import HindcastEnsemble
    CLIMPRED_AVAILABLE = True
except ImportError:
    CLIMPRED_AVAILABLE = False

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logging(
    log_file='rmse_simplified.log',
    module_name=__name__
)

# 日志记录climpred可用性
if not CLIMPRED_AVAILABLE:
    logger.warning("climpred not available, using basic RMSE calculation")

# 全局配置（从 common_config 导入）
MODELS = MODEL_LIST  # "JMA-3-mon" 被排除时仍可独立添加

class SimplifiedRMSE:
    """简化的RMSE分析器，整合计算和绘图功能"""
    
    def __init__(self, var_type: str, data_loader: DataLoader = None):
        """
        初始化RMSE分析器
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
            data_loader: 数据加载器，如果为None则自动创建
        """
        self.var_type = var_type
        self.data_loader = data_loader or DataLoader()
        
        # 直接观测数据路径（用于统计计算，避免插值）
        self.obs_direct_path = f"/sas12t1/ffyan/obs/{var_type}_1deg_199301-202012.nc"
        
        logger.info(f"初始化RMSE分析器: {var_type}")
    
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
    
    
    def calculate_ensemble_spread(self, ensemble_data: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        计算ensemble spread（按照用户示例的方法）
        
        Args:
            ensemble_data: (time, number, lat, lon)
        
        Returns:
            Dictionary with:
            - 'total_spread': mean absolute deviation from ensemble mean (lat, lon)
            - 'spread_per_member': spread for each member (number, lat, lon)
            - 'spread_temporal': spread time series, spatial mean (time,)
            - 'spread_spatial_temporal': spread at each time and location (time, lat, lon)
        """
        try:
            logger.info(f"计算ensemble spread, shape={ensemble_data.shape}, 成员数={ensemble_data.number.size}")
            
            # 检查ensemble数据有效性
            n_valid = np.sum(~np.isnan(ensemble_data.values))
            total_elements = ensemble_data.size
            logger.info(f"Ensemble数据有效性: {n_valid}/{total_elements} ({n_valid/total_elements*100:.1f}%)")
            
            if n_valid == 0:
                logger.error("Ensemble数据全为NaN，无法计算spread")
                return None
            
            # 计算 ensemble mean (每个时间步的平均值)
            ensemble_mean = ensemble_data.mean(dim='number')  # (time, lat, lon)
            
            # 检查是否所有成员相同（spread=0的情况）
            sample_variance = ensemble_data.var(dim='number').mean().values
            if sample_variance < 1e-10:
                logger.warning(f"Ensemble成员间方差极小({sample_variance:.2e})，可能所有成员值相同")
            
            # 计算每个成员的 model spread: |ensemble - ensemble_mean|
            # 广播：ensemble_data (time, number, lat, lon) - ensemble_mean (time, lat, lon)
            spread = np.abs(ensemble_data - ensemble_mean)  # (time, number, lat, lon)
            
            # 1. Total spread: 对所有成员和时间求平均
            total_spread = spread.mean(dim=['time', 'number'])  # (lat, lon)
            total_spread.attrs = {
                'long_name': 'Total Ensemble Spread',
                'description': 'Mean absolute deviation from ensemble mean across all members and time',
                'units': ensemble_data.attrs.get('units', '')
            }
            
            # 2. Spread per member: 对时间求平均
            spread_per_member = spread.mean(dim='time')  # (number, lat, lon)
            spread_per_member.attrs = {
                'long_name': 'Spread per Ensemble Member',
                'description': 'Mean absolute deviation from ensemble mean for each member',
                'units': ensemble_data.attrs.get('units', '')
            }
            
            # 3. Temporal spread: 空间平均后的时间序列
            spread_spatial_temporal = spread.mean(dim='number')  # (time, lat, lon)
            spread_temporal = spread_spatial_temporal.mean(dim=['lat', 'lon'])  # (time,)
            spread_temporal.attrs = {
                'long_name': 'Temporal Ensemble Spread',
                'description': 'Spatial mean of ensemble spread time series',
                'units': ensemble_data.attrs.get('units', '')
            }
            
            # 4. Spread at each time and location (averaged over members)
            spread_spatial_temporal.attrs = {
                'long_name': 'Spatial-Temporal Ensemble Spread',
                'description': 'Ensemble spread at each time and location',
                'units': ensemble_data.attrs.get('units', '')
            }
            
            results = {
                'total_spread': total_spread,
                'spread_per_member': spread_per_member,
                'spread_temporal': spread_temporal,
                'spread_spatial_temporal': spread_spatial_temporal
            }
            
            logger.info(f"Ensemble spread计算完成")
            logger.info(f"  Total spread范围: [{float(total_spread.min()):.4f}, {float(total_spread.max()):.4f}]")
            logger.info(f"  Temporal spread范围: [{float(spread_temporal.min()):.4f}, {float(spread_temporal.max()):.4f}]")
            
            return results
            
        except Exception as e:
            logger.error(f"计算ensemble spread失败: {e}")
            return None
    
    def calculate_rmse_climpred(self, obs_data: xr.DataArray, 
                                fcst_data: xr.DataArray,
                                ensemble_data: xr.DataArray = None) -> Dict:
        """
        使用climpred计算RMSE和相关指标
        
        Args:
            obs_data: 观测数据 (time, lat, lon)
            fcst_data: 预报数据 (time, lat, lon) - ensemble mean
            ensemble_data: ensemble数据 (time, number, lat, lon) - 可选
        
        Returns:
            Dictionary with:
            - 'rmse': RMSE from climpred
            - 'spread': ensemble spread (if ensemble available)
            - 'spread_error_ratio': spread/RMSE ratio (if ensemble available)
        """
        if not CLIMPRED_AVAILABLE:
            logger.warning("climpred不可用，无法使用climpred方法")
            return None
        
        try:
            logger.info("使用climpred计算RMSE...")
            
            # 时间对齐
            obs_aligned, fcst_aligned = self.align_time_data(obs_data, fcst_data)
            if obs_aligned is None:
                return None
            
            # 空间对齐
            obs_aligned, fcst_aligned = self.align_spatial_grid(obs_aligned, fcst_aligned)
            
            results = {}
            
            # 使用climpred计算RMSE
            # 注意：climpred需要特定的坐标命名和结构
            # 这里我们使用基本的方法，因为hindcast ensemble需要init维度
            
            # 计算标准RMSE作为参考
            diff = fcst_aligned - obs_aligned
            rmse_standard = np.sqrt((diff ** 2).mean(dim='time', skipna=True))
            results['rmse'] = rmse_standard
            
            logger.info("标准RMSE计算完成（使用climpred框架）")
            
            # 如果有ensemble数据，计算spread和spread-error ratio
            if ensemble_data is not None:
                # 对齐ensemble数据
                common_times = obs_aligned.time.to_index().intersection(ensemble_data.time.to_index())
                if len(common_times) > 0:
                    ensemble_aligned = ensemble_data.sel(time=common_times)
                    
                    # 空间对齐
                    if not np.array_equal(ensemble_aligned.lat.values, obs_aligned.lat.values) or \
                       not np.array_equal(ensemble_aligned.lon.values, obs_aligned.lon.values):
                        ensemble_aligned = ensemble_aligned.interp(lat=obs_aligned.lat, lon=obs_aligned.lon, method='linear')
                    
                    # 计算spread
                    spread_calc = self.calculate_ensemble_spread(ensemble_aligned)
                    if spread_calc:
                        results['spread'] = spread_calc['total_spread']
                        results['spread_temporal'] = spread_calc['spread_temporal']
                        results['spread_per_member'] = spread_calc['spread_per_member']
                        results['spread_spatial_temporal'] = spread_calc['spread_spatial_temporal']
                        
                        # 计算spread-error ratio
                        # spread/RMSE比率：理想情况下应接近1（校准良好的ensemble）
                        spread_error_ratio = spread_calc['total_spread'] / (rmse_standard + 1e-10)
                        spread_error_ratio.attrs = {
                            'long_name': 'Spread-Error Ratio',
                            'description': 'Ratio of ensemble spread to RMSE. Ideal value ~1.',
                            'units': 'dimensionless'
                        }
                        results['spread_error_ratio'] = spread_error_ratio
                        
                        logger.info(f"Spread-Error Ratio范围: [{float(spread_error_ratio.min()):.4f}, {float(spread_error_ratio.max()):.4f}]")
                        logger.info(f"Spread-Error Ratio平均: {float(spread_error_ratio.mean()):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"climpred RMSE计算失败: {e}")
            return None
    
    def calculate_spatial_rmse(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> xr.DataArray:
        """计算空间RMSE"""
        try:
            # 时间对齐
            obs_aligned, fcst_aligned = self.align_time_data(obs_data, fcst_data)
            if obs_aligned is None:
                return None
            # 空间网格对齐
            obs_aligned, fcst_aligned = self.align_spatial_grid(obs_aligned, fcst_aligned)
            
            # 内存优化：分块计算RMSE，避免大数组一次性加载
            try:
                # 计算RMSE
                diff = fcst_aligned - obs_aligned
                # 有效覆盖度阈值：至少min_valid_count个时间点共同有效
                total_time = int(obs_aligned.sizes.get('time', 0))
                min_valid_count = max(12, int(0.2 * total_time)) if total_time else 0
                valid_mask = (~xr.apply_ufunc(np.isnan, obs_aligned)) & (~xr.apply_ufunc(np.isnan, fcst_aligned))
                valid_count = valid_mask.sum(dim='time') if total_time else None

                rmse = np.sqrt(((diff ** 2).where(valid_mask)).mean(dim='time', skipna=True))
                if valid_count is not None:
                    rmse = rmse.where(valid_count >= min_valid_count)
            except MemoryError:
                logger.warning("内存不足，使用分块计算RMSE")
                # 分块计算避免内存溢出
                chunk_size = min(DEFAULT_TIME_CHUNK, max(6, int(obs_aligned.sizes.get('time', 24) // 12)))
                time_chunks = [obs_aligned.time[i:i+chunk_size] for i in range(0, len(obs_aligned.time), chunk_size)]
                
                squared_diffs = []
                valid_counts = []
                for chunk_times in time_chunks:
                    obs_chunk = obs_aligned.sel(time=chunk_times)
                    fcst_chunk = fcst_aligned.sel(time=chunk_times)
                    diff_chunk = fcst_chunk - obs_chunk
                    valid_chunk = (~xr.apply_ufunc(np.isnan, obs_chunk)) & (~xr.apply_ufunc(np.isnan, fcst_chunk))
                    squared_diff = ((diff_chunk ** 2).where(valid_chunk)).mean(dim='time', skipna=True)
                    squared_diffs.append(squared_diff)
                    valid_counts.append(valid_chunk.sum(dim='time'))
                
                # 计算所有块的平均值
                mean_squared_diff = xr.concat(squared_diffs, dim='chunk').mean(dim='chunk')
                valid_count = xr.concat(valid_counts, dim='chunk').sum(dim='chunk')
                rmse = np.sqrt(mean_squared_diff)
                rmse = rmse.where(valid_count >= max(12, int(0.2 * int(obs_aligned.sizes.get('time', 0)))))
            
            # 设置属性
            rmse.attrs = {
                'long_name': f'{self.var_type.upper()} Spatial RMSE',
                'units': obs_data.attrs.get('units', ''),
                'description': 'Root Mean Square Error'
            }
            
            return rmse
            
        except Exception as e:
            logger.error(f"计算空间RMSE失败: {e}")
            return None

    def calculate_spatial_field_rmse_ts(self, obs_data: xr.DataArray, fcst_data: xr.DataArray, use_direct_obs: bool = True) -> Tuple[np.ndarray, np.ndarray, xr.DataArray]:
        """逐时间点计算空间场RMSE时间序列，返回 (rmse_ts, n_ts, time_coord)
        Args:
            use_direct_obs: 是否使用直接观测数据（避免插值）
        """
        if use_direct_obs:
            # 使用直接观测数据，避免插值
            try:
                obs_direct = self.load_obs_data_direct()
                obs_data = obs_direct
            except Exception as e:
                logger.warning(f"直接加载观测数据失败，使用原始观测数据: {e}")
        
        common_times = np.intersect1d(obs_data.time.values, fcst_data.time.values)
        if len(common_times) == 0:
            return np.array([]), np.array([]), xr.DataArray(common_times, dims=['time'])
        obs_aligned = obs_data.sel(time=common_times)
        fcst_aligned = fcst_data.sel(time=common_times)
        
        # 对于统计计算，使用坐标选择而非插值
        try:
            # 选择观测网格覆盖的预报数据区域
            fcst_aligned = fcst_aligned.sel(
                lat=obs_aligned.lat, 
                lon=obs_aligned.lon, 
                method='nearest'
            )
        except Exception:
            # 如果坐标选择失败，保持原有插值逻辑
            try:
                obs_aligned = obs_aligned.interp_like(fcst_aligned)
            except Exception:
                try:
                    fcst_aligned = fcst_aligned.interp_like(obs_aligned)
                except Exception:
                    pass
        ntime = len(common_times)
        rmse_ts = np.full(ntime, np.nan)
        n_ts = np.full(ntime, np.nan)
        for t in range(ntime):
            obs_field = obs_aligned.isel(time=t).values.flatten()
            fcst_field = fcst_aligned.isel(time=t).values.flatten()
            valid = ~(np.isnan(obs_field) | np.isnan(fcst_field))
            n_valid = int(np.sum(valid))
            if n_valid < 10:
                continue
            mse = np.mean((fcst_field[valid] - obs_field[valid]) ** 2)
            rmse_ts[t] = float(np.sqrt(mse))
            n_ts[t] = n_valid
        return rmse_ts, n_ts, obs_aligned.time
    
    def calculate_temporal_rmse(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> pd.Series:
        """计算时间RMSE"""
        try:
            # 时间对齐
            obs_aligned, fcst_aligned = self.align_time_data(obs_data, fcst_data)
            if obs_aligned is None:
                return None
            # 空间网格对齐
            obs_aligned, fcst_aligned = self.align_spatial_grid(obs_aligned, fcst_aligned)
            
            # 空间平均
            obs_spatial_mean = obs_aligned.mean(dim=['lat', 'lon'], skipna=True)
            fcst_spatial_mean = fcst_aligned.mean(dim=['lat', 'lon'], skipna=True)
            
            # 计算时间RMSE（滑动窗口）
            window_size = 12  # 12个月窗口
            rmse_series = []
            times = []
            
            for i in range(len(obs_spatial_mean) - window_size + 1):
                obs_window = obs_spatial_mean[i:i+window_size]
                fcst_window = fcst_spatial_mean[i:i+window_size]
                
                # 使用xarray带skipna的均值计算，避免NaN导致全NaN
                diff = (fcst_window - obs_window) ** 2
                rmse_val = float(np.sqrt(diff.mean(skipna=True).values))
                rmse_series.append(rmse_val)
                times.append(obs_spatial_mean.time[i+window_size-1].values)
            
            # 创建Series
            rmse_ts = pd.Series(rmse_series, index=pd.to_datetime(times))
            rmse_ts.name = f'{self.var_type.upper()}_RMSE'
            
            return rmse_ts
            
        except Exception as e:
            logger.error(f"计算时间RMSE失败: {e}")
            return None
    
    def calculate_seasonal_rmse(self, obs_data: xr.DataArray, fcst_data: xr.DataArray, season: str) -> xr.DataArray:
        """计算季节RMSE"""
        try:
            if season is None or season == 'annual':
                # None或annual表示年平均，直接计算所有月份的RMSE
                return self.calculate_spatial_rmse(obs_data, fcst_data)
            
            if season not in SEASONS:
                raise ValueError(f"不支持的季节: {season}")
            
            # 选择季节数据
            months = SEASONS[season]
            obs_season = obs_data.sel(time=obs_data.time.dt.month.isin(months))
            fcst_season = fcst_data.sel(time=fcst_data.time.dt.month.isin(months))
            
            # 计算季节RMSE
            return self.calculate_spatial_rmse(obs_season, fcst_season)
            
        except Exception as e:
            logger.error(f"计算季节RMSE失败: {e}")
            return None
    
    def process_model_leadtime(self, model: str, leadtime: int, season: str = None) -> Dict:
        """处理单个模型和提前期的RMSE计算"""
        try:
            logger.info(f"处理 {model} L{leadtime}" + (f" {season}" if season else ""))
            
            # 加载数据 - 使用1度网格观测数据
            obs_data = self.load_obs_data_direct()  # 直接加载1度网格，避免插值到精细网格
            fcst_data = self.data_loader.load_forecast_data(model, self.var_type, leadtime)
            
            if obs_data is None or fcst_data is None:
                logger.warning(f"数据加载失败: {model} L{leadtime}")
                return None
            
            # 尝试加载ensemble数据（稍后对齐）
            ensemble_data = None
            try:
                ensemble_data = self.data_loader.load_forecast_data_ensemble(
                    model,
                    self.var_type,
                    leadtime
                )
                if ensemble_data is not None:
                    logger.info(f"成功加载ensemble数据，成员数: {ensemble_data.number.size}")
                    logger.info(f"Ensemble原始shape: {ensemble_data.shape}")
            except Exception as e:
                logger.debug(f"Ensemble数据加载失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # 计算RMSE
            # 注意：climpred在处理大ensemble数据时会导致内存溢出，因此禁用
            # 使用标准RMSE计算方法
            if False:  # 禁用climpred以避免内存问题
                pass
            else:
                # 标准RMSE计算
                if season:
                    spatial_rmse = self.calculate_seasonal_rmse(obs_data, fcst_data, season)
                else:
                    spatial_rmse = self.calculate_spatial_rmse(obs_data, fcst_data)
            
            # *** 对齐ensemble到RMSE网格并计算spread ***
            spread_results = None
            if ensemble_data is not None and spatial_rmse is not None and not season:
                try:
                    logger.info("对齐ensemble到RMSE网格...")
                    
                    # 时间对齐：使用fcst_data的时间轴（已经对齐过）
                    common_times = fcst_data.time.to_index().intersection(ensemble_data.time.to_index())
                    if len(common_times) > 0:
                        ensemble_aligned = ensemble_data.sel(time=common_times)
                        logger.info(f"时间对齐后ensemble shape: {ensemble_aligned.shape}")
                        
                        # 空间对齐：插值到spatial_rmse的网格（已对齐的观测网格）
                        if not np.array_equal(ensemble_aligned.lat.values, spatial_rmse.lat.values) or \
                           not np.array_equal(ensemble_aligned.lon.values, spatial_rmse.lon.values):
                            logger.info(f"空间对齐：将ensemble插值到RMSE网格...")
                            logger.info(f"  Ensemble网格: lat={len(ensemble_aligned.lat)}, lon={len(ensemble_aligned.lon)}")
                            logger.info(f"  RMSE网格: lat={len(spatial_rmse.lat)}, lon={len(spatial_rmse.lon)}")
                            
                            # *** 插值到1度观测网格 ***
                            # 计算预期内存需求
                            n_time = len(ensemble_aligned.time)
                            n_member = len(ensemble_aligned.number)
                            total_size_gb = (n_time * n_member * len(spatial_rmse.lat) * len(spatial_rmse.lon) * 8) / (1024**3)
                            logger.info(f"预计插值后内存需求: {total_size_gb:.2f}GB")
                            
                            # 插值到1度观测网格（内存需求小，无需分块）
                            ensemble_aligned = ensemble_aligned.interp(
                                lat=spatial_rmse.lat,
                                lon=spatial_rmse.lon,
                                method='linear'
                            )
                            
                            logger.info(f"空间对齐后ensemble shape: {ensemble_aligned.shape}")
                        else:
                            logger.info("Ensemble已在RMSE网格上，无需插值")
                        
                        # 使用对齐后的ensemble计算spread
                        spread_results = self.calculate_ensemble_spread(ensemble_aligned)
                        logger.info(f"Spread计算完成，shape: {spread_results['total_spread'].shape}")
                    else:
                        logger.warning("Ensemble时间对齐失败：没有共同时间点")
                        
                except Exception as e:
                    logger.warning(f"Ensemble对齐和Spread计算失败: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # 如果有spread_results，计算spread-error ratio
            if spread_results is not None and spatial_rmse is not None and not season:
                try:
                    total_spread = spread_results['total_spread']
                    
                    # *** 验证网格一致性 ***
                    # Ensemble已经对齐到观测网格，spread和RMSE应该在相同网格上
                    logger.info(f"验证网格一致性: spread{total_spread.shape} vs rmse{spatial_rmse.shape}")
                    if total_spread.shape != spatial_rmse.shape:
                        logger.error(f"网格对齐失败！Spread{total_spread.shape} != RMSE{spatial_rmse.shape}")
                        logger.error("这不应该发生 - ensemble应该已经对齐到观测网格")
                        raise ValueError(f"Grid mismatch after alignment: {total_spread.shape} vs {spatial_rmse.shape}")
                    
                    logger.info("网格验证通过，Spread和RMSE在相同网格上")
                    
                    # 直接计算ratio（数据已经对齐，无需插值）
                    spread_error_ratio = total_spread / (spatial_rmse + 1e-10)
                    spread_error_ratio.attrs = {
                        'long_name': 'Spread-Error Ratio',
                        'description': 'Ratio of ensemble spread to RMSE. Ideal value ~1.',
                        'units': 'dimensionless'
                    }
                    spread_results['spread_error_ratio'] = spread_error_ratio
                    logger.info(f"Spread-Error Ratio计算完成")
                    logger.info(f"Spread-Error Ratio范围: [{float(spread_error_ratio.min()):.4f}, {float(spread_error_ratio.max()):.4f}]")
                    logger.info(f"Spread-Error Ratio平均: {float(spread_error_ratio.mean()):.4f}")
                    
                except Exception as e:
                    logger.warning(f"计算spread-error ratio失败: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            temporal_rmse = self.calculate_temporal_rmse(obs_data, fcst_data)
            
            # 保存结果
            results = {}
            if spatial_rmse is not None:
                spatial_path = get_rmse_spatial_data_path(self.var_type, model, leadtime, season)
                spatial_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 确保变量名为rmse，以便绘图模块正确读取
                spatial_rmse.name = 'rmse'
                spatial_rmse.to_netcdf(spatial_path)
                results['spatial_path'] = spatial_path
                logger.info(f"空间RMSE已保存: {spatial_path}")
            
            if temporal_rmse is not None:
                temporal_path = get_rmse_temporal_data_path(self.var_type, model, leadtime, season)
                temporal_path.parent.mkdir(parents=True, exist_ok=True)
                temporal_rmse.to_csv(temporal_path)
                results['temporal_path'] = temporal_path
                logger.info(f"时间RMSE已保存: {temporal_path}")

                # 额外导出：基于时间序列的年/季/月聚合（仅季节为None时，避免重复）
                try:
                    if season in (None, 'annual') and len(temporal_rmse) > 0:
                        da = xr.DataArray(temporal_rmse.values, coords={'time': pd.to_datetime(temporal_rmse.index)}, dims=['time'])
                        agg = compute_aggregates(da)
                        
                        # 年度 - 使用文件锁避免并行处理时覆盖
                        out_annual = get_aggregated_csv_path('rmse', self.var_type, leadtime, 'annual')
                        out_annual.parent.mkdir(parents=True, exist_ok=True)
                        lock_file = out_annual.with_suffix('.lock')
                        try:
                            with open(lock_file, 'w') as lock:
                                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                                try:
                                    if out_annual.exists():
                                        existing_df = pd.read_csv(out_annual, index_col=0)
                                        new_data = pd.Series({model: agg['annual']}, name='Annual').to_frame()
                                        # 避免重复添加同一模型
                                        if model not in existing_df.index:
                                            combined_df = pd.concat([existing_df, new_data])
                                            combined_df.to_csv(out_annual)
                                    else:
                                        pd.Series({model: agg['annual']}, name='Annual').to_frame().to_csv(out_annual)
                                finally:
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        except Exception as lock_e:
                            logger.warning(f"文件锁操作失败，使用直接写入: {lock_e}")
                            if out_annual.exists():
                                existing_df = pd.read_csv(out_annual, index_col=0)
                                if model not in existing_df.index:
                                    new_data = pd.Series({model: agg['annual']}, name='Annual').to_frame()
                                    combined_df = pd.concat([existing_df, new_data])
                                    combined_df.to_csv(out_annual)
                            else:
                                pd.Series({model: agg['annual']}, name='Annual').to_frame().to_csv(out_annual)
                        
                        # 季节 - 使用文件锁
                        out_seasonal = get_aggregated_csv_path('rmse', self.var_type, leadtime, 'seasonal')
                        lock_file = out_seasonal.with_suffix('.lock')
                        try:
                            with open(lock_file, 'w') as lock:
                                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                                try:
                                    if out_seasonal.exists():
                                        existing_df = pd.read_csv(out_seasonal, index_col=0)
                                        if model not in existing_df.index:
                                            new_data = pd.DataFrame({model: agg['seasonal']}).T
                                            combined_df = pd.concat([existing_df, new_data])
                                            combined_df.to_csv(out_seasonal)
                                    else:
                                        pd.DataFrame({model: agg['seasonal']}).T.to_csv(out_seasonal)
                                finally:
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        except Exception as lock_e:
                            logger.warning(f"文件锁操作失败，使用直接写入: {lock_e}")
                            if out_seasonal.exists():
                                existing_df = pd.read_csv(out_seasonal, index_col=0)
                                if model not in existing_df.index:
                                    new_data = pd.DataFrame({model: agg['seasonal']}).T
                                    combined_df = pd.concat([existing_df, new_data])
                                    combined_df.to_csv(out_seasonal)
                            else:
                                pd.DataFrame({model: agg['seasonal']}).T.to_csv(out_seasonal)
                        
                        # 月度 - 使用文件锁
                        out_monthly = get_aggregated_csv_path('rmse', self.var_type, leadtime, 'monthly')
                        lock_file = out_monthly.with_suffix('.lock')
                        try:
                            with open(lock_file, 'w') as lock:
                                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                                try:
                                    if out_monthly.exists():
                                        existing_df = pd.read_csv(out_monthly, index_col=0)
                                        if model not in existing_df.index:
                                            new_data = pd.DataFrame({model: agg['monthly']}).T
                                            combined_df = pd.concat([existing_df, new_data])
                                            combined_df.to_csv(out_monthly)
                                    else:
                                        pd.DataFrame({model: agg['monthly']}).T.to_csv(out_monthly)
                                finally:
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        except Exception as lock_e:
                            logger.warning(f"文件锁操作失败，使用直接写入: {lock_e}")
                            if out_monthly.exists():
                                existing_df = pd.read_csv(out_monthly, index_col=0)
                                if model not in existing_df.index:
                                    new_data = pd.DataFrame({model: agg['monthly']}).T
                                    combined_df = pd.concat([existing_df, new_data])
                                    combined_df.to_csv(out_monthly)
                            else:
                                pd.DataFrame({model: agg['monthly']}).T.to_csv(out_monthly)
                except Exception as e:
                    logger.debug(f"导出RMSE聚合CSV失败: {e}")

            # 额外：保存空间场RMSE时间序列
            try:
                rmse_ts, n_ts, time_coord = self.calculate_spatial_field_rmse_ts(obs_data, fcst_data)
                if len(rmse_ts) > 0 and np.any(~np.isnan(rmse_ts)) and season in (None, 'annual'):
                    ts_path = get_spatial_rmse_ts_path(self.var_type, model, leadtime)
                    ts_path.parent.mkdir(parents=True, exist_ok=True)
                    ts_ds = xr.Dataset({
                        'spatial_rmse': xr.DataArray(rmse_ts, dims=['time'], coords={'time': time_coord},
                            attrs={'description': '逐时间点空间场RMSE', 'units': ''}),
                        'sample_size': xr.DataArray(n_ts, dims=['time'], coords={'time': time_coord},
                            attrs={'description': '有效格点数', 'units': 'count'}),
                    })
                    ts_ds.attrs.update({'lead_month': leadtime, 'model': model, 'variable': self.var_type,
                                        'analysis_type': 'spatial_field_rmse_ts'})
                    ts_ds.to_netcdf(ts_path)
                    logger.info(f"空间RMSE时间序列已保存: {ts_path}")
            except Exception as e:
                logger.debug(f"空间RMSE时间序列保存失败: {e}")
            
            # 保存ensemble spread结果（仅在非季节模式下）
            if spread_results is not None and not season:
                try:
                    output_dir = Path("/sas12t1/ffyan/output/rmse_analysis/spread")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存total spread (空间分布)
                    spread_spatial_path = output_dir / f"spread_spatial_{model}_L{leadtime}_{self.var_type}.nc"
                    spread_results['total_spread'].to_netcdf(spread_spatial_path)
                    results['spread_spatial_path'] = spread_spatial_path
                    logger.info(f"Ensemble spread空间分布已保存: {spread_spatial_path}")
                    
                    # 保存temporal spread (时间序列)
                    if 'spread_temporal' in spread_results and spread_results['spread_temporal'] is not None:
                        spread_temporal_path = output_dir / f"spread_temporal_{model}_L{leadtime}_{self.var_type}.csv"
                        spread_ts = spread_results['spread_temporal'].to_series()
                        spread_ts.to_csv(spread_temporal_path)
                        results['spread_temporal_path'] = spread_temporal_path
                        logger.info(f"Ensemble spread时间序列已保存: {spread_temporal_path}")
                    
                    # 如果有spread-error ratio，也保存
                    if 'spread_error_ratio' in spread_results:
                        ratio_path = output_dir / f"spread_error_ratio_{model}_L{leadtime}_{self.var_type}.nc"
                        spread_results['spread_error_ratio'].to_netcdf(ratio_path)
                        results['spread_error_ratio_path'] = ratio_path
                        logger.info(f"Spread-Error Ratio已保存: {ratio_path}")
                    
                except Exception as e:
                    logger.warning(f"保存spread结果失败: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"处理 {model} L{leadtime} 失败: {e}")
            return None
    
    def run_analysis(self, models: List[str] = None, leadtimes: List[int] = None, 
                    seasons: List[str] = None, parallel: bool = True, n_jobs: int = None):
        """运行RMSE分析"""
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        seasons = seasons or [None]  # None表示所有月份
        
        logger.info(f"开始RMSE分析: {len(models)} 模型, {len(leadtimes)} 提前期")
        
        # 准备任务列表
        tasks = []
        for model in models:
            for leadtime in leadtimes:
                for season in seasons:
                    tasks.append((model, leadtime, season))
        
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
                        executor.submit(self.process_model_leadtime, model, leadtime, season): (model, leadtime, season)
                        for model, leadtime, season in tasks
                    }
                    
                    completed = 0
                    failed = 0
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result(timeout=900)  # 15分钟超时，减少反复重试
                            completed += 1
                            logger.info(f"完成 {completed}/{len(tasks)}: {task}")
                        except Exception as e:
                            failed += 1
                            logger.error(f"任务失败 {task}: {e}")
                            
                            # 记录错误类型
                            if "allocate" in str(e).lower() or "memory" in str(e).lower():
                                logger.warning("检测到内存相关错误，但系统内存充足，可能是临时问题")
                    
                    logger.info(f"并行处理完成: {completed} 成功, {failed} 失败")
                    
            except Exception as e:
                logger.error(f"并行处理失败: {e}")
                logger.info("回退到串行处理...")
                parallel = False
        
        if not parallel or n_jobs == 1:
            # 串行处理（自动降并发）
            logger.info("使用串行处理（已自动降并发）")
            completed = 0
            failed = 0
            for i, (model, leadtime, season) in enumerate(tasks):
                try:
                    result = self.process_model_leadtime(model, leadtime, season)
                    if result:
                        completed += 1
                    else:
                        failed += 1
                    logger.info(f"完成 {i+1}/{len(tasks)}: {model} L{leadtime}")
                except Exception as e:
                    failed += 1
                    logger.error(f"任务失败 {model} L{leadtime}: {e}")
            
            logger.info(f"串行处理完成: {completed} 成功, {failed} 失败")
        
        logger.info("RMSE分析完成")
    
    def plot_spread_results(self, models: List[str] = None, leadtimes: List[int] = None, 
                            plot_monthly_contour: bool = True):
        """
        绘制spread相关图表
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        logger.info("="*60)
        logger.info(f"开始绘制Spread图表: {self.var_type}")
        logger.info("="*60)
        
        plotter = SpreadPlotter(self.var_type)
        
        # 1. Spread-Error Ratio空间分布图（L0和L3）
        target_leadtimes = [0, 3] if 0 in leadtimes and 3 in leadtimes else leadtimes[:2] if len(leadtimes) >= 2 else leadtimes
        logger.info("\n绘制Spread-Error Ratio空间分布图...")
        plotter.plot_spread_error_ratio_spatial_distribution(target_leadtimes, models)
        
        # 2. Spread vs RMSE散点图（L0和L3）
        logger.info("\n绘制Spread vs RMSE散点图...")
        plotter.plot_spread_vs_rmse_scatter(target_leadtimes, models)
        
        # 3. Spread-Error Ratio柱状图（L0和L3）
        logger.info("\n绘制Spread-Error Ratio柱状图...")
        plotter.plot_spread_error_ratio_bar(target_leadtimes, models)
        
        # 4. RMSE随leadtime变化的折线图
        logger.info("\n绘制RMSE随leadtime变化的折线图...")
        plotter.plot_rmse_leadtime_timeseries(models, leadtimes)
        
        # 5. RMSE逐月等高线图
        if plot_monthly_contour:
            logger.info("\n绘制RMSE逐月等高线图...")
            plotter.plot_rmse_monthly_contour(models, leadtimes)
        
        logger.info("\n" + "="*60)
        logger.info("Spread图表绘制完成")
        logger.info("="*60)


# 颜色配置（与block_bootstrap_score.py一致）
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



class SpreadPlotter:
    """Ensemble Spread可视化类（整合版）"""
    
    def __init__(self, var_type: str):
        """
        初始化SpreadPlotter
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
        """
        self.var_type = var_type
        self.spread_data_dir = Path("/sas12t1/ffyan/output/rmse_analysis/spread")
        self.rmse_data_dir = Path("/sas12t1/ffyan/outputdata/rmse_spatial")
        self.output_dir = Path("/sas12t1/ffyan/output/rmse_analysis/plots/spread")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化SpreadPlotter: {var_type}")
    
    def _load_models_data(self, leadtimes: List[int], models: List[str]) -> Dict[int, Dict[str, Dict]]:
        """
        加载多个leadtime的模型数据
        
        Args:
            leadtimes: 提前期列表
            models: 模型列表
            
        Returns:
            {leadtime: {model: {'ratio': ratio_data, 'spread': spread_data, 'rmse': rmse_data}}}
        """
        all_leadtimes_data = {}
        
        for leadtime in leadtimes:
            leadtime_data = {}
            for model in models:
                ratio_file = self.spread_data_dir / f"spread_error_ratio_{model}_L{leadtime}_{self.var_type}.nc"
                spread_file = self.spread_data_dir / f"spread_spatial_{model}_L{leadtime}_{self.var_type}.nc"
                
                # RMSE文件命名处理
                model_variants = [model, model.replace('-mon', '').replace('mon-', '')]
                rmse_file = None
                for model_variant in model_variants:
                    rmse_candidate = self.rmse_data_dir / self.var_type / f"rmse_spatial_{self.var_type}_{model_variant}_lead{leadtime}.nc"
                    if rmse_candidate.exists():
                        rmse_file = rmse_candidate
                        break
                
                if not (ratio_file.exists() and spread_file.exists() and rmse_file):
                    logger.debug(f"{model} L{leadtime}: 数据文件不完整，跳过")
                    continue
                
                try:
                    # 加载ratio数据
                    ratio_ds = xr.open_dataset(ratio_file)
                    var_candidates = ['t2m', 'tprate', 'tp', 'tm', 'temp', 'prec', '__xarray_dataarray_variable__']
                    ratio_data = None
                    for var in var_candidates:
                        if var in ratio_ds:
                            ratio_data = ratio_ds[var]
                            break
                    if ratio_data is None:
                        data_vars = [v for v in ratio_ds.data_vars]
                        if data_vars:
                            ratio_data = ratio_ds[data_vars[0]]
                        else:
                            ratio_ds.close()
                            continue
                    
                    # 加载spread数据
                    spread_ds = xr.open_dataset(spread_file)
                    spread_data = None
                    for var in var_candidates:
                        if var in spread_ds:
                            spread_data = spread_ds[var]
                            break
                    if spread_data is None:
                        data_vars = [v for v in spread_ds.data_vars]
                        if data_vars:
                            spread_data = spread_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            ratio_ds.close()
                            continue
                    
                    # 加载RMSE数据
                    rmse_ds = xr.open_dataset(rmse_file)
                    rmse_candidates = ['rmse'] + var_candidates
                    rmse_data = None
                    for var in rmse_candidates:
                        if var in rmse_ds:
                            rmse_data = rmse_ds[var]
                            break
                    if rmse_data is None:
                        data_vars = [v for v in rmse_ds.data_vars]
                        if data_vars:
                            rmse_data = rmse_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            ratio_ds.close()
                            rmse_ds.close()
                            continue
                    
                    # 验证网格一致性
                    if spread_data.shape != rmse_data.shape or ratio_data.shape != rmse_data.shape:
                        logger.warning(f"{model} L{leadtime}: 网格不匹配，跳过")
                        spread_ds.close()
                        ratio_ds.close()
                        rmse_ds.close()
                        continue
                    
                    leadtime_data[model] = {
                        'ratio': ratio_data,
                        'spread': spread_data,
                        'rmse': rmse_data
                    }
                    
                    spread_ds.close()
                    ratio_ds.close()
                    rmse_ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model} L{leadtime}数据失败: {e}")
                    continue
            
            if leadtime_data:
                all_leadtimes_data[leadtime] = leadtime_data
        
        return all_leadtimes_data
    
    def plot_spread_vs_leadtime(self, models: List[str], leadtimes: List[int]):
        """
        绘制ensemble spread随lead time变化的折线图（上下两个子图）
        
        Args:
            models: 模型列表
            leadtimes: 提前期列表
        """
        try:
            logger.info(f"绘制Spread vs Lead Time折线图: {self.var_type}")
            
            # 收集所有模型在不同leadtime的spread数据
            model_leadtime_spreads = {}
            
            for model in models:
                model_spreads = {}
                
                for leadtime in leadtimes:
                    # 查找spread spatial文件
                    spread_file = self.spread_data_dir / f"spread_spatial_{model}_L{leadtime}_{self.var_type}.nc"
                    
                    if not spread_file.exists():
                        logger.debug(f"Spread文件不存在，跳过: {spread_file}")
                        continue
                    
                    try:
                        # 加载NetCDF文件
                        ds = xr.open_dataset(spread_file)
                        
                        # 获取total_spread数据
                        var_candidates = ['t2m', 'tprate', 'tp', 'tm', 'temp', 'prec',
                                        '__xarray_dataarray_variable__']
                        spread_data = None
                        
                        for var in var_candidates:
                            if var in ds:
                                spread_data = ds[var]
                                break
                        
                        if spread_data is None:
                            data_vars = [v for v in ds.data_vars]
                            if data_vars:
                                spread_data = ds[data_vars[0]]
                            else:
                                logger.warning(f"在文件 {spread_file} 中未找到任何数据变量")
                                ds.close()
                                continue
                        
                        # 计算空间加权平均
                        weights = np.cos(np.deg2rad(spread_data.lat))
                        avg_spread = spread_data.weighted(weights).mean(dim=['lat', 'lon'])
                        
                        if not np.isnan(avg_spread.values):
                            model_spreads[leadtime] = float(avg_spread.values)
                        
                        ds.close()
                        
                    except Exception as e:
                        logger.error(f"处理文件 {spread_file} 时出错: {e}")
                        continue
                
                if model_spreads:
                    model_leadtime_spreads[model] = model_spreads
            
            if not model_leadtime_spreads:
                logger.warning(f"没有有效的spread数据用于折线图: {self.var_type}")
                return
            
            # 计算所有模型在每个leadtime的平均值
            mean_spreads = {}
            for leadtime in leadtimes:
                spreads_at_lt = []
                for model in model_leadtime_spreads:
                    if leadtime in model_leadtime_spreads[model]:
                        spreads_at_lt.append(model_leadtime_spreads[model][leadtime])
                if spreads_at_lt:
                    mean_spreads[leadtime] = np.mean(spreads_at_lt)
            
            # 创建上下两个子图，共享x轴
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})
            
            models_list = list(model_leadtime_spreads.keys())
            all_spread_values = []
            all_anomaly_values = []
            
            # 绘制每个模型的折线
            for i, model in enumerate(models_list):
                leadtime_list = sorted(model_leadtime_spreads[model].keys())
                spread_list = [model_leadtime_spreads[model][lt] for lt in leadtime_list]
                
                display_name = model.replace('-mon', '').replace('mon-', '')
                color = COLORS[i % len(COLORS)]
                
                # 上面的子图：原始spread值
                ax1.plot(leadtime_list, spread_list,
                        color=color, linewidth=2.5, linestyle='-',
                        label=display_name, alpha=0.85)
                
                all_spread_values.extend(spread_list)
                
                # 下面的子图：偏差
                anomaly_list = [model_leadtime_spreads[model][lt] - mean_spreads[lt]
                               for lt in leadtime_list if lt in mean_spreads]
                ax2.plot(leadtime_list, anomaly_list,
                        color=color, linewidth=2.5, linestyle='-', alpha=0.85)
                
                all_anomaly_values.extend(anomaly_list)
            
            # 设置上面子图的属性
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            ax1.set_ylabel(f'Ensemble Spread ({unit})', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_axisbelow(True)
            
            if all_spread_values:
                y_min = min(all_spread_values)
                y_max = max(all_spread_values)
                y_range = y_max - y_min
                ax1.set_ylim(max(0, y_min - y_range * 0.15), y_max + y_range * 0.15)
            
            ax1.legend(loc='best', framealpha=0.9, fontsize=10)
            
            # 设置下面子图的属性
            ax2.set_xlabel('Lead month', fontsize=12)
            ax2.set_ylabel(f'Anomaly ({unit})', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            if all_anomaly_values:
                abs_max = max(abs(min(all_anomaly_values)), abs(max(all_anomaly_values)))
                ax2.set_ylim(-abs_max * 1.15, abs_max * 1.15)
            
            ax2.set_xticks(leadtimes)
            plt.tight_layout()
            
            output_file = self.output_dir / f"spread_vs_leadtime_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spread vs Lead Time折线图已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制Spread vs Lead Time折线图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spread_error_ratio_spatial_distribution(self, leadtimes: List[int], models: List[str]):
        """
        绘制Spread-Error Ratio空间分布图
        分为上下两半（L0和L3），每个lead占2行，第1行留空+3模型，第2行4模型
        子图之间不留空隙，仅在最外围绘制经纬度标签和脊线，最下方绘制colorbar
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            models: 模型列表
        """
        try:
            logger.info(f"绘制Spread-Error Ratio空间分布图: L{leadtimes} {self.var_type}")
            
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
                        ratio = model_data['ratio']
                        ratio = ratio.where(np.isfinite(ratio))
                        valid_values = ratio.values[~np.isnan(ratio.values)]
                        all_ratio_values.extend(valid_values)
            
            # 计算统一范围
            if all_ratio_values:
                data_min = np.min(all_ratio_values)
                data_max = np.max(all_ratio_values)
                # 向下取整到小数点后1位
                ratio_min = np.floor(data_min * 10) / 10
                # 向上取整到小数点后1位
                ratio_max = np.ceil(data_max * 10) / 10
                logger.info(f"Spread-Error Ratio范围: [{data_min:.4f}, {data_max:.4f}], 显示范围: [{ratio_min:.1f}, {ratio_max:.1f}]")
            else:
                ratio_min, ratio_max = 0.5, 1.5
            
            cmap = 'Blues'  # 白色到蓝色渐变（低值白色，高值蓝色）
            
            # 计算布局
            n_leadtimes = len(leadtimes)
            n_cols = 4  # 固定4列：留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 基于第一个模型的数据计算经纬度边界
            first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
            sample_ratio = first_model_data['ratio']
            lon_centers = sample_ratio.lon.values if hasattr(sample_ratio, 'lon') else None
            lat_centers = sample_ratio.lat.values if hasattr(sample_ratio, 'lat') else None
            
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
                        model_lon_centers = model_data['ratio'].lon.values
                        model_lat_centers = model_data['ratio'].lat.values
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
                    
                    # 绘制ratio数据
                    ratio_data = model_data['ratio']
                    ratio_data = ratio_data.where(np.isfinite(ratio_data))
                    
                    im = ax_spatial.pcolormesh(model_lon_edges, model_lat_edges, ratio_data.values,
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmap, vmin=ratio_min, vmax=ratio_max, shading='flat')
                    
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
                        model_lon_centers = model_data['ratio'].lon.values
                        model_lat_centers = model_data['ratio'].lat.values
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
                    
                    ratio_data = model_data['ratio']
                    ratio_data = ratio_data.where(np.isfinite(ratio_data))
                    
                    im = ax_spatial.pcolormesh(model_lon_edges, model_lat_edges, ratio_data.values,
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmap, vmin=ratio_min, vmax=ratio_max, shading='flat')
                    
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
                cbar.set_label('Spread/RMSE Ratio', fontsize=11, labelpad=5)
                cbar.ax.tick_params(labelsize=9)
            
            # 保存图像
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.output_dir / f"spread_error_ratio_spatial_distribution_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"spread_error_ratio_spatial_distribution_{leadtimes_str}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            logger.info(f"Spread-Error Ratio空间分布图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制Spread-Error Ratio空间分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spread_error_ratio_spatial(self, leadtime: int, models: List[str]):
        """
        绘制Spread-Error Ratio的空间分布图（多模型）- 保留作为备用
        
        Args:
            leadtime: 提前期
            models: 模型列表
        """
        try:
            logger.info(f"绘制Spread-Error Ratio空间分布图: L{leadtime} {self.var_type}")
            
            # 收集所有模型的ratio数据
            all_models_ratio = {}
            
            for model in models:
                ratio_file = self.spread_data_dir / f"spread_error_ratio_{model}_L{leadtime}_{self.var_type}.nc"
                
                if not ratio_file.exists():
                    logger.debug(f"Ratio文件不存在，跳过: {ratio_file.name}")
                    continue
                
                try:
                    ds = xr.open_dataset(ratio_file)
                    
                    var_candidates = ['t2m', 'tprate', 'tp', 'tm', 'temp', 'prec',
                                    '__xarray_dataarray_variable__']
                    ratio_data = None
                    
                    for var in var_candidates:
                        if var in ds:
                            ratio_data = ds[var]
                            break
                    
                    if ratio_data is None:
                        data_vars = [v for v in ds.data_vars]
                        if data_vars:
                            ratio_data = ds[data_vars[0]]
                        else:
                            logger.warning(f"在文件 {ratio_file} 中未找到任何数据变量")
                            ds.close()
                            continue
                    
                    all_models_ratio[model] = ratio_data
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"处理文件 {ratio_file} 时出错: {e}")
                    continue
            
            if not all_models_ratio:
                logger.info(f"没有ratio数据用于空间分布图 L{leadtime}（请先运行计算）")
                return
            
            n_models = len(all_models_ratio)
            logger.info(f"准备绘制 {n_models} 个模型的Spread-Error Ratio空间分布图")
            
            # 计算所有模型的ratio范围（向上下取整）
            all_values = []
            for model, data in all_models_ratio.items():
                valid_values = data.values[~np.isnan(data.values)]
                all_values.extend(valid_values)
            
            if all_values:
                data_min = np.min(all_values)
                data_max = np.max(all_values)
                # 向下取整到小数点后1位
                vmin = np.floor(data_min * 10) / 10
                # 向上取整到小数点后1位
                vmax = np.ceil(data_max * 10) / 10
                logger.info(f"Spread-Error Ratio范围: [{data_min:.4f}, {data_max:.4f}], 显示范围: [{vmin:.1f}, {vmax:.1f}]")
            else:
                vmin, vmax = 0.5, 1.5
            
            # 动态计算子图布局
            if n_models <= 3:
                rows, cols = 1, n_models
            elif n_models <= 6:
                rows, cols = 2, 3
            else:
                rows = (n_models + 2) // 3
                cols = 3
            
            last_row_models = n_models - (rows - 1) * cols
            fig_width = 6 * cols
            fig_height = 5 * rows
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 使用单色渐变（白蓝渐变）
            cmap = 'Blues'  # 白色到蓝色渐变
            
            # 用于保存第一个pcolormesh对象以创建统一colorbar
            im_for_cbar = None
            
            # 为每个模型创建子图
            for i, (model, data) in enumerate(all_models_ratio.items()):
                if i < (rows - 1) * cols:
                    subplot_pos = i + 1
                else:
                    row = i // cols
                    col_in_row = i % cols
                    
                    if last_row_models < cols:
                        offset = (cols - last_row_models) // 2
                        col_in_row += offset
                    
                    subplot_pos = row * cols + col_in_row + 1
                
                ax = fig.add_subplot(rows, cols, subplot_pos, projection=ccrs.PlateCarree())
                
                ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.add_feature(cfeature.LAND, alpha=0.1)
                ax.add_feature(cfeature.OCEAN, alpha=0.1)
                
                if i >= (rows-1)*cols:
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                else:
                    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
                
                im = ax.pcolormesh(data.lon, data.lat, data,
                                  transform=ccrs.PlateCarree(),
                                  cmap=cmap, vmin=vmin, vmax=vmax)
                
                # 保存第一个im对象用于创建统一colorbar
                if im_for_cbar is None:
                    im_for_cbar = im
                
                display_name = model.replace('-mon', '').replace('mon-', '')
                ax.set_title(f'{display_name}', fontsize=12, pad=10)
            
            # 隐藏多余的子图
            if last_row_models < cols:
                last_row_start = (rows - 1) * cols + 1
                offset = (cols - last_row_models) // 2
                
                for i in range(offset):
                    ax = fig.add_subplot(rows, cols, last_row_start + i, projection=ccrs.PlateCarree())
                    ax.set_visible(False)
                
                for i in range(offset + last_row_models, cols):
                    ax = fig.add_subplot(rows, cols, last_row_start + i, projection=ccrs.PlateCarree())
                    ax.set_visible(False)
            
            # *** 添加统一的colorbar ***
            if im_for_cbar is not None:
                # 调整子图布局，为colorbar腾出空间
                fig.subplots_adjust(right=0.90)
                # 在整个图的右侧添加统一的colorbar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
                cbar.set_label('Spread/RMSE Ratio', fontsize=11, rotation=270, labelpad=20)
                cbar.ax.tick_params(labelsize=9)
            else:
                plt.tight_layout()
            
            output_file = self.output_dir / f"spread_error_ratio_spatial_L{leadtime}_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spread-Error Ratio空间分布图已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制Spread-Error Ratio空间分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spread_vs_rmse_scatter(self, leadtimes: List[int], models: List[str]):
        """
        绘制Spread vs RMSE散点图
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

            logger.info(f"绘制Spread vs RMSE散点图: L{leadtimes} {self.var_type}")

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
            all_rmse_vals = []
            all_spread_vals = []
            all_densities = []

            for leadtime in leadtimes:
                if leadtime in all_leadtimes_data:
                    for model_data in all_leadtimes_data[leadtime].values():
                        spread_flat = model_data['spread'].values.flatten()
                        rmse_flat = model_data['rmse'].values.flatten()
                        valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                        spread_valid = spread_flat[valid_mask]
                        rmse_valid = rmse_flat[valid_mask]
                        all_rmse_vals.extend(rmse_valid)
                        all_spread_vals.extend(spread_valid)

                        if len(spread_valid) > 100:
                            try:
                                xy = np.vstack([rmse_valid, spread_valid])
                                kde = gaussian_kde(xy)
                                z = kde(xy)
                                all_densities.extend(z)
                            except:
                                pass

            # 计算统一的坐标范围
            if all_rmse_vals and all_spread_vals:
                # X轴使用0-99%的数据范围
                scatter_x_min = max(0, np.percentile(all_rmse_vals, 0.5))
                scatter_x_max = np.percentile(all_rmse_vals, 99)
                
                # Y轴使用全部数据范围
                scatter_y_min = np.min(all_spread_vals)
                scatter_y_max = np.max(all_spread_vals)
                
                # 添加适当边距
                x_range = scatter_x_max - scatter_x_min
                y_range = scatter_y_max - scatter_y_min
                scatter_x_min = max(0, scatter_x_min - x_range * 0.02)
                scatter_x_max = scatter_x_max + x_range * 0.02
                scatter_y_min = max(0, scatter_y_min - y_range * 0.05)
                scatter_y_max = scatter_y_max + y_range * 0.05
            else:
                scatter_x_min, scatter_x_max = 0, 5
                scatter_y_min, scatter_y_max = 0, 5

            # 计算统一的密度范围
            if all_densities:
                raw_density_min = np.percentile(all_densities, 5)
                raw_density_max = np.percentile(all_densities, 95)
                density_min = 0.0
                density_max = 1.0
            else:
                density_min, density_max = 0.0, 1.0
                raw_density_min, raw_density_max = 0, 1

            logger.info(f"散点图RMSE范围: [{scatter_x_min:.2f}, {scatter_x_max:.2f}]")
            logger.info(f"散点图Spread范围: [{scatter_y_min:.2f}, {scatter_y_max:.2f}]")

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
                    spread_flat = model_data['spread'].values.flatten()
                    rmse_flat = model_data['rmse'].values.flatten()
                    valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                    spread_valid = spread_flat[valid_mask]
                    rmse_valid = rmse_flat[valid_mask]

                    if len(spread_valid) > 0:
                        # 分离X轴范围内外的点
                        in_range_mask = rmse_valid <= scatter_x_max
                        out_range_mask = rmse_valid > scatter_x_max
                        
                        rmse_in = rmse_valid[in_range_mask]
                        spread_in = spread_valid[in_range_mask]
                        rmse_out = rmse_valid[out_range_mask]
                        spread_out = spread_valid[out_range_mask]
                        
                        # 绘制范围内的点（使用密度着色）
                        if len(rmse_in) > 100:
                            try:
                                xy = np.vstack([rmse_in, spread_in])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)

                                scatter = ax.scatter(rmse_in, spread_in, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')

                                if scatter_for_cbar is None:
                                    scatter_for_cbar = scatter
                            except:
                                ax.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')
                        
                        # 在右侧边框标记超出范围的点（使用叉号）
                        if len(rmse_out) > 0:
                            ax.scatter([scatter_x_max] * len(rmse_out), spread_out, 
                                     marker='x', s=30, c='red', alpha=0.7, 
                                     linewidths=1.5, zorder=10)

                        # y=x参考线
                        ax.plot([scatter_x_min, scatter_x_max], 
                               [scatter_x_min, scatter_x_max],
                               'r--', linewidth=1.2, alpha=0.7)

                        # 拟合线
                        if len(rmse_in) > 2:
                            slope_all, intercept_all, _, _, _ = stats.linregress(rmse_in, spread_in)
                            x_fit = np.array([scatter_x_min, scatter_x_max])
                            y_fit_all = slope_all * x_fit + intercept_all
                            ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)

                            # Robust拟合线（残差IQR筛选）
                            if len(rmse_in) > 10:
                                residuals = spread_in - (slope_all * rmse_in + intercept_all)
                                q1, q3 = np.percentile(residuals, [25, 75])
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                mask_inliers = (residuals >= lower_bound) & (residuals <= upper_bound)
                                rmse_robust = rmse_in[mask_inliers]
                                spread_robust = spread_in[mask_inliers]

                                if len(rmse_robust) > 3:
                                    slope_robust, intercept_robust, _, _, _ = stats.linregress(rmse_robust, spread_robust)
                                    y_fit_robust = slope_robust * x_fit + intercept_robust
                                    ax.plot(x_fit, y_fit_robust, 'b-', linewidth=1.2, alpha=0.7)
                            
                            # 去除高RMSE值（10%）的拟合线
                            if len(rmse_in) > 10:
                                rmse_threshold = np.percentile(rmse_in, 90)
                                mask_low_rmse = rmse_in <= rmse_threshold
                                rmse_low = rmse_in[mask_low_rmse]
                                spread_low = spread_in[mask_low_rmse]
                                
                                if len(rmse_low) > 3:
                                    slope_low, intercept_low, _, _, _ = stats.linregress(rmse_low, spread_low)
                                    y_fit_low = slope_low * x_fit + intercept_low
                                    ax.plot(x_fit, y_fit_low, color='purple', linestyle='-', linewidth=1.2, alpha=0.7)

                    # 设置坐标范围（所有子图统一）
                    ax.set_xlim(scatter_x_min, scatter_x_max)
                    ax.set_ylim(scatter_y_min, scatter_y_max)

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
                    spread_flat = model_data['spread'].values.flatten()
                    rmse_flat = model_data['rmse'].values.flatten()
                    valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                    spread_valid = spread_flat[valid_mask]
                    rmse_valid = rmse_flat[valid_mask]

                    if len(spread_valid) > 0:
                        # 分离X轴范围内外的点
                        in_range_mask = rmse_valid <= scatter_x_max
                        out_range_mask = rmse_valid > scatter_x_max
                        
                        rmse_in = rmse_valid[in_range_mask]
                        spread_in = spread_valid[in_range_mask]
                        rmse_out = rmse_valid[out_range_mask]
                        spread_out = spread_valid[out_range_mask]
                        
                        # 绘制散点
                        if len(rmse_in) > 100:
                            try:
                                xy = np.vstack([rmse_in, spread_in])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)

                                scatter = ax.scatter(rmse_in, spread_in, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')
                            except:
                                ax.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')
                        
                        # 在右侧边框标记超出范围的点
                        if len(rmse_out) > 0:
                            ax.scatter([scatter_x_max] * len(rmse_out), spread_out, 
                                     marker='x', s=30, c='red', alpha=0.7, 
                                     linewidths=1.5, zorder=10)

                        # y=x参考线
                        ax.plot([scatter_x_min, scatter_x_max], 
                               [scatter_x_min, scatter_x_max],
                               'r--', linewidth=1.2, alpha=0.7)

                        # 拟合线
                        if len(rmse_in) > 2:
                            slope_all, intercept_all, _, _, _ = stats.linregress(rmse_in, spread_in)
                            x_fit = np.array([scatter_x_min, scatter_x_max])
                            y_fit_all = slope_all * x_fit + intercept_all
                            ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)

                            # Robust拟合线
                            if len(rmse_in) > 10:
                                residuals = spread_in - (slope_all * rmse_in + intercept_all)
                                lower_bound = np.percentile(residuals, 5)
                                upper_bound = np.percentile(residuals, 95)
                                mask_inliers = (residuals >= lower_bound) & (residuals <= upper_bound)
                                rmse_robust = rmse_in[mask_inliers]
                                spread_robust = spread_in[mask_inliers]

                                if len(rmse_robust) > 3:
                                    slope_robust, intercept_robust, _, _, _ = stats.linregress(rmse_robust, spread_robust)
                                    y_fit_robust = slope_robust * x_fit + intercept_robust
                                    ax.plot(x_fit, y_fit_robust, 'b-', linewidth=1.2, alpha=0.7)
                            
                            # Low RMSE拟合线
                            if len(rmse_in) > 10:
                                rmse_threshold = np.percentile(rmse_in, 90)
                                mask_low_rmse = rmse_in <= rmse_threshold
                                rmse_low = rmse_in[mask_low_rmse]
                                spread_low = spread_in[mask_low_rmse]
                                
                                if len(rmse_low) > 3:
                                    slope_low, intercept_low, _, _, _ = stats.linregress(rmse_low, spread_low)
                                    y_fit_low = slope_low * x_fit + intercept_low
                                    ax.plot(x_fit, y_fit_low, color='purple', linestyle='-', linewidth=1.2, alpha=0.7)

                    # 设置坐标范围（所有子图统一）
                    ax.set_xlim(scatter_x_min, scatter_x_max)
                    ax.set_ylim(scatter_y_min, scatter_y_max)

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
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            fig.text(0.025, 0.63, f'Spread ({unit})', va='center', rotation='vertical', fontsize=11)
            fig.text(0.5, 0.07, f'RMSE ({unit})', ha='center', va='center', fontsize=11)

            # 图例（左侧）和密度colorbar（右侧）
            legend_elements = [
                Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='ideal'),
                Line2D([0], [0], color='#ff7f0e', linestyle='-', linewidth=2, label='All data'),
                Line2D([0], [0], color='b', linestyle='-', linewidth=2, label='Robust (±5%)'),
                Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='Low RMSE (90%)'),
            ]
            fig.legend(handles=legend_elements, loc='lower left', ncol=4, frameon=False,
                       bbox_to_anchor=(0.05, 0.05))

            # 密度colorbar
            if scatter_for_cbar is not None:
                cbar_density_ax = fig.add_axes([0.65, 0.05, 0.25, 0.02])
                cbar_density = fig.colorbar(scatter_for_cbar, cax=cbar_density_ax, orientation='horizontal')
                cbar_density.set_label('Point Density', fontsize=10)
                cbar_density.ax.tick_params(labelsize=9)

            # 保存图像
            leadtimes_str = '_'.join([f'L{lt}' for lt in leadtimes])
            output_file_png = self.output_dir / f"spread_vs_rmse_scatter_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"spread_vs_rmse_scatter_{leadtimes_str}_{self.var_type}.pdf"

            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()

            logger.info(f"Spread vs RMSE散点图已保存: {output_file_png}")

        except Exception as e:
            logger.error(f"绘制Spread vs RMSE散点图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spread_vs_rmse_scatter_all_models(self, leadtime: int, models: List[str]):
        """
        绘制所有模型的Spread vs RMSE散点图（多子图布局，与空间分布图一致）- 保留作为备用
        
        Args:
            leadtime: 提前期
            models: 模型列表
        """
        try:
            logger.info(f"绘制Spread vs RMSE散点图（多模型）: L{leadtime} {self.var_type}")
            
            # 收集所有模型的数据
            all_models_data = {}
            
            for model in models:
                # 加载spread数据
                spread_file = self.spread_data_dir / f"spread_spatial_{model}_L{leadtime}_{self.var_type}.nc"
                
                # RMSE文件命名可能去掉了"mon"，尝试多种命名
                model_variants = [
                    model,  # 原名
                    model.replace('-mon', '').replace('mon-', ''),  # 去掉mon
                ]
                
                rmse_file = None
                for model_variant in model_variants:
                    rmse_candidate = self.rmse_data_dir / self.var_type / f"rmse_spatial_{self.var_type}_{model_variant}_lead{leadtime}.nc"
                    if rmse_candidate.exists():
                        rmse_file = rmse_candidate
                        break
                
                if not spread_file.exists() or rmse_file is None:
                    logger.info(f"数据文件不存在，跳过: {model} L{leadtime} (spread:{spread_file.exists()}, rmse:{rmse_file is not None})")
                    continue
                
                try:
                    # 加载spread数据
                    spread_ds = xr.open_dataset(spread_file)
                    var_candidates = ['t2m', 'tprate', 'tp', 'tm', 'temp', 'prec', '__xarray_dataarray_variable__']
                    spread_data = None
                    for var in var_candidates:
                        if var in spread_ds:
                            spread_data = spread_ds[var]
                            break
                    if spread_data is None:
                        data_vars = [v for v in spread_ds.data_vars]
                        if data_vars:
                            spread_data = spread_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            continue
                    
                    # 加载RMSE数据
                    rmse_ds = xr.open_dataset(rmse_file)
                    rmse_candidates = ['rmse', 't2m', 'tprate', 'tp', 'tm', 'temp', 'prec', '__xarray_dataarray_variable__']
                    rmse_data = None
                    for var in rmse_candidates:
                        if var in rmse_ds:
                            rmse_data = rmse_ds[var]
                            break
                    if rmse_data is None:
                        data_vars = [v for v in rmse_ds.data_vars]
                        if data_vars:
                            rmse_data = rmse_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            rmse_ds.close()
                            continue
                    
                    # 网格验证
                    if spread_data.shape != rmse_data.shape:
                        logger.info(f"{model}: 网格不匹配，跳过 - Spread{spread_data.shape} != RMSE{rmse_data.shape}")
                        spread_ds.close()
                        rmse_ds.close()
                        continue
                    
                    logger.debug(f"{model}: 数据加载成功，shape={spread_data.shape}")
                    
                    # 存储数据
                    all_models_data[model] = {
                        'spread': spread_data,
                        'rmse': rmse_data
                    }
                    
                    spread_ds.close()
                    rmse_ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model}数据失败: {e}")
                    continue
            
            if not all_models_data:
                logger.info(f"没有有效的数据用于散点图 L{leadtime}")
                return
            
            n_models = len(all_models_data)
            logger.info(f"准备绘制 {n_models} 个模型的散点图")
            
            # 动态计算子图布局（与空间分布图一致）
            if n_models <= 3:
                rows, cols = 1, n_models
            elif n_models <= 6:
                rows, cols = 2, 3
            else:
                rows = (n_models + 2) // 3
                cols = 3
            
            last_row_models = n_models - (rows - 1) * cols
            fig_width = 6 * cols
            fig_height = 3.5 * rows
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            
            # *** 第一遍：计算所有模型的统一密度范围和坐标范围 ***
            all_densities = []
            all_rmse_values = []
            all_spread_values = []
            
            for model, data in all_models_data.items():
                spread_values = data['spread'].values.flatten()
                rmse_values = data['rmse'].values.flatten()
                valid_mask = ~(np.isnan(spread_values) | np.isnan(rmse_values))
                spread_valid = spread_values[valid_mask]
                rmse_valid = rmse_values[valid_mask]
                
                # 收集所有有效值用于计算统一坐标范围
                all_rmse_values.extend(rmse_valid)
                all_spread_values.extend(spread_valid)
                
                if len(spread_valid) > 100:
                    try:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([rmse_valid, spread_valid])
                        z = gaussian_kde(xy)(xy)
                        all_densities.extend(z)
                    except:
                        pass
            
            # 计算统一的密度范围
            if all_densities:
                density_min = np.min(all_densities)
                density_max = np.max(all_densities)
                logger.info(f"统一密度范围: [{density_min:.4e}, {density_max:.4e}]")
            else:
                density_min, density_max = None, None
            
            # 计算统一的坐标范围
            if all_rmse_values and all_spread_values:
                axis_min = min(np.min(all_rmse_values), np.min(all_spread_values))
                axis_max = max(np.max(all_rmse_values), np.max(all_spread_values))
                # 添加5%的边距
                axis_range = axis_max - axis_min
                axis_min = max(0, axis_min - axis_range * 0.05)
                axis_max = axis_max + axis_range * 0.05
                logger.info(f"统一坐标范围: [{axis_min:.4f}, {axis_max:.4f}]")
            else:
                axis_min, axis_max = None, None
            
            # *** 第二遍：绘制所有模型的散点图 ***
            scatter_for_cbar = None  # 用于创建统一colorbar
            legend_handles = None  # 用于创建统一图例
            legend_labels = None
            
            for i, (model, data) in enumerate(all_models_data.items()):
                # 计算子图位置
                if i < (rows - 1) * cols:
                    subplot_pos = i + 1
                else:
                    row = i // cols
                    col_in_row = i % cols
                    if last_row_models < cols:
                        offset = (cols - last_row_models) // 2
                        col_in_row += offset
                    subplot_pos = row * cols + col_in_row + 1
                
                ax = fig.add_subplot(rows, cols, subplot_pos)
                
                # 展平数据
                spread_values = data['spread'].values.flatten()
                rmse_values = data['rmse'].values.flatten()
                
                # 移除NaN值
                valid_mask = ~(np.isnan(spread_values) | np.isnan(rmse_values))
                spread_valid = spread_values[valid_mask]
                rmse_valid = rmse_values[valid_mask]
                
                if len(spread_valid) == 0:
                    continue
                
                # 绘制散点（使用统一密度范围）
                try:
                    if len(spread_valid) > 100 and density_min is not None:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([rmse_valid, spread_valid])
                        z = gaussian_kde(xy)(xy)
                        idx = z.argsort()
                        
                        # 使用统一的密度范围
                        scatter = ax.scatter(rmse_valid[idx], spread_valid[idx], c=z[idx],
                                           s=15, alpha=0.6, cmap='viridis', 
                                           vmin=density_min, vmax=density_max,
                                           edgecolors='none')
                        
                        # 保存第一个scatter对象用于创建统一colorbar
                        if scatter_for_cbar is None:
                            scatter_for_cbar = scatter
                    else:
                        # 数据点少时使用简单散点
                        ax.scatter(rmse_valid, spread_valid, s=15, alpha=0.5,
                                  color='steelblue', edgecolors='none')
                except Exception as e:
                    logger.debug(f"{model}密度计算失败，使用简单散点: {e}")
                    ax.scatter(rmse_valid, spread_valid, s=15, alpha=0.5,
                              color='steelblue', edgecolors='none')
                
                # 添加y=x参考线（使用统一坐标范围）
                if axis_min is not None and axis_max is not None:
                    min_val, max_val = axis_min, axis_max
                else:
                    max_val = max(np.max(rmse_valid), np.max(spread_valid))
                    min_val = min(np.min(rmse_valid), np.min(spread_valid))
                
                line_ideal = ax.plot([min_val, max_val], [min_val, max_val],
                                    'r--', linewidth=1.5, alpha=0.7, label='y=x')
                
                # 计算相关系数和拟合线（全部数据）
                correlation = np.corrcoef(rmse_valid, spread_valid)[0, 1]
                r_squared = correlation ** 2
                slope, intercept, _, _, _ = stats.linregress(rmse_valid, spread_valid)
                
                x_fit = np.array([min_val, max_val])
                y_fit = slope * x_fit + intercept
                line_all = ax.plot(x_fit, y_fit, 'g-', linewidth=1.5, alpha=0.7,
                                  label=f'All (R²={r_squared:.2f})')
                
                # *** 去除离群值后的robust拟合 ***
                # 使用IQR方法识别离群值
                q1_x, q3_x = np.percentile(rmse_valid, [25, 75])
                q1_y, q3_y = np.percentile(spread_valid, [25, 75])
                iqr_x = q3_x - q1_x
                iqr_y = q3_y - q1_y
                
                # 定义离群值范围（1.5倍IQR）
                lower_x, upper_x = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x
                lower_y, upper_y = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y
                
                # 过滤离群值
                outlier_mask = (rmse_valid >= lower_x) & (rmse_valid <= upper_x) & \
                               (spread_valid >= lower_y) & (spread_valid <= upper_y)
                rmse_robust = rmse_valid[outlier_mask]
                spread_robust = spread_valid[outlier_mask]
                
                # 添加统计信息（包含robust拟合信息）
                stats_text = f'N={len(spread_valid)}\n'
                stats_text += f'All: R²={r_squared:.2f}, S={slope:.2f}\n'
                
                line_robust = None
                if len(rmse_robust) > 10:  # 确保有足够数据点
                    slope_robust, intercept_robust, _, _, _ = stats.linregress(rmse_robust, spread_robust)
                    r_robust = np.corrcoef(rmse_robust, spread_robust)[0, 1]
                    r2_robust = r_robust ** 2
                    
                    y_fit_robust = slope_robust * x_fit + intercept_robust
                    line_robust = ax.plot(x_fit, y_fit_robust, 'b-', linewidth=1.5, alpha=0.7,
                                         label=f'Robust (R²={r2_robust:.2f})')
                    
                    outlier_pct = (1 - len(rmse_robust) / len(rmse_valid)) * 100
                    stats_text += f'Robust: R²={r2_robust:.2f}, S={slope_robust:.2f}'
                    logger.debug(f"{model}: 去除{outlier_pct:.1f}%离群值，Slope {slope:.2f}→{slope_robust:.2f}")
                
                # 保存第一个子图的图例句柄（用于创建统一图例）
                if legend_handles is None:
                    legend_handles = line_ideal + line_all
                    if line_robust is not None:
                        legend_handles += line_robust
                    legend_labels = [h.get_label() for h in legend_handles]
                
                # 添加统计信息框（不添加图例）
                ax.text(0.05, 0.95, stats_text,
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                       fontsize=8)
                
                # 设置轴标签（只在边缘子图）
                if i >= (rows-1)*cols:  # 最后一行
                    ax.set_xlabel(f'RMSE ({unit})', fontsize=10)
                if i % cols == 0:  # 第一列
                    ax.set_ylabel(f'Spread ({unit})', fontsize=10)
                
                # 标题
                display_name = model.replace('-mon', '').replace('mon-', '')
                ax.set_title(f'{display_name}', fontsize=11, pad=8)
                
                # 设置统一的坐标范围
                if axis_min is not None and axis_max is not None:
                    ax.set_xlim(axis_min, axis_max)
                    ax.set_ylim(axis_min, axis_max)
                
                # 网格和纵横比
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
                ax.set_aspect('equal', adjustable='box')
            
            # *** 添加统一的colorbar和图例 ***
            if scatter_for_cbar is not None:
                # 调整子图位置，为colorbar和图例腾出空间
                fig.subplots_adjust(right=0.92, top=0.92)
                
                # 在图的右侧添加统一的colorbar
                cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                cbar = fig.colorbar(scatter_for_cbar, cax=cbar_ax)
                cbar.set_label('Point Density', fontsize=11, rotation=270, labelpad=20)
                cbar.ax.tick_params(labelsize=9)
            
            # *** 添加统一的图例（在图的上方中央）***
            if legend_handles is not None and legend_labels is not None:
                # 简化图例标签（去掉R²值，因为每个模型不同）
                simple_labels = ['y=x (ideal)', 'All data fit', 'Robust fit (outliers removed)']
                fig.legend(legend_handles, simple_labels,
                          loc='upper center', bbox_to_anchor=(0.5, 0.98),
                          ncol=3, framealpha=0.9, fontsize=10)
            
            # 隐藏多余的子图
            if last_row_models < cols:
                last_row_start = (rows - 1) * cols + 1
                offset = (cols - last_row_models) // 2
                
                for i in range(offset):
                    ax = fig.add_subplot(rows, cols, last_row_start + i)
                    ax.set_visible(False)
                
                for i in range(offset + last_row_models, cols):
                    ax = fig.add_subplot(rows, cols, last_row_start + i)
                    ax.set_visible(False)
            
            # 调整布局（如果有统一colorbar或图例，已经调整过subplots_adjust）
            if scatter_for_cbar is None and legend_handles is None:
                plt.tight_layout()
            
            output_file = self.output_dir / f"spread_vs_rmse_scatter_L{leadtime}_{self.var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spread vs RMSE散点图（多模型）已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制Spread vs RMSE散点图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_spread_error_ratio_bar(self, leadtimes: List[int], models: List[str]):
        """
        绘制Spread-Error Ratio柱状图
        L0和L3绘制为上下两张子图，横轴为Model，子图之间不留间隙，共用横轴，纵轴为指标
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制Spread-Error Ratio柱状图: L{leadtimes} {self.var_type}")
            
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
            
            # 为每个leadtime计算每个模型的平均Spread-Error Ratio
            leadtime_ratios = {}  # {leadtime: {model: ratio}}
            
            for leadtime in leadtimes:
                if leadtime not in all_leadtimes_data:
                    continue
                leadtime_ratios[leadtime] = {}
                for idx, model in enumerate(model_names):
                    if model in all_leadtimes_data[leadtime]:
                        model_data = all_leadtimes_data[leadtime][model]
                        ratio_values = model_data['ratio'].values
                        valid_ratios = ratio_values[np.isfinite(ratio_values)]
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
                
                ax.set_ylabel('Spread/RMSE Ratio', fontsize=11)
                ax.set_ylim(y_min, y_max)
                ax.tick_params(axis='y', labelsize=9)
                
                # 添加y=1参考线（理想情况：Spread=RMSE）
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
            output_file_png = self.output_dir / f"spread_error_ratio_bar_{leadtimes_str}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"spread_error_ratio_bar_{leadtimes_str}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spread-Error Ratio柱状图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制Spread-Error Ratio柱状图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _load_monthly_rmse_data(self, models: List[str], leadtimes: List[int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        读取月度RMSE数据（从rmse_summary文件夹中读取）
        
        数据来源：读取rmse_summary文件夹中的聚合CSV文件
        文件格式：rmse_summary/{var_type}/rmse_monthly_L{leadtime}.csv
        文件格式：第一行为列名（Jan, Feb, ..., Dec），后续行为模型名称和对应的月度RMSE值
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
        
        Returns:
            Dict[str, Dict[int, Dict[int, float]]]: {model: {leadtime: {month: rmse_value}}}
        """
        monthly_rmse_data = {}
        summary_data_dir = Path("/sas12t1/ffyan/outputdata/rmse_summary") / self.var_type
        
        # 月份名称映射
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_name_to_num = {name: i+1 for i, name in enumerate(month_names)}
        
        for leadtime in leadtimes:
            try:
                # 读取汇总文件
                summary_file = summary_data_dir / f"rmse_monthly_L{leadtime}.csv"
                
                if not summary_file.exists():
                    logger.debug(f"月度RMSE汇总文件不存在: {summary_file}")
                    continue
                
                # 读取CSV文件
                df = pd.read_csv(summary_file, index_col=0)
                
                logger.debug(f"读取月度RMSE汇总文件: {summary_file}")
                logger.debug(f"文件中的模型: {list(df.index)}")
                logger.debug(f"文件列名: {list(df.columns)}")
                
                # 处理每个模型
                for model in models:
                    # 处理模型名称变体
                    model_variants = [
                        model,  # 原始名称
                        model.replace('-mon', '').replace('mon-', ''),  # 移除-mon
                        model.replace('DWD-mon-21', 'DWD-21'),  # DWD特殊处理
                        model.replace('ECMWF-51-mon', 'ECMWF-51'),  # ECMWF特殊处理
                    ]
                    # 去重
                    model_variants = list(dict.fromkeys(model_variants))
                    
                    model_found = False
                    for model_variant in model_variants:
                        if model_variant in df.index:
                            # 提取该模型的月度数据
                            monthly_values = {}
                            row_data = df.loc[model_variant]
                            
                            # 处理列名（可能是月份名称或数字）
                            for col_name in df.columns:
                                month_num = None
                                
                                # 尝试匹配月份名称
                                if col_name in month_name_to_num:
                                    month_num = month_name_to_num[col_name]
                                # 尝试匹配数字格式（1-12）
                                elif col_name.isdigit():
                                    month_num = int(col_name)
                                    if month_num < 1 or month_num > 12:
                                        continue
                                # 尝试匹配Month_X格式
                                elif col_name.startswith('Month_'):
                                    try:
                                        month_num = int(col_name.split('_')[1])
                                        if month_num < 1 or month_num > 12:
                                            continue
                                    except:
                                        continue
                                
                                if month_num is not None:
                                    val = row_data[col_name]
                                    if pd.notna(val):
                                        monthly_values[month_num] = float(val)
                                    else:
                                        monthly_values[month_num] = np.nan
                            
                            # 确保所有月份都有值（即使是NaN）
                            for month in range(1, 13):
                                if month not in monthly_values:
                                    monthly_values[month] = np.nan
                            
                            # 初始化模型数据结构
                            if model not in monthly_rmse_data:
                                monthly_rmse_data[model] = {}
                            
                            monthly_rmse_data[model][leadtime] = monthly_values
                            model_found = True
                            valid_months = sum(1 for v in monthly_values.values() if np.isfinite(v))
                            logger.debug(f"成功读取模型 {model} L{leadtime} 的月度数据，有效月份数: {valid_months}")
                            break
                    
                    if not model_found:
                        logger.debug(f"模型 {model} 在汇总文件中未找到: {summary_file}")
                        
            except Exception as e:
                logger.error(f"读取月度RMSE汇总数据失败 L{leadtime}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # 统计加载结果
        for model in models:
            if model in monthly_rmse_data:
                logger.info(f"成功加载模型 {model} 的月度RMSE数据，包含 {len(monthly_rmse_data[model])} 个leadtime")
            else:
                logger.warning(f"模型 {model} 没有加载到任何月度RMSE数据")
        
        logger.info(f"总共加载了 {len(monthly_rmse_data)} 个模型的月度RMSE数据")
        return monthly_rmse_data
    
    def plot_rmse_monthly_contour(self, models: List[str] = None, leadtimes: List[int] = None):
        """
        绘制RMSE逐月等高线图
        横轴为月份（1-12），纵轴为leadtime
        布局：第一行留空+3个模型，第二行4个模型
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        try:
            logger.info(f"开始绘制RMSE逐月等高线图: {self.var_type}")
            
            # 加载月度RMSE数据
            monthly_rmse_data = self._load_monthly_rmse_data(models, leadtimes)
            
            if not monthly_rmse_data:
                logger.warning("没有找到任何月度RMSE数据")
                logger.warning("提示：月度RMSE数据需要先运行计算模式生成。")
                logger.warning("  请运行脚本时不使用--plot-only参数，或确保月度CSV文件已存在。")
                return
            
            # 准备模型列表（按顺序）
            model_names = [m for m in models if m in monthly_rmse_data]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 对每个模型，按月份和leadtime组织数据
            # 数据结构：{model: np.array(shape=(12, n_leadtimes))}
            model_contour_data = {}
            all_values = []
            
            for model in model_names:
                # 创建矩阵 (month, leadtime)
                months = list(range(1, 13))
                contour_matrix = np.full((len(months), len(leadtimes)), np.nan)
                
                for mi, month in enumerate(months):
                    for li, leadtime in enumerate(leadtimes):
                        if leadtime in monthly_rmse_data[model]:
                            if month in monthly_rmse_data[model][leadtime]:
                                val = monthly_rmse_data[model][leadtime][month]
                                if np.isfinite(val):
                                    contour_matrix[mi, li] = val
                                    all_values.append(val)
                
                model_contour_data[model] = contour_matrix
            
            if not all_values:
                logger.warning("没有有效的RMSE数据用于绘制等高线图")
                return
            
            # 计算统一的colorbar范围
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
            
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            logger.info(f"RMSE等高线图数据范围: [{vmin:.3f}, {vmax:.3f}] {unit}")
            
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
                    
                    # 准备数据（转置：横轴为month，纵轴为leadtime）
                    contour_data = model_contour_data[model].T
                    
                    # 绘制等高线图（不填充颜色）
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
                    
                    X, Y = np.meshgrid(months, leadtimes)
                    contours = ax.contour(X, Y, contour_data, levels=levels, colors='black', 
                                         linewidths=1.2, alpha=0.8)
                    # 只标注部分等高线，避免过于密集（自动选择标注位置）
                    ax.clabel(contours, inline=True, fontsize=14, fmt='%.2f', 
                             manual=False, colors='black')
                    
                    # 设置坐标轴
                    ax.set_xticks(months)
                    ax.set_yticks(leadtimes)
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_ylabel('Lead Time', fontsize=16)
                    if col_idx == 1:  # 第一列显示x轴标签
                        ax.set_xlabel('Month', fontsize=16)
                    
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
                    
                    # 准备数据（转置：横轴为month，纵轴为leadtime）
                    contour_data = model_contour_data[model].T
                    
                    # 绘制等高线图（不填充颜色）
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
                    
                    X, Y = np.meshgrid(months, leadtimes)
                    contours = ax.contour(X, Y, contour_data, levels=levels, colors='black', 
                                         linewidths=1.2, alpha=0.8)
                    # 只标注部分等高线，避免过于密集（自动选择标注位置）
                    ax.clabel(contours, inline=True, fontsize=14, fmt='%.2f', 
                             manual=False, colors='black')
                    
                    # 设置坐标轴
                    ax.set_xticks(months)
                    ax.set_yticks(leadtimes)
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_ylabel('Lead Time', fontsize=16)
                    ax.set_xlabel('Month', fontsize=16)
                    
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
            output_file_png = self.output_dir / f"rmse_monthly_contour_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"rmse_monthly_contour_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"RMSE逐月等高线图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制RMSE逐月等高线图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_rmse_leadtime_timeseries(self, models: List[str] = None, leadtimes: List[int] = None):
        """
        绘制RMSE随leadtime变化的折线图
        读取所有模型和leadtime的空间RMSE数据，计算空间加权平均，绘制折线图
        
        Args:
            models: 模型列表
            leadtimes: leadtime列表
        """
        models = models or MODEL_LIST
        leadtimes = leadtimes or LEADTIMES
        
        try:
            logger.info(f"开始绘制RMSE随leadtime变化的折线图...")
            
            # 收集所有模型和leadtime的RMSE数据
            model_leadtime_rmse = {}
            
            for model in models:
                model_rmse = {}
                
                for leadtime in leadtimes:
                    # 查找RMSE spatial文件
                    model_variants = [model, model.replace('-mon', '').replace('mon-', '')]
                    rmse_file = None
                    for model_variant in model_variants:
                        rmse_candidate = self.rmse_data_dir / self.var_type / f"rmse_spatial_{self.var_type}_{model_variant}_lead{leadtime}.nc"
                        if rmse_candidate.exists():
                            rmse_file = rmse_candidate
                            break
                    
                    if not rmse_file:
                        logger.debug(f"RMSE文件不存在，跳过: {model} L{leadtime}")
                        continue
                    
                    try:
                        # 加载NetCDF文件
                        ds = xr.open_dataset(rmse_file)
                        
                        # 获取RMSE数据
                        var_candidates = ['rmse', 't2m', 'tprate', 'tp', 'tm', 'temp', 'prec',
                                        '__xarray_dataarray_variable__']
                        rmse_data = None
                        
                        for var in var_candidates:
                            if var in ds:
                                rmse_data = ds[var]
                                break
                        
                        if rmse_data is None:
                            data_vars = [v for v in ds.data_vars]
                            if data_vars:
                                rmse_data = ds[data_vars[0]]
                            else:
                                logger.warning(f"在文件 {rmse_file} 中未找到任何数据变量")
                                ds.close()
                                continue
                        
                        # 计算空间加权平均（使用纬度权重）
                        weights = np.cos(np.deg2rad(rmse_data.lat))
                        avg_rmse = rmse_data.weighted(weights).mean(dim=['lat', 'lon'])
                        
                        if not np.isnan(avg_rmse.values):
                            model_rmse[leadtime] = float(avg_rmse.values)
                        
                        ds.close()
                        
                    except Exception as e:
                        logger.error(f"处理文件 {rmse_file} 时出错: {e}")
                        continue
                
                if model_rmse:
                    model_leadtime_rmse[model] = model_rmse
            
            if not model_leadtime_rmse:
                logger.warning(f"没有有效的RMSE数据用于折线图: {self.var_type}")
                return
            
            # 创建图形
            fig_height = 6.0
            fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
            
            # 设置颜色映射（使用全局COLORS配置）
            models_list = list(model_leadtime_rmse.keys())
            color_map = {model: COLORS[i % len(COLORS)] for i, model in enumerate(models_list)}
            
            legend_handles = []
            legend_labels = []
            all_y_vals = []
            
            # 绘制每个模型的曲线
            for model in models_list:
                leadtime_list = sorted(model_leadtime_rmse[model].keys())
                rmse_list = [model_leadtime_rmse[model][lt] for lt in leadtime_list]
                
                display_name = model.replace('-mon', '').replace('mon-', '')
                color = color_map[model]
                
                line, = ax.plot(
                    leadtime_list, rmse_list,
                    marker='o', linewidth=2.0, markersize=6,
                    color=color, label=display_name, alpha=0.85
                )
                
                legend_handles.append(line)
                legend_labels.append(display_name)
                all_y_vals.extend(rmse_list)
            
            # 设置坐标轴
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            ax.set_xlabel('Lead Time', fontsize=12)
            ax.set_ylabel(f'RMSE ({unit})', fontsize=12)
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
                    y_min = max(0, y_min - margin)
                    y_max = y_max + margin
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
            output_file_png = self.output_dir / f"rmse_leadtime_timeseries_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"rmse_leadtime_timeseries_{self.var_type}.pdf"
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"RMSE随leadtime折线图已保存: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制RMSE随leadtime折线图失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_combined_spatial_scatter(self, leadtime: int, models: List[str]):
        """
        绘制空间分布+散点组合图
        每个模型包含上方的空间分布图和下方的散点图
        空白位置显示图例和统计表格
        
        Args:
            leadtime: 提前期
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制空间分布+散点组合图: L{leadtime} {self.var_type}")
            
            # 收集所有模型的数据
            all_models_data = {}
            
            for model in models:
                ratio_file = self.spread_data_dir / f"spread_error_ratio_{model}_L{leadtime}_{self.var_type}.nc"
                spread_file = self.spread_data_dir / f"spread_spatial_{model}_L{leadtime}_{self.var_type}.nc"
                
                # RMSE文件命名处理
                model_variants = [model, model.replace('-mon', '').replace('mon-', '')]
                rmse_file = None
                for model_variant in model_variants:
                    rmse_candidate = self.rmse_data_dir / self.var_type / f"rmse_spatial_{self.var_type}_{model_variant}_lead{leadtime}.nc"
                    if rmse_candidate.exists():
                        rmse_file = rmse_candidate
                        break
                
                if not (ratio_file.exists() and spread_file.exists() and rmse_file):
                    logger.debug(f"{model}: 数据文件不完整，跳过")
                    continue
                
                try:
                    # 加载ratio数据
                    ratio_ds = xr.open_dataset(ratio_file)
                    var_candidates = ['t2m', 'tprate', 'tp', 'tm', 'temp', 'prec', '__xarray_dataarray_variable__']
                    ratio_data = None
                    for var in var_candidates:
                        if var in ratio_ds:
                            ratio_data = ratio_ds[var]
                            break
                    if ratio_data is None:
                        data_vars = [v for v in ratio_ds.data_vars]
                        if data_vars:
                            ratio_data = ratio_ds[data_vars[0]]
                        else:
                            ratio_ds.close()
                            continue
                    
                    # 加载spread数据
                    spread_ds = xr.open_dataset(spread_file)
                    spread_data = None
                    for var in var_candidates:
                        if var in spread_ds:
                            spread_data = spread_ds[var]
                            break
                    if spread_data is None:
                        data_vars = [v for v in spread_ds.data_vars]
                        if data_vars:
                            spread_data = spread_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            ratio_ds.close()
                            continue
                    
                    # 加载RMSE数据
                    rmse_ds = xr.open_dataset(rmse_file)
                    rmse_candidates = ['rmse'] + var_candidates
                    rmse_data = None
                    for var in rmse_candidates:
                        if var in rmse_ds:
                            rmse_data = rmse_ds[var]
                            break
                    if rmse_data is None:
                        data_vars = [v for v in rmse_ds.data_vars]
                        if data_vars:
                            rmse_data = rmse_ds[data_vars[0]]
                        else:
                            spread_ds.close()
                            ratio_ds.close()
                            rmse_ds.close()
                            continue
                    
                    # 验证网格一致性
                    if spread_data.shape != rmse_data.shape or ratio_data.shape != rmse_data.shape:
                        logger.warning(f"{model}: 网格不匹配，跳过")
                        spread_ds.close()
                        ratio_ds.close()
                        rmse_ds.close()
                        continue
                    
                    all_models_data[model] = {
                        'ratio': ratio_data,
                        'spread': spread_data,
                        'rmse': rmse_data
                    }
                    
                    spread_ds.close()
                    ratio_ds.close()
                    rmse_ds.close()
                    
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
            fig_width = 6 * cols  # 不需要额外空间
            fig_height = fig_width  # 正方形，1:1的纵横比
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 创建GridSpec：每个模型占2行（空间图3份高度，散点图1份高度，纵横比约1:3）
            n_grid_rows = rows_of_models * 2
            height_ratios = [3, 1] * rows_of_models  # 散点图高度从2改为1，使其更扁平
            gs = GridSpec(n_grid_rows, cols, figure=fig,
                         height_ratios=height_ratios,
                         hspace=0.18, wspace=0.13,  # 缩小间隙
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # 计算统一范围
            # Ratio colorbar范围：使用实际数据范围并向上下取整
            all_ratio_values = []
            for model, data in all_models_data.items():
                valid_values = data['ratio'].values[~np.isnan(data['ratio'].values)]
                all_ratio_values.extend(valid_values)
            
            if all_ratio_values:
                data_min = np.min(all_ratio_values)
                data_max = np.max(all_ratio_values)
                
                # 始终使用实际数据的取整范围
                ratio_min = np.floor(data_min * 10) / 10
                ratio_max = np.ceil(data_max * 10) / 10
            else:
                ratio_min, ratio_max = 0.0, 1.0
            
            # 散点图坐标范围
            all_rmse_vals = []
            all_spread_vals = []
            for model, data in all_models_data.items():
                spread_flat = data['spread'].values.flatten()
                rmse_flat = data['rmse'].values.flatten()
                valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                all_rmse_vals.extend(rmse_flat[valid_mask])
                all_spread_vals.extend(spread_flat[valid_mask])
            
            if all_rmse_vals and all_spread_vals:
                # X轴使用0-99%的数据范围
                scatter_x_min = max(0, np.percentile(all_rmse_vals, 0.5))  # 接近最小值但避免极端值
                scatter_x_max = np.percentile(all_rmse_vals, 99)  # 99%分位数
                
                # Y轴使用全部数据范围
                scatter_y_min = np.min(all_spread_vals)
                scatter_y_max = np.max(all_spread_vals)
                
                # 添加适当边距
                x_range = scatter_x_max - scatter_x_min
                y_range = scatter_y_max - scatter_y_min
                scatter_x_min = max(0, scatter_x_min - x_range * 0.02)
                scatter_x_max = scatter_x_max + x_range * 0.02
                scatter_y_min = max(0, scatter_y_min - y_range * 0.05)
                scatter_y_max = scatter_y_max + y_range * 0.05
            else:
                scatter_x_min, scatter_x_max = 0, 5
                scatter_y_min, scatter_y_max = 0, 5
            
            logger.info(f"Ratio范围: [{ratio_min:.1f}, {ratio_max:.1f}]")
            logger.info(f"散点X范围: [{scatter_x_min:.2f}, {scatter_x_max:.2f}]")
            logger.info(f"散点Y范围: [{scatter_y_min:.2f}, {scatter_y_max:.2f}]")
            
            unit = '°C' if self.var_type == 'temp' else 'mm/day'
            cmap = 'Blues'  # 白色到蓝色渐变（低值白色，高值蓝色）
            
            # 用于统一colorbar
            im_for_cbar = None
            scatter_for_cbar = None
            
            # 用于统计表格
            stats_data = []
            
            # 预计算所有模型的散点密度范围（用于统一colorbar）
            all_densities = []
            for model, data in all_models_data.items():
                spread_flat = data['spread'].values.flatten()
                rmse_flat = data['rmse'].values.flatten()
                valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                spread_valid = spread_flat[valid_mask]
                rmse_valid = rmse_flat[valid_mask]
                
                if len(spread_valid) > 100:  # 足够的点才计算密度
                    try:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([rmse_valid, spread_valid])
                        kde = gaussian_kde(xy)
                        z = kde(xy)
                        all_densities.extend(z)
                    except:
                        pass
            
            # 计算统一的密度范围，归一化到[0,1]
            if all_densities:
                # 先计算实际密度范围（去除极值）
                raw_density_min = np.percentile(all_densities, 5)
                raw_density_max = np.percentile(all_densities, 95)
                
                # 归一化到[0,1]范围
                density_min = 0.0
                density_max = 1.0
                
                logger.info(f"原始密度范围: [{raw_density_min:.2e}, {raw_density_max:.2e}]")
                logger.info(f"归一化密度范围: [0.0, 1.0]")
            else:
                density_min, density_max = 0.0, 1.0
                raw_density_min, raw_density_max = 0, 1
            
            # 绘制每个模型的组合子图
            for i, (model, data) in enumerate(all_models_data.items()):
                # 计算位置，最后一个模型如果单独一行则放在左侧
                row_pair = i // cols
                col = i % cols  # 所有模型都按顺序排列，不特殊处理
                
                # ===== 上半部分：空间分布图 =====
                ax_spatial = fig.add_subplot(gs[row_pair*2, col], projection=ccrs.PlateCarree())
                
                # 找到有效数据边界
                lat_min, lat_max, lon_min, lon_max = find_valid_data_bounds(
                    data['ratio'].values, data['ratio'].lat.values, data['ratio'].lon.values
                )
                
                # 设置地图范围（使用有效数据边界）
                ax_spatial.set_extent([lon_min-1, lon_max+1, lat_min-1, lat_max+1], crs=ccrs.PlateCarree())
                ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                
                # 绘制经纬度网格（每个子图都绘制和显示标签）
                gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False  # 不显示顶部标签
                gl.right_labels = False  # 不显示右侧标签
                gl.left_labels = True  # 所有子图都显示左侧标签
                gl.bottom_labels = True  # 所有子图都显示底部标签
                
                # 绘制数据
                im = ax_spatial.pcolormesh(data['ratio'].lon, data['ratio'].lat, data['ratio'],
                                          transform=ccrs.PlateCarree(),
                                          cmap=cmap, vmin=ratio_min, vmax=ratio_max)
                
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
                spread_flat = data['spread'].values.flatten()
                rmse_flat = data['rmse'].values.flatten()
                valid_mask = ~(np.isnan(spread_flat) | np.isnan(rmse_flat))
                spread_valid = spread_flat[valid_mask]
                rmse_valid = rmse_flat[valid_mask]
                
                if len(spread_valid) > 0:
                    # 分离X轴范围内外的点
                    in_range_mask = rmse_valid <= scatter_x_max
                    out_range_mask = rmse_valid > scatter_x_max
                    
                    rmse_in = rmse_valid[in_range_mask]
                    spread_in = spread_valid[in_range_mask]
                    rmse_out = rmse_valid[out_range_mask]
                    spread_out = spread_valid[out_range_mask]
                    
                    # 绘制范围内的点（使用密度着色）
                    if len(rmse_in) > 100:
                        try:
                            from scipy.stats import gaussian_kde
                            xy = np.vstack([rmse_in, spread_in])
                            kde = gaussian_kde(xy)
                            z_raw = kde(xy)
                            
                            # 归一化密度值到[0,1]范围
                            z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                            z = np.clip(z, 0, 1)  # 确保在[0,1]范围内
                            
                            # 绘制密度着色的散点
                            scatter = ax_scatter.scatter(rmse_in, spread_in, c=z, 
                                                        s=8, alpha=0.6,
                                                        cmap='viridis', 
                                                        vmin=density_min, vmax=density_max,
                                                        edgecolors='none')
                            
                            if scatter_for_cbar is None:
                                scatter_for_cbar = scatter
                        except:
                            # 如果密度计算失败，使用简单散点
                            ax_scatter.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                              color='steelblue', edgecolors='none')
                    else:
                        # 点太少，使用简单散点
                        ax_scatter.scatter(rmse_in, spread_in, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                    
                    # 在右侧边框标记超出范围的点（使用叉号）
                    if len(rmse_out) > 0:
                        ax_scatter.scatter([scatter_x_max] * len(rmse_out), spread_out, 
                                         marker='x', s=30, c='red', alpha=0.7, 
                                         linewidths=1.5, zorder=10)
                    
                    # y=x参考线
                    ax_scatter.plot([scatter_x_min, scatter_x_max], 
                                   [scatter_x_min, scatter_x_max],
                                   'r--', linewidth=1.2, alpha=0.7)
                    
                    # 所有数据的拟合线（只使用范围内的点）
                    if len(rmse_in) > 2:  # 降低阈值，至少3个点就可以拟合
                        slope_all, intercept_all, _, _, _ = stats.linregress(rmse_in, spread_in)
                        r_all = np.corrcoef(rmse_in, spread_in)[0, 1]
                        ax_scatter.plot([scatter_x_min, scatter_x_max],
                                       [slope_all*scatter_x_min + intercept_all, slope_all*scatter_x_max + intercept_all],
                                       color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)  # 橙色
                        
                        # Robust拟合线（残差IQR筛选）
                        if len(rmse_in) > 10:
                            residuals = spread_in - (slope_all * rmse_in + intercept_all)
                            q1, q3 = np.percentile(residuals, [25, 75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            mask_inliers = (residuals >= lower_bound) & (residuals <= upper_bound)
                            rmse_robust = rmse_in[mask_inliers]
                            spread_robust = spread_in[mask_inliers]
                            
                            if len(rmse_robust) > 3:  # 确保有足够的点进行拟合
                                slope_robust, intercept_robust, _, _, _ = stats.linregress(rmse_robust, spread_robust)
                                r_robust = np.corrcoef(rmse_robust, spread_robust)[0, 1]
                                ax_scatter.plot([scatter_x_min, scatter_x_max],
                                               [slope_robust*scatter_x_min + intercept_robust, slope_robust*scatter_x_max + intercept_robust],
                                               'b-', linewidth=1.2, alpha=0.7)
                            else:
                                slope_robust = slope_all
                                r_robust = r_all
                        else:
                            # 点数不够多，不进行robust拟合
                            slope_robust = slope_all
                            r_robust = r_all
                        
                        # 去除高RMSE值（10%）的拟合线
                        if len(rmse_in) > 10:
                            # 只保留RMSE在0-90%分位数范围内的点
                            rmse_threshold = np.percentile(rmse_in, 90)
                            mask_low_rmse = rmse_in <= rmse_threshold
                            rmse_low = rmse_in[mask_low_rmse]
                            spread_low = spread_in[mask_low_rmse]
                            
                            if len(rmse_low) > 3:
                                slope_low, intercept_low, _, _, _ = stats.linregress(rmse_low, spread_low)
                                r_low = np.corrcoef(rmse_low, spread_low)[0, 1]
                                ax_scatter.plot([scatter_x_min, scatter_x_max],
                                               [slope_low*scatter_x_min + intercept_low, slope_low*scatter_x_max + intercept_low],
                                               color='purple', linestyle='-', linewidth=1.2, alpha=0.7)
                            else:
                                slope_low = slope_all
                                r_low = r_all
                        else:
                            slope_low = slope_all
                            r_low = r_all
                    else:
                        # 点数太少，不绘制任何拟合线
                        slope_all = slope_robust = slope_low = 1.0
                        r_all = r_robust = r_low = 0.0
                    
                    # 收集统计数据（包含所有点的信息）
                    stats_data.append({
                        'Model': display_name,
                        'N': len(spread_valid),
                        'N_in_range': len(rmse_in),
                        'N_outliers': len(rmse_out),
                        'R²_All': f'{r_all**2:.2f}',
                        'Slope_All': f'{slope_all:.2f}',
                        'R²_Robust': f'{r_robust**2:.2f}',
                        'Slope_Robust': f'{slope_robust:.2f}',
                        'R²_LowRMSE': f'{r_low**2:.2f}',
                        'Slope_LowRMSE': f'{slope_low:.2f}',
                        'Ratio': f'{np.mean(spread_valid/rmse_valid):.2f}'
                    })
                
                # 设置坐标范围
                ax_scatter.set_xlim(scatter_x_min, scatter_x_max)
                ax_scatter.set_ylim(scatter_y_min, scatter_y_max)
                
                # 轴标签（所有子图都显示）
                ax_scatter.set_xlabel(f'RMSE ({unit})', fontsize=9)
                ax_scatter.set_ylabel(f'Spread ({unit})', fontsize=9)
                
                ax_scatter.grid(True, alpha=0.3, linestyle='--')
                ax_scatter.set_axisbelow(True)
            
            # ===== 保存统计数据为CSV =====
            if stats_data:
                import pandas as pd
                df = pd.DataFrame(stats_data)
                csv_file = self.output_dir / f"combined_statistics_L{leadtime}_{self.var_type}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"统计数据已保存到: {csv_file}")
            
            # ===== 空白位置：Ratio柱状图（中间，占整个组合高度）=====
            if n_models % 3 != 0:  # 有空白位置
                legend_row = rows_of_models - 1
                bar_ax = fig.add_subplot(gs[legend_row*2:legend_row*2+2, 1])  # 中间位置，占用空间图+散点图+间隙的完整高度
                
                # 计算每个模型的平均ratio，保持原顺序
                model_labels = []
                model_ratios = []
                for idx, (model, data) in enumerate(all_models_data.items()):
                    ratio_values = data['ratio'].values
                    valid_ratios = ratio_values[~np.isnan(ratio_values)]
                    if len(valid_ratios) > 0:
                        avg_ratio = np.mean(valid_ratios)
                        subplot_label = chr(97 + idx)  # a, b, c, d...
                        model_labels.append(subplot_label)
                        model_ratios.append(avg_ratio)
                
                # 绘制柱状图（统一颜色）
                x_pos = np.arange(len(model_labels))
                bars = bar_ax.bar(x_pos, model_ratios, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # 设置标签（不需要标题）
                bar_ax.set_xlabel('Model', fontsize=10)
                bar_ax.set_ylabel('Mean Spread/RMSE Ratio', fontsize=10)
                bar_ax.set_xticks(x_pos)
                bar_ax.set_xticklabels(model_labels, fontsize=9)  # 使用a,b,c...标签，不旋转
                bar_ax.tick_params(axis='y', labelsize=9)
                
                # 添加网格
                bar_ax.grid(True, axis='y', alpha=0.3, linestyle='--')
                bar_ax.set_axisbelow(True)
                
                # 设置Y轴范围（完全基于实际数据分布）
                data_min = min(model_ratios)
                data_max = max(model_ratios)
                data_range = data_max - data_min
                
                # 添加10%的相对边距以避免数据贴边
                margin = max(data_range * 0.1, 0.05)  # 至少0.05的边距
                y_min = data_min - margin
                y_max = data_max + margin
                
                # 确保Y轴从0开始（如果数据接近0）
                if y_min < 0.1:
                    y_min = 0
                    
                bar_ax.set_ylim(y_min, y_max)
            
            # ===== 空白位置：Colorbar+图例（右下）=====
            if n_models % 3 != 0:  # 有空白位置
                legend_row = rows_of_models - 1
                legend_ax = fig.add_subplot(gs[legend_row*2:legend_row*2+2, 2])  # 右侧位置
                legend_ax.axis('off')
                
                # 上半部分：两个Colorbar（对应空间分布图高度，高度比=3）
                # Colorbar宽度比子图略窄（0.6而非0.7）
                
                # 添加Ratio Colorbar（上部，与空间图对应）
                if im_for_cbar is not None:
                    cbar_ratio_ax = legend_ax.inset_axes([0.2, 0.62, 0.6, 0.06])  # 上移
                    cbar_ratio = fig.colorbar(im_for_cbar, cax=cbar_ratio_ax, orientation='horizontal')
                    cbar_ratio.set_label('Spread/RMSE Ratio', fontsize=9)
                    cbar_ratio.ax.tick_params(labelsize=8)
                
                # 添加Density Colorbar（中部偏上，与空间图对应）
                if scatter_for_cbar is not None:
                    cbar_density_ax = legend_ax.inset_axes([0.2, 0.42, 0.6, 0.06])
                    cbar_density = fig.colorbar(scatter_for_cbar, cax=cbar_density_ax, orientation='horizontal')
                    cbar_density.set_label('Point Density', fontsize=9)
                    cbar_density.ax.tick_params(labelsize=8)
                
                # 下半部分：线型图例（对应散点图高度，高度比=1，占用更多空间）
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='r', linestyle='--', linewidth=4, label='ideal'),
                    Line2D([0], [0], color='#ff7f0e', linestyle='-', linewidth=4, label='All data'),
                    Line2D([0], [0], color='b', linestyle='-', linewidth=4, label='Robust (±5%)'),
                    Line2D([0], [0], color='purple', linestyle='-', linewidth=4, label='Low RMSE (90%)'),
                ]
                
                # 图例放在下方，与散点图对应，横向排列，每行2个
                legend = legend_ax.legend(handles=legend_elements, loc='center',
                                        ncol=2,  # 每行2个元素
                                        frameon=True, fontsize=11,  # 字体增大
                                        bbox_to_anchor=(0.5, 0.12),  # 下移到底部区域
                                        framealpha=0.9, edgecolor='gray',
                                        handlelength=3.5,  # 线的长度
                                        handletextpad=1.0,  # 线和文本之间的间距
                                        columnspacing=2.0)  # 列之间的间距
            
            # 保存图像（PNG + PDF）
            output_file_png = self.output_dir / f"combined_spatial_scatter_L{leadtime}_{self.var_type}.png"
            output_file_pdf = self.output_dir / f"combined_spatial_scatter_L{leadtime}_{self.var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"组合图已保存到: {output_file_png}")
            logger.info(f"矢量图已保存到: {output_file_pdf}")
            
        except Exception as e:
            logger.error(f"绘制组合图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """主函数"""
    parser = create_parser(
        description="简化的RMSE分析",
        include_seasons=True,
        var_default=None  # 允许不指定，默认处理temp和prec
    )
    args = parser.parse_args()
    
    # 解析参数
    models = parse_models(args.models, MODEL_LIST) if args.models else MODEL_LIST
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_types = parse_vars(args.var) if args.var else ["temp", "prec"]
    
    # 确定要处理的季节
    if args.all_seasons:
        seasons = ["DJF", "MAM", "JJA", "SON", "annual", None]  # None表示月度数据
    elif args.seasons:
        seasons = args.seasons
        # 如果指定了annual，转换为None（表示年平均）
        if "annual" in seasons:
            seasons = [s if s != "annual" else None for s in seasons]
    else:
        # 默认处理年平均和所有季节
        seasons = ["DJF", "MAM", "JJA", "SON", None]  # None表示年平均
    
    # 标准化绘图参数
    normalize_plot_args(args)
    
    logger.info(f"将处理变量: {var_types}")
    logger.info(f"将处理模型: {models}")
    logger.info(f"将处理提前期: {leadtimes}")
    logger.info(f"将处理季节: {[s if s else 'annual' for s in seasons]}")
    logger.info(f"绘图模式: no_plot={args.no_plot}, plot_only={args.plot_only}")
    
    # 为每个变量运行分析
    for var_type in var_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始处理变量: {var_type.upper()}")
        logger.info(f"{'='*50}")
        
        # 创建分析器
        analyzer = SimplifiedRMSE(var_type)
        
        # 运行RMSE分析计算（除非是仅绘图模式）
        if not args.plot_only:
            # 标准化并行参数
            parallel = normalize_parallel_args(args)
            
            analyzer.run_analysis(
                models=models,
                leadtimes=leadtimes,
                seasons=seasons,
                parallel=parallel,
                n_jobs=args.n_jobs
            )
            logger.info(f"{var_type.upper()} 计算完成")
        
        # 绘制spread图表（默认绘图，除非指定了--no-plot）
        if not args.no_plot:
            try:
                analyzer.plot_spread_results(
                    models=models,
                    leadtimes=leadtimes
                )
            except Exception as e:
                logger.error(f"绘制spread图表时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*50}")
    logger.info("所有任务完成！")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

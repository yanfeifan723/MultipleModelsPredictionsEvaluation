"""
主分析器类
整合所有分析功能，提供统一的接口
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

from .config.settings import (
    OBS_DIR, FORECAST_DIR, OUTPUT_DIR, MODELS, VAR_CONFIG,
    get_eof_config, get_correlation_config, get_rmse_config, 
    get_spectrum_config, get_plotting_config
)
from .core.eof_analysis import EOFAnalyzer
from .core.correlation_analysis import CorrelationAnalyzer
from .core.rmse_analysis import RMSEAnalyzer
from .core.spectrum_analysis import SpectrumAnalyzer
from .core.common_eof_analysis import CommonEOFAnalyzer
from .core.crpss_analysis import CRPSSAnalyzer
from .utils.data_utils import find_variable, dynamic_coord_sel, validate_data
from .config.settings import setup_logging, create_output_dirs
from .plotting import *

class ClimateAnalyzer:
    """气候分析主类"""
    
    def __init__(self, 
                 obs_dir: str = OBS_DIR,
                 forecast_dir: str = FORECAST_DIR,
                 output_dir: str = OUTPUT_DIR,
                 log_file: Optional[str] = None):
        """
        初始化分析器
        
        Args:
            obs_dir: 观测数据目录
            forecast_dir: 预测数据目录
            output_dir: 输出目录
            log_file: 日志文件路径
        """
        self.obs_dir = Path(obs_dir)
        self.forecast_dir = Path(forecast_dir)
        self.output_dir = Path(output_dir)
        
        # 设置日志
        if log_file is None:
            log_file = self.output_dir / "climate_analysis.log"
        self.logger = setup_logging(log_file)
        
        # 创建输出目录
        self.output_dirs = create_output_dirs(self.output_dir)
        
        # 检查环境
        try:
            from .config.settings import check_environment
            if not check_environment():
                self.logger.warning("环境检查未通过，某些功能可能无法正常工作")
        except ImportError:
            self.logger.warning("无法导入环境检查函数")
        
        # 初始化分析器
        self.eof_analyzer = EOFAnalyzer()
        self.corr_analyzer = CorrelationAnalyzer()
        self.rmse_analyzer = RMSEAnalyzer()
        self.spectrum_analyzer = SpectrumAnalyzer()
        self.common_eof_analyzer = CommonEOFAnalyzer()
        self.crpss_analyzer = CRPSSAnalyzer()
        
        self.logger.info("ClimateAnalyzer初始化完成")
    
    def load_observation_data(self, var_type: str) -> xr.DataArray:
        """
        加载观测数据
        
        Args:
            var_type: 变量类型 ('temp' 或 'prec')
            
        Returns:
            观测数据
        """
        var_config = get_var_config(var_type)
        obs_file = self.obs_dir / f"{var_type}_1deg_199301-202012.nc"
        
        if not obs_file.exists():
            raise FileNotFoundError(f"观测文件不存在: {obs_file}")
        
        with xr.open_dataset(obs_file, mask_and_scale=False) as ds:
            obs_var_name = find_variable(ds, var_config['obs_names'])
            obs_data = ds[obs_var_name].where(
                ~ds[obs_var_name].isin([1e20, ds[obs_var_name].attrs.get('_FillValue', 1e20)]), 
                np.nan
            )
            
            # 应用单位转换
            if var_type == 'prec' and 'obs_conv' in var_config:
                obs_data = obs_data * var_config['obs_conv'](1)
                obs_data = obs_data.clip(min=0).fillna(0)
            
            # 空间裁剪
            obs_data = dynamic_coord_sel(obs_data, SPATIAL_BOUNDS)
            
            # 时间处理
            obs_data = obs_data.resample(time='1MS').mean()
            start_time, end_time = DATE_RANGE
            obs_data = obs_data.sel(time=slice(start_time, end_time))
            
            # 数据验证
            validate_data(obs_data, f"{var_type}观测数据")
            
            self.logger.info(f"观测数据加载完成: {obs_data.shape}")
            return obs_data
    
    def load_forecast_data(self, var_type: str, model: str, leadtime: int) -> xr.DataArray:
        """
        加载预测数据
        
        Args:
            var_type: 变量类型
            model: 模型名称
            leadtime: 提前期
            
        Returns:
            预测数据
        """
        var_config = get_var_config(var_type)
        model_config = get_model_config(model)
        suffix = model_config[var_config['file_type']]
        model_dir = self.forecast_dir / model
        
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        # 收集所有时间步的数据
        all_data = []
        valid_times = []
        
        for year in range(1993, 2021):
            for month in range(1, 13):
                file_path = model_dir / f"{year}{month:02d}.{suffix}.nc"
                if not file_path.exists():
                    continue
                
                try:
                    with xr.open_dataset(file_path) as ds:
                        fcst_var_name = find_variable(ds, var_config['fcst_names'])
                        data = ds[fcst_var_name]
                        
                        # 处理预报时效
                        if 'time' in data.dims:
                            init_time = pd.Timestamp(year, month, 1)
                            valid_time = (init_time + pd.DateOffset(months=leadtime)).replace(day=1)
                            
                            if valid_time.year < 1993 or valid_time.year > 2020:
                                continue
                            
                            data = data.sel(time=valid_time, method='nearest', tolerance='15D')
                        
                        # 空间处理
                        data = dynamic_coord_sel(data, SPATIAL_BOUNDS)
                        
                        # 单位转换
                        if var_type == 'prec' and 'fcst_conv' in var_config:
                            data = data * var_config['fcst_conv'](1)
                            data = data.clip(min=0).fillna(0)
                        
                        all_data.append(data)
                        valid_times.append(valid_time)
                        
                except Exception as e:
                    self.logger.warning(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
        
        if not all_data:
            raise ValueError(f"没有找到有效的预测数据: {model} L{leadtime}")
        
        # 合并数据
        fcst_data = xr.concat(all_data, dim=xr.DataArray(valid_times, dims='time', name='time'))
        fcst_data = fcst_data.sortby('time')
        
        # 插值到观测网格
        obs_data = self.load_observation_data(var_type)
        fcst_data = fcst_data.interp_like(obs_data)
        
        self.logger.info(f"预测数据加载完成: {model} L{leadtime}, 形状: {fcst_data.shape}")
        return fcst_data
    
    def run_eof_analysis(self, var_type: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行EOF分析
        
        Args:
            var_type: 变量类型
            models: 模型列表，None表示所有模型
            
        Returns:
            EOF分析结果
        """
        if models is None:
            models = get_model_names()
        
        self.logger.info(f"开始EOF分析: {var_type}")
        
        # 加载观测数据
        obs_data = self.load_observation_data(var_type)
        
        # 观测数据EOF分析
        obs_results = self.eof_analyzer.fit(obs_data)
        
        # 保存观测结果
        obs_save_path = self.output_dirs['eof'] / f"{var_type}_obs_eof.pkl"
        self.eof_analyzer.save_results(obs_save_path)
        
        # 预测数据EOF分析
        forecast_results = {}
        
        for model in models:
            for leadtime in LEADTIMES:
                try:
                    fcst_data = self.load_forecast_data(var_type, model, leadtime)
                    
                    # 确保时间对齐
                    common_time = obs_data.time.intersection(fcst_data.time)
                    if len(common_time) < 10:  # 至少需要10个时间点
                        continue
                    
                    fcst_aligned = fcst_data.sel(time=common_time)
                    obs_aligned = obs_data.sel(time=common_time)
                    
                    # EOF分析
                    fcst_eof = EOFAnalyzer()
                    fcst_results_key = f"{model}_lead{leadtime}"
                    forecast_results[fcst_results_key] = fcst_eof.fit(fcst_aligned)
                    
                    # 保存结果
                    fcst_save_path = self.output_dirs['eof'] / f"{var_type}_{fcst_results_key}_eof.pkl"
                    fcst_eof.save_results(fcst_save_path)
                    
                except Exception as e:
                    self.logger.error(f"EOF分析失败 {model} L{leadtime}: {str(e)}")
                    continue
        
        # 合并结果
        all_results = {'obs': obs_results, **forecast_results}
        
        # 保存完整结果
        complete_save_path = self.output_dirs['eof'] / f"{var_type}_complete_eof.pkl"
        with open(complete_save_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        self.logger.info(f"EOF分析完成: {var_type}")
        return all_results
    
    def run_correlation_analysis(self, var_type: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行相关性分析
        
        Args:
            var_type: 变量类型
            models: 模型列表
            
        Returns:
            相关性分析结果
        """
        if models is None:
            models = get_model_names()
        
        self.logger.info(f"开始相关性分析: {var_type}")
        
        # 加载观测数据
        obs_data = self.load_observation_data(var_type)
        
        results = {}
        
        for model in models:
            for leadtime in LEADTIMES:
                try:
                    fcst_data = self.load_forecast_data(var_type, model, leadtime)
                    
                    # 确保时间对齐
                    common_time = obs_data.time.intersection(fcst_data.time)
                    if len(common_time) < 10:
                        continue
                    
                    fcst_aligned = fcst_data.sel(time=common_time)
                    obs_aligned = obs_data.sel(time=common_time)
                    
                    # 计算相关性
                    corr_result = self.corr_analyzer.compute_correlation(obs_aligned, fcst_aligned)
                    
                    # 保存结果
                    result_key = f"{model}_lead{leadtime}"
                    results[result_key] = corr_result
                    
                    save_path = self.output_dirs['correlation'] / model / f"lead_{leadtime}_corr.nc"
                    self.corr_analyzer.save_results(corr_result, save_path)
                    
                except Exception as e:
                    self.logger.error(f"相关性分析失败 {model} L{leadtime}: {str(e)}")
                    continue
        
        self.logger.info(f"相关性分析完成: {var_type}")
        return results
    
    def run_rmse_analysis(self, var_type: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行RMSE分析
        
        Args:
            var_type: 变量类型
            models: 模型列表
            
        Returns:
            RMSE分析结果
        """
        if models is None:
            models = get_model_names()
        
        self.logger.info(f"开始RMSE分析: {var_type}")
        
        # 加载观测数据
        obs_data = self.load_observation_data(var_type)
        
        results = {}
        
        for model in models:
            for leadtime in LEADTIMES:
                try:
                    fcst_data = self.load_forecast_data(var_type, model, leadtime)
                    
                    # 确保时间对齐
                    common_time = obs_data.time.intersection(fcst_data.time)
                    if len(common_time) < 10:
                        continue
                    
                    fcst_aligned = fcst_data.sel(time=common_time)
                    obs_aligned = obs_data.sel(time=common_time)
                    
                    # 计算RMSE
                    rmse_result = self.rmse_analyzer.compute_rmse(obs_aligned, fcst_aligned)
                    
                    # 保存结果
                    result_key = f"{model}_lead{leadtime}"
                    results[result_key] = rmse_result
                    
                    save_path = self.output_dirs['rmse'] / model / f"lead_{leadtime}_rmse.nc"
                    self.rmse_analyzer.save_results(rmse_result, save_path)
                    
                except Exception as e:
                    self.logger.error(f"RMSE分析失败 {model} L{leadtime}: {str(e)}")
                    continue
        
        self.logger.info(f"RMSE分析完成: {var_type}")
        return results
    
    def run_common_eof_analysis(self, var_type: str, 
                               leadtimes: Optional[List[int]] = None,
                               models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行Common EOF分析
        
        Args:
            var_type: 变量类型
            leadtimes: 提前期列表
            models: 模型列表
            
        Returns:
            Common EOF分析结果
        """
        if leadtimes is None:
            leadtimes = LEADTIMES
        if models is None:
            models = get_model_names()
        
        self.logger.info(f"开始Common EOF分析: {var_type}")
        
        results = {}
        
        for leadtime in leadtimes:
            try:
                # 收集所有数据集
                datasets = {}
                
                # 观测数据
                obs_data = self.load_observation_data(var_type)
                datasets['OBS'] = obs_data
                
                # 预测数据
                for model in models:
                    try:
                        fcst_data = self.load_forecast_data(var_type, model, leadtime)
                        datasets[model] = fcst_data
                    except Exception as e:
                        self.logger.warning(f"跳过模型 {model} L{leadtime}: {str(e)}")
                        continue
                
                if len(datasets) < 2:
                    self.logger.warning(f"L{leadtime} 有效数据集不足，跳过")
                    continue
                
                # Common EOF分析
                common_eof_result = self.common_eof_analyzer.compute_common_eofs(datasets)
                
                # 保存结果
                results[f"lead_{leadtime}"] = common_eof_result
                
                save_path = self.output_dirs['common_eof'] / var_type / f"lead_{leadtime}"
                self.common_eof_analyzer.save_results(common_eof_result, save_path)
                
            except Exception as e:
                self.logger.error(f"Common EOF分析失败 L{leadtime}: {str(e)}")
                continue
        
        self.logger.info(f"Common EOF分析完成: {var_type}")
        return results
    
    def run_spectrum_analysis(self, var_type: str, 
                             region: str = 'global',
                             models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行功率谱分析
        
        Args:
            var_type: 变量类型
            region: 分析区域
            models: 模型列表
            
        Returns:
            功率谱分析结果
        """
        if models is None:
            models = get_model_names()
        
        self.logger.info(f"开始功率谱分析: {var_type} {region}")
        
        # 加载观测数据
        obs_data = self.load_observation_data(var_type)
        
        # 观测数据功率谱
        obs_spectrum = self.spectrum_analyzer.compute_power_spectrum(obs_data, region)
        
        results = {'obs': obs_spectrum}
        
        # 预测数据功率谱
        for model in models:
            for leadtime in LEADTIMES:
                try:
                    fcst_data = self.load_forecast_data(var_type, model, leadtime)
                    
                    # 确保时间对齐
                    common_time = obs_data.time.intersection(fcst_data.time)
                    if len(common_time) < 10:
                        continue
                    
                    fcst_aligned = fcst_data.sel(time=common_time)
                    
                    # 功率谱分析
                    fcst_spectrum = self.spectrum_analyzer.compute_power_spectrum(fcst_aligned, region)
                    
                    result_key = f"{model}_lead{leadtime}"
                    results[result_key] = fcst_spectrum
                    
                except Exception as e:
                    self.logger.error(f"功率谱分析失败 {model} L{leadtime}: {str(e)}")
                    continue
        
        # 保存结果
        save_path = self.output_dirs['spectrum'] / f"{var_type}_{region}_spectrum.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"功率谱分析完成: {var_type} {region}")
        return results
    
    def create_all_plots(self, var_type: str, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        创建所有类型的图形
        
        Args:
            var_type: 变量类型
            save_dir: 保存目录
        
        Returns:
            图形文件路径字典
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        try:
            # 1. 创建地图图形
            logger.info(f"开始创建{var_type}的地图图形...")
            map_dir = save_dir / 'maps'
            map_dir.mkdir(exist_ok=True)
            
            # 加载观测和预测数据
            obs_data = self.load_observation_data(var_type)
            
            # 创建空间分布图
            from .plotting.map_plots import plot_spatial_field
            fig, ax = plot_spatial_field(
                obs_data.isel(time=0),  # 使用第一个时间点
                title=f"{var_type.upper()} Spatial Distribution",
                save_path=map_dir / f"{var_type}_spatial_distribution.png"
            )
            plot_paths['spatial_distribution'] = str(map_dir / f"{var_type}_spatial_distribution.png")
            
            # 2. 创建EOF图形
            logger.info(f"开始创建{var_type}的EOF图形...")
            eof_dir = save_dir / 'eof'
            eof_dir.mkdir(exist_ok=True)
            
            # 运行EOF分析
            eof_results = self.run_eof_analysis(var_type)
            
            if eof_results:
                from .plotting.eof_plots import plot_eof_modes, plot_pc_timeseries, plot_eof_variance_spectrum
                
                # EOF模态图
                fig, axes = plot_eof_modes(
                    eof_results['eofs'],
                    eof_results['coords']['lat'],
                    eof_results['coords']['lon'],
                    eof_results['explained_variance_ratio'],
                    var_type=var_type,
                    save_path=eof_dir / f"{var_type}_eof_modes.png"
                )
                plot_paths['eof_modes'] = str(eof_dir / f"{var_type}_eof_modes.png")
                
                # PC时间序列图
                fig, axes = plot_pc_timeseries(
                    eof_results['pcs'],
                    np.arange(len(eof_results['pcs'])),
                    eof_results['explained_variance_ratio'],
                    var_type=var_type,
                    save_path=eof_dir / f"{var_type}_pc_timeseries.png"
                )
                plot_paths['pc_timeseries'] = str(eof_dir / f"{var_type}_pc_timeseries.png")
                
                # EOF方差谱图
                fig, ax = plot_eof_variance_spectrum(
                    eof_results['explained_variance_ratio'],
                    save_path=eof_dir / f"{var_type}_eof_variance_spectrum.png"
                )
                plot_paths['eof_variance_spectrum'] = str(eof_dir / f"{var_type}_eof_variance_spectrum.png")
            
            # 3. 创建相关性图形
            logger.info(f"开始创建{var_type}的相关性图形...")
            corr_dir = save_dir / 'correlation'
            corr_dir.mkdir(exist_ok=True)
            
            # 运行相关性分析
            corr_results = self.run_correlation_analysis(var_type)
            
            if corr_results:
                from .plotting.map_plots import plot_correlation_map
                
                # 相关性地图
                for lead_time in [0, 1, 2]:
                    if f'lead_{lead_time}' in corr_results:
                        fig, ax = plot_correlation_map(
                            corr_results[f'lead_{lead_time}']['correlation'],
                            corr_results[f'lead_{lead_time}']['p_value'],
                            var_type=var_type,
                            lead_time=lead_time,
                            save_path=corr_dir / f"{var_type}_correlation_lead{lead_time}.png"
                        )
                        plot_paths[f'correlation_lead{lead_time}'] = str(corr_dir / f"{var_type}_correlation_lead{lead_time}.png")
            
            # 4. 创建RMSE图形
            logger.info(f"开始创建{var_type}的RMSE图形...")
            rmse_dir = save_dir / 'rmse'
            rmse_dir.mkdir(exist_ok=True)
            
            # 运行RMSE分析
            rmse_results = self.run_rmse_analysis(var_type)
            
            if rmse_results:
                from .plotting.map_plots import plot_rmse_map
                from .plotting.statistical_plots import plot_rmse_comparison
                
                # RMSE地图
                for lead_time in [0, 1, 2]:
                    if f'lead_{lead_time}' in rmse_results:
                        fig, ax = plot_rmse_map(
                            rmse_results[f'lead_{lead_time}']['rmse'],
                            var_type=var_type,
                            lead_time=lead_time,
                            save_path=rmse_dir / f"{var_type}_rmse_lead{lead_time}.png"
                        )
                        plot_paths[f'rmse_lead{lead_time}'] = str(rmse_dir / f"{var_type}_rmse_lead{lead_time}.png")
                
                # RMSE比较图
                fig, ax = plot_rmse_comparison(
                    rmse_results,
                    var_type=var_type,
                    save_path=rmse_dir / f"{var_type}_rmse_comparison.png"
                )
                plot_paths['rmse_comparison'] = str(rmse_dir / f"{var_type}_rmse_comparison.png")
            
            # 5. 创建功率谱图形
            logger.info(f"开始创建{var_type}的功率谱图形...")
            spectrum_dir = save_dir / 'spectrum'
            spectrum_dir.mkdir(exist_ok=True)
            
            # 运行功率谱分析
            spectrum_results = self.run_spectrum_analysis(var_type)
            
            if spectrum_results:
                from .plotting.spectrum_plots import plot_power_spectrum, plot_spectrum_analysis
                
                # 功率谱图
                fig, ax = plot_power_spectrum(
                    spectrum_results['frequencies'],
                    spectrum_results['power_spectrum'],
                    var_type=var_type,
                    save_path=spectrum_dir / f"{var_type}_power_spectrum.png"
                )
                plot_paths['power_spectrum'] = str(spectrum_dir / f"{var_type}_power_spectrum.png")
                
                # 谱分析图（时间序列+功率谱）
                if 'time_series' in spectrum_results:
                    fig, axes = plot_spectrum_analysis(
                        spectrum_results['time_series'],
                        spectrum_results['frequencies'],
                        spectrum_results['power_spectrum'],
                        var_type=var_type,
                        save_path=spectrum_dir / f"{var_type}_spectrum_analysis.png"
                    )
                    plot_paths['spectrum_analysis'] = str(spectrum_dir / f"{var_type}_spectrum_analysis.png")
            
            # 6. 创建Taylor图
            logger.info(f"开始创建{var_type}的Taylor图...")
            taylor_dir = save_dir / 'taylor'
            taylor_dir.mkdir(exist_ok=True)
            
            if eof_results and corr_results:
                from .plotting.taylor_plots import plot_taylor_diagram, calc_taylor_metrics
                
                # 计算Taylor指标
                taylor_metrics = {}
                obs_pcs = eof_results['pcs']
                
                for model in MODELS.keys():
                    for lead_time in [0, 1, 2]:
                        key = f"{model}_lead{lead_time}"
                        if key in corr_results:
                            # 使用第一个EOF模态进行Taylor图分析
                            obs_pc = obs_pcs[:, 0]
                            fcst_pc = corr_results[key].get('pcs', obs_pc)  # 简化处理
                            
                            metrics = calc_taylor_metrics(obs_pc, fcst_pc)
                            taylor_metrics[f"{model}_lead{lead_time}"] = metrics
                
                if taylor_metrics:
                    fig, dia = plot_taylor_diagram(
                        taylor_metrics,
                        np.std(obs_pcs[:, 0]),
                        var_type=var_type,
                        save_path=taylor_dir / f"{var_type}_taylor_diagram.png"
                    )
                    plot_paths['taylor_diagram'] = str(taylor_dir / f"{var_type}_taylor_diagram.png")
            
            # 7. 创建统计图形
            logger.info(f"开始创建{var_type}的统计图形...")
            stats_dir = save_dir / 'statistics'
            stats_dir.mkdir(exist_ok=True)
            
            from .plotting.statistical_plots import plot_timeseries, plot_boxplot
            
            # 时间序列图
            if 'time' in obs_data.dims:
                fig, ax = plot_timeseries(
                    obs_data.mean(dim=['lat', 'lon']),
                    title=f"{var_type.upper()} Time Series",
                    save_path=stats_dir / f"{var_type}_timeseries.png"
                )
                plot_paths['timeseries'] = str(stats_dir / f"{var_type}_timeseries.png")
            
            logger.info(f"{var_type}的所有图形创建完成")
            
        except Exception as e:
            logger.error(f"创建{var_type}图形时发生错误: {str(e)}")
            raise
        
        return plot_paths
    
    def _create_eof_plots(self, var_type: str) -> None:
        """创建EOF相关图形"""
        # 这里实现EOF图形的创建
        pass
    
    def _create_correlation_plots(self, var_type: str) -> None:
        """创建相关性图形"""
        # 这里实现相关性图形的创建
        pass
    
    def _create_rmse_plots(self, var_type: str) -> None:
        """创建RMSE图形"""
        # 这里实现RMSE图形的创建
        pass
    
    def _create_spectrum_plots(self, var_type: str) -> None:
        """创建功率谱图形"""
        # 这里实现功率谱图形的创建
        pass

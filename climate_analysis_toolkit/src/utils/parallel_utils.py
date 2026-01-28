"""
并行计算工具模块
提供统一的并行处理功能，支持多核计算
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, List, Dict, Any, Optional, Union
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """并行处理器类"""
    
    def __init__(self, n_jobs: Optional[int] = None, backend: str = "process"):
        """
        初始化并行处理器
        
        Args:
            n_jobs: 并行作业数，None表示使用所有可用CPU
            backend: 后端类型，"process"或"thread"
        """
        self.n_jobs = n_jobs or min(mp.cpu_count(), 32)  # 限制最大32个进程
        self.backend = backend
        self.executor_class = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        
        logger.info(f"初始化并行处理器: {self.n_jobs} 个 {backend} 进程")
    
    def parallel_map(self, func: Callable, iterable: List, **kwargs) -> List:
        """
        并行映射函数
        
        Args:
            func: 要执行的函数
            iterable: 迭代对象
            **kwargs: 传递给函数的额外参数
            
        Returns:
            结果列表
        """
        if self.n_jobs == 1:
            # 单进程执行
            return [func(item, **kwargs) for item in iterable]
        
        results = []
        with self.executor_class(max_workers=self.n_jobs) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(func, item, **kwargs): item 
                for item in iterable
            }
            
            # 收集结果
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"完成处理: {item}")
                except Exception as e:
                    logger.error(f"处理 {item} 时出错: {str(e)}")
                    results.append(None)
        
        return results
    
    def parallel_apply(self, func: Callable, data: Union[xr.DataArray, pd.DataFrame], 
                      dim: str = "time", **kwargs) -> Union[xr.DataArray, pd.DataFrame]:
        """
        并行应用函数到数据
        
        Args:
            func: 要应用的函数
            data: 输入数据
            dim: 并行处理的维度
            **kwargs: 传递给函数的额外参数
            
        Returns:
            处理后的数据
        """
        if isinstance(data, xr.DataArray):
            return self._parallel_apply_xarray(func, data, dim, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return self._parallel_apply_dataframe(func, data, **kwargs)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _parallel_apply_xarray(self, func: Callable, data: xr.DataArray, 
                              dim: str, **kwargs) -> xr.DataArray:
        """并行应用函数到xarray数据"""
        # 将数据分割成块
        chunks = np.array_split(data[dim], self.n_jobs)
        
        def process_chunk(chunk_data):
            return func(chunk_data, **kwargs)
        
        # 并行处理
        results = self.parallel_map(process_chunk, chunks)
        
        # 合并结果
        return xr.concat(results, dim=dim)
    
    def _parallel_apply_dataframe(self, func: Callable, data: pd.DataFrame, 
                                 **kwargs) -> pd.DataFrame:
        """并行应用函数到DataFrame数据"""
        # 将数据分割成块
        chunks = np.array_split(data, self.n_jobs)
        
        def process_chunk(chunk_data):
            return func(chunk_data, **kwargs)
        
        # 并行处理
        results = self.parallel_map(process_chunk, chunks)
        
        # 合并结果
        return pd.concat(results, ignore_index=True)


def parallel_analysis(func: Callable, analysis_params: List[Dict], 
                     n_jobs: Optional[int] = None, 
                     backend: str = "process") -> List[Any]:
    """
    并行执行分析任务
    
    Args:
        func: 分析函数
        analysis_params: 分析参数列表
        n_jobs: 并行作业数
        backend: 后端类型
        
    Returns:
        分析结果列表
    """
    processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)
    
    def analysis_wrapper(params):
        """分析函数包装器"""
        try:
            return func(**params)
        except Exception as e:
            logger.error(f"分析任务失败: {str(e)}")
            return None
    
    return processor.parallel_map(analysis_wrapper, analysis_params)


def parallel_model_analysis(func: Callable, models: List[str], 
                          leadtimes: List[int], var_type: str,
                          n_jobs: Optional[int] = None,
                          backend: str = "process") -> Dict[str, Any]:
    """
    并行执行模型分析任务
    
    Args:
        func: 分析函数
        models: 模型列表
        leadtimes: 提前期列表
        var_type: 变量类型
        n_jobs: 并行作业数
        backend: 后端类型
        
    Returns:
        分析结果字典
    """
    # 生成任务参数
    tasks = []
    for model in models:
        for leadtime in leadtimes:
            tasks.append({
                'model': model,
                'leadtime': leadtime,
                'var_type': var_type
            })
    
    # 并行执行
    results = parallel_analysis(func, tasks, n_jobs, backend)
    
    # 整理结果
    organized_results = {}
    for i, (model, leadtime) in enumerate([(m, lt) for m in models for lt in leadtimes]):
        if model not in organized_results:
            organized_results[model] = {}
        organized_results[model][leadtime] = results[i]
    
    return organized_results


def get_optimal_n_jobs(data_size: int, memory_per_item: float = 1e9) -> int:
    """
    根据数据大小和内存需求计算最优并行作业数
    
    Args:
        data_size: 数据大小
        memory_per_item: 每个项目的内存需求（字节）
        
    Returns:
        最优并行作业数
    """
    # 获取系统信息
    cpu_count = mp.cpu_count()
    available_memory = _get_available_memory()
    
    # 基于内存限制计算
    memory_based_jobs = max(1, int(available_memory / memory_per_item))
    
    # 基于CPU数量计算
    cpu_based_jobs = cpu_count
    
    # 取较小值
    optimal_jobs = min(memory_based_jobs, cpu_based_jobs, 32)
    
    logger.info(f"系统信息: CPU={cpu_count}, 内存={available_memory/1e9:.1f}GB")
    logger.info(f"计算最优并行作业数: {optimal_jobs}")
    
    return optimal_jobs


def _get_available_memory() -> float:
    """获取可用内存（字节）"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        # 如果没有psutil，使用保守估计
        return 8e9  # 假设8GB可用内存


def _process_single_task(task_params):
    """单个处理任务 - 模块级函数，可以被pickle序列化"""
    data_loader = task_params['data_loader']
    model = task_params['model']
    leadtime = task_params['leadtime']
    var_type = task_params['var_type']
    
    try:
        # 加载数据
        obs_data = data_loader.load_obs_data(var_type)
        fcst_data = data_loader.load_forecast_data(model, var_type, leadtime)
        
        if fcst_data is None:
            logger.warning(f"跳过 {model} L{leadtime}, 无数据")
            return None
        
        # 注意：这里不再调用func，因为func可能是绑定方法
        # 具体的处理逻辑应该在调用方实现
        return {
            'obs_data': obs_data,
            'fcst_data': fcst_data,
            'model': model,
            'leadtime': leadtime,
            'var_type': var_type
        }
        
    except Exception as e:
        logger.error(f"处理 {model} L{leadtime} 时出错: {str(e)}")
        return None


def parallel_data_processing(data_loader, 
                           models: List[str], leadtimes: List[int],
                           var_type: str, n_jobs: Optional[int] = None) -> Dict:
    """
    并行数据处理
    
    Args:
        data_loader: 数据加载器
        models: 模型列表
        leadtimes: 提前期列表
        var_type: 变量类型
        n_jobs: 并行作业数
        
    Returns:
        处理结果字典
    """
    # 生成任务列表
    tasks = []
    for model in models:
        for leadtime in leadtimes:
            tasks.append({
                'data_loader': data_loader,
                'model': model,
                'leadtime': leadtime,
                'var_type': var_type
            })
    
    # 并行执行
    processor = ParallelProcessor(n_jobs=n_jobs)
    results = processor.parallel_map(_process_single_task, tasks)
    
    # 整理结果
    organized_results = {}
    for i, (model, leadtime) in enumerate([(m, lt) for m in models for lt in leadtimes]):
        if model not in organized_results:
            organized_results[model] = {}
        organized_results[model][leadtime] = results[i]
    
    return organized_results


def _apply_func_with_kwargs(func, args, task_kwargs):
    """应用函数到任务参数的辅助函数"""
    return func(*args, **task_kwargs)

def _process_task_with_func(task_data):
    """处理单个任务的辅助函数"""
    func, args, task_kwargs = task_data
    return _apply_func_with_kwargs(func, args, task_kwargs)

# 装饰器：自动并行化
def parallelize(n_jobs: Optional[int] = None, backend: str = "process"):
    """
    装饰器：自动并行化函数
    
    Args:
        n_jobs: 并行作业数
        backend: 后端类型
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 检查是否有可并行化的参数
            if 'models' in kwargs and 'leadtimes' in kwargs:
                models = kwargs['models']
                leadtimes = kwargs['leadtimes']
                
                # 生成任务参数
                tasks = []
                for model in models:
                    for leadtime in leadtimes:
                        task_kwargs = kwargs.copy()
                        task_kwargs['model'] = model
                        task_kwargs['leadtime'] = leadtime
                        tasks.append((func, args, task_kwargs))
                
                # 并行执行
                processor = ParallelProcessor(n_jobs=n_jobs, backend=backend)
                results = processor.parallel_map(_process_task_with_func, tasks)
                
                return results
            else:
                # 如果没有可并行化的参数，直接执行
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

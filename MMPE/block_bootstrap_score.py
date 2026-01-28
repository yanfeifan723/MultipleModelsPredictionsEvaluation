#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的 Block Bootstrap 评分计算脚本 - 修复版本

修复问题：
1. 对每个时间 block 单独进行统计计算，而不是拼接后计算
2. 实现真正的独立 block bootstrap
3. 正确计算 Fisher-z 变换和反变换

使用方法：
完整分析（计算+绘图）：
python block_bootstrap_score.py --models all --leadtimes all
python block_bootstrap_score.py --models all --leadtimes all --var temp prec
python block_bootstrap_score.py --models all --leadtimes all --var temp

仅绘图（基于已有NetCDF结果文件）：
python block_bootstrap_score.py --models all --leadtimes all --plot-only
python block_bootstrap_score.py --models CMCC-35 --leadtimes 0 1 --plot-only
"""

import os
import sys
import gc
import json
import logging
import warnings
import argparse
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from concurrent.futures import as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da

# 统一导入toolkit路径
sys.path.insert(0, '/sas12t1/ffyan/climate_analysis_toolkit')
from src.utils.parallel_utils import ParallelProcessor
from src.utils.logging_config import setup_logging

from common_config import (
    MODEL_LIST,
    MODEL_FILE_MAP,
    LEADTIMES,
    SPATIAL_BOUNDS as COMMON_SPATIAL_BOUNDS,
    DATA_PATHS,
    VAR_NAMES,
)

# 配置日志
logger = setup_logging(
    log_file='block_bootstrap_score.log',
    module_name=__name__
)

# GPU支持已移除，保留变量定义以兼容现有代码
GPU_AVAILABLE = False
GPU_DEVICE_COUNT = 0
GPU_MEMORY_INFO = {}
from src.utils.data_loader import DataLoader
from src.utils.alignment import align_time_to_monthly, align_spatial_to_obs, align_multiple_datasets_to_common_time
from src.utils.cli_args import create_parser, parse_models, parse_leadtimes, parse_vars, normalize_plot_args, normalize_parallel_args
from src.utils.data_utils import create_land_mask, compute_data_extent

warnings.filterwarnings('ignore')

# 模型配置
MODELS = MODEL_FILE_MAP

def obs_prec_conv(x):
    """观测降水单位转换函数"""
    return x * 86400

def fcst_prec_conv(x):
    """预报降水单位转换函数"""
    return x * 86400 * 1000

# 扩展VAR_CONFIG（使用common_config中的VAR_NAMES）
def obs_prec_conv(x):
    """观测降水单位转换函数"""
    return x * 86400

def fcst_prec_conv(x):
    """预报降水单位转换函数"""
    return x * 86400 * 1000

VAR_CONFIG = {
    "temp": {
        "file_type": "sfc",
        **VAR_NAMES["temp"],  # 使用common_config中的变量名
        "unit": "°C",
        "accuracy_delta": 0.05,  # ±5% for temperature
        "accuracy_eps": 1e-6
    },
    "prec": {
        "file_type": "sfc",
        **VAR_NAMES["prec"],  # 使用common_config中的变量名
        "obs_conv": obs_prec_conv,
        "fcst_conv": fcst_prec_conv,
        "unit": "mm/day",
        "accuracy_delta": 0.3,   # ±30% for precipitation (more lenient)
        "accuracy_eps": 1e-6      # Larger epsilon for precipitation
    }
}

# LEADTIMES 已从 common_config 导入
SPATIAL_BOUNDS = {
    "lat": list(COMMON_SPATIAL_BOUNDS["lat"]),
    "lon": list(COMMON_SPATIAL_BOUNDS["lon"]),
}
MAX_MEMORY_GB = 230
def get_memory_usage_gb():
    """获取当前内存使用量（GB）"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024**3  # 转换为GB

def check_memory_limit():
    """检查内存是否超过限制"""
    current_memory = get_memory_usage_gb()
    if current_memory > MAX_MEMORY_GB:
        logger.warning(f"内存使用量 {current_memory:.1f}GB 超过限制 {MAX_MEMORY_GB}GB")
        return False
    return True

def log_memory_usage(step_name: str):
    """记录内存使用情况"""
    current_memory = get_memory_usage_gb()
    logger.info(f"{step_name}: 内存使用 {current_memory:.1f}GB / {MAX_MEMORY_GB}GB ({current_memory/MAX_MEMORY_GB*100:.1f}%)")

def force_garbage_collection():
    """强制垃圾回收"""
    logger.info("执行强制垃圾回收...")
    gc.collect()
    current_memory = get_memory_usage_gb()
    logger.info(f"垃圾回收后内存使用: {current_memory:.1f}GB")

def fisher_z(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Fisher z 变换"""
    return np.arctanh(np.clip(r, -0.9999999, 0.9999999))


def fisher_z_inv(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Fisher z 逆变换"""
    return np.tanh(z)


def normalize_rmse(b_rmse: np.ndarray, rmse_min: float, rmse_max: float) -> np.ndarray:
    """归一化 RMSE 到 [0,1]，并反向定义为 s_rmse"""
    # 检查边界情况
    if np.isnan(rmse_min) or np.isnan(rmse_max) or rmse_min == rmse_max:
        logger.warning(f"RMSE归一化参数异常: min={rmse_min}, max={rmse_max}")
        return np.full_like(b_rmse, np.nan)
    
    # 归一化到[0,1]
    range_val = rmse_max - rmse_min
    if range_val <= 0:
        logger.warning(f"RMSE范围无效: {range_val}")
        return np.full_like(b_rmse, np.nan)
    
    n_rmse = (b_rmse - rmse_min) / range_val
    # 反向定义：RMSE越小，得分越高
    s_rmse = 1 - np.clip(n_rmse, 0, 1)
    
    # 处理NaN值
    s_rmse = np.where(np.isnan(b_rmse), np.nan, s_rmse)
    
    return s_rmse


def compute_score(s_accuracy: np.ndarray, s_rmse: np.ndarray, 
                 s_pcc: np.ndarray, weights: Tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
    """计算综合得分"""
    w_accuracy, w_rmse, w_pcc = weights
    
    # 检查输入数据
    if s_accuracy.shape != s_rmse.shape or s_accuracy.shape != s_pcc.shape:
        logger.error(f"输入数组形状不匹配: s_accuracy{s_accuracy.shape}, s_rmse{s_rmse.shape}, s_pcc{s_pcc.shape}")
        return np.full_like(s_accuracy, np.nan)
    
    
    # 将PCC从[-1,1]映射到[0,1]
    s_pcc_normalized = (s_pcc + 1) / 2
    
    # 计算加权平均
    total_weight = sum(weights)
    if total_weight == 0:
        logger.error("权重总和为0")
        return np.full_like(s_accuracy, np.nan)
    
    score = (w_accuracy * s_accuracy + w_rmse * s_rmse + w_pcc * s_pcc_normalized) / total_weight
    
    # 处理NaN值：如果任何输入为NaN，输出也为NaN
    score = np.where(
        np.isnan(s_accuracy) | np.isnan(s_rmse) | np.isnan(s_pcc),
        np.nan,
        score
    )
    
    # 检查结果范围
    valid_scores = score[~np.isnan(score)]
    if len(valid_scores) > 0:
        if np.any(valid_scores < 0) or np.any(valid_scores > 1):
            logger.warning(f"得分超出[0,1]范围: min={np.min(valid_scores):.4f}, max={np.max(valid_scores):.4f}")
    else:
        logger.warning(f"所有得分都是NaN!")
    
    return score


def ensure_member_dimension(data: xr.DataArray) -> xr.DataArray:
    """确保数据有 member 维度"""
    if 'member' not in data.dims:
        data = data.expand_dims('member')
        data['member'] = [0]
    return data


def _compute_block_metrics(model_block: np.ndarray, obs_block: np.ndarray,
                          accuracy_delta: float, accuracy_eps: float,
                          model_climatology: float = None, obs_climatology: float = None) -> Tuple[float, float, float, float, float]:
    """
    计算单个时间 block 的指标（包含equitable accuracy）
    
    Args:
        model_block: 模型数据 block
        obs_block: 观测数据 block  
        accuracy_delta: accuracy 相对误差阈值
        accuracy_eps: 防止除零的小常数
        model_climatology: 模型气候态（用于计算equitable accuracy）
        obs_climatology: 观测气候态（保留用于未来扩展）
        
    Returns:
        (accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z)
    """
    # 检查有效数据
    mask = ~(np.isnan(model_block) | np.isnan(obs_block))
    n_valid = np.sum(mask)
    if n_valid < 2:  # 至少需要2个有效点计算PCC
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    valid_model = model_block[mask]
    valid_obs = obs_block[mask]
    
    # ===== 向量化计算：一次性计算所有指标 =====
    
    # 1. Raw Accuracy - 使用动态accuracy_eps
    dynamic_eps = np.maximum(accuracy_eps, np.minimum(valid_obs * 0.01, accuracy_eps * 1000))
    rel_error_fcst = np.abs(valid_model - valid_obs) / (np.abs(valid_obs) + dynamic_eps)
    accuracy_raw = np.clip(np.mean(rel_error_fcst <= accuracy_delta), 0.0, 1.0)
    
    # 2. RMSE（向量化，避免循环）
    rmse = np.sqrt(np.mean((valid_model - valid_obs) ** 2))
    
    # 3. Equitable Accuracy - 使用动态accuracy_eps
    if obs_climatology is not None and not np.isnan(obs_climatology):
        rel_error_clim = np.abs(obs_climatology - valid_obs) / (np.abs(valid_obs) + dynamic_eps)
        accuracy_clim = np.clip(np.mean(rel_error_clim <= accuracy_delta), 0.0, 1.0)
        
        if accuracy_clim < 1.0:
            equitable_accuracy = (accuracy_raw - accuracy_clim) / (1.0 - accuracy_clim)
        else:
            equitable_accuracy = 0.0
        
        equitable_accuracy = np.clip(equitable_accuracy, -1.0, 1.0)
    else:
        accuracy_clim = np.nan
        equitable_accuracy = np.nan
    
    # 4. Pearson相关系数（需要逐个计算，但优化条件检查）
    if n_valid >= 2:
        std_model = np.std(valid_model)
        std_obs = np.std(valid_obs)
        if std_model > 1e-10 and std_obs > 1e-10:
            r, _ = pearsonr(valid_model, valid_obs)
            pcc_fisher_z = fisher_z(r)
        else:
            pcc_fisher_z = fisher_z(0.0)
    else:
        pcc_fisher_z = fisher_z(0.0)
    
    return accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z


def _compute_block_metrics_gpu(model_block: np.ndarray, obs_block: np.ndarray,
                              accuracy_delta: float, accuracy_eps: float,
                              model_climatology: float = None, obs_climatology: float = None) -> Tuple[float, float, float, float, float]:
    """
    GPU加速版本的block指标计算
    
    Args:
        model_block: 模型数据 block (numpy array)
        obs_block: 观测数据 block (numpy array)
        accuracy_delta: accuracy 相对误差阈值
        accuracy_eps: 防止除零的小常数
        model_climatology: 模型气候态
        obs_climatology: 观测气候态
        
    Returns:
        (accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z)
    """
    if not GPU_AVAILABLE:
        # 回退到CPU计算
        return _compute_block_metrics(model_block, obs_block, accuracy_delta, accuracy_eps, 
                                    model_climatology, obs_climatology)
    
    try:
        # 转换为GPU数组
        model_gpu = cp.asarray(model_block)
        obs_gpu = cp.asarray(obs_block)
        
        # 检查有效数据
        mask = ~(cp.isnan(model_gpu) | cp.isnan(obs_gpu))
        n_valid = cp.sum(mask)
        
        if n_valid < 2:
            return cp.nan, cp.nan, cp.nan, cp.nan, cp.nan
        
        valid_model = model_gpu[mask]
        valid_obs = obs_gpu[mask]
        
        # 1. Raw Accuracy - GPU向量化计算
        dynamic_eps = cp.maximum(accuracy_eps, cp.minimum(valid_obs * 0.01, accuracy_eps * 1000))
        rel_error_fcst = cp.abs(valid_model - valid_obs) / (cp.abs(valid_obs) + dynamic_eps)
        accuracy_raw = float(cp.clip(cp.mean(rel_error_fcst <= accuracy_delta), 0.0, 1.0))
        
        # 2. RMSE - GPU向量化计算
        rmse = float(cp.sqrt(cp.mean((valid_model - valid_obs) ** 2)))
        
        # 3. Equitable Accuracy
        if obs_climatology is not None and not cp.isnan(obs_climatology):
            rel_error_clim = cp.abs(obs_climatology - valid_obs) / (cp.abs(valid_obs) + dynamic_eps)
            accuracy_clim = float(cp.clip(cp.mean(rel_error_clim <= accuracy_delta), 0.0, 1.0))
            
            if accuracy_clim < 1.0:
                equitable_accuracy = float((accuracy_raw - accuracy_clim) / (1.0 - accuracy_clim))
            else:
                equitable_accuracy = 0.0
            
            equitable_accuracy = float(cp.clip(equitable_accuracy, -1.0, 1.0))
        else:
            accuracy_clim = cp.nan
            equitable_accuracy = cp.nan
        
        # 4. Pearson相关系数 - 使用CuPy的scipy.stats
        if n_valid >= 2:
            std_model = cp.std(valid_model)
            std_obs = cp.std(valid_obs)
            if std_model > 1e-10 and std_obs > 1e-10:
                # 使用CuPy计算相关系数
                r = cp_stats.pearsonr(valid_model, valid_obs)[0]
                pcc_fisher_z = float(fisher_z_gpu(r))
            else:
                pcc_fisher_z = float(fisher_z_gpu(0.0))
        else:
            pcc_fisher_z = float(fisher_z_gpu(0.0))
        
        # 清理GPU内存
        del model_gpu, obs_gpu, valid_model, valid_obs
        cp.get_default_memory_pool().free_all_blocks()
        
        return accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z
        
    except Exception as e:
        logger.warning(f"GPU计算失败，回退到CPU: {e}")
        # 回退到CPU计算
        return _compute_block_metrics(model_block, obs_block, accuracy_delta, accuracy_eps, 
                                    model_climatology, obs_climatology)


def fisher_z_gpu(r):
    """GPU版本的Fisher Z变换"""
    if not GPU_AVAILABLE:
        return fisher_z(r)
    
    try:
        r_gpu = cp.asarray(r)
        # 避免数值溢出
        r_clipped = cp.clip(r_gpu, -0.999999, 0.999999)
        z = 0.5 * cp.log((1 + r_clipped) / (1 - r_clipped))
        return float(z)
    except Exception:
        return fisher_z(r)


def _compute_block_metrics_smart(model_block: np.ndarray, obs_block: np.ndarray,
                                accuracy_delta: float, accuracy_eps: float,
                                model_climatology: float = None, obs_climatology: float = None,
                                use_gpu: bool = None) -> Tuple[float, float, float, float, float]:
    """
    智能选择GPU或CPU算法的block指标计算
    
    Args:
        model_block: 模型数据 block
        obs_block: 观测数据 block
        accuracy_delta: accuracy阈值
        accuracy_eps: 防除零常数
        model_climatology: 模型气候态
        obs_climatology: 观测气候态
        use_gpu: 是否使用GPU，None表示自动选择
    
    Returns:
        (accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z)
    """
    # 自动选择GPU或CPU
    if use_gpu is None:
        # 根据数据大小和GPU可用性自动选择
        data_size = len(model_block) * len(obs_block)
        use_gpu = GPU_AVAILABLE and data_size > 1000  # 数据量足够大才使用GPU
    
    if use_gpu and GPU_AVAILABLE:
        return _compute_block_metrics_gpu(model_block, obs_block, accuracy_delta, accuracy_eps, 
                                        model_climatology, obs_climatology)
    else:
        return _compute_block_metrics(model_block, obs_block, accuracy_delta, accuracy_eps, 
                                    model_climatology, obs_climatology)


def _compute_brier_metrics_smart(ensemble_block: np.ndarray, obs_block: np.ndarray,
                                accuracy_delta: float, accuracy_eps: float,
                                climatology_hit_rate: float = None,
                                use_gpu: bool = None) -> Tuple[float, float, float]:
    """
    智能选择GPU或CPU算法的Brier metrics计算
    """
    # 自动选择GPU或CPU
    if use_gpu is None:
        data_size = ensemble_block.size + obs_block.size
        use_gpu = GPU_AVAILABLE and data_size > 1000
    
    if use_gpu and GPU_AVAILABLE:
        return _compute_brier_metrics_gpu(ensemble_block, obs_block, accuracy_delta, accuracy_eps, climatology_hit_rate)
    else:
        return _compute_brier_metrics(ensemble_block, obs_block, accuracy_delta, accuracy_eps, climatology_hit_rate)


MIN_BS_REF = 1e-4  # 避免BSS参考分数过小导致除零


def _compute_brier_metrics(ensemble_block: np.ndarray, obs_block: np.ndarray,
                          accuracy_delta: float, accuracy_eps: float,
                          climatology_hit_rate: float = None) -> Tuple[float, float, float]:
    """
    计算Brier Score和Brier Skill Score
    
    Args:
        ensemble_block: ensemble成员数据 (shape: time x n_members)
        obs_block: 观测数据 (shape: time)
        accuracy_delta: accuracy相对误差阈值
        accuracy_eps: 防止除零的小常数
        climatology_hit_rate: 气候态命中率（用于计算BSS的参考预报）
    
    Returns:
        (BS, BS_ref, BSS)
        BS: Brier Score
        BS_ref: 参考预报的Brier Score (基于气候态)
        BSS: Brier Skill Score
    """
    # 检查有效数据
    if ensemble_block.shape[0] != obs_block.shape[0]:
        return np.nan, np.nan, np.nan
    
    # 仅根据观测是否为NaN筛选时间点，其余缺测成员在后续逐个处理
    obs_valid_mask = ~np.isnan(obs_block)
    if np.sum(obs_valid_mask) < 2:
        return np.nan, np.nan, np.nan
    
    ensemble_aligned = ensemble_block[obs_valid_mask, :]
    obs_aligned = obs_block[obs_valid_mask]
    
    prob_list: List[float] = []
    binary_list: List[float] = []
    
    for t in range(len(obs_aligned)):
        obs_t = obs_aligned[t]
        members_t = ensemble_aligned[t, :]
        member_mask = ~np.isnan(members_t)
        n_valid_members = np.sum(member_mask)
        if n_valid_members == 0:
            continue
        
        members_valid = members_t[member_mask]
        # 动态调整accuracy_eps以避免数值不稳定
        dynamic_eps = max(accuracy_eps, min(obs_t * 0.01, accuracy_eps * 1000))
        
        rel_errors = np.abs(members_valid - obs_t) / (np.abs(obs_t) + dynamic_eps)
        hits = rel_errors <= accuracy_delta
        prob_list.append(np.mean(hits))
        
        ensemble_mean = np.mean(members_valid)
        dynamic_eps_mean = max(accuracy_eps, min(obs_t * 0.01, accuracy_eps * 1000))
        rel_error_mean = np.abs(ensemble_mean - obs_t) / (np.abs(obs_t) + dynamic_eps_mean)
        binary_list.append(1.0 if rel_error_mean <= accuracy_delta else 0.0)
    
    if len(prob_list) < 2:
        return np.nan, np.nan, np.nan
    
    prob_forecasts = np.array(prob_list)
    binary_obs = np.array(binary_list)
    
    # 计算Brier Score: BS = mean((p - o)^2)
    bs = np.mean((prob_forecasts - binary_obs) ** 2)
    bs = np.clip(bs, 0.0, 1.0)  # BS 应该在 [0, 1] 范围内
    
    # ===== 计算Brier Skill Score =====
    if climatology_hit_rate is None or np.isnan(climatology_hit_rate):
        # 若无气候态命中率，则使用观测的命中频率作为参考概率
        p_clim = np.mean(binary_obs) if len(binary_obs) > 0 else np.nan
    else:
        p_clim = climatology_hit_rate
    
    if np.isnan(p_clim):
        return bs, np.nan, np.nan
    
    bs_ref = np.mean((p_clim - binary_obs) ** 2)
    if bs_ref < MIN_BS_REF:
        bs_ref = MIN_BS_REF
    
    bss = 1.0 - bs / bs_ref
    bss = np.clip(bss, -5.0, 5.0)
    
    return bs, bs_ref, bss


def _compute_brier_metrics_gpu(ensemble_block: np.ndarray, obs_block: np.ndarray,
                              accuracy_delta: float, accuracy_eps: float,
                              climatology_hit_rate: float = None) -> Tuple[float, float, float]:
    """
    GPU加速版本的Brier Score和Brier Skill Score计算
    """
    if not GPU_AVAILABLE:
        return _compute_brier_metrics(ensemble_block, obs_block, accuracy_delta, accuracy_eps, climatology_hit_rate)
    
    try:
        # 转换为GPU数组
        ensemble_gpu = cp.asarray(ensemble_block)
        obs_gpu = cp.asarray(obs_block)
        
        # 检查有效数据
        mask = ~(cp.isnan(ensemble_gpu).any(axis=1) | cp.isnan(obs_gpu))
        if cp.sum(mask) < 2:
            return cp.nan, cp.nan, cp.nan
        
        ensemble_valid = ensemble_gpu[mask]
        obs_valid = obs_gpu[mask]
        n_members = ensemble_valid.shape[1]
        n_time = ensemble_valid.shape[0]
        
        # 计算每个时间步的命中概率
        prob_forecasts = cp.zeros(n_time)
        
        for t in range(n_time):
            obs_t = obs_valid[t]
            # 动态调整accuracy_eps
            dynamic_eps = cp.maximum(accuracy_eps, cp.minimum(obs_t * 0.01, accuracy_eps * 1000))
            
            # 检查每个成员是否命中
            rel_errors = cp.abs(ensemble_valid[t, :] - obs_t) / (cp.abs(obs_t) + dynamic_eps)
            hits = rel_errors <= accuracy_delta
            prob_forecasts[t] = cp.sum(hits) / n_members
        
        # 计算ensemble mean的二元观测
        ensemble_mean = cp.mean(ensemble_valid, axis=1)
        dynamic_eps_mean = cp.maximum(accuracy_eps, cp.minimum(obs_valid * 0.01, accuracy_eps * 1000))
        rel_error_mean = cp.abs(ensemble_mean - obs_valid) / (cp.abs(obs_valid) + dynamic_eps_mean)
        binary_obs = (rel_error_mean <= accuracy_delta).astype(float)
        
        # 计算Brier Score
        bs = float(cp.mean((prob_forecasts - binary_obs) ** 2))
        
        # 计算Brier Skill Score（与CPU逻辑保持一致）
        if climatology_hit_rate is None or cp.isnan(climatology_hit_rate):
            if len(binary_obs) == 0:
                return float(bs), float('nan'), float('nan')
            p_clim = float(cp.mean(binary_obs))
        else:
            p_clim = float(climatology_hit_rate)
        
        bs_ref = float(cp.mean((p_clim - binary_obs) ** 2))
        if bs_ref < MIN_BS_REF:
            bs_ref = MIN_BS_REF
        
        bss = float(1.0 - bs / bs_ref)
        bss = float(cp.clip(bss, -5.0, 5.0))
        
        # 清理GPU内存
        del ensemble_gpu, obs_gpu, ensemble_valid, obs_valid
        cp.get_default_memory_pool().free_all_blocks()
        
        return float(bs), float(bs_ref), float(bss)
        
    except Exception as e:
        logger.warning(f"GPU Brier计算失败，回退到CPU: {e}")
        return _compute_brier_metrics(ensemble_block, obs_block, accuracy_delta, accuracy_eps, climatology_hit_rate)


# def _process_single_point_independent_blocks(args):
#     """
#     处理单个格点的函数 - 独立 block 版本
#     对每个时间 block 单独计算统计量，然后进行平均
#     """
#     i, j, model_data, obs_data, ensemble_data, climatology_hit_rate, block_size, accuracy_delta, accuracy_eps, n_bootstrap, confidence_level = args
    
#     # 为每个进程创建独立的随机数生成器，避免线程安全问题
#     rng = np.random.RandomState()
    
#     try:
#         model_ts = model_data.isel(lat=i, lon=j).values
#         obs_ts = obs_data.isel(lat=i, lon=j).values
        
#         # 提取ensemble数据（如果有）
#         has_ensemble = ensemble_data is not None
#         if has_ensemble:
#             # ensemble_data shape: (time, number, lat, lon)
#             ensemble_ts = ensemble_data.isel(lat=i, lon=j).values  # (time, number)
#         else:
#             ensemble_ts = None
        
#         # 检查有效数据
#         mask = ~(np.isnan(model_ts) | np.isnan(obs_ts))
#         valid_model = model_ts[mask]
#         valid_obs = obs_ts[mask]
#         if has_ensemble:
#             valid_ensemble = ensemble_ts[mask, :]  # (n_valid, n_members)
#         else:
#             valid_ensemble = None
#         n_time = len(valid_obs)
        
#         if n_time < block_size * 2:
#             return i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, {}
        
#         # 计算可能的 block 数量
#         n_blocks_total = n_time - block_size + 1
        
#         # 如果 block 数量太少，无法进行有意义的 bootstrap
#         if n_blocks_total < 10:
#             return i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, {}
        
#         # 计算气候态（整个时间序列的平均值）
#         model_climatology = np.mean(valid_model)
#         obs_climatology = np.mean(valid_obs)
        
#         # 计算指标 - 独立 block bootstrap
#         accuracy_list, accuracy_clim_list, equitable_accuracy_list, rmse_list, pcc_fisher_z_list = [], [], [], [], []
#         bs_list, bss_list = [], []
        
#         for _ in range(int(n_bootstrap)):
#             # 随机选择 block 进行 bootstrap
#             selected_blocks = rng.choice(n_blocks_total, 
#                                        size=min(n_blocks_total, n_bootstrap), 
#                                        replace=True)
            
#             block_accuracy_list, block_accuracy_clim_list, block_equitable_accuracy_list, block_rmse_list, block_pcc_fisher_z_list = [], [], [], [], []
#             block_bs_list, block_bss_list = [], []
            
#             for block_idx in selected_blocks:
#                 start_idx = block_idx
#                 end_idx = start_idx + block_size
                
#                 model_block = valid_model[start_idx:end_idx]
#                 obs_block = valid_obs[start_idx:end_idx]
                
#                 # 计算单个 block 的指标（包含equitable accuracy）- 智能选择GPU/CPU
#                 accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z = _compute_block_metrics_smart(
#                     model_block, obs_block, accuracy_delta, accuracy_eps,
#                     model_climatology, obs_climatology
#                 )
                
#                 # 计算BS/BSS（如果有ensemble数据）
#                 if has_ensemble and valid_ensemble is not None:
#                     ensemble_block = valid_ensemble[start_idx:end_idx, :]
#                     bs, bs_ref, bss = _compute_brier_metrics_smart(
#                         ensemble_block, obs_block, accuracy_delta, accuracy_eps,
#                         climatology_hit_rate
#                     )
#                 else:
#                     bs, bss = np.nan, np.nan
                
#                 if not np.isnan(accuracy_raw) and not np.isnan(rmse) and not np.isnan(pcc_fisher_z):
#                     block_accuracy_list.append(accuracy_raw)
#                     if not np.isnan(accuracy_clim):
#                         block_accuracy_clim_list.append(accuracy_clim)
#                     if not np.isnan(equitable_accuracy):
#                         block_equitable_accuracy_list.append(equitable_accuracy)
#                     block_rmse_list.append(rmse)
#                     block_pcc_fisher_z_list.append(pcc_fisher_z)
                    
#                     if has_ensemble and not np.isnan(bs):
#                         block_bs_list.append(bs)
#                         if not np.isnan(bss):
#                             block_bss_list.append(bss)
            
#             # 对当前 bootstrap 样本的所有 block 进行平均
#             if block_accuracy_list and block_rmse_list and block_pcc_fisher_z_list:
#                 accuracy_list.append(np.mean(block_accuracy_list))
#                 if block_accuracy_clim_list:
#                     accuracy_clim_list.append(np.mean(block_accuracy_clim_list))
#                 if block_equitable_accuracy_list:
#                     equitable_accuracy_list.append(np.mean(block_equitable_accuracy_list))
#                 rmse_list.append(np.mean(block_rmse_list))
#                 pcc_fisher_z_list.append(np.mean(block_pcc_fisher_z_list))
                
#                 if block_bs_list:
#                     bs_list.append(np.mean(block_bs_list))
#                     if block_bss_list:
#                         bss_list.append(np.mean(block_bss_list))
        
#         # 如果没有任何有效的 bootstrap 样本，返回 NaN
#         if not accuracy_list or not rmse_list or not pcc_fisher_z_list:
#             return i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, {}
        
#         # 计算所有 bootstrap 样本的均值
#         b_accuracy = np.clip(np.mean(accuracy_list), 0.0, 1.0)  # 确保在 [0, 1] 范围内
#         b_accuracy_clim = np.mean(accuracy_clim_list) if accuracy_clim_list else np.nan
#         b_equitable_accuracy = np.mean(equitable_accuracy_list) if equitable_accuracy_list else np.nan
#         b_rmse = np.mean(rmse_list)
#         b_pcc = np.clip(fisher_z_inv(np.mean(pcc_fisher_z_list)), -1.0, 1.0)  # 确保在 [-1, 1] 范围内
        
#         # 计算BS/BSS均值
#         b_bs = np.mean(bs_list) if bs_list else np.nan
#         b_bss = np.mean(bss_list) if bss_list else np.nan
        
#         # 计算置信区间
#         alpha = (1 - confidence_level) / 2
#         bootstrap_results = {
#             'accuracy_ci': np.percentile(accuracy_list, [alpha * 100, (1 - alpha) * 100]),
#             'rmse_ci': np.percentile(rmse_list, [alpha * 100, (1 - alpha) * 100]),
#             'pcc_ci': fisher_z_inv(np.percentile(pcc_fisher_z_list, [alpha * 100, (1 - alpha) * 100]))
#         }
#         if equitable_accuracy_list:
#             bootstrap_results['equitable_accuracy_ci'] = np.percentile(equitable_accuracy_list, [alpha * 100, (1 - alpha) * 100])
#         if bs_list:
#             bootstrap_results['bs_ci'] = np.percentile(bs_list, [alpha * 100, (1 - alpha) * 100])
#         if bss_list:
#             bootstrap_results['bss_ci'] = np.percentile(bss_list, [alpha * 100, (1 - alpha) * 100])
        
#         return i, j, b_accuracy, b_accuracy_clim, b_equitable_accuracy, b_rmse, b_pcc, b_bs, b_bss, bootstrap_results
        
#     except Exception as e:
#         logger.debug(f"处理格点 ({i}, {j}) 时出错: {e}")
#         return i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, {}


def _process_chunk_numpy(chunk_args):
    """
    处理一个分块的行数据 - 批量处理多行以提高效率
    """
    results = []
    for args in chunk_args:
        result = _process_single_row_numpy(args)
        results.append(result)
    return results


def _process_single_row_numpy(args):
    """
    处理单行格点的函数 - 使用numpy数组版本
    按行并行处理，提高效率
    支持网格不匹配的情况：根据经纬度坐标匹配格点
    """
    i, model_values, obs_values, obs_climatology_values, ensemble_values, climatology_hit_rates, \
    model_lat_values, model_lon_values, obs_lat_values, obs_lon_values, grids_match, \
    block_size, accuracy_delta, accuracy_eps, n_bootstrap, confidence_level = args
    
    # 为每个进程创建独立的随机数生成器，避免线程安全问题
    rng = np.random.RandomState()
    
    # 获取网格维度
    model_n_lat, model_n_lon = model_values.shape[1], model_values.shape[2]
    obs_n_lat, obs_n_lon = obs_values.shape[1], obs_values.shape[2]
    
    # 检查ensemble数据是否存在
    has_ensemble = ensemble_values is not None
    
    # 检查行索引是否有效（使用观测数据的维度，因为输出基于观测网格）
    if i >= obs_n_lat:
        return (i, np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan),
                np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan),
                np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan),
                np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan),
                np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan),
                np.full(obs_n_lon, np.nan), np.full(obs_n_lon, np.nan))
    
    # 初始化结果数组（使用观测数据的经度维度，因为输出基于观测网格，保持观测数据的完整范围）
    row_accuracy = np.full(obs_n_lon, np.nan)
    row_accuracy_clim = np.full(obs_n_lon, np.nan)
    row_equitable_accuracy = np.full(obs_n_lon, np.nan)
    row_rmse = np.full(obs_n_lon, np.nan)
    row_pcc = np.full(obs_n_lon, np.nan)
    row_bs = np.full(obs_n_lon, np.nan)
    row_bss = np.full(obs_n_lon, np.nan)
    row_accuracy_ci_lower = np.full(obs_n_lon, np.nan)
    row_accuracy_ci_upper = np.full(obs_n_lon, np.nan)
    row_rmse_ci_lower = np.full(obs_n_lon, np.nan)
    row_rmse_ci_upper = np.full(obs_n_lon, np.nan)
    row_pcc_ci_lower = np.full(obs_n_lon, np.nan)
    row_pcc_ci_upper = np.full(obs_n_lon, np.nan)
    row_equitable_accuracy_ci_lower = np.full(obs_n_lon, np.nan)
    row_equitable_accuracy_ci_upper = np.full(obs_n_lon, np.nan)
    row_bs_ci_lower = np.full(obs_n_lon, np.nan)
    row_bs_ci_upper = np.full(obs_n_lon, np.nan)
    row_bss_ci_lower = np.full(obs_n_lon, np.nan)
    row_bss_ci_upper = np.full(obs_n_lon, np.nan)
    
    # 循环遍历观测数据的格点（保持观测数据的完整范围）
    # 对于每个观测格点，找到对应的预报格点
    obs_i = i  # 观测数据的行索引
    for obs_j in range(obs_n_lon):
        try:
            # 检查观测格点是否有有效数据（海面等区域可能全为NaN，应跳过）
            obs_ts_check = obs_values[:, obs_i, obs_j]
            if np.all(np.isnan(obs_ts_check)):
                # 如果该格点所有时间步都是NaN（如海面），跳过该格点的计算
                continue
            
            # 根据网格是否匹配来选择数据获取方式
            # 注意：网格对齐只基于经纬度坐标，不受数据内容（NaN值）影响
            if grids_match:
                # 网格匹配：直接使用索引
                if obs_i >= model_n_lat or obs_j >= model_n_lon:
                    continue
                model_i, model_j = obs_i, obs_j
            else:
                # 网格不匹配：根据经纬度坐标找到最近的预报格点
                if obs_lat_values is None or obs_lon_values is None or \
                   model_lat_values is None or model_lon_values is None:
                    continue
                obs_lat = obs_lat_values[obs_i]
                obs_lon = obs_lon_values[obs_j]
                
                # 找到最近的预报格点（仅基于坐标，不受数据内容影响）
                lat_distances = np.abs(model_lat_values - obs_lat)
                lon_distances = np.abs(model_lon_values - obs_lon)
                model_i = np.argmin(lat_distances)
                model_j = np.argmin(lon_distances)
                
                # 检查距离是否合理（不超过0.5度，1度网格间距的一半）
                # 对于1度网格，0.5度容差可以匹配所有合理的网格变体
                if lat_distances[model_i] > 0.5 or lon_distances[model_j] > 0.5:
                    continue
                
                if model_i >= model_n_lat or model_j >= model_n_lon:
                    continue
                
            # 从numpy数组获取时间序列
            model_ts = model_values[:, model_i, model_j]  # 时间, lat, lon
            obs_ts = obs_values[:, obs_i, obs_j]
            
            # 提取ensemble数据（如果有）
            if has_ensemble:
                ensemble_ts = ensemble_values[:, :, model_i, model_j]  # (time, number)
            else:
                ensemble_ts = None
            
            # 获取气候态命中率（如果有）
            # climatology_hit_rates基于观测数据的网格，使用obs_i和obs_j
            if climatology_hit_rates is not None:
                # 注意：climatology_hit_rates的维度应该与obs_climatology一致，基于观测网格
                if obs_i < climatology_hit_rates.shape[0] and obs_j < climatology_hit_rates.shape[1]:
                    clim_hit_rate = climatology_hit_rates[obs_i, obs_j]
                else:
                    clim_hit_rate = None
            else:
                clim_hit_rate = None
            
            # 获取观测气候态值
            obs_clim_value = obs_climatology_values[obs_i, obs_j]
            
            # 检查有效数据
            mask = ~(np.isnan(model_ts) | np.isnan(obs_ts))
            valid_model = model_ts[mask]
            valid_obs = obs_ts[mask]
            if has_ensemble:
                valid_ensemble = ensemble_ts[mask, :]  # (n_valid, n_members)
            else:
                valid_ensemble = None
            n_time = len(valid_obs)
            
            if n_time < block_size * 2:
                continue
            
            # 计算可能的 block 数量
            n_blocks_total = n_time - block_size + 1
            
            if n_blocks_total < 10:
                continue
            
            # *** 优化：使用zeros预分配，减少内存分配开销 ***
            accuracy_array = np.zeros(int(n_bootstrap))
            accuracy_clim_array = np.zeros(int(n_bootstrap))
            equitable_accuracy_array = np.zeros(int(n_bootstrap))
            rmse_array = np.zeros(int(n_bootstrap))
            pcc_fisher_z_array = np.zeros(int(n_bootstrap))
            bs_array = np.zeros(int(n_bootstrap))
            bss_array = np.zeros(int(n_bootstrap))
            
            # 预计算block采样数量
            n_blocks_per_bootstrap = min(n_blocks_total, n_bootstrap)
            
            for bootstrap_idx in range(int(n_bootstrap)):
                # 随机选择 block 进行 bootstrap
                selected_blocks = rng.choice(n_blocks_total, 
                                           size=n_blocks_per_bootstrap, 
                                           replace=True)
                
                # *** 优化：使用zeros预分配，减少内存开销 ***
                block_accuracy = np.zeros(n_blocks_per_bootstrap)
                block_accuracy_clim = np.zeros(n_blocks_per_bootstrap)
                block_equitable_accuracy = np.zeros(n_blocks_per_bootstrap)
                block_rmse = np.zeros(n_blocks_per_bootstrap)
                block_pcc_fisher_z = np.zeros(n_blocks_per_bootstrap)
                block_bs = np.zeros(n_blocks_per_bootstrap)
                block_bss = np.zeros(n_blocks_per_bootstrap)
                
                valid_count = 0
                
                for idx, block_idx in enumerate(selected_blocks):
                    start_idx = block_idx
                    end_idx = start_idx + block_size
                    
                    model_block = valid_model[start_idx:end_idx]
                    obs_block = valid_obs[start_idx:end_idx]
                    
                    accuracy_raw, accuracy_clim, equitable_accuracy, rmse, pcc_fisher_z = _compute_block_metrics_smart(
                        model_block, obs_block, accuracy_delta, accuracy_eps,
                        None, obs_clim_value
                    )
                    
                    # 计算BS/BSS（如果有ensemble数据）
                    if has_ensemble and valid_ensemble is not None:
                        ensemble_block = valid_ensemble[start_idx:end_idx, :]
                        bs, bs_ref, bss = _compute_brier_metrics_smart(
                            ensemble_block, obs_block, accuracy_delta, accuracy_eps,
                            clim_hit_rate
                        )
                    else:
                        bs, bss = np.nan, np.nan
                    
                    # *** 优化：直接存储到数组，避免list append ***
                    if not np.isnan(accuracy_raw) and not np.isnan(rmse) and not np.isnan(pcc_fisher_z):
                        block_accuracy[idx] = accuracy_raw
                        block_accuracy_clim[idx] = accuracy_clim
                        block_equitable_accuracy[idx] = equitable_accuracy
                        block_rmse[idx] = rmse
                        block_pcc_fisher_z[idx] = pcc_fisher_z
                        
                        if has_ensemble:
                            block_bs[idx] = bs
                            block_bss[idx] = bss
                        
                        valid_count += 1
                
                # *** 优化：使用nanmean向量化计算平均值 ***
                if valid_count > 0:
                    accuracy_array[bootstrap_idx] = np.nanmean(block_accuracy)
                    accuracy_clim_array[bootstrap_idx] = np.nanmean(block_accuracy_clim)
                    equitable_accuracy_array[bootstrap_idx] = np.nanmean(block_equitable_accuracy)
                    rmse_array[bootstrap_idx] = np.nanmean(block_rmse)
                    pcc_fisher_z_array[bootstrap_idx] = np.nanmean(block_pcc_fisher_z)
                    
                    if has_ensemble:
                        bs_array[bootstrap_idx] = np.nanmean(block_bs)
                        bss_array[bootstrap_idx] = np.nanmean(block_bss)
            
            # *** 优化：使用数组进行向量化计算 ***
            # 计算所有 bootstrap 样本的均值和置信区间
            valid_bootstraps = ~np.isnan(accuracy_array)
            if np.sum(valid_bootstraps) > 0:
                # 使用观测数据的列索引（obs_j）来存储结果，保持观测数据的完整范围
                row_accuracy[obs_j] = np.clip(np.nanmean(accuracy_array), 0.0, 1.0)
                row_accuracy_clim[obs_j] = np.nanmean(accuracy_clim_array)
                row_equitable_accuracy[obs_j] = np.nanmean(equitable_accuracy_array)
                row_rmse[obs_j] = np.nanmean(rmse_array)
                row_pcc[obs_j] = np.clip(fisher_z_inv(np.nanmean(pcc_fisher_z_array)), -1.0, 1.0)
                
                # 计算BS/BSS均值
                row_bs[obs_j] = np.nanmean(bs_array)
                row_bss[obs_j] = np.nanmean(bss_array)
                
                # *** 优化：使用nanpercentile向量化计算置信区间 ***
                alpha = (1 - confidence_level) / 2
                row_accuracy_ci_lower[obs_j] = np.nanpercentile(accuracy_array, alpha * 100)
                row_accuracy_ci_upper[obs_j] = np.nanpercentile(accuracy_array, (1 - alpha) * 100)
                row_equitable_accuracy_ci_lower[obs_j] = np.nanpercentile(equitable_accuracy_array, alpha * 100)
                row_equitable_accuracy_ci_upper[obs_j] = np.nanpercentile(equitable_accuracy_array, (1 - alpha) * 100)
                row_rmse_ci_lower[obs_j] = np.nanpercentile(rmse_array, alpha * 100)
                row_rmse_ci_upper[obs_j] = np.nanpercentile(rmse_array, (1 - alpha) * 100)
                row_pcc_ci_lower[obs_j] = fisher_z_inv(np.nanpercentile(pcc_fisher_z_array, alpha * 100))
                row_pcc_ci_upper[obs_j] = fisher_z_inv(np.nanpercentile(pcc_fisher_z_array, (1 - alpha) * 100))
                row_bs_ci_lower[obs_j] = np.nanpercentile(bs_array, alpha * 100)
                row_bs_ci_upper[obs_j] = np.nanpercentile(bs_array, (1 - alpha) * 100)
                row_bss_ci_lower[obs_j] = np.nanpercentile(bss_array, alpha * 100)
                row_bss_ci_upper[obs_j] = np.nanpercentile(bss_array, (1 - alpha) * 100)
            
        except Exception as e:
            continue
    
    return (i, row_accuracy, row_accuracy_clim, row_equitable_accuracy, row_rmse, row_pcc, row_bs, row_bss,
            row_accuracy_ci_lower, row_accuracy_ci_upper, 
            row_equitable_accuracy_ci_lower, row_equitable_accuracy_ci_upper,
            row_rmse_ci_lower, row_rmse_ci_upper, 
            row_pcc_ci_lower, row_pcc_ci_upper,
            row_bs_ci_lower, row_bs_ci_upper,
            row_bss_ci_lower, row_bss_ci_upper)


class UnifiedBlockBootstrapAnalyzer:
    """统一的 Block Bootstrap 分析器"""
    
    def __init__(self, 
                 block_size: int = 36,
                 n_bootstrap: int = 100,
                 confidence_level: float = 0.95,
                 n_jobs: int = 64,
                 parallel_backend: str = 'process',
                 parallel_strategy: str = 'auto',
                 accuracy_delta: float = 0.015,
                 accuracy_eps: float = 1e-3):
        """初始化分析器"""
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        # 智能并行度设置
        total_cores = mp.cpu_count()
        if n_jobs > total_cores:
            logger.warning(f"请求的并行作业数({n_jobs})超过系统核心数({total_cores})，调整为{total_cores}")
            n_jobs = total_cores
        
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend if parallel_backend in ('process','thread') else 'process'
        self.parallel_strategy = parallel_strategy
        
        # 记录并行配置
        logger.info(f"并行配置: {self.parallel_backend}后端, {self.n_jobs}个作业, 系统总核心数: {total_cores}")
        self.accuracy_delta = accuracy_delta
        self.accuracy_eps = accuracy_eps
        
        # 区域定义
        self.regions = {
            "NorthEast": {"lat": [35, 45], "lon": [110, 125], "name": "NE"},
            "NorthChina": {"lat": [35, 41], "lon": [110, 118], "name": "NC"},
            "EastChina": {"lat": [23, 36], "lon": [115, 123], "name": "EC"},
            "SouthChina": {"lat": [18, 24], "lon": [109, 121], "name": "SC"},
            "SouthWest": {"lat": [21, 30], "lon": [98, 107], "name": "SW"},
            "NorthWest": {"lat": [34, 48], "lon": [75, 105], "name": "NW"},
            "Tibetan": {"lat": [26, 40], "lon": [80, 100], "name": "TP"},
            "WholeChina": {"lat": [15, 55], "lon": [70, 140], "name": "All"}
        }
        
        # 数据加载器
        self.data_loader = DataLoader(
            obs_dir=DATA_PATHS["obs_dir"],
            forecast_dir=DATA_PATHS["forecast_dir"]
        )
        
        # 输出目录
        self.output_dir = Path(DATA_PATHS["output_dir"]) / "block_bootstrap_score"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.results_dir = self.output_dir / "results"  # NetCDF结果文件
        self.plots_dir = self.output_dir / "plots"      # 图像文件
        self.summary_dir = self.output_dir / "summary"  # 汇总文件
        
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.summary_dir.mkdir(exist_ok=True)
        
        # 存储所有模型的 RMSE 范围，用于统一归一化
        self.global_rmse_min = np.nan
        self.global_rmse_max = np.nan
        
        logger.info(f"UnifiedBlockBootstrapAnalyzer 初始化完成")
        logger.info(f"Block 大小: {block_size} 个月, Bootstrap 次数: {n_bootstrap}")
        logger.info(f"并行后端: {self.parallel_backend}, 并行作业数: {self.n_jobs}")
        logger.info(f"使用独立 block bootstrap 方法")
        
        # 初始化并行处理器
        self.parallel_processor = ParallelProcessor(
            n_jobs=self.n_jobs,
            backend=self.parallel_backend
        )
    
    def _resample_time(self, data: Optional[xr.DataArray]) -> Optional[xr.DataArray]:
        if data is None or 'time' not in data.coords:
            return data
        return data.resample(time='1MS').mean().sel(time=slice('1993-01-01', '2020-12-31'))

    def _load_obs_data(self, var_type: str) -> Optional[xr.DataArray]:
        try:
            data = self.data_loader.load_obs_data(var_type)
            data = self._resample_time(data)
            return data.load()
        except Exception as exc:
            logger.error(f"加载观测数据失败: {exc}")
            return None

    def _load_forecast_data(self, model: str, var_type: str, leadtime: int) -> Optional[xr.DataArray]:
        try:
            data = self.data_loader.load_forecast_data(model, var_type, leadtime)
            data = self._resample_time(data)
            return data.load()
        except Exception as exc:
            logger.error(f"加载预报数据失败 {model} L{leadtime}: {exc}")
            return None

    def _load_forecast_ensemble(self, model: str, var_type: str, leadtime: int) -> Optional[xr.DataArray]:
        try:
            data = self.data_loader.load_forecast_data_ensemble(model, var_type, leadtime)
            return data.load() if data is not None else None
        except Exception as exc:
            logger.error(f"加载ensemble数据失败 {model} L{leadtime}: {exc}")
            return None
    
    def _create_land_mask(self, var_type: str, target_data: xr.DataArray) -> Optional[xr.DataArray]:
        """
        创建陆地掩膜（陆地为True，海洋为False）
        基于观测数据，如果观测数据在某个格点为NaN，则认为是海洋
        使用 toolkit 中的 create_land_mask 函数
        
        Args:
            var_type: 变量类型
            target_data: 目标数据（用于对齐掩膜的网格）
            
        Returns:
            陆地掩膜（与target_data对齐），陆地为True，海洋为False
        """
        try:
            # 加载观测数据（用于掩膜，不fillna，保留原始NaN）
            obs_data_raw = self.data_loader.load_obs_data(var_type, for_mask=True)
            if obs_data_raw is None:
                logger.warning(f"无法加载观测数据来创建掩膜，将使用数据本身的NaN作为掩膜")
                return None
            
            # 重采样时间（如果需要）
            obs_data_raw = self._resample_time(obs_data_raw)
            obs_data_raw = obs_data_raw.load()
            
            # 使用 toolkit 中的函数创建掩膜，传入var_type参数
            return create_land_mask(obs_data_raw, target_data, var_type=var_type)
            
        except Exception as exc:
            logger.warning(f"创建陆地掩膜失败: {exc}，将使用数据本身的NaN作为掩膜")
            return None
    
    def _compute_data_extent(self, data: xr.DataArray, land_mask: Optional[xr.DataArray] = None) -> Tuple[float, float, float, float]:
        """
        计算数据的实际范围（经度、纬度）
        根据有效数据的存在范围来决定，并进行取整
        使用 toolkit 中的 compute_data_extent 函数
        
        Args:
            data: 数据数组
            land_mask: 可选的陆地掩膜（如果提供，只考虑陆地区域）
            
        Returns:
            (lon_min, lon_max, lat_min, lat_max) 取整后的范围
        """
        try:
            # 使用 toolkit 中的函数计算数据范围
            return compute_data_extent(data, land_mask)
        except Exception as exc:
            logger.warning(f"计算数据范围失败: {exc}，使用整个数据范围")
            try:
                lon_min = float(data.lon.min().values)
                lon_max = float(data.lon.max().values)
                lat_min = float(data.lat.min().values)
                lat_max = float(data.lat.max().values)
            except:
                # 如果连这个都失败，返回默认值
                lon_min, lon_max, lat_min, lat_max = 0.0, 360.0, -90.0, 90.0
            return lon_min, lon_max, lat_min, lat_max

    def _align_data(self, obs_data: xr.DataArray, fcst_data: xr.DataArray) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
        if obs_data is None or fcst_data is None:
            return None, None
        obs_time, fcst_time = align_time_to_monthly(obs_data, fcst_data, min_common_months=12)
        if obs_time is None or fcst_time is None:
            return None, None
        # 使用no_interp=True，只裁剪不重采样，保持预报数据原始网格
        return align_spatial_to_obs(obs_time, fcst_time, no_interp=True)
    
    def _compute_climatology_hit_rate(self, obs_data: xr.DataArray, obs_climatology: xr.DataArray,
                                     accuracy_delta: float, accuracy_eps: float) -> np.ndarray:
        """
        计算每个格点的气候态命中率（用于BSS参考预报）
        使用观测数据的气候态作为参考，确保所有模型使用统一基准
        
        Args:
            obs_data: 观测数据 (time, lat, lon)
            obs_climatology: 观测的气候态 (lat, lon)
            accuracy_delta: accuracy阈值
            accuracy_eps: 防止除零的小常数
        
        Returns:
            climatology_hit_rates: 每个格点的气候态命中率 (lat, lon)
        """
        logger.info("计算气候态命中率（基于观测气候态）...")
        
        # 初始化结果数组
        n_lat, n_lon = len(obs_data.lat), len(obs_data.lon)
        climatology_hit_rates = np.full((n_lat, n_lon), np.nan)
        
        # 对每个格点计算气候态命中率
        for i in range(n_lat):
            for j in range(n_lon):
                try:
                    # 获取该格点的观测时间序列
                    obs_ts = obs_data.isel(lat=i, lon=j).values
                    
                    # 获取该格点的观测气候态值
                    obs_clim_value = obs_climatology.isel(lat=i, lon=j).values
                    
                    # 检查有效数据
                    mask = ~np.isnan(obs_ts)
                    if np.sum(mask) < 2:
                        continue
                    
                    valid_obs = obs_ts[mask]
                    
                    # 计算气候态预报（总是预报观测气候态值）对每个观测的相对误差
                    rel_errors = np.abs(obs_clim_value - valid_obs) / (np.abs(valid_obs) + accuracy_eps)
                    
                    # 计算命中率
                    hits = rel_errors <= accuracy_delta
                    hit_rate = np.mean(hits)
                    
                    climatology_hit_rates[i, j] = hit_rate
                    
                except Exception as e:
                    continue
        
        return climatology_hit_rates
    
    def process_gridded_data_independent_blocks(self,
                                              model_data: xr.DataArray,
                                              obs_data: xr.DataArray,
                                              obs_climatology: xr.DataArray,
                                              ensemble_data: xr.DataArray = None,
                                              climatology_hit_rates: np.ndarray = None,
                                              accuracy_delta: float = None,
                                              accuracy_eps: float = None) -> xr.Dataset:
        """处理网格数据 - 使用独立 block bootstrap 方法（优化内存使用）"""
        # 使用传入的参数或默认值
        if accuracy_delta is None:
            accuracy_delta = self.accuracy_delta
        if accuracy_eps is None:
            accuracy_eps = self.accuracy_eps
        
        # 转换obs_climatology为numpy数组
        obs_climatology_values = obs_climatology.values  # (lat, lon)
        
        # 使用观测数据的网格作为输出网格，保持观测数据的完整范围
        obs_n_lat, obs_n_lon = len(obs_data.lat), len(obs_data.lon)
        n_lat, n_lon = obs_n_lat, obs_n_lon  # 输出网格使用观测数据的维度
        
        # 初始化结果数组（使用观测数据的维度）
        shape = (n_lat, n_lon)
        b_accuracy = np.full(shape, np.nan)
        b_accuracy_clim = np.full(shape, np.nan)
        b_equitable_accuracy = np.full(shape, np.nan)
        b_rmse = np.full(shape, np.nan)
        b_pcc = np.full(shape, np.nan)
        b_bs = np.full(shape, np.nan)
        b_bss = np.full(shape, np.nan)
        accuracy_ci_lower = np.full(shape, np.nan)
        accuracy_ci_upper = np.full(shape, np.nan)
        equitable_accuracy_ci_lower = np.full(shape, np.nan)
        equitable_accuracy_ci_upper = np.full(shape, np.nan)
        rmse_ci_lower = np.full(shape, np.nan)
        rmse_ci_upper = np.full(shape, np.nan)
        pcc_ci_lower = np.full(shape, np.nan)
        pcc_ci_upper = np.full(shape, np.nan)
        bs_ci_lower = np.full(shape, np.nan)
        bs_ci_upper = np.full(shape, np.nan)
        bss_ci_lower = np.full(shape, np.nan)
        bss_ci_upper = np.full(shape, np.nan)
        
        logger.info(f"使用独立 block bootstrap 方法 (n_jobs={self.n_jobs})")
        logger.info(f"数据形状: model_data={model_data.shape}, obs_data={obs_data.shape}")
        logger.info(f"数据内存占用: model_data={model_data.nbytes / 1024**2:.1f} MB, obs_data={obs_data.nbytes / 1024**2:.1f} MB")
        
        # 记录内存使用
        log_memory_usage("开始数据处理")
        
        # 检查内存限制
        if not check_memory_limit():
            logger.error("内存使用量过高，无法进行数据处理")
            return None
        
        # 注意：观测数据应该保持完整范围，不应再进行裁剪
        # 在 _align_data 中已经完成对齐，观测数据保持原始范围，预报数据已裁剪到观测范围内
        # 这里只对 ensemble 数据进行必要的裁剪（如果它还没有对齐）
        
        # 如果 ensemble 数据的空间范围超出观测范围，将其裁剪到观测范围内
        # 但观测数据本身不应被裁剪
        if ensemble_data is not None and 'lat' in ensemble_data.coords and 'lon' in ensemble_data.coords:
            # 获取观测数据的空间范围
            obs_lat_min = float(obs_data.lat.min())
            obs_lat_max = float(obs_data.lat.max())
            obs_lon_min = float(obs_data.lon.min())
            obs_lon_max = float(obs_data.lon.max())
            
            # 获取ensemble数据的空间范围
            ens_lat_min = float(ensemble_data.lat.min())
            ens_lat_max = float(ensemble_data.lat.max())
            ens_lon_min = float(ensemble_data.lon.min())
            ens_lon_max = float(ensemble_data.lon.max())
            
            # 如果ensemble数据超出观测范围，裁剪到观测范围内
            if (ens_lat_min < obs_lat_min or ens_lat_max > obs_lat_max or 
                ens_lon_min < obs_lon_min or ens_lon_max > obs_lon_max):
                lat_min = max(obs_lat_min, ens_lat_min)
                lat_max = min(obs_lat_max, ens_lat_max)
                lon_min = max(obs_lon_min, ens_lon_min)
                lon_max = min(obs_lon_max, ens_lon_max)
                
                if lat_max > lat_min and lon_max > lon_min:
                    ensemble_data = ensemble_data.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max)
                    )
                    logger.info(f"裁剪ensemble数据到观测范围: ensemble={ensemble_data.shape}")
        
        logger.info(f"数据处理前: obs={obs_data.shape}, model={model_data.shape}")
        if ensemble_data is not None:
            logger.info(f"数据处理前: ensemble={ensemble_data.shape}")
        
        # 将数据转换为numpy数组以提高访问速度
        model_values = model_data.values  # 已经是内存中的数组
        obs_values = obs_data.values
        
        # 转换ensemble数据（如果有）
        if ensemble_data is not None:
            ensemble_values = ensemble_data.values  # shape: (time, number, lat, lon)
            logger.info(f"Ensemble数据形状: {ensemble_values.shape}, 成员数={ensemble_values.shape[1]}")
            logger.info(f"Ensemble数据内存占用: {ensemble_data.nbytes / 1024**2:.1f} MB")
        else:
            ensemble_values = None
            logger.info("无Ensemble数据，跳过BS/BSS计算")
        
        # 检查时间维度一致性，确保所有数据使用共同的时间范围
        # 时间长度不完全一致是正常的，只对能够进行计算的时间范围进行计算
        time_shapes = [model_values.shape[0], obs_values.shape[0]]
        if ensemble_values is not None:
            time_shapes.append(ensemble_values.shape[0])
        
        min_time = min(time_shapes)
        max_time = max(time_shapes)
        
        if min_time != max_time:
            logger.info(f"时间维度不完全一致: model={model_values.shape[0]}, obs={obs_values.shape[0]}, "
                       f"ensemble={ensemble_values.shape[0] if ensemble_values is not None else 'N/A'}, "
                       f"使用共同时间范围: {min_time}个时间点")
        
        # 所有数据都裁剪到共同的时间范围
        if model_values.shape[0] > min_time:
            model_values = model_values[:min_time, :, :]
        if obs_values.shape[0] > min_time:
            obs_values = obs_values[:min_time, :, :]
        if ensemble_values is not None and ensemble_values.shape[0] > min_time:
            ensemble_values = ensemble_values[:min_time, :, :, :]
        
        # 获取预报数据的网格维度（用于循环和匹配）
        model_n_lat, model_n_lon = len(model_data.lat), len(model_data.lon)
        
        # 检查网格是否匹配
        if obs_n_lat != model_n_lat or obs_n_lon != model_n_lon:
            logger.info(f"网格不匹配: obs=({obs_n_lat}, {obs_n_lon}), model=({model_n_lat}, {model_n_lon})")
            logger.info("将使用观测网格作为输出网格，通过经纬度匹配计算预报格点")
        
        # 自适应调整并行度
        # 智能并行度调整策略
        current_memory = get_memory_usage_gb()
        memory_usage_ratio = current_memory / MAX_MEMORY_GB
        total_cores = mp.cpu_count()
        
        # 基于数据大小和系统资源的动态调整
        data_size_mb = (model_data.nbytes + obs_data.nbytes) / (1024 * 1024)
        complexity_factor = (n_lat * n_lon * self.n_bootstrap) / 1000000  # 百万级复杂度
        
        # 基础并行度
        base_jobs = min(self.n_jobs, total_cores)
        
        # 内存限制调整
        if memory_usage_ratio > 0.8:
            adjusted_jobs = max(1, base_jobs // 2)
            logger.warning(f"内存使用率 {memory_usage_ratio*100:.1f}%，降低并行度到 {adjusted_jobs}")
        elif memory_usage_ratio > 0.6:
            adjusted_jobs = max(1, int(base_jobs * 0.75))
            logger.info(f"内存使用率 {memory_usage_ratio*100:.1f}%，降低并行度到 {adjusted_jobs}")
        else:
            adjusted_jobs = base_jobs
            
        # *** 优化：数据复杂度调整 ***
        if complexity_factor > 10:  # 高复杂度任务
            # 高复杂度时适当限制并行度，避免内存竞争
            adjusted_jobs = min(adjusted_jobs, total_cores // 2)
            logger.info(f"高复杂度任务(复杂度={complexity_factor:.1f})，限制并行度到 {adjusted_jobs}")
        elif complexity_factor < 1:  # 低复杂度任务
            # 低复杂度时使用最大并行度
            adjusted_jobs = min(adjusted_jobs, total_cores)
            logger.info(f"低复杂度任务(复杂度={complexity_factor:.1f})，使用最大并行度 {adjusted_jobs}")
        else:  # 中等复杂度
            # 保持默认并行度，不额外增加
            logger.info(f"中等复杂度任务(复杂度={complexity_factor:.1f})，保持并行度 {adjusted_jobs}")
            
        logger.info(f"最终并行度: {adjusted_jobs} (数据大小: {data_size_mb:.1f}MB, 复杂度: {complexity_factor:.1f})")
        
        # 准备参数列表 - 按行并行，传递numpy数组而不是xarray对象
        # 如果网格不同，需要传递观测数据的经纬度坐标用于格点匹配
        obs_lat_values = obs_data.lat.values if hasattr(obs_data, 'lat') else None
        obs_lon_values = obs_data.lon.values if hasattr(obs_data, 'lon') else None
        model_lat_values = model_data.lat.values
        model_lon_values = model_data.lon.values
        
        # 检查网格是否匹配（先检查长度，再检查值）
        grids_match = False
        if obs_lat_values is not None and obs_lon_values is not None and \
           model_lat_values is not None and model_lon_values is not None:
            if (obs_n_lat == model_n_lat and obs_n_lon == model_n_lon and
                len(obs_lat_values) == len(model_lat_values) and
                len(obs_lon_values) == len(model_lon_values)):
                # 长度匹配，再检查值是否相同
                try:
                    grids_match = (np.allclose(obs_lat_values, model_lat_values, rtol=0, atol=1e-6) and
                                  np.allclose(obs_lon_values, model_lon_values, rtol=0, atol=1e-6))
                except (ValueError, TypeError):
                    # 如果比较失败（如形状不匹配），则网格不匹配
                    grids_match = False
            else:
                grids_match = False
        else:
            grids_match = False
        
        if grids_match:
            logger.info("观测和预报网格匹配，使用直接索引访问")
        else:
            logger.info(f"观测和预报网格不匹配: obs=({obs_n_lat}, {obs_n_lon}), model=({model_n_lat}, {model_n_lon})，将使用经纬度匹配")
        
        args_list = []
        for i in range(n_lat):
            args_list.append((i, model_values, obs_values, obs_climatology_values, ensemble_values, climatology_hit_rates,
                            model_lat_values, model_lon_values, obs_lat_values, obs_lon_values, grids_match,
                            self.block_size, accuracy_delta, accuracy_eps, self.n_bootstrap, self.confidence_level))
        
        # 使用优化的并行处理策略
        logger.info(f"开始并行处理 {n_lat} 行数据，使用 {adjusted_jobs} 个进程...")
        
        completed_rows = 0
        start_time = datetime.now()
        
        # 根据任务复杂度和用户选择确定并行策略
        use_chunked = False
        if hasattr(self, 'parallel_strategy'):
            if self.parallel_strategy == 'chunked':
                use_chunked = True
            elif self.parallel_strategy == 'standard':
                use_chunked = False
            else:  # auto
                # *** 优化：更严格的chunked触发条件 ***
                # 只有在数据量大且并行度高时才使用chunked
                # 小数据集使用standard策略更快（减少进程间通信开销）
                use_chunked = (complexity_factor > 10 and n_lat > 100 and adjusted_jobs > 16)
        else:
            use_chunked = (complexity_factor > 10 and n_lat > 100 and adjusted_jobs > 16)
        
        logger.info(f"并行策略选择: {'chunked' if use_chunked else 'standard'} (n_lat={n_lat}, complexity={complexity_factor:.1f})")
            
        if use_chunked:
            # 高复杂度任务：使用chunked processing减少进程间通信开销
            # 优化：确保每个块至少有4行数据，减少进程间通信开销
            min_chunk_size = 4  # 最小块大小
            chunk_size = max(min_chunk_size, n_lat // adjusted_jobs)
            # 如果总行数少，适当减小块大小但不小于2
            if n_lat < adjusted_jobs * min_chunk_size:
                chunk_size = max(2, n_lat // adjusted_jobs)
            logger.info(f"使用分块处理策略，块大小: {chunk_size} (优化后)")
            
            # 将任务分块
            chunked_args = []
            for i in range(0, n_lat, chunk_size):
                end_i = min(i + chunk_size, n_lat)
                chunk_args = []
                for j in range(i, end_i):
                    chunk_args.append((j, model_values, obs_values, obs_climatology_values, ensemble_values, climatology_hit_rates,
                                     model_lat_values, model_lon_values, obs_lat_values, obs_lon_values, grids_match,
                                     self.block_size, accuracy_delta, accuracy_eps, self.n_bootstrap, self.confidence_level))
                chunked_args.append(chunk_args)
            
            # 使用分块并行处理 - 使用ParallelProcessor
            processor = ParallelProcessor(n_jobs=adjusted_jobs, backend=self.parallel_backend)
            with processor.executor_class(max_workers=adjusted_jobs) as executor:
                futures = {executor.submit(_process_chunk_numpy, chunk): i for i, chunk in enumerate(chunked_args)}
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        chunk_idx = futures[future]
                        
                        # 处理分块结果
                        for result in chunk_results:
                            i, row_accuracy, row_accuracy_clim, row_equitable_accuracy, row_rmse, row_pcc, row_bs, row_bss, \
                            row_accuracy_ci_lower, row_accuracy_ci_upper, \
                            row_equitable_accuracy_ci_lower, row_equitable_accuracy_ci_upper, \
                            row_rmse_ci_lower, row_rmse_ci_upper, \
                            row_pcc_ci_lower, row_pcc_ci_upper, \
                            row_bs_ci_lower, row_bs_ci_upper, \
                            row_bss_ci_lower, row_bss_ci_upper = result
                            
                            # 更新结果数组
                            b_accuracy[i, :] = row_accuracy
                            b_accuracy_clim[i, :] = row_accuracy_clim
                            b_equitable_accuracy[i, :] = row_equitable_accuracy
                            b_rmse[i, :] = row_rmse
                            b_pcc[i, :] = row_pcc
                            b_bs[i, :] = row_bs
                            b_bss[i, :] = row_bss
                            accuracy_ci_lower[i, :] = row_accuracy_ci_lower
                            accuracy_ci_upper[i, :] = row_accuracy_ci_upper
                            equitable_accuracy_ci_lower[i, :] = row_equitable_accuracy_ci_lower
                            equitable_accuracy_ci_upper[i, :] = row_equitable_accuracy_ci_upper
                            rmse_ci_lower[i, :] = row_rmse_ci_lower
                            rmse_ci_upper[i, :] = row_rmse_ci_upper
                            pcc_ci_lower[i, :] = row_pcc_ci_lower
                            pcc_ci_upper[i, :] = row_pcc_ci_upper
                            bs_ci_lower[i, :] = row_bs_ci_lower
                            bs_ci_upper[i, :] = row_bs_ci_upper
                            bss_ci_lower[i, :] = row_bss_ci_lower
                            bss_ci_upper[i, :] = row_bss_ci_upper
                            
                            completed_rows += 1
                            progress = completed_rows / n_lat * 100
                            
                            if completed_rows % 5 == 0:
                                elapsed = datetime.now() - start_time
                                eta = elapsed * (n_lat - completed_rows) / completed_rows if completed_rows > 0 else timedelta(0)
                                logger.info(f"已完成 {completed_rows}/{n_lat} 行 ({progress:.1f}%) - 预计剩余时间: {eta}")
                                
                        # 定期检查内存使用
                        if completed_rows % 20 == 0:
                            log_memory_usage(f"处理进度 {completed_rows}/{n_lat} ({progress:.1f}%)")
                            
                    except Exception as e:
                        logger.error(f"处理分块 {futures[future]} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            # 标准并行处理 - 使用ParallelProcessor
            processor = ParallelProcessor(n_jobs=adjusted_jobs, backend=self.parallel_backend)
            with processor.executor_class(max_workers=adjusted_jobs) as executor:
                futures = {executor.submit(_process_single_row_numpy, args): args[0] for args in args_list}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        i, row_accuracy, row_accuracy_clim, row_equitable_accuracy, row_rmse, row_pcc, row_bs, row_bss, \
                        row_accuracy_ci_lower, row_accuracy_ci_upper, \
                        row_equitable_accuracy_ci_lower, row_equitable_accuracy_ci_upper, \
                        row_rmse_ci_lower, row_rmse_ci_upper, \
                        row_pcc_ci_lower, row_pcc_ci_upper, \
                        row_bs_ci_lower, row_bs_ci_upper, \
                        row_bss_ci_lower, row_bss_ci_upper = result
                        
                        # 更新结果数组
                        b_accuracy[i, :] = row_accuracy
                        b_accuracy_clim[i, :] = row_accuracy_clim
                        b_equitable_accuracy[i, :] = row_equitable_accuracy
                        b_rmse[i, :] = row_rmse
                        b_pcc[i, :] = row_pcc
                        b_bs[i, :] = row_bs
                        b_bss[i, :] = row_bss
                        accuracy_ci_lower[i, :] = row_accuracy_ci_lower
                        accuracy_ci_upper[i, :] = row_accuracy_ci_upper
                        equitable_accuracy_ci_lower[i, :] = row_equitable_accuracy_ci_lower
                        equitable_accuracy_ci_upper[i, :] = row_equitable_accuracy_ci_upper
                        rmse_ci_lower[i, :] = row_rmse_ci_lower
                        rmse_ci_upper[i, :] = row_rmse_ci_upper
                        pcc_ci_lower[i, :] = row_pcc_ci_lower
                        pcc_ci_upper[i, :] = row_pcc_ci_upper
                        bs_ci_lower[i, :] = row_bs_ci_lower
                        bs_ci_upper[i, :] = row_bs_ci_upper
                        bss_ci_lower[i, :] = row_bss_ci_lower
                        bss_ci_upper[i, :] = row_bss_ci_upper
                        
                        completed_rows += 1
                        progress = completed_rows / n_lat * 100
                        
                        if completed_rows % 5 == 0:
                            elapsed = datetime.now() - start_time
                            eta = elapsed * (n_lat - completed_rows) / completed_rows if completed_rows > 0 else timedelta(0)
                            logger.info(f"已完成 {completed_rows}/{n_lat} 行 ({progress:.1f}%) - 预计剩余时间: {eta}")

                        # 定期检查内存使用
                        if completed_rows % 20 == 0:
                            log_memory_usage(f"处理进度 {completed_rows}/{n_lat} ({progress:.1f}%)")

                    except Exception as e:
                        logger.error(f"处理行 {futures[future]} 时出错: {e}")
                        import traceback
                        traceback.print_exc()

        logger.info("所有行处理完成")
        log_memory_usage("数据处理完成")

        # 更新全局 RMSE 范围
        valid_rmse = b_rmse[~np.isnan(b_rmse)]
        if len(valid_rmse) > 0:
            rmse_min = np.min(valid_rmse)
            rmse_max = np.max(valid_rmse)
            if np.isnan(self.global_rmse_min):
                self.global_rmse_min = rmse_min
                self.global_rmse_max = rmse_max
            else:
                self.global_rmse_min = min(self.global_rmse_min, rmse_min)
                self.global_rmse_max = max(self.global_rmse_max, rmse_max)
        else:
            rmse_min, rmse_max = np.nan, np.nan

        # 确保所有指标在合理范围内
        b_accuracy = np.clip(b_accuracy, 0.0, 1.0)
        b_accuracy_clim = np.clip(b_accuracy_clim, 0.0, 1.0)
        b_equitable_accuracy = np.clip(b_equitable_accuracy, -1.0, 1.0)
        b_pcc = np.clip(b_pcc, -1.0, 1.0)

        # 创建结果数据集
        result_ds = xr.Dataset({
            'b_accuracy': (['lat', 'lon'], b_accuracy),
            'b_accuracy_clim': (['lat', 'lon'], b_accuracy_clim),
            'b_equitable_accuracy': (['lat', 'lon'], b_equitable_accuracy),
            'b_rmse': (['lat', 'lon'], b_rmse),
            'b_pcc': (['lat', 'lon'], b_pcc),
            'b_bs': (['lat', 'lon'], b_bs),  # Brier Score
            'b_bss': (['lat', 'lon'], b_bss),  # Brier Skill Score
            's_accuracy': (['lat', 'lon'], b_accuracy),  # s_accuracy = b_accuracy
            's_equitable_accuracy': (['lat', 'lon'], b_equitable_accuracy),  # 标准化的equitable accuracy
            's_rmse': (['lat', 'lon'], b_rmse),  # 将在后续更新
            's_pcc': (['lat', 'lon'], b_pcc),  # s_pcc = b_pcc
            'score': (['lat', 'lon'], np.full_like(b_accuracy, np.nan)),  # 将在后续更新
            'score_raw': (['lat', 'lon'], np.full_like(b_accuracy, np.nan)),  # 基于raw accuracy的综合得分（用于对比）
            'accuracy_ci_lower': (['lat', 'lon'], accuracy_ci_lower),
            'accuracy_ci_upper': (['lat', 'lon'], accuracy_ci_upper),
            'equitable_accuracy_ci_lower': (['lat', 'lon'], equitable_accuracy_ci_lower),
            'equitable_accuracy_ci_upper': (['lat', 'lon'], equitable_accuracy_ci_upper),
            'rmse_ci_lower': (['lat', 'lon'], rmse_ci_lower),
            'rmse_ci_upper': (['lat', 'lon'], rmse_ci_upper),
            'pcc_ci_lower': (['lat', 'lon'], pcc_ci_lower),
            'pcc_ci_upper': (['lat', 'lon'], pcc_ci_upper),
            'bs_ci_lower': (['lat', 'lon'], bs_ci_lower),  # BS置信区间下限
            'bs_ci_upper': (['lat', 'lon'], bs_ci_upper),  # BS置信区间上限
            'bss_ci_lower': (['lat', 'lon'], bss_ci_lower),  # BSS置信区间下限
            'bss_ci_upper': (['lat', 'lon'], bss_ci_upper),  # BSS置信区间上限
        }, coords={
            'lat': obs_data.lat,  # 使用观测数据的坐标，保持观测数据的完整范围
            'lon': obs_data.lon   # 使用观测数据的坐标，保持观测数据的完整范围
        })

        # 添加属性
        result_ds.attrs = {
            'block_size': self.block_size,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'rmse_min': float(rmse_min),
            'rmse_max': float(rmse_max),
            'method': 'independent_block_bootstrap'
        }

        return result_ds


    def plot_spatial_distribution_all_models(self, leadtime: int, var_type: str, models: List[str], metric: str = 'score'):
        """绘制所有模型的空间分布图（同一lead time）- 修复版"""
        return

    
    def _compute_timeseries_metrics(self, model: str, leadtime: int, var_type: str, obs_data_cache: Dict = None) -> Optional[Dict]:
        """计算单个模型的时间序列指标
        
        Args:
            model: 模型名称
            leadtime: 提前期
            var_type: 变量类型
            obs_data_cache: 观测数据缓存（避免重复加载）
            
        Returns:
            包含时间序列数据的字典，或None（如果计算失败）
        """
        try:
            logger.info(f"计算时间序列指标: {model} L{leadtime} {var_type}")
            
            # 使用缓存的观测数据或加载新数据
            if obs_data_cache is not None and var_type in obs_data_cache:
                obs_data = obs_data_cache[var_type]
                logger.debug(f"使用缓存的观测数据")
            else:
                obs_data = self._load_obs_data(var_type)
                if obs_data_cache is not None:
                    obs_data_cache[var_type] = obs_data
            
            # 加载预报数据
            fcst_data = self._load_forecast_data(model, var_type, leadtime)
            
            if obs_data is None or fcst_data is None:
                logger.warning(f"数据加载失败: {model} L{leadtime}")
                return None
            
            # 对齐数据
            obs_aligned, fcst_aligned = self._align_data(obs_data, fcst_data)
            
            if obs_aligned is None or fcst_aligned is None:
                logger.warning(f"数据对齐失败: {model} L{leadtime}")
                return None
            
            # 获取变量特定的accuracy参数
            var_config = VAR_CONFIG.get(var_type, {})
            accuracy_delta = var_config.get('accuracy_delta', self.accuracy_delta)
            accuracy_eps = var_config.get('accuracy_eps', self.accuracy_eps)
            
            # 计算每个时间点的空间平均指标
            n_time = len(obs_aligned.time)
            s_accuracy_ts = np.full(n_time, np.nan)
            s_rmse_ts = np.full(n_time, np.nan)
            s_pcc_ts = np.full(n_time, np.nan)
            
            for t in range(n_time):
                try:
                    obs_t = obs_aligned.isel(time=t).values.flatten()
                    fcst_t = fcst_aligned.isel(time=t).values.flatten()
                    
                    # 移除NaN值和无穷值
                    valid_mask = ~(np.isnan(obs_t) | np.isnan(fcst_t) | np.isinf(obs_t) | np.isinf(fcst_t))
                    obs_valid = obs_t[valid_mask]
                    fcst_valid = fcst_t[valid_mask]
                    
                    if len(obs_valid) < 10:  # 至少需要10个有效格点
                        continue
                    
                    # 计算accuracy（添加异常值检查）
                    rel_error = np.abs(fcst_valid - obs_valid) / (np.abs(obs_valid) + accuracy_eps)
                    # 防止除零或极端值导致的问题
                    rel_error = np.clip(rel_error, 0, 1e6)
                    accuracy = np.mean(rel_error <= accuracy_delta)
                    s_accuracy_ts[t] = np.clip(accuracy, 0.0, 1.0)
                    
                    # 计算RMSE（添加数值稳定性检查）
                    diff_sq = (fcst_valid - obs_valid) ** 2
                    # 防止极端值
                    diff_sq = np.clip(diff_sq, 0, 1e10)
                    rmse = np.sqrt(np.mean(diff_sq))
                    if not np.isnan(rmse) and not np.isinf(rmse):
                        s_rmse_ts[t] = rmse
                    
                    # 计算PCC（添加异常值检查）
                    if np.std(obs_valid) > 1e-10 and np.std(fcst_valid) > 1e-10:
                        # 检查数据范围是否合理
                        if np.max(np.abs(obs_valid)) < 1e10 and np.max(np.abs(fcst_valid)) < 1e10:
                            pcc, _ = pearsonr(fcst_valid, obs_valid)
                            s_pcc_ts[t] = np.clip(pcc, -1.0, 1.0)
                        else:
                            s_pcc_ts[t] = 0.0
                    else:
                        s_pcc_ts[t] = 0.0
                        
                except Exception as e:
                    logger.debug(f"计算时间点 {t} 的指标时出错: {e}")
                    continue
            
            # 归一化RMSE为score（需要该模型的RMSE范围）
            valid_rmse = s_rmse_ts[~np.isnan(s_rmse_ts)]
            if len(valid_rmse) > 0:
                rmse_min = np.min(valid_rmse)
                rmse_max = np.max(valid_rmse)
                if rmse_max > rmse_min:
                    s_rmse_score_ts = 1 - (s_rmse_ts - rmse_min) / (rmse_max - rmse_min)
                    s_rmse_score_ts = np.clip(s_rmse_score_ts, 0.0, 1.0)
                else:
                    s_rmse_score_ts = np.full_like(s_rmse_ts, 0.5)
            else:
                s_rmse_score_ts = np.full_like(s_rmse_ts, np.nan)
            
            logger.info(f"时间序列计算完成: {model} L{leadtime}, {n_time}个时间点")
            
            return {
                'time': pd.to_datetime(obs_aligned.time.values),
                's_accuracy': s_accuracy_ts,
                's_rmse': s_rmse_score_ts,  # 归一化后的RMSE score
                's_pcc': s_pcc_ts
            }
            
        except Exception as e:
            logger.error(f"计算时间序列指标失败 {model} L{leadtime}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    # ===== 以下函数已注释，被拆分为独立函数：plot_score_spatial_distribution, plot_score_cdf_distribution, plot_score_bar =====
    def plot_combined_spatial_timeseries(self, leadtime: int, var_type: str, models: List[str], metric: str = 'score'):
        return

    
    def _load_bss_data(self, leadtimes: List[int], models: List[str], var_type: str) -> Dict[int, Dict[str, Dict]]:
        """
        加载多个leadtime的BSS和Accuracy数据
        
        Args:
            leadtimes: 提前期列表
            models: 模型列表
            var_type: 变量类型
            
        Returns:
            {leadtime: {model: {'bss': bss_data, 'accuracy': accuracy_data}}}
        """
        all_leadtimes_data = {}
        
        logger.info(f"开始加载BSS数据: leadtimes={leadtimes}, models={models}, var_type={var_type}")
        
        for leadtime in leadtimes:
            leadtime_data = {}
            for model in models:
                nc_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
                
                if not nc_file.exists():
                    logger.debug(f"{model} L{leadtime}: BSS文件不存在，跳过")
                    continue
                
                try:
                    ds = xr.open_dataset(nc_file)
                    
                    # 获取BSS变量
                    if 'b_bss' not in ds:
                        logger.warning(f"{model} L{leadtime}: 文件中没有b_bss变量")
                        ds.close()
                        continue
                    
                    bss_data = ds['b_bss']
                    
                    # 检查BSS数据是否有效（至少有一个非NaN值）
                    if not np.isfinite(bss_data.values).any():
                        logger.warning(f"{model} L{leadtime}: b_bss变量全为NaN或无效值，跳过")
                        ds.close()
                        continue
                    
                    # 获取Accuracy变量（优先使用s_accuracy，如果没有则使用b_accuracy）
                    if 's_accuracy' in ds:
                        accuracy_data = ds['s_accuracy']
                    elif 'b_accuracy' in ds:
                        accuracy_data = ds['b_accuracy']
                    else:
                        logger.warning(f"{model} L{leadtime}: 文件中没有accuracy变量")
                        ds.close()
                        continue
                    
                    # 验证网格一致性
                    if bss_data.shape != accuracy_data.shape:
                        logger.warning(f"{model} L{leadtime}: 网格不匹配，跳过")
                        ds.close()
                        continue
                    
                    valid_bss_count = np.sum(~np.isnan(bss_data.values))
                    logger.debug(f"{model} L{leadtime}: 成功加载BSS数据，有效值: {valid_bss_count}/{bss_data.size}")
                    
                    leadtime_data[model] = {
                        'bss': bss_data,
                        'accuracy': accuracy_data
                    }
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model} L{leadtime} BSS数据失败: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            if leadtime_data:
                all_leadtimes_data[leadtime] = leadtime_data
                logger.info(f"Leadtime {leadtime}: 成功加载 {len(leadtime_data)} 个模型的数据")
            else:
                logger.warning(f"Leadtime {leadtime}: 没有加载到任何有效的BSS数据")
        
        logger.info(f"BSS数据加载完成: 成功加载 {len(all_leadtimes_data)} 个leadtimes的数据")
        return all_leadtimes_data
    
    def _load_score_data(self, leadtimes: List[int], models: List[str], var_type: str) -> Dict[int, Dict[str, Dict]]:
        """
        加载多个leadtime的Score数据
        
        Args:
            leadtimes: 提前期列表
            models: 模型列表
            var_type: 变量类型
            
        Returns:
            {leadtime: {model: {'score': score_data, 's_accuracy': accuracy_data, 's_pcc': pcc_data, 's_rmse': rmse_data}}}
        """
        all_leadtimes_data = {}
        
        for leadtime in leadtimes:
            leadtime_data = {}
            for model in models:
                nc_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
                
                if not nc_file.exists():
                    logger.debug(f"{model} L{leadtime}: Score文件不存在，跳过")
                    continue
                
                try:
                    ds = xr.open_dataset(nc_file)
                    
                    # 获取需要的变量
                    required_vars = ['score', 's_accuracy', 's_pcc', 's_rmse']
                    model_vars = {}
                    
                    for var in required_vars:
                        if var not in ds:
                            logger.warning(f"{model} L{leadtime}: 文件中没有{var}变量")
                            ds.close()
                            break
                        model_vars[var] = ds[var]
                    else:
                        # 所有变量都存在
                        leadtime_data[model] = model_vars
                    
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"加载{model} L{leadtime} Score数据失败: {e}")
                    continue
            
            if leadtime_data:
                all_leadtimes_data[leadtime] = leadtime_data
        
        return all_leadtimes_data
    
    def plot_bss_scatter(self, leadtimes: List[int], var_type: str, models: List[str]):
        """
        绘制BSS空间分布图
        分为上下两半（L0和L3），每个lead占2行，第1行留空+3模型，第2行4模型
        子图之间不留空隙，仅在最外围绘制经纬度标签和脊线，最下方绘制colorbar
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            var_type: 变量类型
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
            
            logger.info(f"绘制BSS空间分布图: L{leadtimes} {var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_bss_data(leadtimes, models, var_type)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于BSS空间分布图")
                return

            available_leadtimes = [lt for lt in leadtimes if lt in all_leadtimes_data]
            if not available_leadtimes:
                logger.warning("指定的lead time均无可用的BSS数据")
                return
            
            # 准备模型列表，按顺序排列（只要任一lead有数据即保留）
            def _model_has_data(model: str) -> bool:
                return any(
                    lead in all_leadtimes_data and model in all_leadtimes_data[lead]
                    for lead in available_leadtimes
                )
            
            model_names = [m for m in models if _model_has_data(m)]
            n_models = len(model_names)
            n_leadtimes = len(available_leadtimes)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            n_leadtimes = len(available_leadtimes)
            
            # 创建陆地掩膜（基于第一个模型的数据）
            land_mask = None
            if var_type == 'prec':
                first_leadtime = available_leadtimes[0]
                first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
                sample_bss = first_model_data['bss']
                land_mask = self._create_land_mask(var_type, sample_bss)
            
            # 收集所有leadtime的所有BSS数据，用于计算统一的colorbar范围（不做IQR去极值）
            # 只考虑陆地区域的数据
            all_bss_values = []
            for leadtime in leadtimes:
                if leadtime in all_leadtimes_data:
                    for model_data in all_leadtimes_data[leadtime].values():
                        bss_data = model_data['bss']
                        # 应用陆地掩膜（如果存在）
                        if land_mask is not None:
                            # 对齐掩膜到数据网格
                            if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                                mask_aligned = land_mask.reindex(
                                    lat=bss_data.lat,
                                    lon=bss_data.lon,
                                    method='nearest',
                                    tolerance=0.5
                                )
                                bss_masked = bss_data.where(mask_aligned)
                            else:
                                bss_masked = bss_data
                        else:
                            bss_masked = bss_data
                        
                        bss_flat = bss_masked.values.flatten()
                        valid_values = bss_flat[~np.isnan(bss_flat)]
                        all_bss_values.extend(valid_values)
            
            # 计算统一范围（不去除极值，直接基于全部有效值）
            if all_bss_values:
                data_min = np.min(all_bss_values)
                data_max = np.max(all_bss_values)
                bss_min = np.floor(data_min * 10) / 10
                bss_max = np.ceil(data_max * 10) / 10
            else:
                bss_min, bss_max = -1.0, 1.0
            
            logger.info(f"BSS范围: [{bss_min:.1f}, {bss_max:.1f}]")
            
            # 使用高对比度蓝色分段 colormap + 分段 colorbar
            from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
            cmap_base = LinearSegmentedColormap.from_list(
                'blue_strong',
                [
                    "#ffffff",  # 低值白
                    "#dbe9f6",
                    "#b5d0f0",
                    "#8cb9e6",
                    "#5fa0da",
                    "#3c87c6",
                    "#236daf",
                    "#12538f",
                    "#08306b"   # 高值深蓝
                ]
            ).reversed()  # 反转cbar，深色代表更高的BSS
            
            # 分段设置：默认11段，若范围无效则回退到[-1,1]
            if np.isfinite(bss_min) and np.isfinite(bss_max) and bss_max > bss_min:
                levels = np.linspace(bss_min, bss_max, 11)
            else:
                levels = np.linspace(-1.0, 1.0, 11)
                bss_min, bss_max = -1.0, 1.0
            norm = BoundaryNorm(levels, cmap_base.N, clip=True)
            cmap = cmap_base
            
            # 计算布局
            n_leadtimes = len(available_leadtimes)
            n_cols = 4  # 固定4列：留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 基于第一个模型的数据计算经纬度边界
            # 根据实际数据范围（考虑陆地掩膜）来决定绘制范围
            first_leadtime = available_leadtimes[0]
            first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
            sample_bss = first_model_data['bss']
            
            # 计算实际数据范围（取整）
            lon_min, lon_max, lat_min, lat_max = self._compute_data_extent(sample_bss, land_mask)
            
            # 创建边界数组（用于绘制）
            def _compute_edges(center_coords: np.ndarray) -> np.ndarray:
                center_coords = np.asarray(center_coords)
                diffs = np.diff(center_coords)
                first_edge = center_coords[0] - diffs[0] / 2.0 if diffs.size > 0 else center_coords[0] - 0.5
                last_edge = center_coords[-1] + diffs[-1] / 2.0 if diffs.size > 0 else center_coords[-1] + 0.5
                mid_edges = center_coords[:-1] + diffs / 2.0 if diffs.size > 0 else np.array([])
                return np.concatenate([[first_edge], mid_edges, [last_edge]])
            
            # 获取所有数据的经纬度中心点
            lon_centers = sample_bss.lon.values if hasattr(sample_bss, 'lon') else None
            lat_centers = sample_bss.lat.values if hasattr(sample_bss, 'lat') else None
            
            if lon_centers is None or lat_centers is None:
                logger.error("数据缺少经纬度坐标")
                return
            
            # 根据实际范围筛选经纬度
            lon_centers_filtered = lon_centers[(lon_centers >= lon_min) & (lon_centers <= lon_max)]
            lat_centers_filtered = lat_centers[(lat_centers >= lat_min) & (lat_centers <= lat_max)]
            
            # 如果筛选后为空，使用原始范围
            if len(lon_centers_filtered) == 0:
                lon_centers_filtered = lon_centers
            if len(lat_centers_filtered) == 0:
                lat_centers_filtered = lat_centers
            
            lon_edges = _compute_edges(lon_centers_filtered)
            lat_edges = _compute_edges(lat_centers_filtered)
            
            # 计算画布大小
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
            for lt_idx, leadtime in enumerate(available_leadtimes):
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
                    model_data = model_data_dict.get(model)
                    if model_data is None:
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        ax_blank.text(
                            0.5,
                            0.5,
                            'NO DATA',
                            ha='center',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='red',
                            transform=ax_blank.transAxes,
                        )
                        continue
                    
                    bss_data = model_data['bss']
                    if not np.isfinite(bss_data.values).any():
                        ax_blank = fig.add_subplot(gs[row_obs, col_idx + 1])
                        ax_blank.axis('off')
                        ax_blank.text(
                            0.5,
                            0.5,
                            'NO DATA',
                            ha='center',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='red',
                            transform=ax_blank.transAxes,
                        )
                        continue
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_obs, col_idx + 1], projection=ccrs.PlateCarree())
                    
                    # 基于模型自身网格计算边界（根据实际数据范围）
                    try:
                        model_bss = model_data['bss']
                        model_lon_min, model_lon_max, model_lat_min, model_lat_max = self._compute_data_extent(model_bss, land_mask)
                        ax_spatial.set_extent([model_lon_min, model_lon_max, model_lat_min, model_lat_max], crs=ccrs.PlateCarree())
                    except Exception:
                        # 如果计算失败，使用全局范围
                        ax_spatial.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 只在外围显示坐标轴标签
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    if lt_idx == 0 and col_idx == 0:  # 第一个leadtime的第一行第一列
                        gl.left_labels = True
                        gl.bottom_labels = True
                    elif lt_idx == n_leadtimes - 1 and col_idx == 0:  # 最后一个leadtime的第一行第一列
                        gl.left_labels = True
                        gl.bottom_labels = True
                    else:
                        gl.left_labels = False
                        gl.bottom_labels = False
                    
                    # 绘制BSS数据
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=bss_data.lat,
                                lon=bss_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            bss_data_masked = bss_data.where(mask_aligned)
                        else:
                            bss_data_masked = bss_data
                    else:
                        # 如果没有掩膜，对于降水数据使用NaN作为掩膜
                        if var_type == 'prec':
                            ocean_mask = np.isnan(bss_data.values)
                            bss_data_masked = bss_data.where(~ocean_mask)
                        else:
                            bss_data_masked = bss_data
                    
                    # 获取vmin和vmax从norm
                    if hasattr(norm, 'boundaries') and len(norm.boundaries) > 0:
                        vmin, vmax = norm.boundaries[0], norm.boundaries[-1]
                    else:
                        vmin, vmax = bss_min, bss_max
                    
                    # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                    levels = ticker.MaxNLocator(nbins=11, prune=None).tick_values(vmin, vmax)
                    
                    im = ax_spatial.contour(
                        bss_data_masked.lon, bss_data_masked.lat, bss_data_masked,
                        levels=levels, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, linewidths=1.2, alpha=0.8
                    )
                    
                    if im_for_cbar is None:
                        im_for_cbar = im
                    
                    # 模型标签
                    label = chr(97 + col_idx)  # a, b, c
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name}', 
                                  transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                  verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_spatial)
                    _expand_axes_vertically(ax_spatial, lt_idx == 0, lt_idx == n_leadtimes - 1)
                
                # 第2行：4个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(model_names):
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[model_idx]
                    model_data = model_data_dict.get(model)
                    if model_data is None:
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        ax_blank.text(
                            0.5,
                            0.5,
                            'NO DATA',
                            ha='center',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='red',
                            transform=ax_blank.transAxes,
                        )
                        continue
                    
                    bss_data = model_data['bss']
                    if not np.isfinite(bss_data.values).any():
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        ax_blank.text(
                            0.5,
                            0.5,
                            'NO DATA',
                            ha='center',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='red',
                            transform=ax_blank.transAxes,
                        )
                        continue
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_models2, col_idx], projection=ccrs.PlateCarree())
                    
                    # 基于模型自身网格计算边界（根据实际数据范围）
                    try:
                        model_bss = model_data['bss']
                        model_lon_min, model_lon_max, model_lat_min, model_lat_max = self._compute_data_extent(model_bss, land_mask)
                        ax_spatial.set_extent([model_lon_min, model_lon_max, model_lat_min, model_lat_max], crs=ccrs.PlateCarree())
                    except Exception:
                        # 如果计算失败，使用全局范围
                        ax_spatial.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 只在外围显示坐标轴标签
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    if lt_idx == n_leadtimes - 1:  # 最后一个leadtime
                        gl.bottom_labels = True
                    else:
                        gl.bottom_labels = False
                    if col_idx == 0:  # 第一列
                        gl.left_labels = True
                    else:
                        gl.left_labels = False
                    
                    # 绘制BSS数据
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=bss_data.lat,
                                lon=bss_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            bss_data_masked = bss_data.where(mask_aligned)
                        else:
                            bss_data_masked = bss_data
                    else:
                        # 如果没有掩膜，对于降水数据使用NaN作为掩膜
                        if var_type == 'prec':
                            ocean_mask = np.isnan(bss_data.values)
                            bss_data_masked = bss_data.where(~ocean_mask)
                        else:
                            bss_data_masked = bss_data
                    
                    # 获取vmin和vmax从norm
                    if hasattr(norm, 'boundaries') and len(norm.boundaries) > 0:
                        vmin, vmax = norm.boundaries[0], norm.boundaries[-1]
                    else:
                        vmin, vmax = bss_min, bss_max
                    
                    # 使用 MaxNLocator 自动选择等间距且“美观”的刻度
                    levels = ticker.MaxNLocator(nbins=11, prune=None).tick_values(vmin, vmax)
                    
                    ax_spatial.contour(
                        bss_data_masked.lon, bss_data_masked.lat, bss_data_masked,
                        levels=levels, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, linewidths=1.2, alpha=0.8
                    )
                    
                    # 模型标签
                    label = chr(97 + model_idx)  # d, e, f, g
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name}', 
                                  transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                  verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_spatial)
                    _expand_axes_vertically(ax_spatial, lt_idx == 0, lt_idx == n_leadtimes - 1)
            
            # 在最下方添加colorbar
            if im_for_cbar is not None:
                cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
                cbar = fig.colorbar(
                    im_for_cbar, cax=cbar_ax, orientation='horizontal',
                    boundaries=levels, ticks=levels
                )
                cbar.set_label('Brier Skill Score (BSS)', fontsize=11)
                cbar.ax.tick_params(labelsize=9)
            
            # 保存图像
            output_file_png = self.plots_dir / f"bss_scatter_{var_type}.png"
            output_file_pdf = self.plots_dir / f"bss_scatter_{var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"BSS空间分布图已保存到: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制BSS空间分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_bss_vs_accuracy_scatter(self, leadtimes: List[int], var_type: str, models: List[str]):
        """
        绘制BSS vs Accuracy散点图
        分为上下两半（L0和L3），模式排列与空间分布图一致
        子图之间不留空隙，仅在最外围绘制横纵坐标标签和脊线
        所有子图同步横纵坐标，内部不绘制网格
        最下方左右排列绘制图例和密度colorbar
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            var_type: 变量类型
            models: 模型列表
        """
        try:
            from scipy.stats import gaussian_kde
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制BSS vs Accuracy散点图: L{leadtimes} {var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_bss_data(leadtimes, models, var_type)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于BSS vs Accuracy散点图")
                return
            
            available_leadtimes = [lt for lt in leadtimes if lt in all_leadtimes_data]
            if not available_leadtimes:
                logger.warning("指定的lead time均无BSS/Accuracy数据")
                return
            
            def _model_has_data(model: str) -> bool:
                return any(
                    lead in all_leadtimes_data and model in all_leadtimes_data[lead]
                    for lead in available_leadtimes
                )
            
            model_names = [m for m in models if _model_has_data(m)]
            n_models = len(model_names)
            n_leadtimes = len(available_leadtimes)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 创建陆地掩膜（基于第一个模型的数据，与CDF图保持一致）
            land_mask = None
            if var_type == 'prec':
                first_leadtime = available_leadtimes[0]
                first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
                sample_accuracy = first_model_data['accuracy']
                land_mask = self._create_land_mask(var_type, sample_accuracy)
            
            # 收集所有数据，用于计算统一的坐标范围和密度范围
            # 只考虑陆地区域的数据（应用掩膜）
            all_bss_vals = []
            all_accuracy_vals = []
            all_densities = []
            
            for leadtime in available_leadtimes:
                for model_data in all_leadtimes_data[leadtime].values():
                    bss_data = model_data['bss']
                    accuracy_data = model_data['accuracy']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=bss_data.lat,
                                lon=bss_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            bss_data = bss_data.where(mask_aligned)
                            accuracy_data = accuracy_data.where(mask_aligned)
                    
                    bss_flat = bss_data.values.flatten()
                    accuracy_flat = accuracy_data.values.flatten()
                    valid_mask = ~(np.isnan(bss_flat) | np.isnan(accuracy_flat))
                    bss_valid = bss_flat[valid_mask]
                    accuracy_valid = accuracy_flat[valid_mask]
                    all_bss_vals.extend(bss_valid)
                    all_accuracy_vals.extend(accuracy_valid)
                    
                    if len(bss_valid) > 100:
                        try:
                            xy = np.vstack([bss_valid, accuracy_valid])
                            kde = gaussian_kde(xy)
                            z = kde(xy)
                            all_densities.extend(z)
                        except:
                            pass
            
            # 计算统一的坐标范围（基于实际数据分布，范围略放宽以减少越界点）
            # 对于温度变量，使用更聚焦的范围以改善分布显示
            if all_bss_vals and all_accuracy_vals:
                bss_array = np.array(all_bss_vals)
                acc_array = np.array(all_accuracy_vals)
                
                # 根据变量类型选择不同的分位数策略
                if var_type == 'temp':
                    # 温度：数据高度集中在高值，使用更聚焦的分位数（5%-100%）
                    bss_low = np.nanpercentile(bss_array, 5.0)
                    bss_high = np.nanpercentile(bss_array, 100.0)  # 使用最大值
                    # 如果大部分值接近1.0，聚焦在0.85-1.0范围
                    if bss_high >= 0.95:
                        bss_low = max(bss_low, 0.85)
                        bss_high = 1.0
                    
                    acc_low = np.nanpercentile(acc_array, 5.0)
                    acc_high = np.nanpercentile(acc_array, 100.0)
                    # 如果大部分值接近1.0，聚焦在0.85-1.0范围
                    if acc_high >= 0.95:
                        acc_low = max(acc_low, 0.85)
                        acc_high = 1.0
                else:
                    # 降水：使用更宽的分位数区间（0.1% ~ 99.9%）
                    bss_low = np.nanpercentile(bss_array, 0.1)
                    bss_high = np.nanpercentile(bss_array, 99.9)
                    acc_low = np.nanpercentile(acc_array, 0.1)
                    acc_high = np.nanpercentile(acc_array, 99.9)
                
                # BSS范围处理
                if not np.isfinite(bss_low) or not np.isfinite(bss_high):
                    bss_low, bss_high = -10.0, 1.1
                if bss_low == bss_high:
                    delta = max(abs(bss_high), 0.1) * 0.1
                    bss_low -= delta
                    bss_high += delta
                bss_margin = (bss_high - bss_low) * 0.10
                scatter_bss_min = bss_low - bss_margin
                scatter_bss_max = bss_high + bss_margin
                
                # Accuracy范围处理
                acc_low = float(np.clip(acc_low, 0.0, 1.0))
                acc_high = float(np.clip(acc_high, 0.0, 1.0))
                if acc_low == acc_high:
                    delta = max(acc_high, 0.05) * 0.1
                    acc_low = max(0.0, acc_low - delta)
                    acc_high = min(1.0, acc_high + delta)
                acc_margin = (acc_high - acc_low) * 0.08
                scatter_accuracy_min = max(0.0, acc_low - acc_margin)
                scatter_accuracy_max = min(1.0, acc_high + acc_margin)
            else:
                scatter_bss_min, scatter_bss_max = -10.0, 1.1
                scatter_accuracy_min, scatter_accuracy_max = -0.1, 1.1
            
            # 计算统一的密度范围
            if all_densities:
                raw_density_min = np.percentile(all_densities, 5)
                raw_density_max = np.percentile(all_densities, 95)
                density_min = 0.0
                density_max = 1.0
            else:
                density_min, density_max = 0.0, 1.0
                raw_density_min, raw_density_max = 0, 1
            
            logger.info(f"散点图BSS范围: [{scatter_bss_min:.2f}, {scatter_bss_max:.2f}]")
            logger.info(f"散点图Accuracy范围: [{scatter_accuracy_min:.2f}, {scatter_accuracy_max:.2f}]")
            
            # 布局：4行×4列（上两行为L0，下两行为L3）
            fig = plt.figure(figsize=(16, 10))
            n_rows = n_leadtimes * 2
            gs = GridSpec(n_rows, 4, figure=fig, hspace=0.0, wspace=0.0,
                          left=0.06, right=0.97, top=0.95, bottom=0.12)
            axes_grid = [[None]*4 for _ in range(n_rows)]
            
            # 用于保存colorbar的绘图对象
            scatter_for_cbar = None
            content_axes = []
            
            # 绘制每个lead的两行
            for lead_idx, leadtime in enumerate(available_leadtimes):
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
                    
                    # 准备散点数据（应用陆地掩膜，与CDF图保持一致）
                    bss_data = model_data['bss']
                    accuracy_data = model_data['accuracy']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=bss_data.lat,
                                lon=bss_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            bss_data = bss_data.where(mask_aligned)
                            accuracy_data = accuracy_data.where(mask_aligned)
                    
                    bss_flat = bss_data.values.flatten()
                    accuracy_flat = accuracy_data.values.flatten()
                    valid_mask = ~(np.isnan(bss_flat) | np.isnan(accuracy_flat))
                    bss_all = bss_flat[valid_mask]
                    accuracy_all = accuracy_flat[valid_mask]
                    bss_valid = bss_all.copy()
                    accuracy_valid = accuracy_all.copy()
                    
                    if len(bss_valid) > 0:
                        # 识别超出绘制范围的点
                        out_x_low = bss_all < scatter_bss_min
                        out_x_high = bss_all > scatter_bss_max
                        out_y_low = accuracy_all < scatter_accuracy_min
                        out_y_high = accuracy_all > scatter_accuracy_max
                        out_x = out_x_low | out_x_high
                        out_y = out_y_low | out_y_high
                        out_any = out_x | out_y
                        
                        # 标记超出范围的点在坐标轴上
                        if np.any(out_any):
                            if np.any(out_x_low):
                                x_mark_low = np.full(np.sum(out_x_low), scatter_bss_min)
                                y_mark_low = np.clip(accuracy_all[out_x_low], scatter_accuracy_min, scatter_accuracy_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_x_high):
                                x_mark_high = np.full(np.sum(out_x_high), scatter_bss_max)
                                y_mark_high = np.clip(accuracy_all[out_x_high], scatter_accuracy_min, scatter_accuracy_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_low):
                                y_mark_low = np.full(np.sum(out_y_low), scatter_accuracy_min)
                                x_mark_low = np.clip(bss_all[out_y_low], scatter_bss_min, scatter_bss_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_high):
                                y_mark_high = np.full(np.sum(out_y_high), scatter_accuracy_max)
                                x_mark_high = np.clip(bss_all[out_y_high], scatter_bss_min, scatter_bss_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                        
                        # 过滤极端值
                        if len(bss_valid) > 10:
                            bss_q1, bss_q3 = np.percentile(bss_valid, [25, 75])
                            bss_iqr = bss_q3 - bss_q1
                            bss_lower = bss_q1 - 1.5 * bss_iqr
                            bss_upper = bss_q3 + 1.5 * bss_iqr
                            
                            accuracy_q1, accuracy_q3 = np.percentile(accuracy_valid, [25, 75])
                            accuracy_iqr = accuracy_q3 - accuracy_q1
                            accuracy_lower = accuracy_q1 - 1.5 * accuracy_iqr
                            accuracy_upper = accuracy_q3 + 1.5 * accuracy_iqr
                            
                            outlier_mask = (
                                (bss_valid >= bss_lower) & (bss_valid <= bss_upper) &
                                (accuracy_valid >= accuracy_lower) & (accuracy_valid <= accuracy_upper)
                            )
                            
                            if outlier_mask.sum() > 5:
                                bss_valid = bss_valid[outlier_mask]
                                accuracy_valid = accuracy_valid[outlier_mask]
                        
                        # 绘制散点（使用密度着色）
                        if len(bss_valid) > 100:
                            try:
                                xy = np.vstack([bss_valid, accuracy_valid])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)
                                
                                scatter = ax.scatter(bss_valid, accuracy_valid, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')
                                
                                if scatter_for_cbar is None:
                                    scatter_for_cbar = scatter
                            except:
                                ax.scatter(bss_valid, accuracy_valid, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(bss_valid, accuracy_valid, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')
                        
                        # 拟合线
                        if len(bss_valid) > 2:
                            # 检查x值是否全部相同
                            bss_std = np.std(bss_valid)
                            if bss_std > 1e-10:  # 如果标准差足够大，说明x值有变化
                                try:
                                    slope_all, intercept_all, _, _, _ = stats.linregress(bss_valid, accuracy_valid)
                                    x_fit = np.array([scatter_bss_min, scatter_bss_max])
                                    y_fit_all = slope_all * x_fit + intercept_all
                                    ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)
                                    
                                    # Robust拟合线
                                    residuals = accuracy_valid - (slope_all * bss_valid + intercept_all)
                                    q1, q3 = np.percentile(residuals, [25, 75])
                                    iqr = q3 - q1
                                    lower_bound = q1 - 1.5 * iqr
                                    upper_bound = q3 + 1.5 * iqr
                                    outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
                                    bss_robust = bss_valid[outlier_mask]
                                    accuracy_robust = accuracy_valid[outlier_mask]
                                    
                                    if len(bss_robust) > 2:
                                        bss_robust_std = np.std(bss_robust)
                                        if bss_robust_std > 1e-10:  # 检查robust数据的x值是否有变化
                                            try:
                                                slope_robust, intercept_robust, _, _, _ = stats.linregress(bss_robust, accuracy_robust)
                                                y_fit_robust = slope_robust * x_fit + intercept_robust
                                                ax.plot(x_fit, y_fit_robust, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
                                            except (ValueError, np.linalg.LinAlgError):
                                                pass  # 如果robust拟合失败，跳过
                                except (ValueError, np.linalg.LinAlgError):
                                    pass  # 如果线性回归失败，跳过拟合线绘制
                    
                    # 设置坐标范围（所有子图统一）
                    ax.set_xlim(scatter_bss_min, scatter_bss_max)
                    ax.set_ylim(scatter_accuracy_min, scatter_accuracy_max)
                    
                    # 标注模型名与lead
                    label = chr(97 + col_idx)
                    ax.text(0.02, 0.95, f"({label}) {display_name}", transform=ax.transAxes, ha='left', va='top',
                            fontsize=10, fontweight='bold')
                    if col_idx == 0:
                        ax.text(0.98, 0.95, f"L{leadtime}", transform=ax.transAxes, ha='right', va='top',
                                fontsize=11, fontweight='bold')
                    
                    # 坐标轴：仅外侧显示刻度标签，但所有子图保留边框
                    is_bottom = (row_start == n_rows - 2) and (lead_idx == n_leadtimes - 1)
                    ax.set_xlabel('BSS' if is_bottom else '', fontsize=10)
                    ax.set_ylabel('', fontsize=10)
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax.spines[spine].set_visible(True)
                    ax.tick_params(axis='y', labelleft=False, left=True, labelsize=8)
                    ax.tick_params(axis='x', labelbottom=is_bottom, bottom=True, labelsize=8)
                    
                    ax.tick_params(axis='both', labelsize=8)
                    ax.grid(False)  # 不绘制网格
                
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
                    
                    # 准备散点数据（应用陆地掩膜，与CDF图保持一致）
                    bss_data = model_data['bss']
                    accuracy_data = model_data['accuracy']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=bss_data.lat,
                                lon=bss_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            bss_data = bss_data.where(mask_aligned)
                            accuracy_data = accuracy_data.where(mask_aligned)
                    
                    bss_flat = bss_data.values.flatten()
                    accuracy_flat = accuracy_data.values.flatten()
                    valid_mask = ~(np.isnan(bss_flat) | np.isnan(accuracy_flat))
                    bss_all = bss_flat[valid_mask]
                    accuracy_all = accuracy_flat[valid_mask]
                    bss_valid = bss_all.copy()
                    accuracy_valid = accuracy_all.copy()
                    
                    if len(bss_valid) > 0:
                        # 识别超出绘制范围的点
                        out_x_low = bss_all < scatter_bss_min
                        out_x_high = bss_all > scatter_bss_max
                        out_y_low = accuracy_all < scatter_accuracy_min
                        out_y_high = accuracy_all > scatter_accuracy_max
                        out_x = out_x_low | out_x_high
                        out_y = out_y_low | out_y_high
                        out_any = out_x | out_y
                        
                        # 标记超出范围的点
                        if np.any(out_any):
                            if np.any(out_x_low):
                                x_mark_low = np.full(np.sum(out_x_low), scatter_bss_min)
                                y_mark_low = np.clip(accuracy_all[out_x_low], scatter_accuracy_min, scatter_accuracy_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_x_high):
                                x_mark_high = np.full(np.sum(out_x_high), scatter_bss_max)
                                y_mark_high = np.clip(accuracy_all[out_x_high], scatter_accuracy_min, scatter_accuracy_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_low):
                                y_mark_low = np.full(np.sum(out_y_low), scatter_accuracy_min)
                                x_mark_low = np.clip(bss_all[out_y_low], scatter_bss_min, scatter_bss_max)
                                ax.scatter(x_mark_low, y_mark_low, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                            
                            if np.any(out_y_high):
                                y_mark_high = np.full(np.sum(out_y_high), scatter_accuracy_max)
                                x_mark_high = np.clip(bss_all[out_y_high], scatter_bss_min, scatter_bss_max)
                                ax.scatter(x_mark_high, y_mark_high, marker='x', s=30, color='red', 
                                          linewidths=1.5, alpha=0.7, zorder=100)
                        
                        # 过滤极端值
                        if len(bss_valid) > 10:
                            bss_q1, bss_q3 = np.percentile(bss_valid, [25, 75])
                            bss_iqr = bss_q3 - bss_q1
                            bss_lower = bss_q1 - 1.5 * bss_iqr
                            bss_upper = bss_q3 + 1.5 * bss_iqr
                            
                            accuracy_q1, accuracy_q3 = np.percentile(accuracy_valid, [25, 75])
                            accuracy_iqr = accuracy_q3 - accuracy_q1
                            accuracy_lower = accuracy_q1 - 1.5 * accuracy_iqr
                            accuracy_upper = accuracy_q3 + 1.5 * accuracy_iqr
                            
                            outlier_mask = (
                                (bss_valid >= bss_lower) & (bss_valid <= bss_upper) &
                                (accuracy_valid >= accuracy_lower) & (accuracy_valid <= accuracy_upper)
                            )
                            
                            if outlier_mask.sum() > 5:
                                bss_valid = bss_valid[outlier_mask]
                                accuracy_valid = accuracy_valid[outlier_mask]
                        
                        # 绘制散点
                        if len(bss_valid) > 100:
                            try:
                                xy = np.vstack([bss_valid, accuracy_valid])
                                kde = gaussian_kde(xy)
                                z_raw = kde(xy)
                                z = (z_raw - raw_density_min) / (raw_density_max - raw_density_min)
                                z = np.clip(z, 0, 1)
                                
                                scatter = ax.scatter(bss_valid, accuracy_valid, c=z, 
                                                    s=8, alpha=0.6,
                                                    cmap='viridis', 
                                                    vmin=density_min, vmax=density_max,
                                                    edgecolors='none')
                            except:
                                ax.scatter(bss_valid, accuracy_valid, s=8, alpha=0.4,
                                          color='steelblue', edgecolors='none')
                        else:
                            ax.scatter(bss_valid, accuracy_valid, s=8, alpha=0.4,
                                      color='steelblue', edgecolors='none')
                        
                        # 拟合线
                        if len(bss_valid) > 2:
                            # 检查x值是否全部相同
                            bss_std = np.std(bss_valid)
                            if bss_std > 1e-10:  # 如果标准差足够大，说明x值有变化
                                try:
                                    slope_all, intercept_all, _, _, _ = stats.linregress(bss_valid, accuracy_valid)
                                    x_fit = np.array([scatter_bss_min, scatter_bss_max])
                                    y_fit_all = slope_all * x_fit + intercept_all
                                    ax.plot(x_fit, y_fit_all, color='#ff7f0e', linestyle='-', linewidth=1.2, alpha=0.7)
                                    
                                    # Robust拟合线
                                    residuals = accuracy_valid - (slope_all * bss_valid + intercept_all)
                                    q1, q3 = np.percentile(residuals, [25, 75])
                                    iqr = q3 - q1
                                    lower_bound = q1 - 1.5 * iqr
                                    upper_bound = q3 + 1.5 * iqr
                                    outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
                                    bss_robust = bss_valid[outlier_mask]
                                    accuracy_robust = accuracy_valid[outlier_mask]
                                    
                                    if len(bss_robust) > 2:
                                        bss_robust_std = np.std(bss_robust)
                                        if bss_robust_std > 1e-10:  # 检查robust数据的x值是否有变化
                                            try:
                                                slope_robust, intercept_robust, _, _, _ = stats.linregress(bss_robust, accuracy_robust)
                                                y_fit_robust = slope_robust * x_fit + intercept_robust
                                                ax.plot(x_fit, y_fit_robust, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
                                            except (ValueError, np.linalg.LinAlgError):
                                                pass  # 如果robust拟合失败，跳过
                                except (ValueError, np.linalg.LinAlgError):
                                    pass  # 如果线性回归失败，跳过拟合线绘制
                    
                    # 设置坐标范围
                    ax.set_xlim(scatter_bss_min, scatter_bss_max)
                    ax.set_ylim(scatter_accuracy_min, scatter_accuracy_max)
                    
                    # 标注模型名
                    label = chr(97 + model_idx)
                    ax.text(0.02, 0.95, f"({label}) {display_name}", transform=ax.transAxes, ha='left', va='top',
                            fontsize=10, fontweight='bold')
                    
                    # 坐标轴：仅外侧显示刻度标签，但所有子图保留边框
                    is_bottom = (row_start + 1 == n_rows - 1)
                    show_left = (col_idx == 0)
                    ax.set_xlabel('BSS' if is_bottom else '', fontsize=10)
                    ax.set_ylabel('Accuracy' if show_left else '', fontsize=10)
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax.spines[spine].set_visible(True)
                    ax.tick_params(axis='y', labelleft=show_left, left=True, labelsize=8)
                    ax.tick_params(axis='x', labelbottom=is_bottom, bottom=True, labelsize=8)
                    ax.grid(False)  # 不绘制网格
            
            # 在最下方添加colorbar和图例
            if scatter_for_cbar is not None:
                # Colorbar
                cbar_ax = fig.add_axes([0.15, 0.04, 0.3, 0.015])
                cbar = fig.colorbar(scatter_for_cbar, cax=cbar_ax, orientation='horizontal')
                cbar.set_label('Density', fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                
                # 图例
                from matplotlib.lines import Line2D
                legend_ax = fig.add_axes([0.55, 0.04, 0.4, 0.015])
                legend_ax.axis('off')
                legend_elements = [
                    Line2D([0], [0], color='#ff7f0e', linewidth=2, label='Linear Fit'),
                    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Robust Fit')
                ]
                legend_ax.legend(handles=legend_elements, loc='center', ncol=2, 
                               frameon=True, fontsize=9, framealpha=0.9)
            
            # 保存图像
            output_file_png = self.plots_dir / f"bss_vs_accuracy_scatter_{var_type}.png"
            output_file_pdf = self.plots_dir / f"bss_vs_accuracy_scatter_{var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"BSS vs Accuracy散点图已保存到: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制BSS vs Accuracy散点图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_score_spatial_distribution(self, leadtimes: List[int], var_type: str, models: List[str]):
        """
        绘制Score空间分布图
        分为上下两半（L0和L3），每个lead占2行，第1行留空+3模型，第2行4模型
        子图之间不留空隙，仅在最外围绘制经纬度标签和脊线，最下方绘制colorbar
        
        辅助函数：计算等高线级别数量
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            var_type: 变量类型
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
            from matplotlib.colors import LinearSegmentedColormap
            
            logger.info(f"绘制Score空间分布图: L{leadtimes} {var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_score_data(leadtimes, models, var_type)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于Score空间分布图")
                return
            
            available_leadtimes = [lt for lt in leadtimes if lt in all_leadtimes_data]
            if not available_leadtimes:
                logger.warning("指定的lead time均无Score数据用于空间分布")
                return
            
            def _model_has_data(model: str) -> bool:
                return any(
                    lead in all_leadtimes_data and model in all_leadtimes_data[lead]
                    for lead in available_leadtimes
                )
            
            model_names = [m for m in models if _model_has_data(m)]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 创建陆地掩膜（基于第一个模型的数据）
            land_mask = None
            if var_type == 'prec':
                first_leadtime = available_leadtimes[0]
                first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
                sample_score = first_model_data['score']
                land_mask = self._create_land_mask(var_type, sample_score)
            
            # 收集所有leadtime的所有score数据，用于计算统一的colorbar范围
            # 只考虑陆地区域的数据
            all_score_values = []
            for leadtime in leadtimes:
                if leadtime in all_leadtimes_data:
                    for model_data in all_leadtimes_data[leadtime].values():
                        score_data = model_data['score']
                        # 应用陆地掩膜（如果存在）
                        if land_mask is not None:
                            # 对齐掩膜到数据网格
                            if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                                mask_aligned = land_mask.reindex(
                                    lat=score_data.lat,
                                    lon=score_data.lon,
                                    method='nearest',
                                    tolerance=0.5
                                )
                                score_masked = score_data.where(mask_aligned)
                            else:
                                score_masked = score_data
                        else:
                            score_masked = score_data
                        
                        score_flat = score_masked.values.flatten()
                        valid_values = score_flat[~np.isnan(score_flat)]
                        all_score_values.extend(valid_values)
            
            # 计算统一范围
            if all_score_values:
                data_min = np.min(all_score_values)
                data_max = np.max(all_score_values)
                data_range = data_max - data_min
                
                # 动态选择取整精度
                if data_range > 10:
                    round_factor = 1
                elif data_range > 1:
                    round_factor = 10
                else:
                    round_factor = 100
                
                score_min = np.floor(data_min * round_factor) / round_factor
                score_max = np.ceil(data_max * round_factor) / round_factor
            else:
                score_min, score_max = 0.0, 1.0
            
            logger.info(f"Score范围: [{score_min:.2f}, {score_max:.2f}]")
            
            # 使用绿色系colormap
            cmap = LinearSegmentedColormap.from_list(
                'white_to_green',
                [
                    (0.0, "#f7fcf5"),
                    (0.25, "#e5f5e0"),
                    (0.50, "#a1d99b"),
                    (0.75, "#41ab5d"),
                    (1.0, "#006d2c")
                ]
            )
            
            # 计算布局
            n_leadtimes = len(available_leadtimes)
            n_cols = 4  # 固定4列：留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 基于第一个模型的数据计算经纬度边界
            # 根据实际数据范围（考虑陆地掩膜）来决定绘制范围
            first_leadtime = available_leadtimes[0]
            first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
            sample_score = first_model_data['score']
            
            # 计算实际数据范围（取整）
            lon_min, lon_max, lat_min, lat_max = self._compute_data_extent(sample_score, land_mask)
            
            # 创建边界数组（用于绘制）
            def _compute_edges(center_coords: np.ndarray) -> np.ndarray:
                center_coords = np.asarray(center_coords)
                diffs = np.diff(center_coords)
                first_edge = center_coords[0] - diffs[0] / 2.0 if diffs.size > 0 else center_coords[0] - 0.5
                last_edge = center_coords[-1] + diffs[-1] / 2.0 if diffs.size > 0 else center_coords[-1] + 0.5
                mid_edges = center_coords[:-1] + diffs / 2.0 if diffs.size > 0 else np.array([])
                return np.concatenate([[first_edge], mid_edges, [last_edge]])
            
            # 获取所有数据的经纬度中心点
            lon_centers = sample_score.lon.values if hasattr(sample_score, 'lon') else None
            lat_centers = sample_score.lat.values if hasattr(sample_score, 'lat') else None
            
            if lon_centers is None or lat_centers is None:
                logger.error("数据缺少经纬度坐标")
                return
            
            # 根据实际范围筛选经纬度
            lon_centers_filtered = lon_centers[(lon_centers >= lon_min) & (lon_centers <= lon_max)]
            lat_centers_filtered = lat_centers[(lat_centers >= lat_min) & (lat_centers <= lat_max)]
            
            # 如果筛选后为空，使用原始范围
            if len(lon_centers_filtered) == 0:
                lon_centers_filtered = lon_centers
            if len(lat_centers_filtered) == 0:
                lat_centers_filtered = lat_centers
            
            lon_edges = _compute_edges(lon_centers_filtered)
            lat_edges = _compute_edges(lat_centers_filtered)
            
            # 计算画布大小
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
            for lt_idx, leadtime in enumerate(available_leadtimes):
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
                    
                    # 基于模型自身网格计算边界（根据实际数据范围）
                    try:
                        model_score = model_data['score']
                        model_lon_min, model_lon_max, model_lat_min, model_lat_max = self._compute_data_extent(model_score, land_mask)
                        ax_spatial.set_extent([model_lon_min, model_lon_max, model_lat_min, model_lat_max], crs=ccrs.PlateCarree())
                    except Exception:
                        # 如果计算失败，使用全局范围
                        ax_spatial.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 只在外围显示坐标轴标签和脊线
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    # 只在最外侧显示：仅每个lead的第二行才显示纬度（row_models2），最底行显示经度
                    gl.left_labels = (col_idx == 0 and row_obs != row_models2)
                    gl.bottom_labels = (lt_idx == n_leadtimes - 1)
                    gl.xlines = gl.bottom_labels
                    gl.ylines = gl.left_labels
                    
                    # 绘制Score数据
                    score_data = model_data['score']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=score_data.lat,
                                lon=score_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            score_data_masked = score_data.where(mask_aligned)
                        else:
                            score_data_masked = score_data
                    else:
                        # 如果没有掩膜，对于降水数据使用NaN作为掩膜
                        if var_type == 'prec':
                            ocean_mask = np.isnan(score_data.values)
                            score_data_masked = score_data.where(~ocean_mask)
                        else:
                            score_data_masked = score_data
                    
                    n_levels = _compute_n_levels(score_data_masked.values, score_min, score_max)
                    levels = np.linspace(score_min, score_max, n_levels)
                    
                    im = ax_spatial.contour(
                        score_data_masked.lon, score_data_masked.lat, score_data_masked,
                        levels=levels, transform=ccrs.PlateCarree(),
                        cmap=cmap, linewidths=1.2, alpha=0.8
                    )
                    
                    if im_for_cbar is None:
                        im_for_cbar = im
                    
                    # 模型标签
                    label = chr(97 + col_idx)  # a, b, c
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name}', 
                                  transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                  verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_spatial)
                    _expand_axes_vertically(ax_spatial, lt_idx == 0, lt_idx == n_leadtimes - 1)
                
                # 第2行：4个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(model_names):
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[model_idx]
                    if model not in model_data_dict:
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_spatial = fig.add_subplot(gs[row_models2, col_idx], projection=ccrs.PlateCarree())
                    
                    # 基于模型自身网格计算边界（根据实际数据范围）
                    try:
                        model_score = model_data['score']
                        model_lon_min, model_lon_max, model_lat_min, model_lat_max = self._compute_data_extent(model_score, land_mask)
                        ax_spatial.set_extent([model_lon_min, model_lon_max, model_lat_min, model_lat_max], crs=ccrs.PlateCarree())
                    except Exception:
                        # 如果计算失败，使用全局范围
                        ax_spatial.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    ax_spatial.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_spatial.add_feature(cfeature.LAND, alpha=0.1)
                    ax_spatial.add_feature(cfeature.OCEAN, alpha=0.1)
                    
                    # 只在外围显示坐标轴标签和脊线
                    gl = ax_spatial.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    is_last_lead = (lt_idx == n_leadtimes - 1)
                    # 仅第二行（row_models2）显示左侧纬度
                    gl.bottom_labels = is_last_lead  # 只底行显示经度
                    gl.left_labels = (col_idx == 0)
                    gl.xlines = gl.bottom_labels
                    gl.ylines = gl.left_labels
                    
                    # 绘制Score数据
                    score_data = model_data['score']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=score_data.lat,
                                lon=score_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            score_data_masked = score_data.where(mask_aligned)
                        else:
                            score_data_masked = score_data
                    else:
                        # 如果没有掩膜，对于降水数据使用NaN作为掩膜
                        if var_type == 'prec':
                            ocean_mask = np.isnan(score_data.values)
                            score_data_masked = score_data.where(~ocean_mask)
                        else:
                            score_data_masked = score_data
                    
                    n_levels = _compute_n_levels(score_data_masked.values, score_min, score_max)
                    levels = np.linspace(score_min, score_max, n_levels)
                    
                    ax_spatial.contour(
                        score_data_masked.lon, score_data_masked.lat, score_data_masked,
                        levels=levels, transform=ccrs.PlateCarree(),
                        cmap=cmap, linewidths=1.2, alpha=0.8
                    )
                    
                    # 模型标签
                    label = chr(97 + model_idx)  # d, e, f, g
                    ax_spatial.text(0.02, 0.98, f'({label}) {display_name}', 
                                  transform=ax_spatial.transAxes, fontsize=11, fontweight='bold',
                                  verticalalignment='top', horizontalalignment='left')
                    
                    content_axes.append(ax_spatial)
                    _expand_axes_vertically(ax_spatial, lt_idx == 0, lt_idx == n_leadtimes - 1)
            
            # 在最下方添加colorbar
            if im_for_cbar is not None:
                cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
                cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='horizontal')
                cbar.set_label('Combined Score', fontsize=11)
                cbar.ax.tick_params(labelsize=9)
            
            # 保存图像
            output_file_png = self.plots_dir / f"score_spatial_distribution_{var_type}.png"
            output_file_pdf = self.plots_dir / f"score_spatial_distribution_{var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Score空间分布图已保存到: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制Score空间分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_score_cdf_distribution(self, leadtimes: List[int], var_type: str, models: List[str]):
        """
        绘制Score及其三个组成量的CDF累积分布图
        分为上下两半（L0和L3），每个lead占2行，第1行留空+3模型，第2行4模型
        子图之间不留空隙，仅在最外围绘制横纵坐标标签，最下方绘制图例
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            var_type: 变量类型
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            from matplotlib.lines import Line2D
            
            logger.info(f"绘制Score CDF分布图: L{leadtimes} {var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_score_data(leadtimes, models, var_type)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于Score CDF分布图")
                return
            
            available_leadtimes = [lt for lt in leadtimes if lt in all_leadtimes_data]
            if not available_leadtimes:
                logger.warning("指定的lead time均无Score数据用于CDF图")
                return
            
            def _model_has_data(model: str) -> bool:
                return any(
                    lead in all_leadtimes_data and model in all_leadtimes_data[lead]
                    for lead in available_leadtimes
                )
            
            model_names = [m for m in models if _model_has_data(m)]
            n_models = len(model_names)
            
            if n_models == 0:
                logger.warning("没有可用的模型数据")
                return
            
            # 创建陆地掩膜（基于第一个模型的数据）
            land_mask = None
            if var_type == 'prec':
                first_leadtime = available_leadtimes[0]
                first_model_data = list(all_leadtimes_data[first_leadtime].values())[0]
                sample_score = first_model_data['score']
                land_mask = self._create_land_mask(var_type, sample_score)
            
            # 计算布局
            n_leadtimes = len(available_leadtimes)
            n_cols = 4  # 固定4列：留白 + 3个模型，或4个模型
            n_rows = n_leadtimes * 2  # 每个leadtime占2行
            
            # 创建图形（保持原来的画布大小）
            fig_width = n_cols * 4.5
            fig_height = max(n_rows * 2.5, 6.0)
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # GridSpec：紧凑布局，子图之间不留空隙
            gs = GridSpec(
                n_rows,
                n_cols,
                figure=fig,
                height_ratios=[1] * n_rows,
                width_ratios=[1] * n_cols,
                hspace=0.0,
                wspace=0.0,
                left=0.08,
                right=0.95,
                top=0.95,
                bottom=0.18,
            )
            
            # 绘制每个leadtime
            for lt_idx, leadtime in enumerate(available_leadtimes):
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
                    
                    ax_cdf = fig.add_subplot(gs[row_obs, col_idx + 1])
                    
                    # 获取4个变量的空间数据（在展平前应用掩膜）
                    score_data = model_data['score']
                    accuracy_data = model_data['s_accuracy']
                    pcc_data = model_data['s_pcc']
                    rmse_data = model_data['s_rmse']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=score_data.lat,
                                lon=score_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            score_data = score_data.where(mask_aligned)
                            accuracy_data = accuracy_data.where(mask_aligned)
                            pcc_data = pcc_data.where(mask_aligned)
                            rmse_data = rmse_data.where(mask_aligned)
                    
                    # 展平数据
                    score_spatial = score_data.values.flatten()
                    accuracy_spatial = accuracy_data.values.flatten()
                    pcc_spatial = pcc_data.values.flatten()
                    rmse_spatial = rmse_data.values.flatten()
                    
                    # 移除NaN值
                    score_valid = score_spatial[~np.isnan(score_spatial)]
                    accuracy_valid = accuracy_spatial[~np.isnan(accuracy_spatial)]
                    pcc_valid = pcc_spatial[~np.isnan(pcc_spatial)]
                    rmse_valid = rmse_spatial[~np.isnan(rmse_spatial)]
                    
                    # 将PCC从[-1,1]归一化到[0,1]
                    pcc_valid_normalized = (pcc_valid + 1) / 2
                    pcc_valid_normalized = np.clip(pcc_valid_normalized, 0.0, 1.0)
                    
                    # 绘制CDF
                    if len(score_valid) > 10:
                        score_sorted = np.sort(score_valid)
                        cdf_score = np.arange(1, len(score_sorted)+1) / len(score_sorted)
                        ax_cdf.plot(score_sorted, cdf_score, 
                                   label='Score', color='#d62728', linewidth=1.5, alpha=0.8)
                    
                    if len(accuracy_valid) > 10:
                        accuracy_sorted = np.sort(accuracy_valid)
                        cdf_accuracy = np.arange(1, len(accuracy_sorted)+1) / len(accuracy_sorted)
                        ax_cdf.plot(accuracy_sorted, cdf_accuracy, 
                                   label='Accuracy', color='#2ca02c', linewidth=1.5, alpha=0.8)
                    
                    if len(pcc_valid_normalized) > 10:
                        pcc_sorted = np.sort(pcc_valid_normalized)
                        cdf_pcc = np.arange(1, len(pcc_sorted)+1) / len(pcc_sorted)
                        ax_cdf.plot(pcc_sorted, cdf_pcc, 
                                   label='PCC', color='#1f77b4', linewidth=1.5, alpha=0.8)
                    
                    if len(rmse_valid) > 10:
                        rmse_sorted = np.sort(rmse_valid)
                        cdf_rmse = np.arange(1, len(rmse_sorted)+1) / len(rmse_sorted)
                        ax_cdf.plot(rmse_sorted, cdf_rmse, 
                                   label='n_RMSE', color='#ff7f0e', linewidth=1.5, alpha=0.8)
                    
                    # 设置轴标签
                    ax_cdf.set_xlim(0, 1)
                    ax_cdf.set_ylim(0, 1.05)
                    ticks_major = np.arange(0.1, 1.0, 0.1)  # 去除边界刻度
                    ax_cdf.set_xticks(ticks_major)
                    ax_cdf.set_yticks(ticks_major)
                    ax_cdf.tick_params(axis='both', labelsize=8)
                    ax_cdf.grid(True, alpha=0.3, linestyle='--')
                    ax_cdf.set_axisbelow(True)
                    
                    # 所有子图保留边框；上半行不显示刻度标签
                    is_bottom = False
                    show_left = False
                    ax_cdf.set_xlabel('', fontsize=10)
                    ax_cdf.set_ylabel('', fontsize=10)
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax_cdf.spines[spine].set_visible(True)
                    ax_cdf.tick_params(axis='y', labelleft=False, left=True, labelsize=8)
                    ax_cdf.tick_params(axis='x', labelbottom=False, bottom=True, labelsize=8)
                    
                    # 模型标签
                    label = chr(97 + col_idx)
                    ax_cdf.text(0.02, 0.95, f'({label}) {display_name}', 
                              transform=ax_cdf.transAxes, fontsize=11, fontweight='bold',
                              verticalalignment='top', horizontalalignment='left')
                    if col_idx == 0:
                        ax_cdf.text(0.98, 0.95, f'L{leadtime}', 
                                  transform=ax_cdf.transAxes, fontsize=11, fontweight='bold',
                                  verticalalignment='top', horizontalalignment='right')
                
                # 第2行：4个模型
                for col_idx in range(4):
                    model_idx = col_idx + 3
                    if model_idx >= len(model_names):
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model = model_names[model_idx]
                    if model not in model_data_dict:
                        ax_blank = fig.add_subplot(gs[row_models2, col_idx])
                        ax_blank.axis('off')
                        continue
                    
                    model_data = model_data_dict[model]
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    
                    ax_cdf = fig.add_subplot(gs[row_models2, col_idx])
                    
                    # 获取4个变量的空间数据（在展平前应用掩膜）
                    score_data = model_data['score']
                    accuracy_data = model_data['s_accuracy']
                    pcc_data = model_data['s_pcc']
                    rmse_data = model_data['s_rmse']
                    
                    # 应用陆地掩膜（如果存在）
                    if land_mask is not None:
                        # 对齐掩膜到数据网格
                        if 'lat' in land_mask.coords and 'lon' in land_mask.coords:
                            mask_aligned = land_mask.reindex(
                                lat=score_data.lat,
                                lon=score_data.lon,
                                method='nearest',
                                tolerance=0.5
                            )
                            score_data = score_data.where(mask_aligned)
                            accuracy_data = accuracy_data.where(mask_aligned)
                            pcc_data = pcc_data.where(mask_aligned)
                            rmse_data = rmse_data.where(mask_aligned)
                    
                    # 展平数据
                    score_spatial = score_data.values.flatten()
                    accuracy_spatial = accuracy_data.values.flatten()
                    pcc_spatial = pcc_data.values.flatten()
                    rmse_spatial = rmse_data.values.flatten()
                    
                    # 移除NaN值
                    score_valid = score_spatial[~np.isnan(score_spatial)]
                    accuracy_valid = accuracy_spatial[~np.isnan(accuracy_spatial)]
                    pcc_valid = pcc_spatial[~np.isnan(pcc_spatial)]
                    rmse_valid = rmse_spatial[~np.isnan(rmse_spatial)]
                    
                    # 将PCC从[-1,1]归一化到[0,1]
                    pcc_valid_normalized = (pcc_valid + 1) / 2
                    pcc_valid_normalized = np.clip(pcc_valid_normalized, 0.0, 1.0)
                    
                    # 绘制CDF
                    if len(score_valid) > 10:
                        score_sorted = np.sort(score_valid)
                        cdf_score = np.arange(1, len(score_sorted)+1) / len(score_sorted)
                        ax_cdf.plot(score_sorted, cdf_score, 
                                   label='Score', color='#d62728', linewidth=1.5, alpha=0.8)
                    
                    if len(accuracy_valid) > 10:
                        accuracy_sorted = np.sort(accuracy_valid)
                        cdf_accuracy = np.arange(1, len(accuracy_sorted)+1) / len(accuracy_sorted)
                        ax_cdf.plot(accuracy_sorted, cdf_accuracy, 
                                   label='Accuracy', color='#2ca02c', linewidth=1.5, alpha=0.8)
                    
                    if len(pcc_valid_normalized) > 10:
                        pcc_sorted = np.sort(pcc_valid_normalized)
                        cdf_pcc = np.arange(1, len(pcc_sorted)+1) / len(pcc_sorted)
                        ax_cdf.plot(pcc_sorted, cdf_pcc, 
                                   label='PCC', color='#1f77b4', linewidth=1.5, alpha=0.8)
                    
                    if len(rmse_valid) > 10:
                        rmse_sorted = np.sort(rmse_valid)
                        cdf_rmse = np.arange(1, len(rmse_sorted)+1) / len(rmse_sorted)
                        ax_cdf.plot(rmse_sorted, cdf_rmse, 
                                   label='n_RMSE', color='#ff7f0e', linewidth=1.5, alpha=0.8)
                    
                    # 设置轴标签
                    ax_cdf.set_xlim(0, 1)
                    ax_cdf.set_ylim(0, 1.05)
                    ticks_major = np.arange(0.1, 1.0, 0.1)  # 去除边界刻度
                    ax_cdf.set_xticks(ticks_major)
                    ax_cdf.set_yticks(ticks_major)
                    ax_cdf.tick_params(axis='both', labelsize=8)
                    ax_cdf.grid(True, alpha=0.3, linestyle='--')
                    ax_cdf.set_axisbelow(True)
                    
                    # 所有子图保留边框；仅最左列/最底行显示刻度标签
                    is_bottom = (lt_idx == n_leadtimes - 1)
                    show_left = (col_idx == 0)
                    ax_cdf.set_xlabel('Score Value' if is_bottom else '', fontsize=10)
                    ax_cdf.set_ylabel('Cumulative Probability' if show_left else '', fontsize=10)
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax_cdf.spines[spine].set_visible(True)
                    ax_cdf.tick_params(axis='y', labelleft=show_left, left=True, labelsize=8)
                    ax_cdf.tick_params(axis='x', labelbottom=is_bottom, bottom=True, labelsize=8)
                    
                    # 模型标签
                    label = chr(97 + model_idx)
                    ax_cdf.text(0.02, 0.95, f'({label}) {display_name}', 
                              transform=ax_cdf.transAxes, fontsize=11, fontweight='bold',
                              verticalalignment='top', horizontalalignment='left')
            
            # 在最下方添加图例
            legend_ax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
            legend_ax.axis('off')
            legend_elements = [
                Line2D([0], [0], color='#d62728', linewidth=4, label='Score'),
                Line2D([0], [0], color='#2ca02c', linewidth=4, label='Accuracy'),
                Line2D([0], [0], color='#1f77b4', linewidth=4, label='PCC'),
                Line2D([0], [0], color='#ff7f0e', linewidth=4, label='n_RMSE')
            ]
            legend_ax.legend(handles=legend_elements, loc='center', ncol=4,
                           frameon=True, fontsize=11, framealpha=0.9, edgecolor='gray',
                           handlelength=3.5, handletextpad=1.0, columnspacing=2.0)
            
            # 保存图像
            output_file_png = self.plots_dir / f"score_cdf_distribution_{var_type}.png"
            output_file_pdf = self.plots_dir / f"score_cdf_distribution_{var_type}.pdf"
            
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Score CDF分布图已保存到: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制Score CDF分布图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_score_bar(self, leadtimes: List[int], var_type: str, models: List[str]):
        """
        绘制Score的不同模式的bar图
        分为上下两半（L0和L3），每个lead显示一个bar图，所有模型并排显示
        
        Args:
            leadtimes: 提前期列表（通常为[0, 3]）
            var_type: 变量类型
            models: 模型列表
        """
        try:
            from matplotlib.gridspec import GridSpec
            
            logger.info(f"绘制Score bar图: L{leadtimes} {var_type}")
            
            # 加载数据
            all_leadtimes_data = self._load_score_data(leadtimes, models, var_type)
            if not all_leadtimes_data:
                logger.warning(f"没有有效数据用于Score bar图")
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
            
            # 计算布局：上下两个子图共享X轴
            available_leads = [lt for lt in leadtimes if lt in all_leadtimes_data]
            if not available_leads:
                logger.warning("没有可用的Score数据用于柱状图")
                return
            
            n_leadtimes = len(available_leads)
            fig_width = max(10.0, len(model_names) * 1.2)
            fig_height = max(3.0 * n_leadtimes, 6.0)
            fig, axes = plt.subplots(
                n_leadtimes, 1, sharex=True,
                figsize=(fig_width, fig_height),
                gridspec_kw={'hspace': 0.05}
            )
            if n_leadtimes == 1:
                axes = [axes]
            
            for ax_bar, leadtime in zip(axes, available_leads):
                model_data_dict = all_leadtimes_data[leadtime]
                
                model_labels = []
                model_scores = []
                for idx, model in enumerate(model_names):
                    if model not in model_data_dict:
                        continue
                    model_data = model_data_dict[model]
                    score_values = model_data['score'].values.flatten()
                    valid_scores = score_values[~np.isnan(score_values)]
                    if len(valid_scores) == 0:
                        continue
                    avg_score = np.mean(valid_scores)
                    display_name = model.replace('-mon', '').replace('mon-', '')
                    label_char = chr(97 + len(model_labels))
                    model_labels.append(f"({label_char}) {display_name}")
                    model_scores.append(avg_score)
                
                if not model_scores:
                    ax_bar.axis('off')
                    ax_bar.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                                fontsize=11, fontweight='bold', color='red', transform=ax_bar.transAxes)
                    continue
                
                x_pos = np.arange(len(model_labels))
                bars = ax_bar.bar(
                    x_pos,
                    model_scores,
                    color='#4fa16c',
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                )
                ax_bar.set_ylabel('Mean Score', fontsize=11)
                ax_bar.set_xticks(x_pos)
                ax_bar.set_xticklabels(model_labels, fontsize=9, rotation=45, ha='right')
                ax_bar.tick_params(axis='y', labelsize=9)
                if ax_bar is not axes[-1]:
                    # 隐藏 x 轴标签而不是设置字体大小为 0
                    ax_bar.set_xticklabels([])
                    ax_bar.tick_params(axis='x', length=0)
                ax_bar.grid(True, axis='y', alpha=0.3, linestyle='--')
                ax_bar.set_axisbelow(True)
                
                data_min_bar = min(model_scores)
                data_max_bar = max(model_scores)
                data_range_bar = data_max_bar - data_min_bar
                margin = max(data_range_bar * 0.1, 0.02)
                y_min_bar = max(0, data_min_bar - margin)
                y_max_bar = data_max_bar + margin
                ax_bar.set_ylim(y_min_bar, y_max_bar)
                
                ax_bar.text(
                    0.98,
                    0.92,
                    f'L{leadtime}',
                    transform=ax_bar.transAxes,
                    fontsize=11,
                    fontweight='bold',
                    ha='right',
                    va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none')
                )
            
            axes[-1].set_xlabel('Model', fontsize=12)
            
            output_file_png = self.plots_dir / f"score_bar_{var_type}.png"
            output_file_pdf = self.plots_dir / f"score_bar_{var_type}.pdf"
            
            plt.tight_layout()
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Score bar图已保存到: {output_file_png}")
            
        except Exception as e:
            logger.error(f"绘制Score bar图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_accuracy_comparison(self, leadtime: int, var_type: str, models: List[str]):
        """绘制所有模型的Accuracy Score对比图"""
        return

    
    def plot_national_average_score_bar(self, leadtime: int, var_type: str, models: List[str]):
        """绘制所有模型全格点平均综合得分的柱状图"""
        return

    
    def plot_leadtime_score_lines(self, var_type: str, models: List[str], leadtimes: List[int]):
        """绘制各模式的全图有效格点平均得分随lead time变化的折线图（上下两个子图）"""
        try:
            logger.info(f"绘制lead time得分折线图（上下两个子图）: {var_type}")
            
            # 收集所有模型在不同leadtime的得分
            model_leadtime_scores = {}
            
            for model in models:
                model_scores = {}
                
                for leadtime in leadtimes:
                    # 查找对应的NetCDF文件
                    nc_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
                    
                    if not nc_file.exists():
                        logger.warning(f"结果文件不存在，跳过: {nc_file}")
                        continue
                    
                    try:
                        # 加载NetCDF文件
                        ds = xr.open_dataset(nc_file)
                        
                        # 计算全部有效格点的平均综合得分
                        if 'score' in ds:
                            score_data = ds['score']
                            
                            # 计算加权平均（考虑纬度权重）
                            weights = np.cos(np.deg2rad(score_data.lat))
                            avg_score = score_data.weighted(weights).mean(dim=['lat', 'lon'])
                            
                            if not np.isnan(avg_score.values):
                                model_scores[leadtime] = float(avg_score.values)
                                logger.debug(f"{model} L{leadtime} {var_type} 得分: {avg_score.values:.4f}")
                            else:
                                logger.warning(f"{model} L{leadtime} {var_type} 得分为NaN")
                        else:
                            logger.warning(f"文件 {nc_file} 中没有 'score' 变量")
                        
                        ds.close()
                        
                    except Exception as e:
                        logger.error(f"处理文件 {nc_file} 时出错: {e}")
                        continue
                
                if model_scores:
                    model_leadtime_scores[model] = model_scores
            
            if not model_leadtime_scores:
                logger.warning(f"没有有效的得分数据用于折线图: {var_type}")
                return
            
            # 定义颜色列表（支持多个模型）
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # 模型名称映射（去除mon标签）
            models_list = list(model_leadtime_scores.keys())
            
            # 收集所有得分值和偏差值用于设置Y轴范围
            all_score_values = []
            # all_anomaly_values = []
            
            fig_height = 6.0
            fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
            
            # 绘制每个模型的折线
            for i, model in enumerate(models_list):
                leadtime_list = sorted(model_leadtime_scores[model].keys())
                score_list = [model_leadtime_scores[model][lt] for lt in leadtime_list]
                
                # 去除mon标签显示
                display_name = model.replace('-mon', '').replace('mon-', '')
                color = colors[i % len(colors)]
                
                line, = ax.plot(leadtime_list, score_list, 
                        color=color,
                        marker='o',
                        markersize=6,
                        linewidth=2.0,
                        linestyle='-',
                        label=display_name,
                        alpha=0.85)
                
                all_score_values.extend(score_list)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            

            # 设置坐标轴
            ax.set_xlabel('Lead Time', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            y_min = min(all_score_values)
            y_max = max(all_score_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)
            ax.set_xticks(leadtimes)
            ax.set_xlim(leadtimes[0] - 0.2, leadtimes[-1] + 0.2)
            ax.margins(x=0)

            # 调整布局
            plt.tight_layout()
            
            # 在图像底部添加横排图例，带序号
            # 创建图例标签（带序号）
            legend_labels = []
            legend_handles = []
            for i, model in enumerate(models_list):
                display_name = model.replace('-mon', '').replace('mon-', '')
                color = colors[i % len(colors)]
                # 创建图例句柄
                handle = plt.Line2D([0], [0], color=color, linewidth=2.5, linestyle='-')
                legend_handles.append(handle)
                # 创建带序号的标签
                subplot_label = chr(97 + i)
                legend_labels.append(f"({subplot_label}) {display_name}")
            
            # 在图像底部添加图例
            fig.legend(handles=legend_handles, labels=legend_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, -0.1), 
                      ncol=(len(models_list) + 1) // 2,
                      framealpha=0.9, fontsize=10)
            
            # 保存图像
            output_file = f"{self.output_dir}/plots/leadtime_score_lines_{var_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Lead time得分折线图已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"绘制lead time得分折线图时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def calculate_regional_averages(self, ds: xr.Dataset) -> Dict[str, Dict[str, float]]:
        """计算区域平均值"""
        regional_results = {}
        
        for region_name, region_bounds in self.regions.items():
            region_data = ds.sel(
                lat=slice(region_bounds['lat'][0], region_bounds['lat'][1]),
                lon=slice(region_bounds['lon'][0], region_bounds['lon'][1])
            )
            
            weights = np.cos(np.deg2rad(region_data.lat))
            
            regional_results[region_name] = {
                's_accuracy': float(region_data['s_accuracy'].weighted(weights).mean().values),
                's_equitable_accuracy': float(region_data['s_equitable_accuracy'].weighted(weights).mean().values) if 's_equitable_accuracy' in region_data else np.nan,
                's_rmse': float(region_data['s_rmse'].weighted(weights).mean().values),
                's_pcc': float(region_data['s_pcc'].weighted(weights).mean().values),
                'score': float(region_data['score'].weighted(weights).mean().values),
                'score_raw': float(region_data['score_raw'].weighted(weights).mean().values) if 'score_raw' in region_data else np.nan
            }
        
        return regional_results
    
    def process_single_model_leadtime(self, model: str, leadtime: int, 
                                    var_type: str) -> Optional[xr.Dataset]:
        """处理单个模型和提前期的数据（优化内存使用）"""
        try:
            logger.info(f"处理 {model} L{leadtime} {var_type}")
            
            # 加载数据（强制到内存）
            obs_data = self._load_obs_data(var_type)
            fcst_data = self._load_forecast_data(model, var_type, leadtime)
            
            if fcst_data is None or obs_data is None:
                logger.warning(f"{model} L{leadtime} 无数据")
                return None
            
            # 加载ensemble数据（用于BS/BSS计算）
            ensemble_data = self._load_forecast_ensemble(model, var_type, leadtime)
            
            # 数据对齐
            obs_aligned, fcst_aligned = self._align_data(obs_data, fcst_data)
            
            if obs_aligned is None or fcst_aligned is None:
                logger.warning(f"{model} L{leadtime} 数据对齐失败")
                return None
            
            # 计算观测的气候态（所有模型使用统一参考）
            logger.info("计算观测气候态（所有模型统一参考）...")
            obs_climatology = obs_aligned.mean(dim='time')  # shape: (lat, lon)
            logger.info(f"观测气候态计算完成，范围: [{float(obs_climatology.min()):.2f}, {float(obs_climatology.max()):.2f}]")
            
            # 对齐ensemble数据（如果有）
            ensemble_aligned = None
            climatology_hit_rates = None
            if ensemble_data is not None:
                logger.info("对齐ensemble数据...")
                # ensemble_data shape: (time, number, lat, lon)
                # 需要对齐时间和空间维度
                common_times = obs_aligned.time.to_index().intersection(ensemble_data.time.to_index())
                if len(common_times) > 0:
                    ensemble_aligned = ensemble_data.sel(time=common_times)
                    # 空间对齐：使用经纬度容差匹配（0.5度容差），找到真正重叠的格点
                    # 不进行范围裁剪，直接通过经纬度容差匹配找到匹配的格点
                    # 获取观测数据的坐标值
                    obs_lat_vals = obs_aligned.lat.values
                    obs_lon_vals = obs_aligned.lon.values
                    
                    # 获取ensemble数据的坐标值
                    ensemble_lat_vals = ensemble_aligned.lat.values
                    ensemble_lon_vals = ensemble_aligned.lon.values
                    
                    # 容差阈值（0.5度，1度网格间距的一半）
                    # 对于1度网格，0.5度容差可以匹配所有合理的网格变体（x.5, x.0, x.05等）
                    tolerance = 0.5
                    
                    # 找到ensemble网格中与观测网格匹配的格点（0.5度容差内）
                    ensemble_lat_matched_set = set()
                    ensemble_lon_matched_set = set()
                    
                    for obs_lat in obs_lat_vals:
                        # 找到与观测纬度在0.5度容差内的ensemble纬度格点
                        lat_distances = np.abs(ensemble_lat_vals - obs_lat)
                        matched_lat_indices = np.where(lat_distances <= tolerance)[0]
                        if len(matched_lat_indices) > 0:
                            # 选择最近的格点
                            nearest_lat_idx = matched_lat_indices[np.argmin(lat_distances[matched_lat_indices])]
                            ensemble_lat_matched_set.add(nearest_lat_idx)
                    
                    for obs_lon in obs_lon_vals:
                        # 找到与观测经度在0.5度容差内的ensemble经度格点
                        lon_distances = np.abs(ensemble_lon_vals - obs_lon)
                        matched_lon_indices = np.where(lon_distances <= tolerance)[0]
                        if len(matched_lon_indices) > 0:
                            # 选择最近的格点
                            nearest_lon_idx = matched_lon_indices[np.argmin(lon_distances[matched_lon_indices])]
                            ensemble_lon_matched_set.add(nearest_lon_idx)
                    
                    if len(ensemble_lat_matched_set) > 0 and len(ensemble_lon_matched_set) > 0:
                        # 对索引排序
                        ensemble_lat_matched = sorted(list(ensemble_lat_matched_set))
                        ensemble_lon_matched = sorted(list(ensemble_lon_matched_set))
                        
                        # 只保留匹配的格点，保持原始坐标
                        ensemble_aligned = ensemble_aligned.isel(
                            lat=ensemble_lat_matched,
                            lon=ensemble_lon_matched
                        )
                    # 注意：obs_aligned 和 fcst_aligned 都不进行裁剪，保持完整范围
                    # fcst_aligned 已经在 _align_data 中通过经纬度容差匹配处理过
                    
                    # 确保数据在内存中
                    if hasattr(ensemble_aligned, 'chunks'):
                        ensemble_aligned = ensemble_aligned.compute()
                    
                    logger.info(f"Ensemble数据对齐完成: {ensemble_aligned.shape}")
                else:
                    logger.warning("Ensemble数据时间对齐失败")
                    ensemble_aligned = None
            
            # 确保所有数据对齐到最终共同的时间范围（仅对能够进行计算的时间范围进行计算）
            # 时间长度不完全一致是正常的，只使用共同的时间范围
            if ensemble_aligned is not None:
                # 对齐obs、fcst、ensemble到共同时间范围
                aligned_result = align_multiple_datasets_to_common_time(
                    obs_aligned, fcst_aligned, ensemble_aligned,
                    min_common_times=12,
                    return_climatology=True
                )
                
                if aligned_result[0] is None:
                    logger.warning("多数据集时间对齐失败")
                    return None
                
                obs_aligned, fcst_aligned, ensemble_aligned = aligned_result[0]
                obs_climatology = aligned_result[1]
            else:
                # 没有ensemble数据，只对齐obs和fcst
                aligned_result = align_multiple_datasets_to_common_time(
                    obs_aligned, fcst_aligned,
                    min_common_times=12,
                    return_climatology=True
                )
                
                if aligned_result[0] is None:
                    logger.warning("时间对齐失败")
                    return None
                
                obs_aligned, fcst_aligned = aligned_result[0]
                obs_climatology = aligned_result[1]
            
            # 确保数据在内存中
            if hasattr(obs_aligned, 'chunks'):
                obs_aligned = obs_aligned.compute()
            if hasattr(fcst_aligned, 'chunks'):
                fcst_aligned = fcst_aligned.compute()
            if ensemble_aligned is not None and hasattr(ensemble_aligned, 'chunks'):
                ensemble_aligned = ensemble_aligned.compute()
            
            logger.info(f"数据准备完成，开始应用掩膜...")
            logger.info(f"最终数据维度: obs={obs_aligned.shape}, fcst={fcst_aligned.shape}, "
                       f"ensemble={ensemble_aligned.shape if ensemble_aligned is not None else 'N/A'}")
            
            # 在计算前应用陆地掩膜，将海洋区域设为NaN，这样计算时会自动跳过海洋区域
            logger.info("应用陆地掩膜到观测数据...")
            land_mask = self._create_land_mask(var_type, obs_aligned)
            if land_mask is not None:
                # 对观测数据应用掩膜：海洋区域设为NaN
                obs_aligned = obs_aligned.where(land_mask.reindex_like(obs_aligned, method='nearest', tolerance=0.5))
                logger.info(f"观测数据掩膜应用完成，有效格点数: {np.sum(~np.isnan(obs_aligned.isel(time=0).values))} / {obs_aligned.isel(time=0).size}")
                
                # 对预报数据应用掩膜
                if fcst_aligned is not None:
                    fcst_aligned = fcst_aligned.where(land_mask.reindex_like(fcst_aligned, method='nearest', tolerance=0.5))
                    logger.info(f"预报数据掩膜应用完成")
                
                # 对ensemble数据应用掩膜
                if ensemble_aligned is not None:
                    ensemble_aligned = ensemble_aligned.where(land_mask.reindex_like(ensemble_aligned, method='nearest', tolerance=0.5))
                    logger.info(f"Ensemble数据掩膜应用完成")
                
                # 对观测气候态应用掩膜
                if obs_climatology is not None:
                    obs_climatology = obs_climatology.where(land_mask.reindex_like(obs_climatology, method='nearest', tolerance=0.5))
                    logger.info(f"观测气候态掩膜应用完成")
            else:
                logger.warning("无法创建陆地掩膜，将使用数据本身的NaN作为掩膜")
            
            # 计算气候态命中率（用于BSS参考预报）
            # 使用观测气候态作为参考（在掩膜应用后计算，确保排除海洋区域）
            climatology_hit_rates = None
            if ensemble_aligned is not None:
                logger.info("计算气候态命中率（基于观测气候态，已应用掩膜）...")
                var_config = VAR_CONFIG.get(var_type, {})
                var_accuracy_delta = var_config.get('accuracy_delta', self.accuracy_delta)
                var_accuracy_eps = var_config.get('accuracy_eps', self.accuracy_eps)
                
                climatology_hit_rates = self._compute_climatology_hit_rate(
                    obs_aligned, obs_climatology, var_accuracy_delta, var_accuracy_eps
                )
                logger.info(f"气候态命中率计算完成，范围: [{np.nanmin(climatology_hit_rates):.4f}, {np.nanmax(climatology_hit_rates):.4f}]")
            
            # 获取变量特定的accuracy参数
            var_config = VAR_CONFIG.get(var_type, {})
            var_accuracy_delta = var_config.get('accuracy_delta', self.accuracy_delta)
            var_accuracy_eps = var_config.get('accuracy_eps', self.accuracy_eps)
            
            logger.info(f"使用变量特定参数: accuracy_delta={var_accuracy_delta}, accuracy_eps={var_accuracy_eps}")
            
            # 使用独立 block bootstrap 方法
            result_ds = self.process_gridded_data_independent_blocks(
                fcst_aligned, obs_aligned, obs_climatology, ensemble_aligned, climatology_hit_rates,
                var_accuracy_delta, var_accuracy_eps
            )
            
            result_ds.attrs.update({
                'model': model,
                'leadtime': leadtime,
                'variable': var_type,
                'creation_time': datetime.now().isoformat()
            })
            
            output_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
            result_ds.to_netcdf(output_file)
            logger.info(f"结果已保存到: {output_file}")
            
            # 保存返回值
            return_value = result_ds
            
            # 显式清理内存
            del obs_data, fcst_data, obs_aligned, fcst_aligned, result_ds
            if ensemble_data is not None:
                del ensemble_data
            if ensemble_aligned is not None:
                del ensemble_aligned
            if climatology_hit_rates is not None:
                del climatology_hit_rates
            gc.collect()
            
            return return_value
            
        except Exception as e:
            logger.error(f"处理 {model} L{leadtime} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def normalize_and_update_scores(self, all_results: Dict[str, xr.Dataset]):
        """使用每个模型自己的 RMSE 范围进行归一化并更新得分"""
        logger.info(f"使用每个模型自己的 RMSE 范围进行归一化")
        
        for key, ds in all_results.items():
            # 获取原始数据
            b_accuracy = ds['b_accuracy'].values
            b_equitable_accuracy = ds['b_equitable_accuracy'].values
            b_rmse = ds['b_rmse'].values
            b_pcc = ds['b_pcc'].values
            
            # 检查数据有效性
            valid_mask = ~(np.isnan(b_accuracy) | np.isnan(b_rmse) | np.isnan(b_pcc))
            logger.info(f"{key}: 有效格点数 {np.sum(valid_mask)} / {b_accuracy.size}")
            
            # 使用该模型自己的RMSE范围进行归一化
            local_rmse_min = np.nanmin(b_rmse)
            local_rmse_max = np.nanmax(b_rmse)
            logger.info(f"{key}: 使用模型自己的RMSE范围 [{local_rmse_min:.4f}, {local_rmse_max:.4f}]")
            s_rmse = normalize_rmse(b_rmse, local_rmse_min, local_rmse_max)
            
            # 更新s_rmse
            ds['s_rmse'].values = s_rmse
            
            # 计算综合得分（使用equitable accuracy）
            # 将equitable accuracy从[-1,1]映射到[0,1]
            s_equitable_accuracy_normalized = (b_equitable_accuracy + 1) / 2
            s_equitable_accuracy_normalized = np.clip(s_equitable_accuracy_normalized, 0.0, 1.0)
            ds['s_equitable_accuracy'].values = s_equitable_accuracy_normalized
            
            s_pcc = ds['s_pcc'].values  # 获取标准化的pcc
            
            # 主要得分：使用equitable accuracy
            score = compute_score(s_equitable_accuracy_normalized, s_rmse, s_pcc)
            ds['score'].values = score
            
            # 备用得分：基于raw accuracy（用于对比）
            s_accuracy = ds['s_accuracy'].values  # 获取标准化的raw accuracy
            score_raw = compute_score(s_accuracy, s_rmse, s_pcc)
            ds['score_raw'] = (['lat', 'lon'], score_raw)  # 添加新变量
            
            # 检查得分计算的有效性
            valid_score_mask = ~np.isnan(score)
            logger.info(f"{key}: 有效得分格点数 {np.sum(valid_score_mask)} / {score.size}")
            if np.sum(valid_score_mask) > 0:
                logger.info(f"{key}: 得分范围 (使用equitable accuracy) [{np.nanmin(score):.4f}, {np.nanmax(score):.4f}]")
            
            # 检查raw得分
            valid_raw_score_mask = ~np.isnan(score_raw)
            if np.sum(valid_raw_score_mask) > 0:
                logger.info(f"{key}: Raw得分范围 [{np.nanmin(score_raw):.4f}, {np.nanmax(score_raw):.4f}]")
            
            # 更新属性（记录使用的是模型自己的范围）
            ds.attrs['model_rmse_min'] = float(local_rmse_min)
            ds.attrs['model_rmse_max'] = float(local_rmse_max)
            ds.attrs['rmse_normalization'] = 'model_specific'
            
            # 保存文件
            model = ds.attrs['model']
            leadtime = ds.attrs['leadtime']
            var_type = ds.attrs['variable']
            output_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
            ds.to_netcdf(output_file)
            
            logger.info(f"更新了 {key} 的归一化 RMSE 和综合得分")

    def run_analysis(self, models: List[str], leadtimes: List[int], 
                    var_type: str = 'temp', plot_figures: bool = True):
        """运行完整的分析流程"""
        logger.info(f"开始 Block Bootstrap 评分分析 - 独立 block 版本")
        logger.info(f"模型: {models}")
        logger.info(f"提前期: {leadtimes}")
        logger.info(f"变量: {var_type}")
        logger.info(f"内存限制: {MAX_MEMORY_GB}GB")
        
        # 记录初始内存使用
        log_memory_usage("分析开始")
        
        all_results = {}
        all_regional_results = {}
        
        # 第一阶段：计算所有模型的原始指标
        logger.info("第一阶段：计算所有模型的原始指标...")
        
        for model in models:
            for leadtime in leadtimes:
                logger.info(f"处理 {model} L{leadtime}")
                
                # 检查内存限制
                if not check_memory_limit():
                    logger.error(f"内存使用量过高，跳过 {model} L{leadtime}")
                    continue
                
                ds = self.process_single_model_leadtime(model, leadtime, var_type)
                if ds is not None:
                    key = f"{model}_L{leadtime}"
                    all_results[key] = ds
                    log_memory_usage(f"完成 {model} L{leadtime}")
                else:
                    logger.warning(f"处理失败 {model} L{leadtime}")
                
                # 强制垃圾回收
                force_garbage_collection()
        
        logger.info(f"第一阶段完成: {len(all_results)} 个任务成功")
        log_memory_usage("第一阶段完成")
        
        # 第二阶段：使用全局 RMSE 范围进行归一化
        logger.info("第二阶段：统一归一化 RMSE 并更新得分...")
        if all_results:
            self.normalize_and_update_scores(all_results)
        else:
            logger.warning("all_results为空，跳过归一化步骤")
        
        # 第三阶段：绘图和计算区域平均
        if plot_figures and all_results:
            logger.info("第三阶段：生成图表...")
            # 基于已计算结果直接绘制已实现的综合图（仅lead0/lead3）
            try:
                lead_subset = [lt for lt in leadtimes if lt in (0, 3)]
                if lead_subset:
                    self.plot_score_spatial_distribution(lead_subset, var_type, models)
                    self.plot_score_cdf_distribution(lead_subset, var_type, models)
                    self.plot_score_bar(lead_subset, var_type, models)
                    self.plot_bss_scatter(lead_subset, var_type, models)
                    self.plot_bss_vs_accuracy_scatter(lead_subset, var_type, models)
                else:
                    logger.warning("未找到lead0或lead3，空间分布/散点图跳过")
            except Exception as exc:
                logger.warning(f"批量绘图时出错，继续计算区域平均: {exc}")
            for key, ds in all_results.items():
                model = ds.attrs['model']
                leadtime = ds.attrs['leadtime']
                
                # 绘制空间分布图（只绘制综合得分）
                # 不再绘制单张空间分布图，统一使用多模型组合图
                # self.plot_spatial_distribution(ds, 'score', model, leadtime, var_type)
                
                # 绘制泰勒图
                # self.plot_taylor_diagram(ds, model, leadtime, var_type)  # 已由plot_taylor_diagram_all_models替代
                
                # 计算区域平均
                regional_avgs = self.calculate_regional_averages(ds)
                all_regional_results[key] = regional_avgs
            
            # 绘制所有模型的lead time得分折线图
            self.plot_leadtime_score_lines(var_type, models, leadtimes)
            
            # 保存汇总结果
            summary_file = self.summary_dir / f"regional_averages_{var_type}.json"
            with open(summary_file, 'w') as f:
                json.dump(all_regional_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析完成！")
        logger.info(f"  NetCDF结果文件保存在: {self.results_dir}")
        logger.info(f"  图像文件保存在: {self.plots_dir}")
        logger.info(f"  汇总文件保存在: {self.summary_dir}")
    
    def plot_only_mode(self, models: List[str], leadtimes: List[int], var_types: List[str]):
        """仅绘图模式：基于已有的NetCDF结果文件生成图像"""
        logger.info(f"进入仅绘图模式")
        logger.info(f"模型: {models}")
        logger.info(f"提前期: {leadtimes}")
        logger.info(f"变量: {var_types}")
        
        all_regional_results = {}
        
        for var_type in var_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理变量: {var_type.upper()}")
            logger.info(f"{'='*60}")
            
            var_regional_results = {}
            
            # 为每个leadtime绘制所有模型的accuracy对比图、空间分布图和全国平均得分柱状图
            for leadtime in leadtimes:
                self.plot_accuracy_comparison(leadtime, var_type, models)
                self.plot_spatial_distribution_all_models(leadtime, var_type, models, 'score')
                self.plot_national_average_score_bar(leadtime, var_type, models)
                # 绘制组合图：空间分布+时间序列（已注释，改为使用拆分后的独立函数）
                # self.plot_combined_spatial_timeseries(leadtime, var_type, models, 'score')
            
            # 绘制BSS相关图像（仅lead0和lead3，如果数据存在）
            # 先检查哪些leadtime有BSS数据
            target_leadtimes_candidate = [0, 3]
            available_bss_leadtimes = []
            for lt in target_leadtimes_candidate:
                if lt in leadtimes:
                    # 检查是否有任何模型有该leadtime的BSS数据
                    has_data = False
                    for model in models:
                        nc_file = self.results_dir / f"bootstrap_score_{model}_L{lt}_{var_type}.nc"
                        if nc_file.exists():
                            try:
                                ds = xr.open_dataset(nc_file)
                                if 'b_bss' in ds:
                                    bss_data = ds['b_bss']
                                    if np.isfinite(bss_data.values).any():
                                        has_data = True
                                ds.close()
                                if has_data:
                                    break
                            except:
                                pass
                    if has_data:
                        available_bss_leadtimes.append(lt)
            
            # 只要有至少一个leadtime有数据，就绘制
            if available_bss_leadtimes:
                self.plot_bss_scatter(available_bss_leadtimes, var_type, models)
                self.plot_bss_vs_accuracy_scatter(available_bss_leadtimes, var_type, models)
            else:
                logger.warning(f"没有可用的BSS数据用于绘制散点图 (var_type={var_type})")
            
            # 绘制Score相关图像（仅lead0和lead3，如果数据存在）
            target_leadtimes_score = [0, 3]
            available_score_leadtimes = []
            for lt in target_leadtimes_score:
                if lt in leadtimes:
                    # 检查是否有任何模型有该leadtime的Score数据
                    has_data = False
                    for model in models:
                        nc_file = self.results_dir / f"bootstrap_score_{model}_L{lt}_{var_type}.nc"
                        if nc_file.exists():
                            try:
                                ds = xr.open_dataset(nc_file)
                                if 'score' in ds:
                                    score_data = ds['score']
                                    if np.isfinite(score_data.values).any():
                                        has_data = True
                                ds.close()
                                if has_data:
                                    break
                            except:
                                pass
                    if has_data:
                        available_score_leadtimes.append(lt)
            
            # 只要有至少一个leadtime有数据，就绘制
            if available_score_leadtimes:
                self.plot_score_spatial_distribution(available_score_leadtimes, var_type, models)
                self.plot_score_cdf_distribution(available_score_leadtimes, var_type, models)
                self.plot_score_bar(available_score_leadtimes, var_type, models)
            else:
                logger.warning(f"没有可用的Score数据用于绘制图表 (var_type={var_type})")

            # 额外：对lead0/lead3尝试批量绘制（有数据才会生效）
            try:
                lead_subset = [lt for lt in leadtimes if lt in (0, 3)]
                if lead_subset:
                    self.plot_score_spatial_distribution(lead_subset, var_type, models)
                    self.plot_score_cdf_distribution(lead_subset, var_type, models)
                    # self.plot_score_bar(lead_subset, var_type, models)
                    self.plot_bss_scatter(lead_subset, var_type, models)
                    self.plot_bss_vs_accuracy_scatter(lead_subset, var_type, models)
                else:
                    logger.warning("未找到lead0或lead3，额外批量绘制跳过")
            except Exception as exc:
                logger.warning(f"批量绘制lead0/lead3时出错，已跳过: {exc}")
            
            # 绘制所有模型的lead time得分折线图（使用所有leadtimes）
            self.plot_leadtime_score_lines(var_type, models, leadtimes)
            
            for model in models:
                for leadtime in leadtimes:
                    # 查找对应的NetCDF文件
                    nc_file = self.results_dir / f"bootstrap_score_{model}_L{leadtime}_{var_type}.nc"
                    
                    if not nc_file.exists():
                        logger.warning(f"结果文件不存在，跳过: {nc_file}")
                        continue
                    
                    try:
                        logger.info(f"加载结果文件: {model} L{leadtime} {var_type}")
                        
                        # 加载NetCDF文件
                        ds = xr.open_dataset(nc_file)
                        
                        
                        # 计算区域平均
                        regional_avgs = self.calculate_regional_averages(ds)
                        var_regional_results[f"{model}_L{leadtime}"] = regional_avgs
                        
                        logger.info(f"完成绘图: {model} L{leadtime} {var_type}")
                        
                        # 关闭数据集
                        ds.close()
                        
                    except Exception as e:
                        logger.error(f"处理文件 {nc_file} 时出错: {e}")
                        continue
            
            # 保存该变量的汇总结果
            if var_regional_results:
                summary_file = self.summary_dir / f"regional_averages_{var_type}.json"
                with open(summary_file, 'w') as f:
                    json.dump(var_regional_results, f, indent=2, ensure_ascii=False)
                logger.info(f"区域平均结果已保存到: {summary_file}")
            
            all_regional_results.update(var_regional_results)
        
        logger.info(f"\n{'='*60}")
        logger.info("仅绘图模式完成！")
        logger.info(f"  NetCDF结果文件来源: {self.results_dir}")
        logger.info(f"  图像文件保存在: {self.plots_dir}")
        logger.info(f"  汇总文件保存在: {self.summary_dir}")
        logger.info(f"{'='*60}")


def _process_model_leadtime_task(task):
    """处理单个模型/leadtime任务的函数"""
    model, leadtime, var_type, config = task
    try:
        logger.info(f"开始处理 {model} L{leadtime} {var_type}")
        
        # 创建新的分析器实例
        analyzer = UnifiedBlockBootstrapAnalyzer(
            block_size=config['block_size'],
            n_bootstrap=config['n_bootstrap'],
            confidence_level=config['confidence_level'],
            n_jobs=max(1, config['n_jobs'] // 4),  # 子进程使用1/4的并行度
            parallel_backend=config['parallel_backend'],
            parallel_strategy=config.get('parallel_strategy', 'auto'),
            accuracy_delta=config['accuracy_delta'],
            accuracy_eps=config['accuracy_eps']
        )
        
        ds = analyzer.process_single_model_leadtime(model, leadtime, var_type)
        if ds is not None:
            logger.info(f"完成处理 {model} L{leadtime} {var_type}")
            return f"{model}_L{leadtime}", ds
        else:
            logger.warning(f"处理失败 {model} L{leadtime} {var_type}")
            return None
    except Exception as e:
        logger.error(f"处理 {model} L{leadtime} 时出错: {e}")
        return None


# ===== 命令行参数解析 =====
def parse_args():
    """解析命令行参数"""
    parser = create_parser(
        description='运行 Block Bootstrap 评分分析',
        include_bootstrap=True,
        include_parallel_advanced=True,
        include_gpu=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # block_bootstrap_score 特有的 n_jobs 默认值
    parser.set_defaults(n_jobs=64)
    
    return parser.parse_args()


# ===== 主函数 =====
def main():
    """主函数"""
    args = parse_args()
    
    # 解析参数
    models = parse_models(args.models, list(MODELS.keys())) if args.models else list(MODELS.keys())[:8]
    leadtimes = parse_leadtimes(args.leadtimes, LEADTIMES) if args.leadtimes else LEADTIMES
    var_types = parse_vars(args.var) if args.var else ['temp', 'prec']
    
    # 标准化绘图参数
    normalize_plot_args(args)
    
    # 记录初始内存状态
    log_memory_usage("程序启动")
    
    # GPU配置处理
    global GPU_AVAILABLE
    if args.no_gpu:
        GPU_AVAILABLE = False
        logger.info("用户强制禁用GPU，将使用CPU计算")
    elif args.use_gpu and not GPU_AVAILABLE:
        logger.warning("用户请求GPU加速但GPU不可用，将使用CPU计算")
    
    # GPU内存限制设置已移除（GPU支持已禁用）
    if GPU_AVAILABLE:
        logger.warning("GPU支持已禁用，跳过GPU内存限制设置")
    
    logger.info(f"分析配置：")
    logger.info(f"  模型: {models}")
    logger.info(f"  提前期: {leadtimes}")
    logger.info(f"  变量: {var_types}")
    logger.info(f"  Block 大小: {args.block_size}")
    logger.info(f"  Bootstrap 次数: {args.n_bootstrap}")
    logger.info(f"  置信水平: {args.confidence}")
    logger.info(f"  并行作业数: {args.n_jobs}")
    logger.info(f"  内存限制: {MAX_MEMORY_GB}GB")
    logger.info(f"  Accuracy 相对误差阈值: {args.accuracy_delta} (±{args.accuracy_delta*100:.0f}%)")
    logger.info(f"  Accuracy eps: {args.accuracy_eps}")
    logger.info(f"  GPU加速: {'启用' if GPU_AVAILABLE else '禁用'}")
    logger.info(f"  仅绘图模式: {args.plot_only}")
    
    # 创建分析器
    analyzer = UnifiedBlockBootstrapAnalyzer(
        block_size=args.block_size,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        parallel_strategy=args.parallel_strategy,
        accuracy_delta=args.accuracy_delta,
        accuracy_eps=args.accuracy_eps
    )
    
    # 根据模式选择运行方式
    if args.plot_only:
        # 仅绘图模式
        analyzer.plot_only_mode(
            models=models,
            leadtimes=leadtimes,
            var_types=var_types
        )
    else:
        # 正常计算模式
        for var_type in var_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始分析变量: {var_type.upper()}")
            logger.info(f"{'='*60}")
            
            analyzer.run_analysis(
                models=models,
                leadtimes=leadtimes,
                var_type=var_type,
                plot_figures=not args.no_plot
            )
            
            logger.info(f"变量 {var_type.upper()} 分析完成！")
            
            # 强制垃圾回收以释放内存
            force_garbage_collection()
    
    logger.info(f"\n{'='*60}")
    logger.info("所有变量分析完成！")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
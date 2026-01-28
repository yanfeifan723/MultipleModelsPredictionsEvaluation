"""
数据验证工具函数
提供数据质量检查、异常值检测等功能
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def check_data_quality(data: xr.DataArray, 
                      data_type: str = "unknown") -> Dict[str, Any]:
    """
    检查数据质量
    
    Args:
        data: 输入数据
        data_type: 数据类型标识
        
    Returns:
        质量检查结果字典
    """
    result = {
        "data_type": data_type,
        "shape": data.shape,
        "dims": list(data.dims),
        "total_points": data.size,
        "valid_points": np.sum(~np.isnan(data.values)),
        "nan_ratio": np.isnan(data.values).mean(),
        "finite_ratio": np.isfinite(data.values).mean(),
        "has_infinite": np.isinf(data.values).any(),
        "has_nan": np.isnan(data.values).any(),
        "min_value": float(np.nanmin(data.values)),
        "max_value": float(np.nanmax(data.values)),
        "mean_value": float(np.nanmean(data.values)),
        "std_value": float(np.nanstd(data.values)),
        "median_value": float(np.nanmedian(data.values))
    }
    
    # 检查异常值
    outliers = detect_outliers(data)
    result["outlier_count"] = len(outliers)
    result["outlier_ratio"] = len(outliers) / result["valid_points"] if result["valid_points"] > 0 else 0
    
    # 质量评估
    result["quality_score"] = calculate_quality_score(result)
    result["quality_level"] = assess_quality_level(result["quality_score"])
    
    logger.info(f"数据质量检查完成 - {data_type}: 质量分数 {result['quality_score']:.2f} ({result['quality_level']})")
    
    return result

def detect_outliers(data: xr.DataArray, 
                   method: str = "iqr",
                   threshold: float = 1.5) -> List[Tuple[int, ...]]:
    """
    检测异常值
    
    Args:
        data: 输入数据
        method: 检测方法 ('iqr', 'zscore', 'mad')
        threshold: 阈值
        
    Returns:
        异常值索引列表
    """
    values = data.values.flatten()
    valid_mask = ~np.isnan(values)
    
    if not valid_mask.any():
        return []
    
    valid_values = values[valid_mask]
    
    if method == "iqr":
        # 四分位距方法
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
        
    elif method == "zscore":
        # Z分数方法
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        z_scores = np.abs((valid_values - mean_val) / std_val)
        outlier_mask = z_scores > threshold
        
    elif method == "mad":
        # 中位数绝对偏差方法
        median_val = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median_val))
        modified_z_scores = 0.6745 * (valid_values - median_val) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
    else:
        raise ValueError(f"不支持的异常值检测方法: {method}")
    
    # 获取异常值在原始数组中的索引
    outlier_indices = np.where(valid_mask)[0][outlier_mask]
    outlier_positions = np.unravel_index(outlier_indices, data.shape)
    
    return list(zip(*outlier_positions))

def validate_spatial_bounds(data: xr.DataArray,
                           expected_bounds: Dict[str, Tuple[float, float]],
                           tolerance: float = 0.1) -> bool:
    """
    验证空间边界
    
    Args:
        data: 输入数据
        expected_bounds: 期望的边界 {'lat': (min, max), 'lon': (min, max)}
        tolerance: 容差
        
    Returns:
        边界是否有效
    """
    for coord, (min_val, max_val) in expected_bounds.items():
        if coord not in data.coords:
            logger.warning(f"数据缺少坐标: {coord}")
            return False
        
        coord_values = data[coord].values
        actual_min = float(coord_values.min())
        actual_max = float(coord_values.max())
        
        if abs(actual_min - min_val) > tolerance or abs(actual_max - max_val) > tolerance:
            logger.warning(f"坐标 {coord} 边界不匹配: 期望 ({min_val}, {max_val}), "
                          f"实际 ({actual_min:.2f}, {actual_max:.2f})")
            return False
    
    return True

def check_temporal_consistency(data: xr.DataArray) -> Dict[str, Any]:
    """
    检查时间一致性
    
    Args:
        data: 输入数据
        
    Returns:
        时间一致性检查结果
    """
    if 'time' not in data.dims:
        return {"has_time": False}
    
    time_coord = data.time
    time_values = pd.to_datetime(time_coord.values)
    
    result = {
        "has_time": True,
        "time_points": len(time_values),
        "start_time": time_values[0].isoformat(),
        "end_time": time_values[-1].isoformat(),
        "time_span_days": (time_values[-1] - time_values[0]).days,
        "is_monotonic": time_values.is_monotonic_increasing,
        "has_duplicates": time_values.duplicated().any(),
        "time_frequency": infer_time_frequency(time_values)
    }
    
    # 检查时间间隔一致性
    if len(time_values) > 1:
        time_diffs = np.diff(time_values)
        result["mean_interval_hours"] = np.mean(time_diffs).total_seconds() / 3600
        result["std_interval_hours"] = np.std(time_diffs).total_seconds() / 3600
        result["is_regular"] = result["std_interval_hours"] < 1.0  # 1小时内的标准差认为规则
    
    return result

def infer_time_frequency(time_values: pd.DatetimeIndex) -> str:
    """
    推断时间频率
    
    Args:
        time_values: 时间值
        
    Returns:
        时间频率字符串
    """
    if len(time_values) < 2:
        return "unknown"
    
    # 计算时间间隔
    intervals = np.diff(time_values)
    median_interval = np.median(intervals)
    
    # 转换为小时
    hours = median_interval.total_seconds() / 3600
    
    if hours < 1:
        return "subhourly"
    elif hours == 1:
        return "hourly"
    elif hours == 6:
        return "6hourly"
    elif hours == 12:
        return "12hourly"
    elif hours == 24:
        return "daily"
    elif hours == 24 * 7:
        return "weekly"
    elif hours == 24 * 30:
        return "monthly"
    elif hours == 24 * 365:
        return "yearly"
    else:
        return f"{hours:.1f}hourly"

def calculate_quality_score(quality_dict: Dict[str, Any]) -> float:
    """
    计算数据质量分数
    
    Args:
        quality_dict: 质量检查结果字典
        
    Returns:
        质量分数 (0-1)
    """
    score = 1.0
    
    # 有效数据比例
    valid_ratio = quality_dict["valid_points"] / quality_dict["total_points"]
    score *= valid_ratio
    
    # 异常值比例惩罚
    outlier_ratio = quality_dict["outlier_ratio"]
    score *= (1 - outlier_ratio)
    
    # 无限值惩罚
    if quality_dict["has_infinite"]:
        score *= 0.5
    
    # 数据范围合理性检查
    if quality_dict["std_value"] == 0:
        score *= 0.1  # 常数数据
    
    return max(0.0, min(1.0, score))

def assess_quality_level(score: float) -> str:
    """
    评估质量等级
    
    Args:
        score: 质量分数
        
    Returns:
        质量等级
    """
    if score >= 0.9:
        return "excellent"
    elif score >= 0.7:
        return "good"
    elif score >= 0.5:
        return "fair"
    elif score >= 0.3:
        return "poor"
    else:
        return "very_poor"

def validate_variable_names(data: xr.Dataset,
                           expected_vars: List[str]) -> Dict[str, bool]:
    """
    验证变量名
    
    Args:
        data: 数据集
        expected_vars: 期望的变量名列表
        
    Returns:
        变量存在性字典
    """
    result = {}
    for var in expected_vars:
        result[var] = var in data.data_vars
    
    missing_vars = [var for var, exists in result.items() if not exists]
    if missing_vars:
        logger.warning(f"缺少变量: {missing_vars}")
    
    return result

def check_data_completeness(data: xr.DataArray,
                           min_completeness: float = 0.8) -> bool:
    """
    检查数据完整性
    
    Args:
        data: 输入数据
        min_completeness: 最小完整性要求
        
    Returns:
        是否满足完整性要求
    """
    completeness = 1.0 - np.isnan(data.values).mean()
    
    if completeness < min_completeness:
        logger.warning(f"数据完整性不足: {completeness:.1%} < {min_completeness:.1%}")
        return False
    
    return True

def validate_units(data: xr.DataArray,
                  expected_unit: str,
                  unit_attr: str = "units") -> bool:
    """
    验证数据单位
    
    Args:
        data: 输入数据
        expected_unit: 期望的单位
        unit_attr: 单位属性名
        
    Returns:
        单位是否匹配
    """
    if unit_attr not in data.attrs:
        logger.warning(f"数据缺少单位属性: {unit_attr}")
        return False
    
    actual_unit = data.attrs[unit_attr]
    
    if actual_unit != expected_unit:
        logger.warning(f"单位不匹配: 期望 {expected_unit}, 实际 {actual_unit}")
        return False
    
    return True

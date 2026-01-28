#!/usr/bin/env python3
"""
异常值检测和处理模块

提供多种异常值检测方法，包括：
- IQR方法（四分位距法）
- Z-score方法
- 修正Z-score方法
- 隔离森林方法
- 局部异常因子（LOF）方法

支持按不同维度分组进行异常值检测，适用于气候数据分析。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import warnings

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy未安装，部分异常值检测方法可能不可用")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn未安装，机器学习异常值检测方法不可用")


logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    异常值检测器类
    
    提供多种异常值检测方法，支持分组检测和批量处理。
    """
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, 
                 group_by: Optional[List[str]] = None, min_samples: int = 10):
        """
        初始化异常值检测器
        
        Args:
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof')
            threshold: 检测阈值
            group_by: 分组列名列表
            min_samples: 最小样本数（少于此数量的组将被跳过）
        """
        self.method = method.lower()
        self.threshold = threshold
        self.group_by = group_by
        self.min_samples = min_samples
        
        # 验证方法
        valid_methods = ['iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof']
        if self.method not in valid_methods:
            raise ValueError(f"不支持的检测方法: {method}。支持的方法: {valid_methods}")
        
        # 检查依赖
        if self.method in ['isolation_forest', 'lof'] and not SKLEARN_AVAILABLE:
            raise ImportError(f"使用{method}方法需要安装sklearn")
        
        if self.method in ['zscore', 'modified_zscore'] and not SCIPY_AVAILABLE:
            raise ImportError(f"使用{method}方法需要安装scipy")
    
    def detect_outliers_iqr(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        使用IQR方法检测异常值
        
        Args:
            data: 数值数组
            threshold: IQR倍数阈值，默认使用实例的threshold
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if threshold is None:
            threshold = self.threshold
            
        if len(data) < 4:
            return np.zeros(len(data), dtype=bool)
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return np.zeros(len(data), dtype=bool)
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return outlier_mask
    
    def detect_outliers_zscore(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        使用Z-score方法检测异常值
        
        Args:
            data: 数值数组
            threshold: Z-score阈值，默认使用实例的threshold
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if threshold is None:
            threshold = self.threshold
            
        if len(data) < 3:
            return np.zeros(len(data), dtype=bool)
        
        try:
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            return outlier_mask
        except Exception as e:
            logger.warning(f"Z-score计算失败: {e}")
            return np.zeros(len(data), dtype=bool)
    
    def detect_outliers_modified_zscore(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        使用修正Z-score方法检测异常值（对异常值更鲁棒）
        
        Args:
            data: 数值数组
            threshold: 修正Z-score阈值，默认使用实例的threshold
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if threshold is None:
            threshold = self.threshold
            
        if len(data) < 3:
            return np.zeros(len(data), dtype=bool)
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        return outlier_mask
    
    def detect_outliers_isolation_forest(self, data: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        使用隔离森林方法检测异常值
        
        Args:
            data: 数值数组
            contamination: 预期的异常值比例
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn未安装，无法使用隔离森林方法")
        
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        # 自适应污染率
        contamination = min(0.1, max(0.01, 5.0 / len(data)))
        
        try:
            # 重塑数据为2D数组
            X = data.reshape(-1, 1)
            
            # 创建隔离森林模型
            iso_forest = IsolationForest(
                contamination=contamination, 
                random_state=42,
                n_estimators=100
            )
            predictions = iso_forest.fit_predict(X)
            
            # -1表示异常值，1表示正常值
            outlier_mask = predictions == -1
            return outlier_mask
            
        except Exception as e:
            logger.warning(f"隔离森林检测失败: {e}")
            return np.zeros(len(data), dtype=bool)
    
    def detect_outliers_lof(self, data: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        使用局部异常因子（LOF）方法检测异常值
        
        Args:
            data: 数值数组
            contamination: 预期的异常值比例
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn未安装，无法使用LOF方法")
        
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        # 自适应污染率
        contamination = min(0.1, max(0.01, 5.0 / len(data)))
        
        try:
            # 重塑数据为2D数组
            X = data.reshape(-1, 1)
            
            # 创建LOF模型
            lof = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(20, len(data) // 2),
                metric='euclidean'
            )
            predictions = lof.fit_predict(X)
            
            # -1表示异常值，1表示正常值
            outlier_mask = predictions == -1
            return outlier_mask
            
        except Exception as e:
            logger.warning(f"LOF检测失败: {e}")
            return np.zeros(len(data), dtype=bool)
    
    def detect_outliers(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        根据指定的方法检测异常值
        
        Args:
            data: 数值数组
            **kwargs: 方法特定的参数
        
        Returns:
            outlier_mask: 布尔数组，True表示异常值
        """
        if self.method == 'iqr':
            return self.detect_outliers_iqr(data, kwargs.get('threshold', self.threshold))
        elif self.method == 'zscore':
            return self.detect_outliers_zscore(data, kwargs.get('threshold', self.threshold))
        elif self.method == 'modified_zscore':
            return self.detect_outliers_modified_zscore(data, kwargs.get('threshold', self.threshold))
        elif self.method == 'isolation_forest':
            return self.detect_outliers_isolation_forest(data, kwargs.get('contamination', 0.1))
        elif self.method == 'lof':
            return self.detect_outliers_lof(data, kwargs.get('contamination', 0.1))
        else:
            raise ValueError(f"不支持的检测方法: {self.method}")
    
    def process_dataframe(self, df: pd.DataFrame, value_column: str = 'CRPSS',
                         save_outliers: bool = False, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        处理DataFrame中的异常值
        
        Args:
            df: 包含数据的DataFrame
            value_column: 要检测异常值的列名
            save_outliers: 是否保存异常值
            output_dir: 输出目录
        
        Returns:
            cleaned_df: 清理后的DataFrame
            outliers_df: 异常值DataFrame（如果save_outliers=True）
        """
        if df.empty or value_column not in df.columns:
            return df, None
        
        # 移除缺失值
        valid_data = df.dropna(subset=[value_column])
        
        if valid_data.empty:
            return df, None
        
        outliers_mask = np.zeros(len(valid_data), dtype=bool)
        
        if self.group_by and all(col in valid_data.columns for col in self.group_by):
            # 按指定维度分组检测异常值
            group_stats = {}
            has_valid_groups = False
            
            for name, group in valid_data.groupby(self.group_by):
                if len(group) < self.min_samples:
                    logger.debug(f"组 {name} 样本数不足 ({len(group)} < {self.min_samples})，跳过")
                    continue
                
                has_valid_groups = True
                group_values = group[value_column].values
                group_outliers = self.detect_outliers(group_values)
                
                # 记录组统计信息
                group_stats[name] = {
                    'total': len(group),
                    'outliers': np.sum(group_outliers),
                    'outlier_rate': np.sum(group_outliers) / len(group)
                }
                
                # 将组内异常值标记到全局索引
                group_indices = group.index
                outliers_mask[valid_data.index.get_indexer(group_indices)] = group_outliers
            
            # 如果没有有效组，返回原始数据
            if not has_valid_groups:
                logger.warning(f"所有组的样本数都少于 {self.min_samples}，返回原始数据")
                return valid_data, None
            
            # 记录分组统计
            logger.info(f"分组异常值检测统计:")
            for name, stats in group_stats.items():
                logger.info(f"  {name}: {stats['outliers']}/{stats['total']} ({stats['outlier_rate']:.2%})")
        else:
            # 全局检测异常值
            if len(valid_data) < self.min_samples:
                logger.warning(f"样本数不足 ({len(valid_data)} < {self.min_samples})，返回原始数据")
                return valid_data, None
            
            values = valid_data[value_column].values
            outliers_mask = self.detect_outliers(values)
            
            outlier_count = np.sum(outliers_mask)
            outlier_rate = outlier_count / len(values)
            logger.info(f"全局异常值检测: {outlier_count}/{len(values)} ({outlier_rate:.2%})")
        
        # 分离正常值和异常值
        outliers_df = valid_data[outliers_mask].copy()
        cleaned_df = valid_data[~outliers_mask].copy()
        
        # 保存异常值
        if save_outliers and not outliers_df.empty and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            outliers_file = output_dir / f"outliers_{self.method}_{self.threshold}.csv"
            outliers_df.to_csv(outliers_file, index=False, encoding='utf-8')
            logger.info(f"异常值已保存到: {outliers_file}")
        
        return cleaned_df, outliers_df


def remove_outliers_from_dataframe(df: pd.DataFrame, method: str = 'iqr', 
                                  threshold: float = 1.5, group_by: Optional[List[str]] = None,
                                  value_column: str = 'CRPSS', min_samples: int = 10,
                                  save_outliers: bool = False, output_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    从DataFrame中移除异常值的便捷函数
    
    Args:
        df: 包含数据的DataFrame
        method: 异常值检测方法
        threshold: 检测阈值
        group_by: 分组列名列表
        value_column: 要检测异常值的列名
        min_samples: 最小样本数
        save_outliers: 是否保存异常值
        output_dir: 输出目录
    
    Returns:
        cleaned_df: 清理后的DataFrame
        outliers_df: 异常值DataFrame（如果save_outliers=True）
    """
    detector = OutlierDetector(
        method=method,
        threshold=threshold,
        group_by=group_by,
        min_samples=min_samples
    )
    
    return detector.process_dataframe(
        df=df,
        value_column=value_column,
        save_outliers=save_outliers,
        output_dir=output_dir
    )


def get_outlier_statistics(df: pd.DataFrame, value_column: str = 'CRPSS', 
                          group_by: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    获取异常值统计信息
    
    Args:
        df: 包含数据的DataFrame
        value_column: 要统计的列名
        group_by: 分组列名列表
    
    Returns:
        统计信息字典
    """
    if df.empty or value_column not in df.columns:
        return {}
    
    valid_data = df.dropna(subset=[value_column])
    
    if valid_data.empty:
        return {}
    
    stats = {
        'total_samples': len(valid_data),
        'valid_samples': len(valid_data),
        'missing_samples': len(df) - len(valid_data),
        'value_range': [valid_data[value_column].min(), valid_data[value_column].max()],
        'value_mean': valid_data[value_column].mean(),
        'value_std': valid_data[value_column].std(),
        'value_median': valid_data[value_column].median(),
        'group_statistics': {}
    }
    
    if group_by and all(col in valid_data.columns for col in group_by):
        for name, group in valid_data.groupby(group_by):
            group_values = group[value_column]
            stats['group_statistics'][str(name)] = {
                'count': len(group),
                'mean': group_values.mean(),
                'std': group_values.std(),
                'min': group_values.min(),
                'max': group_values.max(),
                'median': group_values.median()
            }
    
    return stats


def validate_outlier_parameters(method: str, threshold: float, group_by: Optional[List[str]] = None) -> bool:
    """
    验证异常值检测参数
    
    Args:
        method: 检测方法
        threshold: 检测阈值
        group_by: 分组列名列表
    
    Returns:
        参数是否有效
    """
    valid_methods = ['iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof']
    
    if method not in valid_methods:
        logger.error(f"不支持的检测方法: {method}")
        return False
    
    if threshold <= 0:
        logger.error(f"阈值必须大于0: {threshold}")
        return False
    
    # 方法特定的阈值验证
    if method == 'iqr' and threshold > 5:
        logger.warning(f"IQR阈值 {threshold} 可能过大，建议使用1.5-3.0")
    
    if method == 'zscore' and threshold < 2:
        logger.warning(f"Z-score阈值 {threshold} 可能过小，建议使用2.5-3.5")
    
    if method == 'modified_zscore' and threshold < 2:
        logger.warning(f"修正Z-score阈值 {threshold} 可能过小，建议使用3.0-4.0")
    
    return True

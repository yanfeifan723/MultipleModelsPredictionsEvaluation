"""
EOF分析核心模块
提供经验正交函数分析功能
"""

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path
from ..utils.aggregation import compute_aggregates

logger = logging.getLogger(__name__)

class EOFAnalyzer:
    """EOF分析器"""
    
    def __init__(self, n_modes: int = 6, standardize: bool = True):
        """
        初始化EOF分析器
        
        Args:
            n_modes: EOF模态数
            standardize: 是否标准化数据
        """
        self.n_modes = n_modes
        self.standardize = standardize
        self.pca = None
        self.scaler = None
        self.results = {}
        
    def fit(self, data: xr.DataArray) -> Dict[str, Any]:
        """
        执行EOF分析
        
        Args:
            data: 输入数据 (time, lat, lon)
            
        Returns:
            EOF分析结果
        """
        logger.info(f"开始EOF分析，模态数: {self.n_modes}")
        
        # 数据预处理
        data_processed = self._preprocess_data(data)
        
        # 重塑数据为 (time, space) 矩阵
        data_matrix = self._reshape_data(data_processed)
        
        # 标准化
        if self.standardize:
            self.scaler = StandardScaler()
            data_matrix = self.scaler.fit_transform(data_matrix)
        
        # PCA分析
        self.pca = PCA(n_components=self.n_modes)
        pcs = self.pca.fit_transform(data_matrix)
        
        # 计算EOF模态
        eofs = self.pca.components_
        
        # 重塑EOF为空间网格
        eofs_spatial = self._reshape_eofs(eofs, data_processed)
        
        # 计算解释方差
        explained_variance = self.pca.explained_variance_ratio_
        
        # 存储结果
        self.results = {
            'eofs': eofs_spatial,
            'pcs': pcs,
            'explained_variance_ratio': explained_variance,
            'eigenvalues': self.pca.explained_variance_,
            'coords': {
                'lat': data_processed.lat,
                'lon': data_processed.lon,
                'time': data_processed.time
            },
            'n_modes': self.n_modes,
            'standardized': self.standardize
        }
        
        logger.info(f"EOF分析完成，前{self.n_modes}个模态解释方差: {explained_variance[:3]}")
        
        return self.results
    
    def _preprocess_data(self, data: xr.DataArray) -> xr.DataArray:
        """数据预处理"""
        # 移除时间均值
        data_anomaly = data - data.mean(dim='time')
        
        # 处理缺失值
        data_anomaly = data_anomaly.fillna(0)
        
        return data_anomaly
    
    def _reshape_data(self, data: xr.DataArray) -> np.ndarray:
        """重塑数据为矩阵形式"""
        # 堆叠空间维度
        data_stacked = data.stack(space=('lat', 'lon'))
        
        # 转换为 (time, space) 矩阵
        data_matrix = data_stacked.values
        
        # 移除全为NaN的列
        valid_mask = ~np.isnan(data_matrix).all(axis=0)
        data_matrix = data_matrix[:, valid_mask]
        
        logger.info(f"数据重塑: {data.shape} -> {data_matrix.shape}")
        
        return data_matrix
    
    def _reshape_eofs(self, eofs: np.ndarray, data: xr.DataArray) -> np.ndarray:
        """重塑EOF为空间网格"""
        # 获取空间维度
        n_lat, n_lon = data.lat.size, data.lon.size
        
        # 重塑EOF为 (n_modes, lat, lon)
        eofs_spatial = np.zeros((self.n_modes, n_lat, n_lon))
        
        for i in range(self.n_modes):
            eof_flat = eofs[i]
            eof_2d = eof_flat.reshape(n_lat, n_lon)
            eofs_spatial[i] = eof_2d
        
        return eofs_spatial
    
    def get_eof_modes(self) -> np.ndarray:
        """获取EOF模态"""
        if self.results:
            return self.results['eofs']
        else:
            raise ValueError("请先执行fit方法")
    
    def get_pcs(self) -> np.ndarray:
        """获取主成分时间序列"""
        if self.results:
            return self.results['pcs']
        else:
            raise ValueError("请先执行fit方法")
    
    def get_explained_variance(self) -> np.ndarray:
        """获取解释方差"""
        if self.results:
            return self.results['explained_variance_ratio']
        else:
            raise ValueError("请先执行fit方法")
    
    def reconstruct_data(self, n_modes: Optional[int] = None) -> xr.DataArray:
        """
        使用指定数量的EOF模态重构数据
        
        Args:
            n_modes: 使用的模态数量，None表示使用所有模态
        
        Returns:
            重构的数据
        """
        if not hasattr(self, 'eofs') or not hasattr(self, 'pcs'):
            raise ValueError("请先运行fit方法")
        
        if n_modes is None:
            n_modes = self.n_modes
        else:
            n_modes = min(n_modes, self.n_modes)
        
        # 重构数据
        reconstructed = np.dot(self.pcs[:, :n_modes], self.eofs[:n_modes, :])
        
        # 如果数据被标准化，进行逆变换
        if self.standardize and hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
            reconstructed = reconstructed * self.data_std + self.data_mean
        
        # 恢复原始形状
        reconstructed = reconstructed.reshape(self.original_shape)
        
        # 创建DataArray
        result = xr.DataArray(
            reconstructed,
            dims=self.original_dims,
            coords=self.original_coords,
            attrs={
                'description': f'EOF reconstructed data using {n_modes} modes',
                'n_modes_used': n_modes,
                'standardized': self.standardize
            }
        )
        
        logger.info(f"数据重构完成，使用{n_modes}个模态")
        return result
    
    def save_results(self, filepath: str) -> None:
        """保存分析结果"""
        if not self.results:
            raise ValueError("没有可保存的结果")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"EOF分析结果已保存到: {filepath}")

    def compute_pcs_aggregates(self) -> Dict[str, Dict[int, Dict]]:
        """对PC时间序列按模态计算年/季/月聚合，返回 {label: {mode:int -> aggregates(dict)}}。
        label: 'pcs'（主成分）
        """
        if not self.results:
            raise ValueError("请先执行fit方法")
        pcs = self.results.get('pcs')
        time_coord = self.results.get('coords', {}).get('time')
        if pcs is None or time_coord is None:
            return {}
        # 构造 xarray，模式维在第二维 (time, mode)
        out: Dict[str, Dict[int, Dict]] = {'pcs': {}}
        try:
            da = xr.DataArray(pcs, dims=('time','mode'), coords={'time': time_coord, 'mode': np.arange(1, pcs.shape[1]+1)})
            for mode_idx in da['mode'].values:
                ts = da.sel(mode=mode_idx)
                out['pcs'][int(mode_idx)] = compute_aggregates(ts)
        except Exception:
            # 容错：逐列处理
            n_modes = pcs.shape[1] if pcs.ndim == 2 else 0
            for mi in range(n_modes):
                try:
                    ts = xr.DataArray(pcs[:, mi], dims=['time'], coords={'time': time_coord})
                    out['pcs'][mi+1] = compute_aggregates(ts)
                except Exception:
                    out['pcs'][mi+1] = {'annual': np.nan, 'seasonal': {}, 'monthly': {}}
        return out
    
    def load_results(self, filepath: str) -> None:
        """加载分析结果"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        
        logger.info(f"EOF分析结果已从 {filepath} 加载")

def compute_eof(data: xr.DataArray, 
                n_modes: int = 6,
                standardize: bool = True) -> Dict[str, Any]:
    """
    计算EOF分析
    
    Args:
        data: 输入数据
        n_modes: EOF模态数
        standardize: 是否标准化
        
    Returns:
        EOF分析结果
    """
    analyzer = EOFAnalyzer(n_modes=n_modes, standardize=standardize)
    return analyzer.fit(data)

def compute_pcs(data: xr.DataArray,
                eofs: np.ndarray) -> np.ndarray:
    """
    计算主成分时间序列
    
    Args:
        data: 输入数据
        eofs: EOF模态
        
    Returns:
        主成分时间序列
    """
    # 数据预处理
    data_anomaly = data - data.mean(dim='time')
    data_anomaly = data_anomaly.fillna(0)
    
    # 重塑数据
    data_stacked = data_anomaly.stack(space=('lat', 'lon'))
    data_matrix = data_stacked.values
    
    # 计算PCs
    pcs = data_matrix @ eofs.T
    
    return pcs

def compute_explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """
    计算解释方差比例
    
    Args:
        eigenvalues: 特征值
        
    Returns:
        解释方差比例
    """
    return eigenvalues / np.sum(eigenvalues)

def compute_north_test(eigenvalues: np.ndarray,
                      n_samples: int) -> np.ndarray:
    """
    计算North检验的误差范围
    
    Args:
        eigenvalues: 特征值
        n_samples: 样本数
        
    Returns:
        误差范围
    """
    return eigenvalues * np.sqrt(2.0 / n_samples)

def compute_rule_n(eigenvalues: np.ndarray) -> int:
    """
    计算Rule N（确定显著模态数）
    
    Args:
        eigenvalues: 特征值
        
    Returns:
        显著模态数
    """
    # 计算相邻特征值的差值
    diff = np.diff(eigenvalues)
    
    # 找到第一个差值小于阈值的点
    threshold = eigenvalues[0] * 0.1  # 10%阈值
    
    for i, d in enumerate(diff):
        if d < threshold:
            return i + 1
    
    return len(eigenvalues)

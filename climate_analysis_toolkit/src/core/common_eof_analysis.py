"""
Common EOF分析模块
提供多数据集共同EOF模态计算功能
"""

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from ..utils.aggregation import compute_aggregates

logger = logging.getLogger(__name__)

class CommonEOFAnalyzer:
    """Common EOF分析器"""
    
    def __init__(self, n_modes: int = 6, standardize: bool = True):
        """
        初始化Common EOF分析器
        
        Args:
            n_modes: EOF模态数
            standardize: 是否标准化数据
        """
        self.n_modes = n_modes
        self.standardize = standardize
    
    def compute_common_eofs(self, datasets: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        """
        计算Common EOFs
        
        Args:
            datasets: 数据集字典 {name: data}
            
        Returns:
            Common EOF分析结果
        """
        logger.info(f"开始Common EOF分析，模态数: {self.n_modes}")
        
        if len(datasets) < 2:
            raise ValueError("至少需要2个数据集进行Common EOF分析")
        
        # 1) 时间对齐（共同时间交集）
        datasets_aligned = self._align_time(datasets)

        # 2) 数据预处理（异常/去均值）
        processed_datasets = {}
        for name, data in datasets_aligned.items():
            processed_datasets[name] = self._preprocess_data(data)

        # 3) 空间对齐到参考网格
        ref_name = list(processed_datasets.keys())[0]
        ref_data = processed_datasets[ref_name]
        for name in processed_datasets:
            processed_datasets[name] = processed_datasets[name].interp_like(ref_data)

        # 4) 计算共同有效掩膜（所有数据在所有时间均有效）
        common_mask = None
        for name, data in processed_datasets.items():
            valid = (~xr.ufuncs.isnan(data)).all(dim='time')
            common_mask = valid if common_mask is None else (common_mask & valid)
        # 若掩膜过于严格，可在此处放宽策略
        for name in processed_datasets:
            processed_datasets[name] = processed_datasets[name].where(common_mask)

        # 5) 构建组合矩阵（按数据集顺序拼接特征列）
        combined_data, feature_slices = self._combine_datasets_with_slices(processed_datasets, common_mask)
        
        # 标准化
        if self.standardize:
            scaler = StandardScaler()
            combined_matrix = scaler.fit_transform(combined_data)
        else:
            combined_matrix = combined_data
        
        # PCA分析
        pca = PCA(n_components=self.n_modes)
        pcs = pca.fit_transform(combined_matrix)
        
        # 计算EOF模态（特征维度 = 各数据集空间维拼接）
        eofs = pca.components_

        # 重塑EOF为空间网格（按数据集分别返回）
        eofs_spatial_by_dataset = self._reshape_eofs_by_dataset(eofs, ref_data, feature_slices, common_mask)
        
        # 计算解释方差
        explained_variance = pca.explained_variance_ratio_
        
        # 为每个数据集计算PCs（投影到共同EOF），按特征切片做同参数标准化
        individual_pcs = {}
        for name, data in processed_datasets.items():
            data_matrix, _ = self._reshape_data_with_mask(data, common_mask)
            sl = feature_slices[name]
            if self.standardize:
                # 仅对该数据集对应的特征列使用相同的缩放参数
                mean_slice = scaler.mean_[sl]
                scale_slice = scaler.scale_[sl]
                # 防止除零
                scale_slice = np.where(scale_slice == 0, 1.0, scale_slice)
                data_matrix = (data_matrix - mean_slice) / scale_slice
            eofs_block = eofs[:, sl]
            individual_pcs[name] = data_matrix @ eofs_block.T
        
        # 存储结果
        results = {
            'eofs': eofs_spatial_by_dataset,  # dict[name] -> (n_modes, lat, lon)
            'pcs': individual_pcs,            # dict[name] -> (time, n_modes)
            'pcs_common': pcs,                # (time, n_modes)
            'explained_variance_ratio': explained_variance,
            'eigenvalues': pca.explained_variance_,
            'coords': {
                'lat': ref_data.lat,
                'lon': ref_data.lon,
                'time': ref_data.time
            },
            'n_modes': self.n_modes,
            'standardized': self.standardize,
            'dataset_names': list(datasets.keys())
        }
        
        logger.info(f"Common EOF分析完成，前{self.n_modes}个模态解释方差: {explained_variance[:3]}")
        
        return results

    def compute_pcs_aggregates(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[int, Dict]]]:
        """对共同PC（pcs_common）与各数据集PC（pcs[name]）做年/季/月聚合。
        返回 {'pcs_common': {mode:int -> agg}, 'pcs_by_dataset': {name: {mode:int -> agg}}}
        """
        out: Dict[str, Any] = {'pcs_common': {}, 'pcs_by_dataset': {}}
        time_coord = results.get('coords', {}).get('time')
        # 共同PCs
        pcs_common = results.get('pcs_common')
        if pcs_common is not None and time_coord is not None:
            try:
                da = xr.DataArray(pcs_common, dims=('time','mode'), coords={'time': time_coord, 'mode': np.arange(1, pcs_common.shape[1]+1)})
                for mode_idx in da['mode'].values:
                    ts = da.sel(mode=mode_idx)
                    out['pcs_common'][int(mode_idx)] = compute_aggregates(ts)
            except Exception:
                pass
        # 各数据集PCs
        pcs_by = results.get('pcs', {})
        for name, pc in pcs_by.items():
            out['pcs_by_dataset'].setdefault(name, {})
            try:
                da = xr.DataArray(pc, dims=('time','mode'), coords={'time': time_coord, 'mode': np.arange(1, pc.shape[1]+1)})
                for mode_idx in da['mode'].values:
                    ts = da.sel(mode=mode_idx)
                    out['pcs_by_dataset'][name][int(mode_idx)] = compute_aggregates(ts)
            except Exception:
                continue
        return out
    
    def _preprocess_data(self, data: xr.DataArray) -> xr.DataArray:
        """数据预处理"""
        # 移除时间均值
        data_anomaly = data - data.mean(dim='time')
        
        # 处理缺失值
        data_anomaly = data_anomaly.fillna(0)
        
        return data_anomaly
    
    def _combine_datasets_with_slices(self, datasets: Dict[str, xr.DataArray], mask: xr.DataArray) -> (np.ndarray, Dict[str, slice]):
        """合并数据集并记录各数据集在特征列中的切片位置。"""
        combined_matrices = []
        feature_slices: Dict[str, slice] = {}
        col_start = 0
        for name, data in datasets.items():
            data_matrix, valid_idx = self._reshape_data_with_mask(data, mask)
            ncols = data_matrix.shape[1]
            combined_matrices.append(data_matrix)
            feature_slices[name] = slice(col_start, col_start + ncols)
            col_start += ncols
        combined_data = np.hstack(combined_matrices)
        logger.info(f"数据集合并完成，形状: {combined_data.shape}")
        return combined_data, feature_slices
    
    def _reshape_data_with_mask(self, data: xr.DataArray, mask: xr.DataArray) -> (np.ndarray, np.ndarray):
        """按共同掩膜重塑数据为(time, space_valid)矩阵，并返回有效列索引。"""
        data_masked = data.where(mask)
        data_stacked = data_masked.stack(space=('lat', 'lon'))
        mask_stacked = mask.stack(space=('lat', 'lon'))
        valid_idx = np.where(mask_stacked.values.ravel())[0]
        data_matrix = data_stacked.values[:, valid_idx]
        # 若仍有NaN（极少），进一步剔除含NaN列
        col_valid = ~np.isnan(data_matrix).any(axis=0)
        data_matrix = data_matrix[:, col_valid]
        valid_idx = valid_idx[col_valid]
        return data_matrix, valid_idx
    
    def _reshape_eofs_by_dataset(self, eofs: np.ndarray, ref_data: xr.DataArray,
                                 feature_slices: Dict[str, slice], mask: xr.DataArray) -> Dict[str, np.ndarray]:
        """按数据集将EOF特征向量映射回(lat, lon)网格。"""
        n_lat, n_lon = ref_data.lat.size, ref_data.lon.size
        mask_stacked = mask.stack(space=('lat', 'lon')).values.ravel()
        eofs_by_dataset: Dict[str, np.ndarray] = {}
        for name, sl in feature_slices.items():
            block = eofs[:, sl]  # (n_modes, n_valid)
            eofs_spatial = np.full((self.n_modes, n_lat * n_lon), np.nan, dtype=np.float32)
            eofs_spatial[:, mask_stacked] = block.astype(np.float32)
            eofs_by_dataset[name] = eofs_spatial.reshape(self.n_modes, n_lat, n_lon)
        return eofs_by_dataset

    def _align_time(self, datasets: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
        """按RMSE逻辑对齐时间：
        - 统一到月初（Month Start）时间戳
        - 对重复月份取平均
        - 取所有数据集的共同月份交集
        """
        def normalize_to_month_start(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
            periods = pd.to_datetime(times).to_period('M')
            return periods.to_timestamp(how='start')

        # 先规范每个数据集的time为月初，并对重复月份聚合
        normalized: Dict[str, xr.DataArray] = {}
        for name, data in datasets.items():
            times_norm = normalize_to_month_start(data.time.to_index())
            da = data.assign_coords(time=times_norm)
            if da.indexes['time'].duplicated().any():
                da = da.groupby('time').mean('time', skipna=True)
            normalized[name] = da.sortby('time')

        # 求共同月份交集
        common_index = None
        for da in normalized.values():
            t = da.time.to_index()
            common_index = t if common_index is None else common_index.intersection(t)

        if common_index is None or len(common_index) < 12:
            raise ValueError("各数据集无共同时间交集")

        common_index = common_index.sort_values()
        aligned = {name: da.sel(time=common_index) for name, da in normalized.items()}
        logger.info(f"共同时间点数量: {len(common_index)}")
        return aligned
    
    def save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """保存分析结果"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存特征值与解释方差
        import pandas as pd
        df_eigs = pd.DataFrame({
            'eigenvalue': results['eigenvalues'],
            'explained_variance_ratio': results['explained_variance_ratio']
        }, index=[f"Mode_{i+1}" for i in range(self.n_modes)])
        df_eigs.to_csv(save_path / "eigenvalues.csv")
        
        # 保存空间模态
        ds_eofs = xr.Dataset()
        for name, eofs_spatial in results['eofs'].items():
            for i in range(self.n_modes):
                da = xr.DataArray(
                    eofs_spatial[i],
                    dims=('lat', 'lon'),
                    coords={'lat': results['coords']['lat'], 'lon': results['coords']['lon']},
                    name=f"{name}_EOF_{i+1}"
                )
                ds_eofs[f"{name}_EOF_{i+1}"] = da
        ds_eofs.to_netcdf(save_path / "spatial_modes.nc")
        
        # 保存时间序列
        ds_pcs = xr.Dataset()
        # 共同PCs
        ds_pcs['pcs_common'] = xr.DataArray(
            results['pcs_common'],
            dims=('time', 'mode'),
            coords={'time': results['coords']['time'], 'mode': np.arange(1, self.n_modes+1)}
        )
        # 各数据集PCs
        for name, pc_data in results['pcs'].items():
            da = xr.DataArray(
                pc_data,
                dims=('time', 'mode'),
                coords={'time': results['coords']['time'], 'mode': np.arange(1, self.n_modes+1)},
                name=f"{name}_pcs"
            )
            ds_pcs[f"{name}_pcs"] = da
        ds_pcs.to_netcdf(save_path / "temporal_modes.nc")
        
        logger.info(f"Common EOF分析结果已保存到: {save_path}")

def compute_common_eofs(datasets: Dict[str, xr.DataArray],
                       n_modes: int = 6,
                       standardize: bool = True) -> Dict[str, Any]:
    """
    计算Common EOFs
    
    Args:
        datasets: 数据集字典
        n_modes: EOF模态数
        standardize: 是否标准化
        
    Returns:
        Common EOF分析结果
    """
    analyzer = CommonEOFAnalyzer(n_modes=n_modes, standardize=standardize)
    return analyzer.compute_common_eofs(datasets)

def compute_common_pcs(data: xr.DataArray,
                      eofs: np.ndarray) -> np.ndarray:
    """
    计算Common PCs
    
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

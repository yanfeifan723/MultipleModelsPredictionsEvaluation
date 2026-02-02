"""
幂迭代法共同EOF（Common EOF via power-iteration）实现

遵循伪代码：
- 输入各数据集的协方差隐式算子 Σ_k（通过数据矩阵 X_k 实现，不显式构造 p×p 协方差）
- 权重 n_k 为样本数（时间长度）
- 逐模态、带正交投影的幂迭代，求解公共模态 β_j
- 解释方差 Lambdas[k,j] = β_j^T Σ_k β_j

输出结构尽量与 `CommonEOFAnalyzer` 保持一致，便于统一保存与下游使用：
- results['eofs']            : dict[name] -> (n_modes, lat, lon)（此处所有数据集返回相同的公共空间模态）
- results['pcs']             : dict[name] -> (time, n_modes)
- results['pcs_common']      : (time, n_modes)，取参考数据集
- results['explained_variance_ratio'] : 基于各模式的跨数据集均值 λ̄_j 归一化
- results['eigenvalues']     : λ̄_j（各模式在各数据集解释量的平均）
- results['coords']          : {lat, lon, time}
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


class CommonEOFPowerAnalyzer:
    """幂迭代法共同EOF分析器"""

    def __init__(
        self,
        n_modes: int = 6,
        standardize: bool = True,
        tol: float = 1e-6,
        max_iter_per_mode: int = 500,
        eps_denom: float = 1e-12,
    ) -> None:
        self.n_modes = n_modes
        self.standardize = standardize
        self.tol = tol
        self.max_iter_per_mode = max_iter_per_mode
        self.eps_denom = eps_denom

    # ---------- Public API ----------
    def compute_common_eofs(self, datasets: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        if len(datasets) < 2:
            raise ValueError("至少需要2个数据集进行共同EOF分析（幂迭代法）")

        # 1) 时间对齐（共同时间交集）
        datasets_aligned = self._align_time(datasets)

        # 2) 数据预处理（去时间均值、填充缺测）
        processed_datasets: Dict[str, xr.DataArray] = {}
        for name, data in datasets_aligned.items():
            processed_datasets[name] = self._preprocess_data(data)

        # 3) 空间对齐到参考网格
        ref_name = list(processed_datasets.keys())[0]
        ref_data = processed_datasets[ref_name]
        for name in processed_datasets:
            processed_datasets[name] = processed_datasets[name].interp_like(ref_data)

        # 4) 共同有效掩膜（所有时间均有效）
        common_mask = None
        for _, data in processed_datasets.items():
            valid = (~xr.ufuncs.isnan(data)).all(dim='time')
            common_mask = valid if common_mask is None else (common_mask & valid)
        for name in processed_datasets:
            processed_datasets[name] = processed_datasets[name].where(common_mask)

        # 5) 重塑矩阵与标准化（可选）
        #    X_k 形状: (time_k, p_valid)
        X_list: Dict[str, np.ndarray] = {}
        n_list: Dict[str, int] = {}
        for name, da in processed_datasets.items():
            X, _ = self._reshape_data_with_mask(da, common_mask)
            # 去除仍含NaN的列（极少数情况），已在重塑中处理
            if self.standardize:
                # 按列标准化（时间维），防止幅值差异主导
                col_std = np.std(X, axis=0, ddof=1)
                col_std[col_std == 0] = 1.0
                col_mean = np.mean(X, axis=0)
                X = (X - col_mean) / col_std
            X_list[name] = X
            n_list[name] = X.shape[0]

        model_names = list(X_list.keys())
        # 所有 X_k 在共同掩膜下具有相同的特征数 p
        p = X_list[model_names[0]].shape[1]
        n_lat, n_lon = ref_data.lat.size, ref_data.lon.size
        mask_stacked = common_mask.stack(space=('lat', 'lon')).values.ravel()

        # 6) 幂迭代逐模态求解 β_j
        B = np.zeros((p, self.n_modes), dtype=float)
        Lambdas = np.zeros((len(model_names), self.n_modes), dtype=float)

        def sigma_matvec(name: str, vec: np.ndarray) -> np.ndarray:
            X = X_list[name]  # (t, p)
            t = X.shape[0]
            # Σ_k v = (X^T (X v)) / (t - 1)
            Xv = X @ vec  # (t,)
            return (X.T @ Xv) / max(t - 1, 1)

        for j in range(self.n_modes):
            # 6.1 初始化 β：使用 Σ_weighted ≈ (sum n_k Σ_k)/n_sum 的幂迭代近似主方向，再投影
            # 若初始化退化，则退回随机初始化
            n_sum = float(sum(n_list.values())) if len(n_list) > 0 else 1.0
            beta = np.random.randn(p)
            # 小步数幂迭代以接近 Σ_weighted 的第一主方向
            for _ in range(10):
                v_w = np.zeros(p, dtype=float)
                for name in model_names:
                    v_w += n_list[name] * sigma_matvec(name, beta)
                v_w /= max(n_sum, 1.0)
                if j > 0:
                    v_w = v_w - B[:, :j] @ (B[:, :j].T @ v_w)
                nv = np.linalg.norm(v_w)
                if nv == 0:
                    break
                beta = v_w / nv
            # 如仍退化，使用随机向量并投影
            if np.linalg.norm(beta) == 0:
                beta = np.random.randn(p)
                if j > 0:
                    beta = beta - B[:, :j] @ (B[:, :j].T @ beta)
                norm_beta = np.linalg.norm(beta)
                if norm_beta == 0:
                    beta = np.ones(p) / np.sqrt(p)
                else:
                    beta = beta / norm_beta

            # 6.2 幂迭代
            for _ in range(self.max_iter_per_mode):
                denom_k = []
                v_acc = np.zeros(p, dtype=float)
                for name in model_names:
                    Sigma_beta = sigma_matvec(name, beta)
                    denom_val = float(beta.T @ Sigma_beta)
                    if denom_val < self.eps_denom:
                        denom_val = self.eps_denom
                    v_acc += n_list[name] * (Sigma_beta / denom_val)
                    denom_k.append(denom_val)

                # 投影到正交补
                if j > 0:
                    v_acc = v_acc - B[:, :j] @ (B[:, :j].T @ v_acc)

                norm_v = np.linalg.norm(v_acc)
                if norm_v == 0:
                    # 退化时重新随机化
                    beta = np.random.randn(p)
                    if j > 0:
                        beta = beta - B[:, :j] @ (B[:, :j].T @ beta)
                    beta /= np.linalg.norm(beta)
                    continue

                beta_new = v_acc / norm_v
                # 收敛判据：对符号不敏感
                if 1.0 - abs(float(beta_new @ beta)) < self.tol:
                    beta = beta_new
                    break
                beta = beta_new

            # 6.3 固化该模态
            B[:, j] = beta

            # 6.4 计算解释量 Lambdas[:, j]
            for idx, name in enumerate(model_names):
                Sigma_beta = sigma_matvec(name, beta)
                Lambdas[idx, j] = float(beta.T @ Sigma_beta)

            # 6.5 整体再正交化（避免数值漂移）
            B[:, : j + 1] = self._orthonormalize_columns(B[:, : j + 1])

        # 7) 构造空间模态（将 β_j 回填到 (lat, lon)）
        eofs_spatial = np.full((self.n_modes, n_lat * n_lon), np.nan, dtype=np.float32)
        eofs_spatial[:, mask_stacked] = B.T.astype(np.float32)
        eofs_spatial = eofs_spatial.reshape(self.n_modes, n_lat, n_lon)
        eofs_by_dataset: Dict[str, np.ndarray] = {name: eofs_spatial for name in model_names}

        # 8) 计算时间系数（PCs）：PC_k = X_k @ β（各数据集分别返回）
        pcs_by_dataset: Dict[str, np.ndarray] = {}
        for name in model_names:
            X = X_list[name]  # (t, p)
            pcs = X @ B  # (t, r)
            pcs_by_dataset[name] = pcs

        # 共同PCs：取参考数据集
        pcs_common = pcs_by_dataset[ref_name]

        # 9) 解释方差与特征值（跨数据集取均值）
        lambda_mean = Lambdas.mean(axis=0)
        total = np.sum(lambda_mean) if np.sum(lambda_mean) > 0 else 1.0
        explained_ratio = lambda_mean / total

        results = {
            'eofs': eofs_by_dataset,
            'pcs': pcs_by_dataset,
            'pcs_common': pcs_common,
            'explained_variance_ratio': explained_ratio,
            'eigenvalues': lambda_mean,
            'coords': {
                'lat': ref_data.lat,
                'lon': ref_data.lon,
                'time': ref_data.time
            },
            'n_modes': self.n_modes,
            'standardized': self.standardize,
            'dataset_names': list(datasets.keys()),
            'lambdas_by_dataset': {name: Lambdas[i] for i, name in enumerate(model_names)},
            'method': 'power'
        }

        logger.info(
            f"幂迭代共同EOF完成：前{min(3, self.n_modes)}个模态解释方差(均值) = "
            f"{np.round(explained_ratio[:min(3, self.n_modes)], 4)}"
        )
        return results

    def save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """保存分析结果（与 PCA 版对齐的文件结构）"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存特征值与解释方差
        import pandas as pd
        df_eigs = pd.DataFrame({
            'eigenvalue': results['eigenvalues'],
            'explained_variance_ratio': results['explained_variance_ratio']
        }, index=[f"Mode_{i+1}" for i in range(self.n_modes)])
        df_eigs.to_csv(save_path / "eigenvalues.csv")

        # 保存空间模态（按数据集分别命名，但内容相同）
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

        logger.info(f"幂迭代共同EOF结果已保存到: {save_path}")

    # ---------- Helpers ----------
    def _preprocess_data(self, data: xr.DataArray) -> xr.DataArray:
        data_anomaly = data - data.mean(dim='time')
        data_anomaly = data_anomaly.fillna(0)
        return data_anomaly

    def _reshape_data_with_mask(self, data: xr.DataArray, mask: xr.DataArray):
        data_masked = data.where(mask)
        data_stacked = data_masked.stack(space=('lat', 'lon'))
        mask_stacked = mask.stack(space=('lat', 'lon'))
        valid_idx = np.where(mask_stacked.values.ravel())[0]
        X = data_stacked.values[:, valid_idx]
        # 列级清理：去除含NaN列
        col_valid = ~np.isnan(X).any(axis=0)
        X = X[:, col_valid]
        valid_idx = valid_idx[col_valid]
        return X, valid_idx

    def _align_time(self, datasets: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
        def normalize_to_month_start(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
            periods = pd.to_datetime(times).to_period('M')
            return periods.to_timestamp(how='start')

        normalized: Dict[str, xr.DataArray] = {}
        for name, data in datasets.items():
            times_norm = normalize_to_month_start(data.time.to_index())
            da = data.assign_coords(time=times_norm)
            if da.indexes['time'].duplicated().any():
                da = da.groupby('time').mean('time', skipna=True)
            normalized[name] = da.sortby('time')

        common_index = None
        for da in normalized.values():
            t = da.time.to_index()
            common_index = t if common_index is None else common_index.intersection(t)

        if common_index is None or len(common_index) < 12:
            raise ValueError("各数据集无共同时间交集（或不足12个时间点）")

        common_index = common_index.sort_values()
        aligned = {name: da.sel(time=common_index) for name, da in normalized.items()}
        logger.info(f"共同时间点数量: {len(common_index)}")
        return aligned

    def _orthonormalize_columns(self, M: np.ndarray) -> np.ndarray:
        """对列向量做改进的Gram-Schmidt正交-归一化。"""
        Q = np.zeros_like(M)
        for j in range(M.shape[1]):
            v = M[:, j].copy()
            for i in range(j):
                rij = np.dot(Q[:, i], v)
                v = v - rij * Q[:, i]
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                Q[:, j] = 0
            else:
                Q[:, j] = v / norm_v
        return Q


def compute_common_eofs(
    datasets: Dict[str, xr.DataArray],
    n_modes: int = 6,
    standardize: bool = True,
    tol: float = 1e-6,
    max_iter_per_mode: int = 500,
    eps_denom: float = 1e-12,
) -> Dict[str, Any]:
    """函数式封装，便于直接调用。"""
    analyzer = CommonEOFPowerAnalyzer(
        n_modes=n_modes,
        standardize=standardize,
        tol=tol,
        max_iter_per_mode=max_iter_per_mode,
        eps_denom=eps_denom,
    )
    return analyzer.compute_common_eofs(datasets)



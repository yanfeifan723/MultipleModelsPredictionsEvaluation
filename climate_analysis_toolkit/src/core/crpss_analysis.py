"""
CRPSS (Continuous Ranked Probability Skill Score) 分析模块
计算集合预报的连续排序概率技巧分数
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import json
from datetime import datetime
import warnings

from ..config.settings import (
    MODELS, VAR_CONFIG, LEADTIMES, SPATIAL_BOUNDS,
    get_var_config, get_crpss_config
)
from ..utils.data_utils import (
    find_variable, dynamic_coord_sel, validate_data,
    compute_area_mean
)

warnings.filterwarnings("ignore")

class CRPSSAnalyzer:
    """CRPSS分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化CRPSS分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or get_crpss_config()
        self.logger = logging.getLogger(__name__)
        
        # 区域定义
        self.regions = {
            "NorthEast": {"lat": [35, 45], "lon": [110, 125]},
            "NorthChina": {"lat": [35, 41], "lon": [110, 118]},
            "EastChina": {"lat": [23, 36], "lon": [115, 123]},
            "SouthChina": {"lat": [18, 24], "lon": [109, 121]},
            "SouthWest": {"lat": [21, 30], "lon": [98, 107]},
            "NorthWest": {"lat": [34, 48], "lon": [75, 105]},
            "Tibetan": {"lat": [26, 40], "lon": [80, 100]},
            "WholeChina": {"lat": [15, 55], "lon": [70, 140]}
        }
        
        # 季节定义
        self.seasons = {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11]
        }
        
        self.logger.info("CRPSSAnalyzer初始化完成")
    
    def seasonal_group(self, dt: pd.Timestamp) -> str:
        """
        确定时间所属的季节
        
        Args:
            dt: 时间戳
            
        Returns:
            季节字符串
        """
        m = dt.month
        y = dt.year
        if m == 12:
            return f"DJF-{y+1}"
        elif m in (1, 2):
            return f"DJF-{y}"
        elif m in (3, 4, 5):
            return f"MAM-{y}"
        elif m in (6, 7, 8):
            return f"JJA-{y}"
        else:
            return f"SON-{y}"
    
    def crps_ensemble_scalar(self, obs: float, ensemble: np.ndarray) -> float:
        """
        计算集合预报的CRPS
        
        Args:
            obs: 观测值
            ensemble: 集合预报成员
            
        Returns:
            CRPS值
        """
        if np.isnan(obs) or np.any(np.isnan(ensemble)):
            return np.nan
        
        ensemble = np.sort(ensemble)
        n = len(ensemble)
        
        # 计算经验CDF
        crps = 0.0
        for i in range(n):
            if obs <= ensemble[i]:
                crps += (ensemble[i] - obs) ** 2
            else:
                crps += (obs - ensemble[i]) ** 2
        
        # 集合成员间的贡献
        for i in range(n):
            for j in range(i + 1, n):
                crps += (ensemble[j] - ensemble[i]) / (2 * n * n)
        
        return crps / n
    
    def load_and_process_data(self, 
                            obs_dir: Path,
                            forecast_dir: Path,
                            var_type: str,
                            models: List[str],
                            year_start: Optional[int] = None,
                            year_end: Optional[int] = None,
                            months: Optional[List[int]] = None) -> Tuple[xr.DataArray, Dict]:
        """
        加载和处理观测及预报数据（支持年份/月筛选）
        仅返回观测数据与文件清单（避免持有关闭文件的DataArray引用）。
        """
        var_config = get_var_config(var_type)
        
        # 加载观测数据
        obs_file = obs_dir / f"{var_type}_obs.nc"
        if not obs_file.exists():
            obs_file = obs_dir / f"{var_type}_1deg_199301-202012.nc"
        if not obs_file.exists():
            raise FileNotFoundError(f"观测文件不存在: {obs_file}")
        
        with xr.open_dataset(obs_file) as obs_ds:
            obs_var = find_variable(obs_ds, var_config["obs_names"])
            obs_data = obs_ds[obs_var]
            if "obs_conv" in var_config:
                obs_data = var_config["obs_conv"](obs_data)
            obs_data = dynamic_coord_sel(obs_data, {'lat': (15.0, 55.0), 'lon': (70.0, 140.0)})
            self.logger.info(f"观测变量: {obs_var}, 形状: {tuple(obs_data.shape)}")
        
        # 年份 / 月份
        if year_start is None:
            year_start = 1993
        if year_end is None:
            year_end = 2020
        if months is None:
            months = list(range(1, 13))
        
        # 仅收集文件路径元信息
        forecast_index: Dict[str, Dict[str, Any]] = {}
        for model in models:
            model_config = MODELS[model]
            suffix = model_config[var_config["file_type"]]
            model_dir = forecast_dir / model
            if not model_dir.exists():
                self.logger.warning(f"模型目录不存在: {model_dir}")
                continue
            self.logger.info(f"扫描模型目录: {model_dir}")
            forecast_index[model] = {}
            files_found = 0
            for year in range(year_start, year_end + 1):
                for month in months:
                    for path in [
                        model_dir / f"{year}{month:02d}.{suffix}.nc",
                        model_dir / f"{year}{month:02d}.{suffix}",
                        model_dir / f"{year}{month:02d}.nc",
                    ]:
                        if path.exists():
                            key = f"{year}{month:02d}"
                            forecast_index[model][key] = {
                                'path': path,
                                'year': year,
                                'month': month,
                            }
                            files_found += 1
                            break
            self.logger.info(f"模型 {model} 共发现 {files_found} 个文件")
        
        return obs_data, forecast_index

    def _compute_climate_reference(self, obs_data: xr.DataArray) -> Dict[str, Dict[str, List[Tuple[pd.Timestamp, float]]]]:
        """基于观测构建 区域-季节 的气候参考集合 (时间, 值) 列表。"""
        climate: Dict[str, Dict[str, List[Tuple[pd.Timestamp, float]]]] = {}
        self.logger.info(f"构建气候参考，观测数据形状: {obs_data.shape}")
        
        # 遍历区域
        for region_name, region in self.regions.items():
            climate[region_name] = {}
            self.logger.debug(f"处理区域: {region_name}")
            
            # 遍历时间
            if 'time' not in obs_data.dims:
                self.logger.warning("观测数据没有时间维度，跳过气候参考计算")
                continue
                
            times = obs_data['time'].to_index()
            self.logger.debug(f"时间范围: {times[0]} 到 {times[-1]}, 共{len(times)}个时间点")
            
            for t in times:
                try:
                    obs_2d = obs_data.sel(time=t)
                    season = self.seasonal_group(pd.Timestamp(t))
                    if season is None:
                        continue
                    val = compute_area_mean(obs_2d, region)
                    if not np.isnan(val):
                        climate[region_name].setdefault(season, []).append((pd.Timestamp(t), float(val)))
                except Exception as e:
                    self.logger.debug(f"处理时间 {t} 时出错: {e}")
                    continue
            
            # 统计每个区域-季节的数据量
            for season, data_list in climate[region_name].items():
                self.logger.debug(f"区域 {region_name} 季节 {season}: {len(data_list)} 个样本")
        
        return climate

    def _sel_by_year_month(self, da: xr.DataArray, year: int, month: int) -> Optional[xr.DataArray]:
        """按年-月在 time 维上选择最近的单时刻。优先精确年/月匹配；失败则返回 None。"""
        if 'time' not in da.dims:
            return da
        t = da['time']
        try:
            year_arr = t.dt.year
            month_arr = t.dt.month
            mask = (year_arr == year) & (month_arr == month)
            if bool(mask.any()):
                idx = int(mask.argmax())
                return da.isel(time=idx)
        except Exception:
            pass
        return None

    def compute_crpss(self, 
                     obs_data: xr.DataArray,
                     forecast_data: Dict,
                     var_type: str,
                     models: List[str],
                     lead_times: List[int] = None) -> Dict:
        """计算CRPSS"""
        if lead_times is None:
            lead_times = LEADTIMES
        results: Dict[str, Dict[str, Dict]] = {}
        min_samples = self.config.get('min_samples', 1)
        
        # 准备气候参考
        climate = self._compute_climate_reference(obs_data) if 'time' in obs_data.dims else {}
        self.logger.info(f"气候参考构建完成，包含 {len(climate)} 个区域")
        
        for model in models:
            if model not in forecast_data:
                continue
            self.logger.info(f"处理模型: {model}")
            results[model] = {str(lt): {} for lt in lead_times}
            all_data: Dict[str, Dict[str, list]] = {}
            processed = 0
            total_files = len(forecast_data[model])
            
            for key, meta in forecast_data[model].items():
                path: Path = meta['path']
                year = meta['year']
                month = meta['month']
                processed += 1
                if processed % 5 == 0:
                    self.logger.info(f"{model}: 已处理 {processed}/{total_files} 个起报 (最新: {year}-{month:02d})")
                try:
                    with xr.open_dataset(path) as fcst_ds:
                        var_config = get_var_config(var_type)
                        fcst_var = find_variable(fcst_ds, var_config["fcst_names"])
                        fcst = fcst_ds[fcst_var]
                        if "fcst_conv" in var_config:
                            fcst = var_config["fcst_conv"](fcst)
                        # 坐标与观测对齐
                        fcst = dynamic_coord_sel(fcst, {'lat': (15.0, 55.0), 'lon': (70.0, 140.0)})
                        try:
                            fcst = fcst.interp(lat=obs_data.lat, lon=obs_data.lon)
                        except Exception:
                            pass
                        member_dim = next((d for d in ['number','member','ensemble','realization'] if d in fcst.dims), None)
                        if 'time' not in fcst.dims:
                            self.logger.debug(f"预报文件 {path} 没有时间维度")
                            continue
                        for lt in lead_times:
                            valid = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=lt)
                            try:
                                obs_2d = self._sel_by_year_month(obs_data, valid.year, valid.month) if 'time' in obs_data.dims else obs_data
                                fcst_at_valid = self._sel_by_year_month(fcst, valid.year, valid.month)
                                if obs_2d is None or fcst_at_valid is None:
                                    self.logger.debug(f"{model} {year}-{month:02d} LT{lt}: 找不到 {valid.year}-{valid.month:02d} 对应时次，跳过")
                                    continue
                            except Exception as e:
                                self.logger.debug(f"选择时间 {valid} 失败: {e}")
                                continue
                            for region_name, region in self.regions.items():
                                try:
                                    obs_mean = compute_area_mean(obs_2d, region)
                                    if np.isnan(obs_mean):
                                        continue
                                    ensemble_means: List[float] = []
                                    if member_dim and member_dim in fcst_at_valid.dims:
                                        for i in range(fcst_at_valid[member_dim].size):
                                            member_field = fcst_at_valid.isel({member_dim: i})
                                            mval = compute_area_mean(member_field, region)
                                            if not np.isnan(mval):
                                                ensemble_means.append(float(mval))
                                    else:
                                        mval = compute_area_mean(fcst_at_valid, region)
                                        if not np.isnan(mval):
                                            ensemble_means.append(float(mval))
                                    if not ensemble_means:
                                        continue
                                    season = self.seasonal_group(valid)
                                    season_key = season  # 与气候参考中的键一致，如 MAM-1994
                                    data_key = f"{model} LT{lt} {region_name} {season_key}"
                                    if data_key not in all_data:
                                        all_data[data_key] = {"obs": [], "ens": [], "times": []}
                                    all_data[data_key]["obs"].append(float(obs_mean))
                                    all_data[data_key]["ens"].append(np.array(ensemble_means, dtype=float))
                                    all_data[data_key]["times"].append(valid)
                                except Exception as e:
                                    self.logger.debug(f"处理区域 {region_name} 失败: {e}")
                                    continue
                except Exception as e:
                    self.logger.error(f"处理文件失败 {path}: {e}")
                    continue
            
            self.logger.info(f"模型 {model} 数据收集完成，共 {len(all_data)} 个数据组")
            
            # 计算CRPSS
            successful_cases = 0
            for data_key, data in all_data.items():
                parts = data_key.split()
                if len(parts) != 4:
                    continue
                model_name, lt_str, region_name, season = parts[0], parts[1], parts[2], parts[3]
                lt_val = int(lt_str[2:])
                if len(data["obs"]) < min_samples:
                    self.logger.debug(f"{var_type} {model_name} {lt_str} {region_name} {season} 样本数不足: {len(data['obs'])} < {min_samples}")
                    continue
                crps_model_vals: List[float] = []
                crps_ref_vals: List[float] = []
                for i, (obs_val, ens_val, tval) in enumerate(zip(data["obs"], data["ens"], data["times"])):
                    crps_m = self.crps_ensemble_scalar(float(obs_val), np.asarray(ens_val, dtype=float))
                    if not np.isnan(crps_m):
                        crps_model_vals.append(crps_m)
                    # LOO 气候参考
                    clim_list = climate.get(region_name, {}).get(season, []) if climate else []
                    ref_vals = [v for (tt, v) in clim_list if tt != tval]
                    if len(ref_vals) >= min_samples:
                        crps_r = self.crps_ensemble_scalar(float(obs_val), np.asarray(ref_vals, dtype=float))
                        if not np.isnan(crps_r):
                            crps_ref_vals.append(crps_r)
                if len(crps_model_vals) >= min_samples and len(crps_ref_vals) >= min_samples:
                    mean_crps_model = float(np.mean(crps_model_vals))
                    mean_crps_ref = float(np.mean(crps_ref_vals))
                    crpss = float('nan') if mean_crps_ref <= 0 else 1.0 - mean_crps_model / mean_crps_ref
                    if str(lt_val) not in results[model_name]:
                        results[model_name][str(lt_val)] = {}
                    if region_name not in results[model_name][str(lt_val)]:
                        results[model_name][str(lt_val)][region_name] = {}
                    results[model_name][str(lt_val)][region_name][season] = {
                        "CRPS_model": mean_crps_model,
                        "CRPS_ref": mean_crps_ref,
                        "CRPSS": crpss,
                        "n": len(crps_model_vals)
                    }
                    successful_cases += 1
                    self.logger.info(f"计算 CRPSS: {var_type} {model_name} {lt_str} {region_name} {season} pairs={len(crps_model_vals)} -> CRPSS={crpss:.3f}")
            
            self.logger.info(f"模型 {model} 完成，成功计算 {successful_cases} 个CRPSS值")
        return results
    
    def run_analysis(self,
                    obs_dir: Path,
                    forecast_dir: Path,
                    output_dir: Path,
                    var_types: List[str] = None,
                    models: List[str] = None,
                    lead_times: List[int] = None) -> Dict:
        """
        运行CRPSS分析
        
        Args:
            obs_dir: 观测数据目录
            forecast_dir: 预报数据目录
            output_dir: 输出目录
            var_types: 变量类型列表
            models: 模型列表
            lead_times: 预报时效列表
            
        Returns:
            分析结果
        """
        if var_types is None:
            var_types = list(VAR_CONFIG.keys())
        if models is None:
            models = list(MODELS.keys())
        if lead_times is None:
            lead_times = LEADTIMES
        
        self.logger.info(f"开始CRPSS分析: 变量={var_types}, 模型={models}, 时效={lead_times}")
        
        results = {}
        
        for var_type in var_types:
            self.logger.info(f"处理变量: {var_type}")
            
            try:
                # 加载数据
                obs_data, forecast_data = self.load_and_process_data(
                    obs_dir, forecast_dir, var_type, models
                )
                
                # 计算CRPSS
                var_results = self.compute_crpss(
                    obs_data, forecast_data, var_type, models, lead_times
                )
                
                results[var_type] = var_results
                
                # 保存结果
                output_file = output_dir / f"crpss_{var_type}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(var_results, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"结果已保存到: {output_file}")
                
            except Exception as e:
                self.logger.error(f"处理变量 {var_type} 时出错: {e}")
                continue
        
        return results

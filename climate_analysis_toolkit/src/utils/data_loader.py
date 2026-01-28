"""
通用数据加载和处理模块
支持观测数据、模式数据、NetCDF文件等的加载和处理
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.interpolate import griddata
import glob
import os

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def obs_prec_conv(x):
    """观测降水单位转换函数"""
    return x * 86400

def fcst_prec_conv(x):
    """预报降水单位转换函数"""
    return x * 86400 * 1000

class DataLoader:
    """通用数据加载器"""
    
    def __init__(self, obs_dir: str = "./obs", forecast_dir: str = "/raid62/EC-C3S/month"):
        self.obs_dir = Path(obs_dir)
        self.forecast_dir = Path(forecast_dir)
        
        # 模型配置
        self.models = {
            "CMCC-35": {"pl": "cmcc.35", "sfc": "cmcc.35.sfc"},
            "DWD-mon-21": {"pl": "dwd.21", "sfc": "dwd.sfc.21"},
            "ECMWF-51-mon": {"pl": "ecmwf.51", "sfc": "ecmwf.51.sfc"},
            "Meteo-France-8": {"pl": "meteo_france.8", "sfc": "meteo_france.sfc.8"},
            "JMA-3-mon": {"pl": "jma.3", "sfc": "jma.3.sfc"},
            "NCEP-2": {"pl": "ncep.2", "sfc": "ncep.2.sfc"},
            "UKMO-14": {"pl": "ukmo.14", "sfc": "ukmo.sfc.14"},
            "ECCC-Canada-3": {"pl": "eccc.3", "sfc": "eccc.sfc.3"}
        }
        
        # 变量配置
        self.var_config = {
            "temp": {
                "file_type": "sfc",
                "obs_names": ["t", "t2m", "temp", "temperature", "tas", "tm"],
                "fcst_names": ["t", "t2m", "temp", "temperature", "tas", "tm"],
                "unit": "K"
            },
            "prec": {
                "file_type": "sfc",
                "obs_names": ["tp", "prec", "pr", "precip", "tprate", "pre"],
                "fcst_names": ["tp", "tprate", "prec", "pr", "precip", "pre"],
                "obs_conv": obs_prec_conv,
                "fcst_conv": fcst_prec_conv,
                "unit": "mm/day"
            }
        }

    def get_model_suffix(self, model_name: str) -> str:
        """
        根据模式名称推断pressure-level文件后缀。
        逻辑与MMPE脚本保持一致，确保两侧引用同一规则。
        """
        model_lower = model_name.lower()
        parts = model_name.split('-')

        if 'ecmwf' in model_lower:
            return f"ecmwf.{parts[1] if len(parts) > 1 else '51'}"
        if 'cmcc' in model_lower:
            return f"cmcc.{parts[1] if len(parts) > 1 else '35'}"
        if 'dwd' in model_lower:
            return f"dwd.{parts[-1] if len(parts) > 1 else '21'}"
        if 'meteo' in model_lower:
            return f"meteo_france.{parts[-1] if len(parts) > 1 else '8'}"
        if 'ncep' in model_lower:
            return f"ncep.{parts[-1] if len(parts) > 1 else '2'}"
        if 'ukmo' in model_lower:
            return f"ukmo.{parts[-1] if len(parts) > 1 else '14'}"
        if 'eccc' in model_lower:
            return f"eccc.{parts[-1] if len(parts) > 1 else '3'}"
        if len(parts) >= 2:
            return f"{parts[0].lower()}.{parts[-1]}"
        return model_lower

    def load_pressure_level_data(
        self,
        model: str,
        leadtime: int,
        var_name: str,
        pressure_level: int = 850,
        spatial_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        year_range: Tuple[int, int] = (1993, 2020)
    ) -> Optional[xr.DataArray]:
        """
        加载指定模式在给定气压层的变量数据，沿用MMPE脚本的实现作为准则。
        返回(time, number, lat, lon) 或 (time, lat, lon) 的DataArray。
        """
        try:
            model_dir = self.forecast_dir / model
            if not model_dir.exists():
                logger.warning(f"模式目录不存在: {model_dir}")
                return None

            model_suffix = self.get_model_suffix(model)
            latlon_bounds = spatial_bounds or {'lat': (15, 55), 'lon': (70, 140)}

            monthly_da_list = []
            start_year, end_year = year_range

            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    fp = model_dir / f"{year}{month:02d}.{model_suffix}.nc"
                    if not fp.exists():
                        continue
                    try:
                        with xr.open_dataset(fp) as ds:
                            if var_name not in ds:
                                logger.debug(f"变量 {var_name} 不在文件 {fp} 中")
                                continue

                            da = ds[var_name]
                            if 'level' not in da.dims:
                                logger.warning(f"变量 {var_name} 没有level维度，跳过: {fp}")
                                continue
                            if pressure_level not in ds.level.values:
                                logger.warning(
                                    f"气压层 {pressure_level} hPa 不在数据中，可用层: {ds.level.values}"
                                )
                                continue

                            da = da.sel(level=pressure_level).drop_vars('level', errors='ignore')

                            if 'time' in da.dims and 'number' in da.dims:
                                if da.time.size <= leadtime:
                                    continue
                                da = da.isel(time=leadtime)
                            elif 'time' in da.dims:
                                init = pd.Timestamp(year, month, 1)
                                da = da.sel(time=init, method='nearest', tolerance='15D')
                                if 'number' not in da.dims:
                                    da = da.expand_dims('number')
                            else:
                                if 'number' not in da.dims:
                                    da = da.expand_dims('number')

                            da = self.dynamic_coord_sel(da, latlon_bounds)

                            if all(dim in da.dims for dim in ['number', 'lat', 'lon']):
                                if 'time' in da.coords:
                                    forecast_time = pd.Timestamp(da.time.values)
                                else:
                                    forecast_time = pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                                monthly_da_list.append((forecast_time, da))
                    except Exception as e:
                        logger.error(f"处理文件 {fp} 时出错: {e}")
                        continue

            if not monthly_da_list:
                logger.warning(f"无ensemble数据 {model} L{leadtime} {var_name} {pressure_level}hPa")
                return None

            times = xr.DataArray([t for t, _ in monthly_da_list], dims='time', name='time')
            data = xr.concat([da for _, da in monthly_da_list], dim=times)
            data = data.sortby('time')

            logger.info(
                f"Ensemble数据加载成功 {model} L{leadtime} {var_name} {pressure_level}hPa: "
                f"{data.shape}, 成员数={data.number.size if 'number' in data.dims else 1}"
            )
            return data

        except Exception as e:
            logger.error(f"加载pressure-level数据失败: {e}")
            return None
    
    def find_variable(self, ds: xr.Dataset, candidates: List[str]) -> str:
        """在数据集中查找变量"""
        for var in candidates:
            if var in ds:
                return var
        raise ValueError(f"未找到候选变量: {candidates}")
    
    def load_forecast_data_ensemble(
        self,
        model: str,
        var_type: str,
        leadtime: int,
        spatial_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        year_range: Tuple[int, int] = (1993, 2020)
    ) -> Optional[xr.DataArray]:
        """
        加载并返回保留 ensemble 成员维度的预报数据。
        """
        config = self.var_config[var_type]
        suffix = self.models[model][config['file_type']]
        model_dir = self.forecast_dir / model

        if not model_dir.exists():
            logger.warning(f"模式目录不存在: {model_dir}")
            return None

        spatial_bounds = spatial_bounds or {'lat': (15, 55), 'lon': (70, 140)}

        monthly_da_list = []
        start_year, end_year = year_range

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                fp = model_dir / f"{year}{month:02d}.{suffix}.nc"
                if not fp.exists():
                    continue
                try:
                    with xr.open_dataset(fp) as ds:
                        var_name = self.find_variable(ds, config['fcst_names'])
                        da = ds[var_name]

                        if 'level' in da.dims:
                            da = da.isel(level=0)
                            da = da.drop_vars('level', errors='ignore')

                        if 'time' in da.dims:
                            if da.time.size <= leadtime:
                                continue
                            da = da.isel(time=leadtime)
                        if 'number' not in da.dims:
                            da = da.expand_dims('number')

                        if var_type == 'prec' and 'fcst_conv' in config:
                            da = da * config['fcst_conv'](1)
                            da = da.clip(min=0)

                        da = self.dynamic_coord_sel(da, spatial_bounds)

                        forecast_time = (
                            pd.Timestamp(da.time.values)
                            if 'time' in da.coords
                            else pd.Timestamp(year, month, 1) + pd.DateOffset(months=leadtime)
                        )
                        monthly_da_list.append((forecast_time, da))
                except Exception as e:
                    logger.warning(f"处理文件 {fp} 时出错: {e}")
                    continue

        if not monthly_da_list:
            logger.warning(f"无预测数据 {model} L{leadtime} (ensemble)")
            return None

        times = xr.DataArray([t for t, _ in monthly_da_list], dims='time', name='time')
        data = xr.concat([da for _, da in monthly_da_list], dim=times)
        data = data.sortby('time')
        logger.info(
            f"Ensemble 预报加载完成 {model} L{leadtime}: {data.shape}, "
            f"成员数={data.number.size}"
        )
        return data
    
    def dynamic_coord_sel(self, ds: xr.DataArray, coords: Dict[str, Tuple[float, float]]) -> xr.DataArray:
        """动态坐标选择"""
        coord_map = {'lat': ['latitude', 'lats', 'ylat'], 'lon': ['longitude', 'lons', 'xlon']}
        ds = ds.copy()
        
        for target, (start, end) in coords.items():
            real = next((alias for alias in [target] + coord_map.get(target, []) 
                        if alias in ds.coords), None)
            if real is None:
                raise ValueError(f"坐标{target}不存在")
            if real != target:
                ds = ds.rename({real: target})
            
            vals = ds[target].values
            dec = vals[0] > vals[1] if len(vals) > 1 else False
            buf = 0.5 * abs(vals[1] - vals[0]) if len(vals) > 1 else 1.0
            
            if dec:
                slc = slice(end + buf, start - buf)
            else:
                slc = slice(start - buf, end + buf)
            
            ds = ds.sel({target: slc})
            if dec:
                ds = ds.sortby(target)
        
        return ds
    
    def load_obs_data(self, var_type: str, time_range: Optional[Tuple[str, str]] = None, 
                     for_mask: bool = False) -> xr.DataArray:
        """
        加载观测数据 - 使用1度网格数据
        
        Args:
            var_type: 变量类型
            time_range: 时间范围
            for_mask: 如果为True，不进行fillna(0)处理，保留原始NaN用于掩膜创建
        """
        config = self.var_config[var_type]
        # 优先使用obs文件夹中的1度网格数据
        obs_path = self.obs_dir / f"{var_type}_1deg_199301-202012.nc"
        
        if not obs_path.exists():
            # 如果1度网格数据不存在，尝试使用CN05.1数据
            obs_path = self.obs_dir / f"CN05.1_{var_type}_1961_2022_monthly_1x1.nc"
            if not obs_path.exists():
                raise FileNotFoundError(f"观测数据文件不存在: {obs_path}")
        
        with xr.open_dataset(obs_path, mask_and_scale=False) as ds:
            var_name = self.find_variable(ds, config['obs_names'])
            da = ds[var_name].where(
                ~ds[var_name].isin([1e20, ds[var_name].attrs.get('_FillValue', 1e20)]), 
                np.nan
            )
            
            if var_type == 'prec' and 'obs_conv' in config:
                da = da * config['obs_conv'](1)
                da = da.clip(min=0)
                # 只有在非掩膜模式下才fillna(0)
                if not for_mask:
                    da = da.fillna(0)
            
            # 选择目标区域 (15-55N, 70-140E)
            da = self.dynamic_coord_sel(da, {'lat': (15, 55), 'lon': (70, 140)})
            
            if time_range:
                da = da.sel(time=slice(time_range[0], time_range[1]))
            
            logger.info(f"观测数据加载完成（1度网格）: {da.shape}, for_mask={for_mask}")
            return da
    
    def load_forecast_data(self, model: str, var_type: str, leadtime: int, 
                          time_range: Optional[Tuple[str, str]] = None) -> Optional[xr.DataArray]:
        """加载模式预报数据"""
        config = self.var_config[var_type]
        suffix = self.models[model][config['file_type']]
        model_dir = self.forecast_dir / model
        
        if not model_dir.exists():
            logger.warning(f"模式目录不存在: {model_dir}")
            return None
        
        # 收集所有月度数据（保持月度分辨率）
        all_monthly_data = []
        all_monthly_times = []
        actual_lat = None
        actual_lon = None
        
        for year in range(1993, 2021):
            for month in range(1, 13):
                fp = model_dir / f"{year}{month:02d}.{suffix}.nc"
                if not fp.exists():
                    continue
                
                try:
                    with xr.open_dataset(fp) as ds:
                        var_name = self.find_variable(ds, config['fcst_names'])
                        da = ds[var_name]
                        
                        # 处理预报时效
                        if 'number' in da.dims and 'time' in da.dims:
                            # 模型数据的时间维度是预报时效
                            if da.time.size <= leadtime:
                                continue
                            da = da.isel(time=leadtime)
                            if 'number' in da.dims:
                                da = da.mean(dim='number')
                        elif 'time' in da.dims:
                            # 修复：其他数据的时间维度处理
                            # 使用初始化时间而不是有效时间
                            init = pd.Timestamp(year, month, 1)
                            if init.year < 1993 or init.year > 2020:
                                continue
                            sel = da.sel(time=init, method='nearest', tolerance='15D')
                            if sel.time.size == 0:
                                continue
                            da = sel
                        
                        # 数据转换
                        if var_type == 'prec' and 'fcst_conv' in config:
                            da = da * config['fcst_conv'](1)
                            da = da.clip(min=0).fillna(0)
                        
                        # 空间处理 - 选择目标区域（不进行重采样，保持原始网格）
                        da = self.dynamic_coord_sel(da, {'lat': (15, 55), 'lon': (70, 140)})
                        
                        # 保存第一个数据的坐标信息（用于后续构建DataArray）
                        if actual_lat is None and 'lat' in da.coords and 'lon' in da.coords:
                            actual_lat = da.lat.values.copy()
                            actual_lon = da.lon.values.copy()
                        
                        # 获取数据数组（保持原始网格分辨率）
                        arr = da.values
                        if arr.ndim == 3:
                            arr = arr[0]
                        if arr.ndim != 2:
                            continue
                        
                        all_monthly_data.append(arr)
                        # 使用实际的预报时间（从文件内time坐标获取）
                        forecast_time = pd.Timestamp(da.time.values) if hasattr(da, 'time') else pd.Timestamp(year, month, 1)
                        all_monthly_times.append(forecast_time)
                
                except Exception as e:
                    logger.warning(f"处理文件 {fp} 时出错: {str(e)}")
                    continue
        
        if not all_monthly_data:
            logger.warning(f"无预测数据 {model} L{leadtime}")
            return None
        
        try:
            # 检查是否成功获取了坐标信息
            if actual_lat is None or actual_lon is None:
                logger.error("无法获取网格坐标信息")
                return None
            
            # 检查所有数据的网格是否一致
            first_arr = all_monthly_data[0]
            n_lat, n_lon = first_arr.shape
            if len(actual_lat) != n_lat or len(actual_lon) != n_lon:
                logger.warning(f"坐标维度与数据维度不匹配: 坐标=({len(actual_lat)}, {len(actual_lon)}), 数据=({n_lat}, {n_lon})")
            
            for arr in all_monthly_data[1:]:
                if arr.shape != (n_lat, n_lon):
                    logger.warning(f"数据网格不一致: 第一个 {first_arr.shape}, 当前 {arr.shape}")
            
            # 检查时间戳是否有重复并去重
            unique_times = list(dict.fromkeys(all_monthly_times))  # 保持顺序的去重
            if len(unique_times) != len(all_monthly_times):
                logger.warning(f"发现重复时间戳，原始长度: {len(all_monthly_times)}, 去重后: {len(unique_times)}")
                # 去重数据
                unique_data = []
                seen_times = set()
                for i, time in enumerate(all_monthly_times):
                    if time not in seen_times:
                        unique_data.append(all_monthly_data[i])
                        seen_times.add(time)
                all_monthly_data = unique_data
                all_monthly_times = unique_times
            
            data = xr.concat([
                xr.DataArray(arr, dims=('lat', 'lon'), 
                           coords={'lat': actual_lat, 
                                  'lon': actual_lon})
                for arr in all_monthly_data
            ], dim=xr.DataArray(all_monthly_times, dims='time', name='time'))
            
            data = data.sortby('time')
            
            # 检查最终数据的时间长度
            if len(data.time) > 400:  # 超过400个月（约33年）可能有问题
                logger.warning(f"数据时间长度异常: {len(data.time)} 个月，模型: {model}, 预报时效: {leadtime}")
                logger.warning(f"时间范围: {data.time.min().values} 到 {data.time.max().values}")
            
            if time_range:
                data = data.sel(time=slice(time_range[0], time_range[1]))
            
            logger.info(f"模式数据加载完成 {model} L{leadtime}: {data.shape}")
            logger.info(f"时间范围: {data.time.min().values} 到 {data.time.max().values}")
            logger.info(f"时间轴已修复：使用统一的初始化时间轴")
            return data
            
        except Exception as e:
            logger.error(f"处理预测数据时出错 {model} L{leadtime}: {str(e)}")
            return None
    
    def interpolate_to_grid(self, data: xr.DataArray, target_lat: np.ndarray, 
                           target_lon: np.ndarray) -> xr.DataArray:
        """插值到目标网格"""
        try:
            # 创建网格点
            lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
            
            # 提取原始数据点
            lon_orig, lat_orig = np.meshgrid(data.lon, data.lat)
            points = np.column_stack((lon_orig.flatten(), lat_orig.flatten()))
            
            # 对每个时间点进行插值
            interpolated_data = []
            for t in range(len(data.time)):
                values = data.isel(time=t).values.flatten()
                
                # 移除NaN值
                valid_mask = ~np.isnan(values)
                valid_points = points[valid_mask]
                valid_values = values[valid_mask]
                
                if len(valid_values) == 0:
                    interpolated_data.append(np.full((len(target_lat), len(target_lon)), np.nan))
                    continue
                
                # 插值到新网格
                interpolated = griddata(
                    valid_points, 
                    valid_values, 
                    (lon_grid, lat_grid), 
                    method='nearest'
                )
                interpolated_data.append(interpolated)
            
            # 创建新的DataArray
            result = xr.DataArray(
                np.array(interpolated_data),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': data.time,
                    'lat': target_lat,
                    'lon': target_lon
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"插值失败: {str(e)}")
            return data
    
    def load_netcdf_files(self, pattern: str) -> List[xr.Dataset]:
        """加载匹配模式的NetCDF文件"""
        files = sorted(glob.glob(pattern))
        datasets = []
        
        for file_path in files:
            try:
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
                logger.info(f"成功加载: {file_path}")
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {str(e)}")
        
        return datasets
    
    def save_data(self, data: Union[xr.DataArray, xr.Dataset], 
                  output_path: Path, format: str = 'netcdf') -> None:
        """保存数据"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'netcdf':
            if isinstance(data, xr.DataArray):
                data.to_netcdf(output_path)
            else:
                data.to_netcdf(output_path)
        elif format == 'csv':
            if isinstance(data, xr.DataArray):
                data.to_dataframe().to_csv(output_path)
            else:
                data.to_dataframe().to_csv(output_path)
        
        logger.info(f"数据已保存到: {output_path}")
    
    def get_data_info(self, data: xr.DataArray) -> Dict:
        """获取数据信息"""
        info = {
            'shape': data.shape,
            'dims': list(data.dims),
            'coords': {dim: len(data[dim]) for dim in data.dims},
            'attrs': dict(data.attrs),
            'dtype': str(data.dtype)
        }
        
        if 'time' in data.dims:
            info['time_range'] = (str(data.time.min().values), str(data.time.max().values))
        
        if 'lat' in data.dims:
            info['lat_range'] = (float(data.lat.min()), float(data.lat.max()))
        
        if 'lon' in data.dims:
            info['lon_range'] = (float(data.lon.min()), float(data.lon.max()))
        
        return info

"""
气候分析工具包配置文件
包含所有分析模块的配置参数
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Callable
import logging

# ==================== 基础配置 ====================
# 数据目录配置
OBS_DIR = "/sas12t1/ffyan/obs"
FORECAST_DIR = "/raid62/EC-C3S/month"
OUTPUT_DIR = "/sas12t1/ffyan/output"

# 支持的数值模式配置
MODELS = {
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
VAR_CONFIG = {
    "temp": {
        "file_type": "sfc",
        "obs_names": ["temp"],
        "fcst_names": ["t2m"],
        "unit": "K",
        "cmap": "coolwarm",
        "title": "Temperature"
    },
    "prec": {
        "file_type": "sfc",
        "obs_names": ["prec"],
        "fcst_names": ["tprate"],
        "obs_conv": lambda x: x * 86400,  # kg m-2 s-1 → mm/day
        "fcst_conv": lambda x: x * 86400,  # m s-1 → mm/day
        "unit": "mm/day",
        "cmap": "Blues",
        "title": "Precipitation"
    }
}

# 时间配置
LEADTIMES = [0, 1, 2, 3, 4, 5]
DATE_RANGE = ("1993-01", "2020-12")

# 空间配置
SPATIAL_BOUNDS = {
    "lat": (15.0, 55.0),
    "lon": (70.0, 140.0)
}

# ==================== 分析参数配置 ====================
# EOF分析配置
EOF_CONFIG = {
    "n_modes": 6,
    "standardize": True,
    "min_valid_cols": 1,
    "min_time_points": 1,
    "debug_mode": True
}

# Common EOF配置
COMMON_EOF_CONFIG = {
    "n_modes": 6,
    "min_valid_cols": 1,
    "min_time_points": 1,
    "debug_mode": True,
    "output_dir": "./common_eofs"
}

# 相关性分析配置
CORRELATION_CONFIG = {
    "min_valid_points": 3,
    "significance_level": 0.05
}

# RMSE分析配置
RMSE_CONFIG = {
    "high_bias_threshold": 15,
    "high_rmse_threshold": 15
}

# 功率谱分析配置
SPECTRUM_CONFIG = {
    "window_type": "hanning",
    "detrend": True,
    "nperseg": None,  # 自动计算
    "nfft": None,     # 自动计算
    "regions": {
        "global": {"lat": (15, 55), "lon": (70, 140)},
        "east_asia": {"lat": (20, 50), "lon": (100, 140)},
        "south_asia": {"lat": (5, 35), "lon": (70, 100)}
    }
}

# CRPSS分析配置
CRPSS_CONFIG = {
    "min_samples": 1,  # 降低最小样本数要求，从3改为1
    "regions": {
        "NorthEast": {"lat": [35, 45], "lon": [110, 125]},
        "NorthChina": {"lat": [35, 41], "lon": [110, 118]},
        "EastChina": {"lat": [23, 36], "lon": [115, 123]},
        "SouthChina": {"lat": [18, 24], "lon": [109, 121]},
        "SouthWest": {"lat": [21, 30], "lon": [98, 107]},
        "NorthWest": {"lat": [34, 48], "lon": [75, 105]},
        "Tibetan": {"lat": [26, 40], "lon": [80, 100]},
        "WholeChina": {"lat": [15, 55], "lon": [70, 140]}
    },
    "seasons": {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11]
    }
}

# ==================== 可视化配置 ====================
# 绘图配置
PLOT_CONFIG = {
    "figure_dpi": 300,
    "savefig_dpi": 300,
    "default_figsize": (12, 8),
    "font_size": 12,
    "line_width": 2,
    "marker_size": 6
}

# 地图配置
MAP_CONFIG = {
    "projection": "PlateCarree",
    "coastline_resolution": "50m",
    "coastline_linewidth": 1.0,
    "coastline_color": "black",
    "gridline_alpha": 0.5,
    "gridline_linestyle": "--",
    "country_borders_path": "/sas12t1/ffyan/boundaries/国界线.shp"
}

# 颜色配置
COLOR_CONFIG = {
    "models": {
        "CMCC-35": "#1f77b4",
        "DWD-mon-21": "#ff7f0e", 
        "ECMWF-51-mon": "#2ca02c",
        "Meteo-France-8": "#d62728",
        "JMA-3-mon": "#9467bd",
        "NCEP-2": "#8c564b",
        "UKMO-14": "#e377c2",
        "ECCC-Canada-3": "#7f7f7f"
    },
    "lead_times": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "obs_color": "black",
    "obs_alpha": 0.7
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_mode": "w",
    "handlers": ["file", "console"]
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    "create_dirs": True,
    "overwrite": False,
    "compression": "gzip",
    "encoding": "utf-8"
}

# ==================== 配置获取函数 ====================
def get_eof_config() -> Dict[str, Any]:
    """获取EOF分析配置"""
    return EOF_CONFIG.copy()

def get_common_eof_config() -> Dict[str, Any]:
    """获取Common EOF分析配置"""
    return COMMON_EOF_CONFIG.copy()

def get_correlation_config() -> Dict[str, Any]:
    """获取相关性分析配置"""
    return CORRELATION_CONFIG.copy()

def get_rmse_config() -> Dict[str, Any]:
    """获取RMSE分析配置"""
    return RMSE_CONFIG.copy()

def get_spectrum_config() -> Dict[str, Any]:
    """获取功率谱分析配置"""
    return SPECTRUM_CONFIG.copy()

def get_plotting_config() -> Dict[str, Any]:
    """获取绘图配置"""
    return PLOT_CONFIG.copy()

def get_crpss_config() -> Dict[str, Any]:
    """获取CRPSS分析配置"""
    return CRPSS_CONFIG.copy()

def get_model_names() -> List[str]:
    """获取模型名称列表"""
    return list(MODELS.keys())

def get_variable_names() -> List[str]:
    """获取变量名称列表"""
    return list(VAR_CONFIG.keys())

def get_lead_times() -> List[int]:
    """获取提前期列表"""
    return LEADTIMES.copy()

def get_spatial_bounds() -> Dict[str, tuple]:
    """获取空间边界"""
    return SPATIAL_BOUNDS.copy()

def get_var_config(var_type: str) -> Dict[str, Any]:
    """获取变量配置"""
    return VAR_CONFIG.get(var_type, {}).copy()

def get_model_config(model_name: str) -> Dict[str, str]:
    """获取模型配置"""
    return MODELS.get(model_name, {}).copy()

def setup_logging(log_file: str = None, level: int = None) -> logging.Logger:
    """设置日志配置"""
    if level is None:
        level = LOGGING_CONFIG["level"]
    
    if log_file is None:
        log_file = "climate_analysis.log"
    
    logging.basicConfig(
        level=level,
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(log_file, mode=LOGGING_CONFIG["file_mode"]),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_output_dirs(base_dir: str = None) -> Dict[str, Path]:
    """创建输出目录结构"""
    if base_dir is None:
        base_dir = OUTPUT_DIR
    
    base_path = Path(base_dir)
    dirs = {
        "base": base_path,
        "eof": base_path / "eof_analysis",
        "correlation": base_path / "spatial_corr",
        "rmse": base_path / "spatial_rmse",
        "spectrum": base_path / "spectrum_analysis",
        "plots": base_path / "plots",
        "common_eof": base_path / "common_eofs",
        "taylor": base_path / "taylor",
        "timeseries": base_path / "timeseries"
    }
    
    if OUTPUT_CONFIG["create_dirs"]:
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

# ==================== 环境检查 ====================
def check_environment() -> bool:
    """检查运行环境"""
    required_dirs = [OBS_DIR, FORECAST_DIR]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"警告: 以下目录不存在: {missing_dirs}")
        return False
    
    return True

def validate_config() -> bool:
    """验证配置参数"""
    # 检查模型配置
    for model, config in MODELS.items():
        if not isinstance(config, dict) or "pl" not in config or "sfc" not in config:
            print(f"错误: 模型 {model} 配置无效")
            return False
    
    # 检查变量配置
    for var_type, config in VAR_CONFIG.items():
        required_keys = ["file_type", "obs_names", "fcst_names", "unit"]
        for key in required_keys:
            if key not in config:
                print(f"错误: 变量 {var_type} 缺少配置项 {key}")
                return False
    
    return True

# 初始化时验证配置
if not validate_config():
    raise ValueError("配置验证失败，请检查配置文件")

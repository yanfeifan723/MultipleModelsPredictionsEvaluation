"""
MMPE 通用配置
统一模型列表、文件后缀、提前期等常量，避免各脚本重复定义。
"""

MODEL_LIST = [
    "CMCC-35",
    "DWD-mon-21",
    "ECMWF-51-mon",
    "Meteo-France-8",
    "NCEP-2",
    "UKMO-14",
    "ECCC-Canada-3",
]

MODEL_FILE_MAP = {
    "CMCC-35": {"pl": "cmcc.35", "sfc": "cmcc.35.sfc"},
    "DWD-mon-21": {"pl": "dwd.21", "sfc": "dwd.sfc.21"},
    "ECMWF-51-mon": {"pl": "ecmwf.51", "sfc": "ecmwf.51.sfc"},
    "Meteo-France-8": {"pl": "meteo_france.8", "sfc": "meteo_france.sfc.8"},
    "NCEP-2": {"pl": "ncep.2", "sfc": "ncep.2.sfc"},
    "UKMO-14": {"pl": "ukmo.14", "sfc": "ukmo.sfc.14"},
    "ECCC-Canada-3": {"pl": "eccc.3", "sfc": "eccc.sfc.3"},
}

LEADTIMES = [0, 1, 2, 3, 4, 5]
CLIMATOLOGY_PERIOD = "1993-2020"
SPATIAL_BOUNDS = {"lat": (15.0, 55.0), "lon": (70.0, 140.0)}

# 季节配置
SEASONS = {
    'DJF': [12, 1, 2],   # 冬季
    'MAM': [3, 4, 5],    # 春季
    'JJA': [6, 7, 8],    # 夏季
    'SON': [9, 10, 11]   # 秋季
}

# 内存与并行控制
DEFAULT_TIME_CHUNK = 24  # 时间维分块大小（月）
MAX_WORKERS_TEMP = 12     # 温度最大并发进程数
MAX_WORKERS_PREC = 8     # 降水最大并发进程数（更严格）
HARD_WORKER_CAP = 16      # 强制上限，防止外部传入过大并行度

# 异常值处理配置
REMOVE_OUTLIERS = True    # 是否去除异常值
OUTLIER_METHOD = 'IQR'    # 异常值检测方法：'IQR', 'Z-score', 'Modified_Z-score'
OUTLIER_THRESHOLD = 1.5   # IQR方法的阈值倍数

# 颜色配置
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 数据路径配置
DATA_PATHS = {
    "obs_dir": "/sas12t1/ffyan/obs",
    "forecast_dir": "/raid62/EC-C3S/month",
    "output_dir": "/sas12t1/ffyan/output"
}

# 变量名配置（基础变量名列表，各脚本可扩展）
VAR_NAMES = {
    "temp": {
        "obs_names": ["t", "t2m", "temp", "temperature", "tas", "tm"],
        "fcst_names": ["t", "t2m", "temp", "temperature", "tas", "tm"],
    },
    "prec": {
        "obs_names": ["tp", "prec", "pr", "precip", "tprate", "pre"],
        "fcst_names": ["tp", "tprate", "prec", "pr", "precip", "pre"],
    }
}


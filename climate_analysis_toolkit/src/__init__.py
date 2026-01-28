"""
Climate Analysis Toolkit (气候分析工具包)

一个用于气候数据分析和可视化的综合Python工具包
"""

from .config.settings import *
from .core import *
from .utils import *
from .plotting import *

__version__ = "0.1.0"
__author__ = "Climate Analysis Team"
__email__ = "climate@example.com"

# 主要类
from .analyzer import ClimateAnalyzer

__all__ = [
    'ClimateAnalyzer',
    'setup_logging',
    'create_output_dirs',
    'check_environment',
    'get_model_names',
    'get_variable_names',
    'get_lead_times',
    'get_spatial_bounds',
    'get_var_config',
    'get_model_config'
]

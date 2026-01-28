"""
统一的命令行参数解析模块
为MMPE脚本提供标准化的命令行参数定义
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# 动态导入common_config（因为它在MMPE目录下）
def _get_common_config():
    """获取common_config模块"""
    try:
        # 尝试从MMPE目录导入
        mmpe_path = Path(__file__).parent.parent.parent.parent / "MMPE"
        if mmpe_path.exists():
            sys.path.insert(0, str(mmpe_path.parent))
            from MMPE.common_config import MODEL_LIST, LEADTIMES
            return MODEL_LIST, LEADTIMES
    except ImportError:
        pass
    
    # 如果导入失败，使用默认值
    return [
        "CMCC-35", "DWD-mon-21", "ECMWF-51-mon", "Meteo-France-8",
        "NCEP-2", "UKMO-14", "ECCC-Canada-3"
    ], [0, 1, 2, 3, 4, 5]

# 获取默认值
_DEFAULT_MODEL_LIST, _DEFAULT_LEADTIMES = _get_common_config()


def add_common_args(parser: argparse.ArgumentParser, 
                   models_default: str = 'all',
                   leadtimes_default: str = 'all',
                   var_default: List[str] = None,
                   var_required: bool = False) -> argparse.ArgumentParser:
    """
    添加通用命令行参数（所有MMPE脚本都支持）
    
    Args:
        parser: ArgumentParser实例
        models_default: 模型参数的默认值（'all' 或具体模型列表）
        leadtimes_default: 提前期参数的默认值（'all' 或具体提前期列表）
        var_default: 变量参数的默认值（None表示不设置默认值）
        var_required: 变量参数是否必需
        
    Returns:
        配置好的ArgumentParser
    """
    if var_default is None:
        var_default = ['temp', 'prec']
    
    # 模型列表
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=[models_default] if isinstance(models_default, str) else models_default,
        help='要分析的模型列表，使用"all"表示所有模型'
    )
    
    # 提前期列表（支持字符串或整数）
    parser.add_argument(
        '--leadtimes',
        type=str,
        nargs='+',
        default=[leadtimes_default] if isinstance(leadtimes_default, str) else leadtimes_default,
        help='要分析的提前期列表，使用"all"表示所有提前期，或指定具体数值'
    )
    
    # 变量类型（某些脚本可能不需要或允许None）
    var_kwargs = {
        'type': str,
        'nargs': '+' if not var_required else '+',
        'choices': ['temp', 'prec'],
        'help': '变量类型，可以指定多个变量'
    }
    if var_default is not None:
        var_kwargs['default'] = var_default
    if var_required:
        var_kwargs['required'] = True
    
    parser.add_argument('--var', **var_kwargs)
    
    # 并行处理
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=None,  # None表示未指定，由脚本决定默认值
        help='启用并行处理'
    )
    
    # 禁用并行处理（与--parallel互斥）
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='禁用并行处理（与--parallel互斥）'
    )
    
    # 并行作业数
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,  # None表示未指定，由脚本决定默认值
        help='并行作业数（默认根据系统自动调整）'
    )
    
    # 绘图选项
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='仅绘图模式（基于已有数据，不重新计算）'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='仅计算，不绘制图表（默认会自动绘图）'
    )
    
    return parser


def add_season_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加季节相关参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--seasons',
        type=str,
        nargs='+',
        choices=['DJF', 'MAM', 'JJA', 'SON', 'annual', 'all'],
        help='指定季节列表，包括annual表示年平均，all表示所有季节'
    )
    
    parser.add_argument(
        '--all-seasons',
        action='store_true',
        help='处理所有季节（DJF, MAM, JJA, SON）和年平均'
    )
    
    return parser


def add_outlier_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加异常值处理相关参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--no-outliers',
        action='store_true',
        help='禁用异常值去除（默认启用异常值去除）'
    )
    
    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=1.5,
        help='异常值检测阈值倍数（默认1.5）'
    )
    
    return parser


def add_bootstrap_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加Bootstrap相关参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--block-size',
        type=int,
        default=36,
        help='Block 大小（月数，默认36）'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Bootstrap 重采样次数（默认100）'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='置信水平（默认0.95）'
    )
    
    parser.add_argument(
        '--accuracy-delta',
        type=float,
        default=0.015,
        help='Accuracy 相对误差阈值（默认 0.015 表示 ±1.5%）'
    )
    
    parser.add_argument(
        '--accuracy-eps',
        type=float,
        default=1e-6,
        help='Accuracy 计算中防止除零的小常数（默认1e-6）'
    )
    
    return parser


def add_parallel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加高级并行处理参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--parallel-backend',
        type=str,
        default='process',
        choices=['process', 'thread'],
        help='并行后端: process 或 thread（默认: process）'
    )
    
    parser.add_argument(
        '--parallel-strategy',
        choices=['auto', 'standard', 'chunked'],
        default='auto',
        help='并行策略: auto(自动选择), standard(标准), chunked(分块)（默认: auto）'
    )
    
    return parser


def add_gpu_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加GPU相关参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='强制启用GPU加速计算（如果可用）'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='强制禁用GPU，仅使用CPU计算'
    )
    
    parser.add_argument(
        '--gpu-memory-limit',
        type=float,
        default=0.8,
        help='GPU内存使用限制（比例，默认0.8表示80%）'
    )
    
    return parser


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加数据处理相关参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存数据到NetCDF（仅计算和绘图）'
    )
    
    return parser


def add_acc_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加ACC分析特定参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--no-calc-ratio',
        action='store_true',
        help='不计算ratio数据（ACC和IC，用于空间分布图、散点图和柱状图），但如果数据已存在仍会绘图'
    )
    
    parser.add_argument(
        '--no-calc-timeseries',
        action='store_true',
        help='不计算年度距平空间相关系数（用于时间序列图），但如果数据已存在仍会绘图'
    )
    
    parser.add_argument(
        '--use-spatial-mean',
        action='store_true',
        help='使用先时间平均再空间相关方法（方法1）绘制时间序列图，默认使用方法2（时间-空间综合）'
    )
    
    return parser


def add_pearson_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加Pearson相关分析特定参数
    
    Args:
        parser: ArgumentParser实例
        
    Returns:
        配置好的ArgumentParser
    """
    parser.add_argument(
        '--no-anomaly-seasonal',
        action='store_true',
        help='季节相关不去气候态（默认去气候态）'
    )
    
    parser.add_argument(
        '--no-anomaly-monthly',
        action='store_true',
        help='月度相关不去气候态（默认去气候态）'
    )
    
    return parser


def create_parser(
    description: str,
    include_seasons: bool = False,
    include_outliers: bool = False,
    include_bootstrap: bool = False,
    include_parallel_advanced: bool = False,
    include_gpu: bool = False,
    include_data: bool = False,
    include_acc_specific: bool = False,
    include_pearson: bool = False,
    models_default: str = 'all',
    leadtimes_default: str = 'all',
    var_default: List[str] = None,
    var_required: bool = False,
    formatter_class: type = argparse.RawDescriptionHelpFormatter
) -> argparse.ArgumentParser:
    """
    创建并配置ArgumentParser
    
    Args:
        description: 脚本描述
        include_seasons: 是否包含季节参数
        include_outliers: 是否包含异常值处理参数
        include_bootstrap: 是否包含Bootstrap参数
        include_parallel_advanced: 是否包含高级并行参数
        include_gpu: 是否包含GPU参数
        include_data: 是否包含数据处理参数
        include_acc_specific: 是否包含ACC特定参数
        include_pearson: 是否包含Pearson特定参数
        formatter_class: 格式化器类
        
    Returns:
        配置好的ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=formatter_class
    )
    
    # 所有脚本都包含通用参数
    parser = add_common_args(
        parser,
        models_default=models_default,
        leadtimes_default=leadtimes_default,
        var_default=var_default,
        var_required=var_required
    )
    
    # 可选参数组
    if include_seasons:
        parser = add_season_args(parser)
    if include_outliers:
        parser = add_outlier_args(parser)
    if include_bootstrap:
        parser = add_bootstrap_args(parser)
    if include_parallel_advanced:
        parser = add_parallel_args(parser)
    if include_gpu:
        parser = add_gpu_args(parser)
    if include_data:
        parser = add_data_args(parser)
    if include_acc_specific:
        parser = add_acc_specific_args(parser)
    if include_pearson:
        parser = add_pearson_args(parser)
    
    return parser


def parse_models(models_arg: List[str], model_list: List[str] = None) -> List[str]:
    """
    解析模型列表参数
    
    Args:
        models_arg: 命令行传入的模型参数
        model_list: 可用的模型列表（默认使用common_config中的MODEL_LIST）
        
    Returns:
        解析后的模型列表
    """
    if model_list is None:
        model_list = _DEFAULT_MODEL_LIST
    
    if 'all' in models_arg:
        return model_list
    else:
        # 过滤出有效的模型
        return [m for m in models_arg if m in model_list]


def parse_leadtimes(leadtimes_arg, leadtimes_list: List[int] = None) -> List[int]:
    """
    解析提前期列表参数
    
    Args:
        leadtimes_arg: 命令行传入的提前期参数（字符串列表或整数列表）
        leadtimes_list: 可用的提前期列表（默认使用common_config中的LEADTIMES）
        
    Returns:
        解析后的提前期列表
    """
    if leadtimes_list is None:
        leadtimes_list = _DEFAULT_LEADTIMES
    
    # 处理None或空列表
    if not leadtimes_arg:
        return leadtimes_list
    
    # 如果已经是整数列表，直接返回
    if isinstance(leadtimes_arg[0], int):
        return leadtimes_arg
    
    # 处理字符串列表
    if 'all' in [str(lt).lower() for lt in leadtimes_arg]:
        return leadtimes_list
    else:
        # 尝试转换为整数
        try:
            return [int(lt) for lt in leadtimes_arg]
        except (ValueError, TypeError):
            # 如果转换失败，返回默认列表
            return leadtimes_list


def parse_vars(vars_arg: List[str]) -> List[str]:
    """
    解析变量类型参数
    
    Args:
        vars_arg: 命令行传入的变量参数
        
    Returns:
        解析后的变量列表
    """
    if not vars_arg:
        return ['temp', 'prec']
    return vars_arg


def normalize_plot_args(args) -> None:
    """
    标准化绘图参数（保留此函数以保持接口一致性，未来可能需要扩展）
    
    Args:
        args: 解析后的命令行参数对象
    """
    # 目前不需要特殊处理，保留函数以保持接口一致性
    pass


def normalize_parallel_args(args) -> bool:
    """
    标准化并行参数，处理None值
    
    Args:
        args: 解析后的命令行参数对象
        
    Returns:
        处理后的parallel值（bool）
    """
    if hasattr(args, 'parallel'):
        return args.parallel if args.parallel is not None else False
    return False


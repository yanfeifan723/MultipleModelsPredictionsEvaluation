"""
输出配置模块
定义统一的输出目录结构和文件命名规范
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# 基础输出目录
BASE_OUTPUT_DIR = Path("/sas12t1/ffyan")
DATA_OUTPUT_DIR = BASE_OUTPUT_DIR / "outputdata"
PLOT_OUTPUT_DIR = BASE_OUTPUT_DIR / "outputplot"

# 计算方法分类
CALCULATION_METHODS = {
    "eof": "EOF分析",
    "common_eof": "共同EOF分析", 
    "crpss": "CRPSS分析",
    "crps": "CRPS分析",
    "crps_spatial": "空间CRPS分析",
    "crps_temporal": "时间CRPS分析",
    "crps_temporal_raw": "未平滑时间CRPS分析",
    "spatial_correlation": "空间场相关时间序列",
    "spatial_crps": "空间场CRPS时间序列",
    "spatial_rmse": "空间场RMSE时间序列",
    "bias": "Bias分析",
    "bias_spatial": "空间Bias分析",
    "bias_temporal": "时间Bias分析",
    "bias_temporal_raw": "未平滑时间Bias分析",
    "spatial_bias": "空间场Bias时间序列",
    "rmse": "RMSE分析",
    "correlation": "相关性分析",
    "spectrum": "频谱分析",
    "taylor": "Taylor分析",
    "downsampling": "数据降采样",
    "pearson": "Pearson相关分析",
    "auto": "智能绘图", # Added for auto_plot examples
    "rmse_spatial": "空间RMSE分析",
    "rmse_temporal": "时间RMSE分析",
    "nc_combined": "综合NetCDF分析"
}

# 绘制方法分类
PLOTTING_METHODS = {
    "spatial": "空间分布图",
    "timeseries": "时间序列图",
    "boxplot": "箱线图",
    "heatmap": "热力图",
    "scatter": "散点图",
    "histogram": "直方图",
    "eof_modes": "EOF模态图",
    "eof_modes_obs_monthly": "观测月度EOF模态图",
    "eof_modes_obs_DJF": "观测冬季EOF模态图",
    "eof_modes_obs_MAM": "观测春季EOF模态图",
    "eof_modes_obs_JJA": "观测夏季EOF模态图",
    "eof_modes_obs_SON": "观测秋季EOF模态图",
    "eof_modes_monthly": "模型月度EOF模态图",
    "eof_modes_DJF": "模型冬季EOF模态图",
    "eof_modes_MAM": "模型春季EOF模态图",
    "eof_modes_JJA": "模型夏季EOF模态图",
    "eof_modes_SON": "模型秋季EOF模态图",
    "pc_timeseries": "主成分时间序列",
    "variance": "方差解释图",
    "ensemble_pc_timeseries": "集合主成分时间序列",
    "ensemble_eof_comparison": "集合EOF模态对比",
    "ensemble_variance_comparison": "集合方差解释对比",
    "taylor_diagram": "Taylor图",
    "spectrum": "频谱图",
    "comparison": "对比图",
    "summary": "综合分析图",
    "eigenvalues": "特征值图",
    "correlation_matrix": "相关矩阵图",
    "seasonal": "季节性分析图",
    "combined": "综合图表"
}

# 变量类型
VARIABLE_TYPES = {
    "temp": "温度",
    "prec": "降水",
    "pressure": "气压",
    "wind": "风场",
    "humidity": "湿度"
}

# 模型名称映射
MODEL_NAMES = {
    "CMCC-35": "CMCC-35",
    "DWD-mon-21": "DWD-21", 
    "ECMWF-51-mon": "ECMWF-51",
    "Meteo-France-8": "Meteo-France-8",
    "JMA-3-mon": "JMA-3",
    "NCEP-2": "NCEP-2",
    "UKMO-14": "UKMO-14",
    "ECCC-Canada-3": "ECCC-Canada-3",
    "OBS": "观测数据"
}

def create_output_directories():
    """创建所有输出目录 - 已废弃，改为按需创建"""
    logger.warning("create_output_directories() 已废弃，文件夹将在保存文件时按需创建")
    pass

def get_data_output_path(calc_method: str, var_type: str, filename: str) -> Path:
    """
    获取数据输出路径
    
    Args:
        calc_method: 计算方法
        var_type: 变量类型
        filename: 文件名
        
    Returns:
        完整的数据输出路径
    """
    if calc_method not in CALCULATION_METHODS:
        raise ValueError(f"未知的计算方法: {calc_method}")
    if var_type not in VARIABLE_TYPES:
        raise ValueError(f"未知的变量类型: {var_type}")
    
    output_path = DATA_OUTPUT_DIR / calc_method / var_type / filename
    return output_path

def get_plot_output_path(calc_method: str, var_type: str, plot_method: str, 
                        filename: str) -> Path:
    """
    获取图像输出路径
    
    Args:
        calc_method: 计算方法
        var_type: 变量类型
        plot_method: 绘制方法
        filename: 文件名
        
    Returns:
        完整的图像输出路径
    """
    if calc_method not in CALCULATION_METHODS:
        raise ValueError(f"未知的计算方法: {calc_method}")
    if var_type not in VARIABLE_TYPES:
        raise ValueError(f"未知的变量类型: {var_type}")
    if plot_method not in PLOTTING_METHODS:
        raise ValueError(f"未知的绘制方法: {plot_method}")
    
    output_path = PLOT_OUTPUT_DIR / calc_method / var_type / plot_method / filename
    return output_path

def get_standard_filename(calc_method: str, var_type: str, plot_method: str = None,
                         model: str = None, leadtime: int = None, 
                         mode: int = None, suffix: str = "png") -> str:
    """
    生成标准文件名
    
    Args:
        calc_method: 计算方法
        var_type: 变量类型
        plot_method: 绘制方法
        model: 模型名称
        leadtime: 预报时效
        mode: 模态编号
        suffix: 文件后缀
        
    Returns:
        标准文件名
    """
    parts = [calc_method, var_type]
    
    if plot_method:
        parts.append(plot_method)
    
    if model:
        # 清理模型名称
        clean_model = MODEL_NAMES.get(model, model.replace('-mon-', '-').replace('-mon', ''))
        parts.append(clean_model)
    
    if leadtime is not None:
        parts.append(f"lead{leadtime}")
    
    if mode is not None:
        parts.append(f"mode{mode}")
    
    filename = "_".join(parts) + f".{suffix}"
    return filename

def get_eof_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取EOF数据输出路径"""
    filename = get_standard_filename("eof", var_type, model=model, leadtime=leadtime, suffix="pkl")
    return get_data_output_path("eof", var_type, filename)

def get_eof_plot_path(var_type: str, plot_method: str, model: str = None, 
                     leadtime: int = None, mode: int = None) -> Path:
    """获取EOF图像输出路径"""
    filename = get_standard_filename("eof", var_type, plot_method, model, leadtime, mode)
    return get_plot_output_path("eof", var_type, plot_method, filename)

def get_common_eof_data_path(var_type: str, leadtime: int = None) -> Path:
    """获取共同EOF数据输出路径"""
    filename = get_standard_filename("common_eof", var_type, leadtime=leadtime, suffix="pkl")
    return get_data_output_path("common_eof", var_type, filename)

def get_common_eof_plot_path(var_type: str, plot_method: str, 
                           leadtime: int = None, mode: int = None) -> Path:
    """获取共同EOF图像输出路径"""
    filename = get_standard_filename("common_eof", var_type, plot_method, 
                                   leadtime=leadtime, mode=mode)
    return get_plot_output_path("common_eof", var_type, plot_method, filename)

def get_crpss_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取CRPSS数据输出路径"""
    filename = get_standard_filename("crpss", var_type, model=model, leadtime=leadtime, suffix="csv")
    return get_data_output_path("crpss", var_type, filename)

def get_crpss_plot_path(var_type: str, plot_method: str, model: str = None, 
                       leadtime: int = None) -> Path:
    """获取CRPSS图像输出路径"""
    filename = get_standard_filename("crpss", var_type, plot_method, model, leadtime)
    return get_plot_output_path("crpss", var_type, plot_method, filename)

def get_rmse_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取RMSE数据输出路径"""
    filename = get_standard_filename("rmse", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("rmse", var_type, filename)

def get_rmse_plot_path(var_type: str, plot_method: str, model: str = None, 
                      leadtime: int = None) -> Path:
    """获取RMSE图像输出路径"""
    filename = get_standard_filename("rmse", var_type, plot_method, model, leadtime)
    return get_plot_output_path("rmse", var_type, plot_method, filename)

def get_correlation_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取相关性数据输出路径"""
    filename = get_standard_filename("correlation", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("correlation", var_type, filename)

def get_correlation_plot_path(var_type: str, plot_method: str, model: str = None, 
                            leadtime: int = None) -> Path:
    """获取相关性图像输出路径"""
    filename = get_standard_filename("correlation", var_type, plot_method, model, leadtime)
    return get_plot_output_path("correlation", var_type, plot_method, filename)

def get_spectrum_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取频谱数据输出路径"""
    filename = get_standard_filename("spectrum", var_type, model=model, leadtime=leadtime, suffix="pkl")
    return get_data_output_path("spectrum", var_type, filename)

def get_spectrum_plot_path(var_type: str, plot_method: str, model: str = None, 
                          leadtime: int = None) -> Path:
    """获取频谱图像输出路径"""
    filename = get_standard_filename("spectrum", var_type, plot_method, model, leadtime)
    return get_plot_output_path("spectrum", var_type, plot_method, filename)

def get_taylor_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取Taylor数据输出路径"""
    filename = get_standard_filename("taylor", var_type, model=model, leadtime=leadtime, suffix="pkl")
    return get_data_output_path("taylor", var_type, filename)

def get_taylor_plot_path(var_type: str, plot_method: str, model: str = None, 
                        leadtime: int = None) -> Path:
    """获取Taylor图像输出路径"""
    filename = get_standard_filename("taylor", var_type, plot_method, model, leadtime)
    return get_plot_output_path("taylor", var_type, plot_method, filename)

def get_rmse_spatial_data_path(var_type: str, model: str = None, leadtime: int = None, season: str = None) -> Path:
    """获取空间RMSE数据输出路径"""
    if season:
        filename = get_standard_filename("rmse_spatial", var_type, model=model, leadtime=leadtime, suffix="nc")
        # 在文件名中插入季节信息
        if season == 'annual':
            filename = filename.replace(".nc", "_annual.nc")
        else:
            filename = filename.replace(".nc", f"_{season}.nc")
    else:
        filename = get_standard_filename("rmse_spatial", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("rmse_spatial", var_type, filename)

def get_rmse_spatial_plot_path(var_type: str, plot_method: str, model: str = None, 
                              leadtime: int = None) -> Path:
    """获取空间RMSE图像输出路径"""
    filename = get_standard_filename("rmse_spatial", var_type, plot_method, model, leadtime)
    return get_plot_output_path("rmse_spatial", var_type, plot_method, filename)

def get_rmse_temporal_data_path(var_type: str, model: str = None, leadtime: int = None, season: str = None) -> Path:
    """获取时间RMSE数据输出路径"""
    if season:
        filename = get_standard_filename("rmse_temporal", var_type, model=model, leadtime=leadtime, suffix="csv")
        # 在文件名中插入季节信息
        if season == 'annual':
            filename = filename.replace(".csv", "_annual.csv")
        else:
            filename = filename.replace(".csv", f"_{season}.csv")
    else:
        filename = get_standard_filename("rmse_temporal", var_type, model=model, leadtime=leadtime, suffix="csv")
    return get_data_output_path("rmse_temporal", var_type, filename)

def get_rmse_temporal_plot_path(var_type: str, plot_method: str, model: str = None, 
                               leadtime: int = None) -> Path:
    """获取时间RMSE图像输出路径"""
    filename = get_standard_filename("rmse_temporal", var_type, plot_method, model, leadtime)
    return get_plot_output_path("rmse_temporal", var_type, plot_method, filename)

def get_pearson_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取Pearson相关数据输出路径"""
    filename = get_standard_filename("pearson", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("pearson", var_type, filename)

def get_pearson_plot_path(var_type: str, plot_method: str, model: str = None, 
                         leadtime: int = None) -> Path:
    """获取Pearson相关图像输出路径"""
    filename = get_standard_filename("pearson", var_type, plot_method, model, leadtime)
    return get_plot_output_path("pearson", var_type, plot_method, filename)


def get_crps_spatial_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取空间CRPS数据输出路径"""
    filename = get_standard_filename("crps_spatial", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("crps_spatial", var_type, filename)

def get_crps_temporal_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取时间CRPS数据输出路径（12个月滑动平均）"""
    filename = get_standard_filename("crps_temporal", var_type, model=model, leadtime=leadtime, suffix="csv")
    return get_data_output_path("crps_temporal", var_type, filename)

def get_crps_temporal_raw_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取未平滑（无滚动）时间CRPS数据输出路径"""
    filename = get_standard_filename("crps_temporal_raw", var_type, model=model, leadtime=leadtime, suffix="csv")
    return get_data_output_path("crps_temporal_raw", var_type, filename)


# ---- 空间场时间序列结果路径（逐时间点对全空间计算统计量） ----
def get_spatial_correlation_ts_path(var_type: str, model: str, leadtime: int) -> Path:
    """获取空间场相关时间序列输出路径: spatial_corr_{model}_L{leadtime}.nc"""
    filename = f"spatial_corr_{model}_L{leadtime}.nc"
    return get_data_output_path("spatial_correlation", var_type, filename)

def get_spatial_crps_ts_path(var_type: str, model: str, leadtime: int) -> Path:
    """获取空间场CRPS时间序列输出路径: spatial_crps_{model}_L{leadtime}.nc"""
    filename = f"spatial_crps_{model}_L{leadtime}.nc"
    return get_data_output_path("spatial_crps", var_type, filename)

def get_spatial_rmse_ts_path(var_type: str, model: str, leadtime: int) -> Path:
    """获取空间场RMSE时间序列输出路径: spatial_rmse_{model}_L{leadtime}.nc"""
    filename = f"spatial_rmse_{model}_L{leadtime}.nc"
    return get_data_output_path("spatial_rmse", var_type, filename)


# ---- 聚合结果（年/季/月）通用路径 ----
def get_summary_dir(calc_method: str, var_type: str) -> Path:
    """各分析方法的汇总输出目录，如 /sas12t1/ffyan/outputdata/{calc_method}_summary/{var_type}"""
    return DATA_OUTPUT_DIR / f"{calc_method}_summary" / var_type

def get_aggregated_csv_path(calc_method: str, var_type: str, leadtime: int, agg_level: str, prefix: str = None) -> Path:
    """聚合CSV路径：{prefix or calc_method}_{agg_level}_L{lead}.csv"""
    fname_prefix = prefix or calc_method
    filename = f"{fname_prefix}_{agg_level}_L{leadtime}.csv"
    return get_summary_dir(calc_method, var_type) / filename


# ---- Bias 输出路径 ----
def get_bias_spatial_data_path(var_type: str, model: str = None, leadtime: int = None, season: str = None) -> Path:
    """获取空间Bias数据输出路径"""
    filename = get_standard_filename("bias_spatial", var_type, model=model, leadtime=leadtime, suffix="nc")
    if season:
        if season == 'annual':
            filename = filename.replace(".nc", "_annual.nc")
        else:
            filename = filename.replace(".nc", f"_{season}.nc")
    return get_data_output_path("bias_spatial", var_type, filename)

def get_bias_temporal_data_path(var_type: str, model: str = None, leadtime: int = None, season: str = None) -> Path:
    """获取时间Bias数据输出路径（12个月滑动平均）"""
    filename = get_standard_filename("bias_temporal", var_type, model=model, leadtime=leadtime, suffix="csv")
    if season:
        if season == 'annual':
            filename = filename.replace(".csv", "_annual.csv")
        else:
            filename = filename.replace(".csv", f"_{season}.csv")
    return get_data_output_path("bias_temporal", var_type, filename)

def get_bias_temporal_raw_data_path(var_type: str, model: str = None, leadtime: int = None, season: str = None) -> Path:
    """获取未平滑的时间Bias数据输出路径（逐月）"""
    filename = get_standard_filename("bias_temporal_raw", var_type, model=model, leadtime=leadtime, suffix="csv")
    if season:
        if season == 'annual':
            filename = filename.replace(".csv", "_annual.csv")
        else:
            filename = filename.replace(".csv", f"_{season}.csv")
    return get_data_output_path("bias_temporal_raw", var_type, filename)

def get_spatial_bias_ts_path(var_type: str, model: str, leadtime: int) -> Path:
    """获取空间场Bias时间序列输出路径: spatial_bias_{model}_L{leadtime}.nc"""
    filename = f"spatial_bias_{model}_L{leadtime}.nc"
    return get_data_output_path("spatial_bias", var_type, filename)

def get_downsampling_data_path(var_type: str, resolution: str = None) -> Path:
    """获取降采样数据输出路径"""
    filename = get_standard_filename("downsampling", var_type, suffix="nc")
    if resolution:
        filename = filename.replace(".nc", f"_{resolution}.nc")
    return get_data_output_path("downsampling", var_type, filename)

def get_nc_combined_data_path(var_type: str, model: str = None, leadtime: int = None) -> Path:
    """获取综合NetCDF数据输出路径"""
    filename = get_standard_filename("nc_combined", var_type, model=model, leadtime=leadtime, suffix="nc")
    return get_data_output_path("nc_combined", var_type, filename)

def get_nc_combined_plot_path(var_type: str, plot_method: str, model: str = None, 
                             leadtime: int = None) -> Path:
    """获取综合NetCDF图像输出路径"""
    filename = get_standard_filename("nc_combined", var_type, plot_method, model, leadtime)
    return get_plot_output_path("nc_combined", var_type, plot_method, filename)

def list_output_files(calc_method: str = None, var_type: str = None, 
                     plot_method: str = None) -> List[Path]:
    """
    列出输出文件
    
    Args:
        calc_method: 计算方法（可选）
        var_type: 变量类型（可选）
        plot_method: 绘制方法（可选）
        
    Returns:
        文件路径列表
    """
    files = []
    
    if calc_method:
        calc_methods = [calc_method]
    else:
        calc_methods = CALCULATION_METHODS.keys()
    
    if var_type:
        var_types = [var_type]
    else:
        var_types = VARIABLE_TYPES.keys()
    
    for calc in calc_methods:
        for var in var_types:
            # 数据文件
            data_dir = DATA_OUTPUT_DIR / calc / var
            if data_dir.exists():
                files.extend(data_dir.glob("*"))
            
            # 图像文件
            plot_dir = PLOT_OUTPUT_DIR / calc / var
            if plot_dir.exists():
                if plot_method:
                    plot_subdir = plot_dir / plot_method
                    if plot_subdir.exists():
                        files.extend(plot_subdir.glob("*"))
                else:
                    files.extend(plot_dir.rglob("*"))
    
    return files

def cleanup_output_directories():
    """清理输出目录（删除空目录）"""
    for root, dirs, files in os.walk(DATA_OUTPUT_DIR, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.info(f"删除空目录: {dir_path}")
            except OSError:
                pass
    
    for root, dirs, files in os.walk(PLOT_OUTPUT_DIR, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.info(f"删除空目录: {dir_path}")
            except OSError:
                pass

if __name__ == "__main__":
    # 创建输出目录
    create_output_directories()
    print("输出配置模块加载完成")

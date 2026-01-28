"""
统计绘图模块
提供时间序列、箱线图、散点图等统计图表功能
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_timeseries(data: np.ndarray,
                   time: np.ndarray,
                   title: str = "",
                   xlabel: str = "Time",
                   ylabel: str = "Value",
                   figsize: Tuple[float, float] = (12, 6),
                   save_path: Optional[str] = None,
                   **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制时间序列图
    
    Args:
        data: 数据数组
        time: 时间数组
        title: 标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time, data, **kwargs)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_boxplot(data_dict: Dict[str, np.ndarray],
                 title: str = "",
                 xlabel: str = "Groups",
                 ylabel: str = "Value",
                 figsize: Tuple[float, float] = (10, 6),
                 save_path: Optional[str] = None,
                 **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制箱线图
    
    Args:
        data_dict: 数据字典 {name: data}
        title: 标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(data_dict.keys())
    data_list = list(data_dict.values())
    
    ax.boxplot(data_list, labels=labels, **kwargs)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_scatter(x: np.ndarray,
                y: np.ndarray,
                title: str = "",
                xlabel: str = "X",
                ylabel: str = "Y",
                figsize: Tuple[float, float] = (8, 6),
                save_path: Optional[str] = None,
                **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制散点图
    
    Args:
        x: x轴数据
        y: y轴数据
        title: 标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x, y, **kwargs)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_histogram(data: np.ndarray,
                  title: str = "",
                  xlabel: str = "Value",
                  ylabel: str = "Frequency",
                  bins: int = 30,
                  figsize: Tuple[float, float] = (8, 6),
                  save_path: Optional[str] = None,
                  **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制直方图
    
    Args:
        data: 数据数组
        title: 标题
        xlabel: x轴标签
        ylabel: y轴标签
        bins: 直方图箱数
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(data, bins=bins, **kwargs)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_multi_timeseries(data_dict: Dict[str, np.ndarray],
                         time: np.ndarray,
                         title: str = "",
                         xlabel: str = "Time",
                         ylabel: str = "Value",
                         figsize: Tuple[float, float] = (12, 8),
                         save_path: Optional[str] = None,
                         **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制多时间序列图
    
    Args:
        data_dict: 数据字典 {name: data}
        time: 时间数组
        title: 标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    for i, (name, data) in enumerate(data_dict.items()):
        ax.plot(time, data, label=name, color=colors[i], **kwargs)
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_correlation_matrix(correlation_matrix: np.ndarray,
                           labels: Optional[List[str]] = None,
                           title: str = "Correlation Matrix",
                           figsize: Tuple[float, float] = (10, 8),
                           save_path: Optional[str] = None,
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制相关性矩阵热图
    
    Args:
        correlation_matrix: 相关性矩阵
        labels: 标签列表
        title: 标题
        figsize: 图形尺寸
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        图形和坐标轴对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, **kwargs)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # 设置标签
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    # 添加数值标签
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_title(title, fontsize=14, weight='bold')
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_rmse_comparison(rmse_results: Dict[str, Any],
                        var_type: str = "temp",
                        title: str = "RMSE Comparison",
                        save_path: Optional[str] = None,
                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制RMSE比较图
    
    Args:
        rmse_results: RMSE结果字典
        var_type: 变量类型
        title: 图形标题
        save_path: 保存路径
        **kwargs: 其他参数
    
    Returns:
        figure: matplotlib图形对象
        axes: 子图对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取数据
    models = []
    lead_times = []
    rmse_values = []
    
    for key, result in rmse_results.items():
        if key.startswith('lead_'):
            lead_time = int(key.split('_')[1])
            for model, metrics in result.items():
                if isinstance(metrics, dict) and 'rmse' in metrics:
                    models.append(model)
                    lead_times.append(lead_time)
                    rmse_values.append(metrics['rmse'].mean().item())
    
    if not rmse_values:
        logger.warning("没有找到有效的RMSE数据")
        return fig, ax
    
    # 创建数据框
    import pandas as pd
    df = pd.DataFrame({
        'Model': models,
        'Lead Time': lead_times,
        'RMSE': rmse_values
    })
    
    # 绘制箱线图
    df.boxplot(column='RMSE', by='Lead Time', ax=ax)
    
    # 设置标题和标签
    ax.set_title(f"{title} - {var_type.upper()}")
    ax.set_xlabel('Lead Time (months)')
    ax.set_ylabel('RMSE')
    
    # 设置单位
    if var_type == 'temp':
        ax.set_ylabel('RMSE (K)')
    else:
        ax.set_ylabel('RMSE (mm/day)')
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"RMSE比较图已保存到: {save_path}")
    
    return fig, ax

def save_plot(fig: plt.Figure, 
              save_path: str,
              dpi: int = 300,
              bbox_inches: str = 'tight',
              **kwargs) -> None:
    """
    保存图形
    
    Args:
        fig: 图形对象
        save_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界设置
        **kwargs: 其他参数
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    logger.info(f"图形已保存到: {save_path}")
    plt.close(fig)

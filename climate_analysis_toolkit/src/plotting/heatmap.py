"""
Heatmap Plotting Module
Provides general heatmap visualization functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Tuple, Union, Dict
import logging
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Setup plotting style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def save_plot(fig: plt.Figure, save_path, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save figure"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Figure saved to: {save_path}")

def plot_heatmap(data: Union[pd.DataFrame, np.ndarray],
                row_labels: Optional[List[str]] = None,
                col_labels: Optional[List[str]] = None,
                title: str = "Heatmap",
                xlabel: str = "X",
                ylabel: str = "Y",
                cmap: Optional[str] = None,
                center: Optional[float] = None,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                annot: Optional[bool] = None,
                fmt: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                save_path: Optional[str] = None,
                auto_adjust: bool = True,
                **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a general heatmap
    
    Args:
        data: Input data (DataFrame or numpy array)
        row_labels: Row labels
        col_labels: Column labels
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Color map
        center: Center value for color scaling
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        annot: Whether to show annotations
        fmt: Format string for annotations
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for seaborn.heatmap
        
    Returns:
        Figure and axes objects
    """
    setup_plot_style()
    
    # Convert numpy array to DataFrame if needed
    if isinstance(data, np.ndarray):
        if row_labels is None:
            row_labels = [f'Row_{i}' for i in range(data.shape[0])]
        if col_labels is None:
            col_labels = [f'Col_{i}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, index=row_labels, columns=col_labels)
    
    # Auto-adjust parameters based on data characteristics
    if auto_adjust:
        # Auto-determine figure size based on data dimensions
        if figsize is None:
            n_rows, n_cols = data.shape
            if n_rows <= 5 and n_cols <= 5:
                figsize = (8, 6)
            elif n_rows <= 10 and n_cols <= 10:
                figsize = (10, 8)
            else:
                figsize = (max(12, n_cols * 0.8), max(8, n_rows * 0.6))
        
        # Auto-determine color map based on data range
        if cmap is None:
            data_range = data.values.max() - data.values.min()
            if data_range == 0:
                cmap = 'viridis'  # Single value
            elif center is not None or (data.values.min() < 0 and data.values.max() > 0):
                cmap = 'RdBu_r'  # Diverging data
            else:
                cmap = 'viridis'  # Sequential data
        
        # Auto-determine annotation settings
        if annot is None:
            n_elements = data.shape[0] * data.shape[1]
            annot = n_elements <= 100  # Only annotate if not too many elements
        
        # Auto-determine format based on data range
        if fmt is None:
            data_range = abs(data.values.max() - data.values.min())
            if data_range < 0.1:
                fmt = '.3f'
            elif data_range < 10:
                fmt = '.2f'
            else:
                fmt = '.1f'
        
        # Auto-determine color range
        if vmin is None:
            vmin = np.nanpercentile(data.values, 2)
        if vmax is None:
            vmax = np.nanpercentile(data.values, 98)
        
        # Auto-determine center for diverging colormaps
        if center is None and cmap in ['RdBu_r', 'RdBu', 'coolwarm', 'seismic']:
            center = 0
    else:
        # Use default values if not auto-adjusting
        if figsize is None:
            figsize = (12, 8)
        if cmap is None:
            cmap = 'RdBu_r'
        if annot is None:
            annot = True
        if fmt is None:
            fmt = '.2f'
        if vmin is None:
            vmin = np.nanpercentile(data.values, 5)
        if vmax is None:
            vmax = np.nanpercentile(data.values, 95)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = sns.heatmap(data, 
                     annot=annot, 
                     fmt=fmt,
                     cmap=cmap,
                     center=center,
                     vmin=vmin,
                     vmax=vmax,
                     cbar_kws={'label': 'Value', 'shrink': 0.8},
                     annot_kws={'fontsize': 10, 'weight': 'bold'},
                     linewidths=0.5,
                     linecolor='white',
                     ax=ax,
                     **kwargs)
    
    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Auto-adjust label rotation based on label length
    if auto_adjust:
        # Check if labels are long and need rotation
        x_labels = ax.get_xticklabels()
        y_labels = ax.get_yticklabels()
        
        # Rotate x-labels if they're long or if there are many columns
        if len(x_labels) > 8 or any(len(label.get_text()) > 10 for label in x_labels):
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        else:
            ax.set_xticklabels(x_labels, rotation=0)
        
        # Rotate y-labels if they're long
        if any(len(label.get_text()) > 15 for label in y_labels):
            ax.set_yticklabels(y_labels, rotation=0, ha='right')
        else:
            ax.set_yticklabels(y_labels, rotation=0)
    else:
        # Default rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def plot_correlation_heatmap(data: pd.DataFrame,
                           title: str = "Correlation Matrix",
                           figsize: Tuple[float, float] = (10, 8),
                           save_path: Optional[str] = None,
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot correlation matrix heatmap
    
    Args:
        data: Input DataFrame
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        **kwargs: Additional arguments for seaborn.heatmap
        
    Returns:
        Figure and axes objects
    """
    corr_matrix = data.corr()
    return plot_heatmap(corr_matrix, 
                       title=title,
                       xlabel="Variables",
                       ylabel="Variables",
                       cmap='RdBu_r',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       figsize=figsize,
                       save_path=save_path,
                       **kwargs)

def plot_multi_value_heatmap(data_dict: Dict[str, Union[pd.DataFrame, np.ndarray]],
                            row_labels: Optional[List[str]] = None,
                            col_labels: Optional[List[str]] = None,
                            title: str = "Multi-Value Heatmap",
                            xlabel: str = "X",
                            ylabel: str = "Y",
                            layout: str = "quadrant",
                            cmaps: Optional[Dict[str, str]] = None,
                            value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                            show_values: bool = True,
                            value_format: str = ".2f",
                            figsize: Optional[Tuple[float, float]] = None,
                            save_path: Optional[str] = None,
                            alpha: float = 0.8,
                            **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    在同一个cell中绘制多个不同量的热图
    
    Args:
        data_dict: 数据字典，键为变量名，值为对应的数据数组
        row_labels: 行标签
        col_labels: 列标签
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        layout: cell内布局方式 ('quadrant', 'horizontal', 'vertical', 'diagonal', 'corner')
        cmaps: 每个变量的颜色映射字典
        value_ranges: 每个变量的数值范围字典
        show_values: 是否显示数值
        value_format: 数值格式
        figsize: 图形大小
        save_path: 保存路径
        alpha: 透明度
        **kwargs: 其他参数
        
    Returns:
        Figure和axes对象
    """
    setup_plot_style()
    
    # 检查输入数据
    if not data_dict:
        raise ValueError("data_dict不能为空")
    
    # 获取第一个数据的形状作为参考
    first_key = list(data_dict.keys())[0]
    first_data = data_dict[first_key]
    
    if isinstance(first_data, pd.DataFrame):
        n_rows, n_cols = first_data.shape
        if row_labels is None:
            row_labels = first_data.index.tolist()
        if col_labels is None:
            col_labels = first_data.columns.tolist()
    else:
        n_rows, n_cols = first_data.shape
        if row_labels is None:
            row_labels = [f'Row_{i}' for i in range(n_rows)]
        if col_labels is None:
            col_labels = [f'Col_{i}' for i in range(n_cols)]
    
    # 验证所有数据形状一致
    for key, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            data_shape = data.shape
        else:
            data_shape = data.shape
        if data_shape != (n_rows, n_cols):
            raise ValueError(f"数据 '{key}' 的形状 {data_shape} 与参考形状 {(n_rows, n_cols)} 不匹配")
    
    # 设置默认颜色映射
    if cmaps is None:
        default_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        cmaps = {}
        for i, key in enumerate(data_dict.keys()):
            cmaps[key] = default_cmaps[i % len(default_cmaps)]
    
    # 预计算每个变量的颜色范围（若未提供）
    resolved_ranges: Dict[str, Tuple[float, float]] = {}
    for key, data in data_dict.items():
        if value_ranges and key in value_ranges:
            resolved_ranges[key] = value_ranges[key]
        else:
            arr = data.values if isinstance(data, pd.DataFrame) else data
            arr_flat = arr.flatten()
            valid = arr_flat[~np.isnan(arr_flat)]
            if valid.size > 0:
                vmin_k, vmax_k = np.percentile(valid, [5, 95])
            else:
                vmin_k, vmax_k = 0.0, 1.0
            resolved_ranges[key] = (float(vmin_k), float(vmax_k))

    # 设置图形大小
    if figsize is None:
        figsize = (max(12, n_cols * 1.2), max(8, n_rows * 1.0))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 为每个cell绘制多个量
    for i in range(n_rows):
        for j in range(n_cols):
            # 获取cell的边界
            cell_left = j
            cell_right = j + 1
            cell_bottom = n_rows - i - 1
            cell_top = n_rows - i
            
            # 根据布局方式分配子区域
            _draw_multi_value_cell(
                ax,
                data_dict,
                i,
                j,
                cell_left,
                cell_right,
                cell_bottom,
                cell_top,
                layout,
                cmaps,
                resolved_ranges,
                show_values,
                value_format,
                alpha,
            )
    
    # 设置坐标轴
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels[::-1])  # 反转以匹配矩阵显示
    
    # 设置标题和标签
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # 添加网格线
    for i in range(n_rows + 1):
        ax.axhline(i, color='white', linewidth=2)
    for j in range(n_cols + 1):
        ax.axvline(j, color='white', linewidth=2)
    
    # 创建图例
    _create_multi_value_legend(fig, ax, data_dict, cmaps, layout, resolved_ranges)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig, ax

def _draw_multi_value_cell(ax, data_dict, row, col, cell_left, cell_right, 
                          cell_bottom, cell_top, layout, cmaps, value_ranges, 
                          show_values, value_format, alpha):
    """
    在单个cell中绘制多个值
    """
    n_values = len(data_dict)
    
    if layout == "quadrant" and n_values <= 4:
        # 四象限布局
        positions = [
            (0.0, 0.5, 0.5, 1.0),  # 左上
            (0.5, 1.0, 0.5, 1.0),  # 右上
            (0.0, 0.5, 0.0, 0.5),  # 左下
            (0.5, 1.0, 0.0, 0.5),  # 右下
        ]
    elif layout == "horizontal":
        # 水平分割
        width = 1.0 / n_values
        positions = [(i * width, (i + 1) * width, 0.0, 1.0) for i in range(n_values)]
    elif layout == "vertical":
        # 垂直分割
        height = 1.0 / n_values
        positions = [(0.0, 1.0, i * height, (i + 1) * height) for i in range(n_values)]
    elif layout == "diagonal":
        # 对角线分割（仅在2个值时使用真正的三角形分割）
        if n_values == 2:
            positions = None  # 三角形模式下不用矩形位置
        else:
            # 回退到水平分割
            width = 1.0 / n_values
            positions = [(i * width, (i + 1) * width, 0.0, 1.0) for i in range(n_values)]
    elif layout == "corner":
        # 角落布局（用于小的指示器）
        corner_size = 0.3
        positions = [
            (1.0 - corner_size, 1.0, 1.0 - corner_size, 1.0),  # 右上角
            (0.0, corner_size, 1.0 - corner_size, 1.0),        # 左上角
            (1.0 - corner_size, 1.0, 0.0, corner_size),        # 右下角
            (0.0, corner_size, 0.0, corner_size),              # 左下角
        ]
    else:
        # 默认水平分割
        width = 1.0 / n_values
        positions = [(i * width, (i + 1) * width, 0.0, 1.0) for i in range(n_values)]
    
    # 绘制每个值
    diagonal_triangles = (layout == "diagonal" and n_values == 2)
    for idx, (key, data) in enumerate(data_dict.items()):
        if not diagonal_triangles:
            if idx >= len(positions):
                break
            
        # 获取数据值
        if isinstance(data, pd.DataFrame):
            value = data.iloc[row, col]
        else:
            value = data[row, col]
        
        if np.isnan(value):
            continue
        
        # diagonal 2值：用三角形；否则用矩形子区域
        if diagonal_triangles:
            left, right = cell_left, cell_right
            bottom, top = cell_bottom, cell_top
            if idx == 0:
                vertices = [(left, top), (right, top), (left, bottom)]
            else:
                vertices = [(left, bottom), (right, bottom), (right, top)]
        else:
            rel_left, rel_right, rel_bottom, rel_top = positions[idx]
            actual_left = cell_left + rel_left * (cell_right - cell_left)
            actual_right = cell_left + rel_right * (cell_right - cell_left)
            actual_bottom = cell_bottom + rel_bottom * (cell_top - cell_bottom)
            actual_top = cell_bottom + rel_top * (cell_top - cell_bottom)
        
        # 获取颜色映射和数值范围
        cmap = cmaps.get(key, 'viridis')
        if value_ranges and key in value_ranges:
            vmin, vmax = value_ranges[key]
        else:
            # 自动计算范围
            all_values = data.values.flatten() if isinstance(data, pd.DataFrame) else data.flatten()
            valid_values = all_values[~np.isnan(all_values)]
            if len(valid_values) > 0:
                vmin, vmax = np.percentile(valid_values, [5, 95])
            else:
                vmin, vmax = 0, 1
        
        # 标准化颜色值
        norm = Normalize(vmin=vmin, vmax=vmax)
        color = cm.get_cmap(cmap)(norm(value))

        if diagonal_triangles:
            tri = patches.Polygon(vertices, closed=True, linewidth=0.5, edgecolor='white', facecolor=color, alpha=alpha)
            ax.add_patch(tri)
            if show_values:
                cx = sum(v[0] for v in vertices) / 3.0
                cy = sum(v[1] for v in vertices) / 3.0
                luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(cx, cy, format(value, value_format), ha='center', va='center', fontsize=9, color=text_color, weight='bold')
        else:
            # 绘制矩形
            rect = patches.Rectangle(
                (actual_left, actual_bottom),
                actual_right - actual_left,
                actual_top - actual_bottom,
                linewidth=0.5,
                edgecolor='white',
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)
            
            # 添加数值标签
            if show_values:
                text_x = (actual_left + actual_right) / 2
                text_y = (actual_bottom + actual_top) / 2
                
                # 根据区域大小调整字体大小
                area = (actual_right - actual_left) * (actual_top - actual_bottom)
                if area < 0.1:
                    fontsize = 6
                elif area < 0.25:
                    fontsize = 8
                else:
                    fontsize = 10
                
                # 选择合适的文本颜色（基于背景亮度）
                luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                
                ax.text(text_x, text_y, format(value, value_format),
                       ha='center', va='center', fontsize=fontsize,
                       color=text_color, weight='bold')

def _create_multi_value_legend(fig, ax, data_dict, cmaps, layout, value_ranges: Dict[str, Tuple[float, float]]):
    """
    为多值热图创建图例
    """
    # 在右侧创建颜色条
    
    # 计算每个变量的位置
    n_vars = len(data_dict)
    cbar_height = 0.8 / n_vars
    
    for idx, (key, data) in enumerate(data_dict.items()):
        # 使用预先解析的范围，保持与cell内一致
        vmin, vmax = value_ranges.get(key, (0.0, 1.0))
        
        # 创建颜色条
        cmap = cmaps.get(key, 'viridis')
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # 计算颜色条位置
        cbar_bottom = 0.1 + idx * (cbar_height + 0.05)
        cbar_ax = fig.add_axes([0.92, cbar_bottom, 0.02, cbar_height])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(key, fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

def plot_comparison_heatmap(data_dict: Dict[str, Union[pd.DataFrame, np.ndarray]],
                           comparison_type: str = "difference",
                           reference_key: Optional[str] = None,
                           title: str = "Comparison Heatmap",
                           figsize: Optional[Tuple[float, float]] = None,
                           save_path: Optional[str] = None,
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制比较热图，显示数据间的差异或比值
    
    Args:
        data_dict: 数据字典
        comparison_type: 比较类型 ('difference', 'ratio', 'percentage')
        reference_key: 参考数据的键名（如果为None，使用第一个）
        title: 图表标题
        figsize: 图形大小
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        Figure和axes对象
    """
    if len(data_dict) < 2:
        raise ValueError("比较热图至少需要2个数据集")
    
    # 选择参考数据
    if reference_key is None:
        reference_key = list(data_dict.keys())[0]
    
    if reference_key not in data_dict:
        raise ValueError(f"参考键 '{reference_key}' 不在数据字典中")
    
    reference_data = data_dict[reference_key]
    comparison_data = {}
    
    # 计算比较数据
    for key, data in data_dict.items():
        if key == reference_key:
            continue
            
        if comparison_type == "difference":
            comparison_data[f"{key} - {reference_key}"] = data - reference_data
        elif comparison_type == "ratio":
            comparison_data[f"{key} / {reference_key}"] = data / reference_data
        elif comparison_type == "percentage":
            comparison_data[f"{key} vs {reference_key} (%)"] = ((data - reference_data) / reference_data) * 100
        else:
            raise ValueError(f"不支持的比较类型: {comparison_type}")
    
    # 设置合适的颜色映射
    if comparison_type in ["difference", "percentage"]:
        default_cmap = "RdBu_r"
    else:  # ratio
        default_cmap = "viridis"
    
    # 使用多值热图绘制
    return plot_multi_value_heatmap(
        comparison_data,
        title=title,
        layout="horizontal",
        cmaps={key: default_cmap for key in comparison_data.keys()},
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage
    print("Enhanced Heatmap plotting module loaded")
    
    # 创建示例数据
    np.random.seed(42)
    n_rows, n_cols = 5, 4
    
    # 示例：多值热图
    data1 = np.random.randn(n_rows, n_cols)
    data2 = np.random.randn(n_rows, n_cols) * 0.5 + 1
    data3 = np.random.randn(n_rows, n_cols) * 2 - 1
    
    multi_data = {
        'Temperature': data1,
        'Humidity': data2,
        'Pressure': data3
    }
    
    print("示例多值热图数据已创建")
    print("使用方法:")
    print("fig, ax = plot_multi_value_heatmap(multi_data, layout='quadrant')")
    print("fig, ax = plot_comparison_heatmap(multi_data, comparison_type='difference')")

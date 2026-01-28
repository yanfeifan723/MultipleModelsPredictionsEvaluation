#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的绘图工具函数
提供标准化的空间分布图、等高线图等绘图功能

标准配置（参考 combined_pearson_analysis.py 的 acc_spatial_maps）：
- 子图间隙: hspace=0.25, wspace=0.15
- 刻度: 使用 gridlines，字体 12pt
- 绘图: contourf (填色) + contour (轮廓线)
- 显著性: 打点标记
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from typing import Dict, List, Optional, Tuple, Union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============== 标准配置 ==============
STANDARD_CONFIG = {
    'hspace': 0.25,
    'wspace': 0.15,
    'tick_fontsize': 12,
    'title_fontsize': 18,
    'label_fontsize': 14,
    'colorbar_fontsize': 14,
    'grid_linewidth': 0.5,
    'grid_alpha': 0.5,
    'grid_linestyle': '--',
    'contour_linewidth': 0.3,
    'contour_alpha': 0.4,
    'significance_marker_size': 2.0,
    'significance_marker_alpha': 0.8,
}


def setup_cartopy_axes(ax, lon_range: Tuple[float, float], lat_range: Tuple[float, float],
                       lon_ticks: np.ndarray, lat_ticks: np.ndarray,
                       show_features: bool = True, config: dict = None) -> object:
    """
    设置 Cartopy 地图轴的标准配置
    
    参数:
        ax: matplotlib axes (需要有 projection=ccrs.PlateCarree())
        lon_range: (lon_min, lon_max)
        lat_range: (lat_min, lat_max)
        lon_ticks: 经度刻度数组
        lat_ticks: 纬度刻度数组
        show_features: 是否显示海岸线等要素
        config: 配置字典，如果为None则使用STANDARD_CONFIG
        
    返回:
        gridlines 对象
    """
    if config is None:
        config = STANDARD_CONFIG
    
    # 设置地图范围
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                  crs=ccrs.PlateCarree())
    
    # 添加地理要素
    if show_features:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', zorder=2)
        ax.add_feature(cfeature.LAND, alpha=0.1, zorder=1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1, zorder=1)
    
    # 设置网格线和刻度
    gl = ax.gridlines(draw_labels=True, 
                     linewidth=config['grid_linewidth'],
                     alpha=config['grid_alpha'],
                     linestyle=config['grid_linestyle'])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = FixedLocator(lon_ticks)
    gl.ylocator = FixedLocator(lat_ticks)
    gl.xlabel_style = {'size': config['tick_fontsize']}
    gl.ylabel_style = {'size': config['tick_fontsize']}
    
    return gl


def plot_spatial_field_contour(ax, data: xr.DataArray, 
                               levels: Union[int, np.ndarray],
                               cmap = 'RdBu_r',
                               norm: Optional[mcolors.Normalize] = None,
                               add_contour_lines: bool = True,
                               config: dict = None) -> Tuple[object, Optional[object]]:
    """
    绘制空间场的填色等高线图
    
    参数:
        ax: matplotlib axes
        data: xarray.DataArray，包含 lon, lat 坐标
        levels: 等高线层级（数量或数组）
        cmap: colormap 对象或名称
        norm: matplotlib normalizer（如 BoundaryNorm）
        add_contour_lines: 是否添加轮廓线
        config: 配置字典
        
    返回:
        (contourf_object, contour_object)
    """
    if config is None:
        config = STANDARD_CONFIG
    
    # 填色等高线 - 使用 levels 和 cmap，不要同时传 norm
    cf = ax.contourf(data.lon, data.lat, data,
                    levels=levels,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    extend='both', zorder=1)
    
    # 轮廓线
    cs = None
    if add_contour_lines:
        # 使用较稀疏的levels避免太密集
        if isinstance(levels, np.ndarray) and len(levels) > 5:
            contour_levels = levels[::2]
        else:
            contour_levels = levels
            
        cs = ax.contour(data.lon, data.lat, data,
                       levels=contour_levels,
                       transform=ccrs.PlateCarree(),
                       colors='black',
                       linewidths=config['contour_linewidth'],
                       alpha=config['contour_alpha'],
                       zorder=2)
    
    return cf, cs


def add_significance_stippling(ax, data: xr.DataArray, significance_mask: xr.DataArray,
                               config: dict = None):
    """
    在通过显著性检验的格点添加打点标记
    
    参数:
        ax: matplotlib axes
        data: xarray.DataArray（用于获取经纬度网格）
        significance_mask: 布尔数组，True表示显著
        config: 配置字典
    """
    if config is None:
        config = STANDARD_CONFIG
    
    if significance_mask is not None and np.any(significance_mask.values):
        lon_2d, lat_2d = np.meshgrid(data.lon.values, data.lat.values)
        ax.scatter(lon_2d[significance_mask.values], 
                  lat_2d[significance_mask.values],
                  s=config['significance_marker_size'],
                  c='black',
                  marker='.',
                  alpha=config['significance_marker_alpha'],
                  transform=ccrs.PlateCarree(),
                  zorder=4)


def create_spatial_distribution_figure(
    data_dict: Dict[str, Dict[int, xr.DataArray]],
    leadtimes: List[int],
    lon_range: Tuple[float, float] = (70, 140),
    lat_range: Tuple[float, float] = (15, 55),
    levels: Optional[Union[int, np.ndarray]] = None,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    significance_dict: Optional[Dict[str, Dict[int, xr.DataArray]]] = None,
    title: str = '',
    colorbar_label: str = '',
    output_file: Optional[str] = None,
    config: dict = None
) -> plt.Figure:
    """
    创建标准化的空间分布组合图
    
    布局: Lead 0 和 Lead 3，每个lead占2行
        第1行: 空白 + 3个模型
        第2行: 4个模型
    
    参数:
        data_dict: {model_name: {leadtime: DataArray}}
        leadtimes: 要绘制的leadtime列表（通常为[0, 3]）
        lon_range: (lon_min, lon_max)
        lat_range: (lat_min, lat_max)
        levels: 等高线层级
        cmap: colormap
        vmin, vmax: 数据范围
        significance_dict: 显著性掩码字典（同data_dict结构）
        title: 总标题
        colorbar_label: colorbar标签
        output_file: 输出文件路径
        config: 配置字典
        
    返回:
        matplotlib Figure对象
    """
    if config is None:
        config = STANDARD_CONFIG
    
    # 获取模型列表
    plot_models = list(data_dict.keys())
    if not plot_models:
        raise ValueError("没有可绘制的模型数据")
    
    # 计算经纬度刻度
    lon_tick_start = int(np.ceil(lon_range[0] / 15.0) * 15)
    lon_tick_end = int(np.floor(lon_range[1] / 15.0) * 15)
    lon_ticks = np.arange(lon_tick_start, lon_tick_end + 1, 15)
    
    lat_tick_start = int(np.ceil(lat_range[0] / 10.0) * 10)
    lat_tick_end = int(np.floor(lat_range[1] / 10.0) * 10)
    lat_ticks = np.arange(lat_tick_start, lat_tick_end + 1, 10)
    
    # 自动检测数据范围（如果未指定vmin/vmax）
    if vmin is None or vmax is None:
        all_values = []
        for model in plot_models:
            for lt in leadtimes:
                if lt in data_dict[model]:
                    vals = data_dict[model][lt].values
                    all_values.extend(vals[np.isfinite(vals)])
        
        if all_values:
            data_min = np.min(all_values)
            data_max = np.max(all_values)
            if vmin is None:
                vmin = data_min
            if vmax is None:
                vmax = data_max
    
    # 如果没有指定levels，使用vmin/vmax创建
    if levels is None:
        if vmin is not None and vmax is not None:
            n_bins = 20
            levels = np.linspace(vmin, vmax, n_bins + 1)
        else:
            levels = 15  # 默认15个层级
    
    # 对于 contourf，直接使用 cmap 字符串和 levels 即可
    # matplotlib 会自动在 levels 范围内均匀映射颜色
    cmap_obj = cmap  # 保持字符串格式
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 4, figure=fig,
                 hspace=config['hspace'],
                 wspace=config['wspace'],
                 left=0.05, right=0.92, top=0.94, bottom=0.06)
    
    # 绘制子图
    for lt_idx, leadtime in enumerate(leadtimes):
        row_start = lt_idx * 2
        
        # 第一行: 空白 + 3个模型
        ax_blank = fig.add_subplot(gs[row_start, 0])
        ax_blank.axis('off')
        
        for col_idx in range(3):
            if col_idx >= len(plot_models):
                ax = fig.add_subplot(gs[row_start, col_idx + 1])
                ax.axis('off')
                continue
            
            model = plot_models[col_idx]
            if leadtime not in data_dict[model]:
                ax = fig.add_subplot(gs[row_start, col_idx + 1])
                ax.axis('off')
                continue
            
            data = data_dict[model][leadtime]
            display_name = model.replace('-mon', '').replace('mon-', '')
            
            # 创建地图
            ax = fig.add_subplot(gs[row_start, col_idx + 1], 
                               projection=ccrs.PlateCarree())
            setup_cartopy_axes(ax, lon_range, lat_range, lon_ticks, lat_ticks, 
                             show_features=True, config=config)
            
            # 绘制数据
            cf, cs = plot_spatial_field_contour(ax, data, levels, cmap_obj, None,
                                               add_contour_lines=True, config=config)
            
            # 显著性打点
            if significance_dict and model in significance_dict:
                if leadtime in significance_dict[model]:
                    sig_mask = significance_dict[model][leadtime]
                    add_significance_stippling(ax, data, sig_mask, config)
            
            # 添加标题
            title_text = f"({chr(97 + col_idx)}) {display_name}"
            ax.text(0.02, 0.96, title_text,
                   transform=ax.transAxes,
                   fontsize=config['title_fontsize'],
                   fontweight='bold',
                   verticalalignment='top',
                   horizontalalignment='left')
            
            # 添加leadtime标签（第一个模型）
            if col_idx == 0:
                ax.text(0.98, 0.96, f'L{leadtime}',
                       transform=ax.transAxes,
                       fontsize=config['title_fontsize'],
                       fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='right')
        
        # 第二行: 4个模型
        for col_idx in range(4):
            model_idx = col_idx + 3
            if model_idx >= len(plot_models):
                ax = fig.add_subplot(gs[row_start + 1, col_idx])
                ax.axis('off')
                continue
            
            model = plot_models[model_idx]
            if leadtime not in data_dict[model]:
                ax = fig.add_subplot(gs[row_start + 1, col_idx])
                ax.axis('off')
                continue
            
            data = data_dict[model][leadtime]
            display_name = model.replace('-mon', '').replace('mon-', '')
            
            # 创建地图
            ax = fig.add_subplot(gs[row_start + 1, col_idx],
                               projection=ccrs.PlateCarree())
            setup_cartopy_axes(ax, lon_range, lat_range, lon_ticks, lat_ticks,
                             show_features=True, config=config)
            
            # 绘制数据
            cf, cs = plot_spatial_field_contour(ax, data, levels, cmap_obj, None,
                                               add_contour_lines=True, config=config)
            
            # 显著性打点
            if significance_dict and model in significance_dict:
                if leadtime in significance_dict[model]:
                    sig_mask = significance_dict[model][leadtime]
                    add_significance_stippling(ax, data, sig_mask, config)
            
            # 添加标题
            title_text = f"({chr(97 + model_idx)}) {display_name}"
            ax.text(0.02, 0.98, title_text,
                   transform=ax.transAxes,
                   fontsize=config['title_fontsize'],
                   fontweight='bold',
                   verticalalignment='top',
                   horizontalalignment='left')
    
    # 添加colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax, orientation='vertical')
    
    # 设置 colorbar ticks
    if isinstance(levels, np.ndarray) and len(levels) > 10:
        # 如果levels太多，只显示部分ticks
        cbar.set_ticks(levels[::2])
    elif isinstance(levels, np.ndarray):
        cbar.set_ticks(levels)
    
    cbar.set_label(colorbar_label, fontsize=config['colorbar_fontsize'], labelpad=15)
    cbar.ax.tick_params(labelsize=config['tick_fontsize'])
    
    # 添加总标题（已禁用）
    # if title:
    #     fig.suptitle(title, fontsize=16, fontweight='bold', y=0.97)
    
    # 保存
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def create_discrete_colormap_norm(vmin: float, vmax: float, n_bins: int = 20):
    """
    创建离散型colormap和norm（用于固定范围的绘图）
    
    参数:
        vmin: 最小值
        vmax: 最大值
        n_bins: 分段数
        
    返回:
        (cmap, norm, levels)
    """
    levels = np.linspace(vmin, vmax, n_bins + 1)
    cmap = plt.get_cmap('RdBu_r', n_bins)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels


def create_multi_dataset_spatial_figure(
    data_groups: List[Dict],
    leadtimes: List[int],
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    lon_range: Tuple[float, float] = (0, 360),
    lat_range: Tuple[float, float] = (-90, 90),
    lon_ticks: Optional[np.ndarray] = None,
    lat_ticks: Optional[np.ndarray] = None,
    colorbar_orientation: str = 'horizontal',
    config: Optional[dict] = None
):
    """
    创建包含多个数据集（多个colorbar）的空间分布图
    
    适用场景：
    - climatology + bias 图（2个colorbar）
    - observation + model 图（可能需要不同的colorbar）
    
    参数:
        data_groups: 数据组列表，每个组是一个字典，包含:
            - 'data_dict': Dict[str, Dict[int, xr.DataArray]] - {model: {leadtime: data}}
            - 'levels': Union[int, np.ndarray] - 等高线层级
            - 'cmap': str - colormap名称
            - 'vmin': Optional[float] - 最小值（自动检测）
            - 'vmax': Optional[float] - 最大值（自动检测）
            - 'colorbar_label': str - colorbar标签
            - 'significance_dict': Optional[Dict] - 显著性数据
            - 'add_contour_lines': bool - 是否添加轮廓线（默认True）
            - 'column_indices': Optional[List[int]] - 该组数据应用于哪些列（默认全部）
        leadtimes: 提前期列表
        title: 总标题（可选，已禁用）
        output_file: 输出文件路径
        lon_range: 经度范围
        lat_range: 纬度范围
        lon_ticks: 经度刻度
        lat_ticks: 纬度刻度
        colorbar_orientation: colorbar方向 ('horizontal' 或 'vertical')
        config: 配置字典（使用STANDARD_CONFIG）
        
    示例:
        # climatology + bias 图
        data_groups = [
            {
                'data_dict': {'OBS': {0: obs_data, 3: obs_data}},  # 观测数据
                'cmap': 'viridis',
                'colorbar_label': 'Climatology (K)',
                'column_indices': [0],  # 只在第一列
            },
            {
                'data_dict': {'Model1': {0: bias1_L0, 3: bias1_L3}, 'Model2': ...},  # 偏差数据
                'cmap': 'coolwarm',
                'colorbar_label': 'Bias (K)',
                'column_indices': [1, 2, 3, 4, 5, 6],  # 其他列
            }
        ]
    """
    if config is None:
        config = STANDARD_CONFIG
    
    # 参数验证
    if not data_groups or len(data_groups) == 0:
        raise ValueError("data_groups不能为空")
    
    # 收集所有模型（从所有组）
    all_models = []
    for group in data_groups:
        data_dict = group['data_dict']
        all_models.extend(data_dict.keys())
    # 去重并保持顺序
    plot_models = []
    seen = set()
    for m in all_models:
        if m not in seen:
            plot_models.append(m)
            seen.add(m)
    
    n_models = len(plot_models)
    n_leadtimes = len(leadtimes)
    
    # 为每个数据组检测vmin/vmax（如果未提供）和levels
    for group_idx, group in enumerate(data_groups):
        data_dict = group['data_dict']
        
        if 'vmin' not in group or group['vmin'] is None or 'vmax' not in group or group['vmax'] is None:
            all_values = []
            for model in data_dict.keys():
                for lt in leadtimes:
                    if lt in data_dict[model]:
                        vals = data_dict[model][lt].values
                        all_values.extend(vals[np.isfinite(vals)])
            
            if all_values:
                data_min = np.min(all_values)
                data_max = np.max(all_values)
                if 'vmin' not in group or group['vmin'] is None:
                    group['vmin'] = data_min
                if 'vmax' not in group or group['vmax'] is None:
                    group['vmax'] = data_max
        
        # 如果没有指定levels，使用vmin/vmax创建
        if 'levels' not in group or group['levels'] is None:
            if 'vmin' in group and group['vmin'] is not None and 'vmax' in group and group['vmax'] is not None:
                n_bins = 20
                group['levels'] = np.linspace(group['vmin'], group['vmax'], n_bins + 1)
            else:
                group['levels'] = 15
    
    # 创建图形
    fig = plt.figure(figsize=(n_models * 5, n_leadtimes * 3.5))
    gs = GridSpec(n_leadtimes, n_models + 1,
                  figure=fig,
                  hspace=config['hspace'],
                  wspace=config['wspace'],
                  left=0.05, right=0.92,
                  bottom=0.15 if colorbar_orientation == 'horizontal' else 0.08,
                  top=0.95)
    
    # 存储每个数据组的contourf对象（用于创建colorbar）
    cf_objects = [None] * len(data_groups)
    
    # 模型到列索引的映射
    model_to_col = {model: idx for idx, model in enumerate(plot_models)}
    
    # 绘制每个子图
    for row_idx, leadtime in enumerate(leadtimes):
        for model in plot_models:
            col_idx = model_to_col[model]
            
            # 找到该模型对应的数据组
            group_idx = None
            for gidx, group in enumerate(data_groups):
                # 检查column_indices约束
                if 'column_indices' in group:
                    if col_idx not in group['column_indices']:
                        continue
                
                # 检查该组是否包含此模型的数据
                data_dict = group['data_dict']
                if model in data_dict and leadtime in data_dict[model]:
                    group_idx = gidx
                    break
            
            # 如果没有找到对应的数据组，跳过
            if group_idx is None:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis('off')
                continue
            
            group = data_groups[group_idx]
            data_dict = group['data_dict']
            levels = group['levels']
            cmap = group['cmap']
            data = data_dict[model][leadtime]
            
            display_name = model.replace('-mon', '').replace('mon-', '')
            
            # 创建地图
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
            setup_cartopy_axes(ax, lon_range, lat_range, lon_ticks, lat_ticks,
                             show_features=True, config=config)
            
            # 绘制数据
            add_contour_lines = group.get('add_contour_lines', True)
            cf, cs = plot_spatial_field_contour(ax, data, levels, cmap, None,
                                               add_contour_lines=add_contour_lines, config=config)
            
            # 保存第一个有效的cf对象（用于创建colorbar）
            if cf_objects[group_idx] is None:
                cf_objects[group_idx] = cf
            
            # 显著性打点
            significance_dict = group.get('significance_dict', None)
            if significance_dict and model in significance_dict:
                if leadtime in significance_dict[model]:
                    sig_mask = significance_dict[model][leadtime]
                    add_significance_stippling(ax, data, sig_mask, config)
            
            # 添加模型标签（只在每列第一个出现的位置添加）
            if row_idx == 0:
                title_text = f"({chr(97 + col_idx)}) {display_name}"
                ax.text(0.02, 0.96, title_text,
                       transform=ax.transAxes,
                       fontsize=config['title_fontsize'],
                       fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='left')
            
            # 添加leadtime标签（只在第一列）
            if col_idx == 0:
                ax.text(0.98, 0.96, f'L{leadtime}',
                       transform=ax.transAxes,
                       fontsize=config['title_fontsize'],
                       fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='right')
    
    # 添加colorbar
    if colorbar_orientation == 'horizontal':
        # 横向colorbar，在底部排列
        valid_groups = [(idx, group, cf) for idx, (group, cf) in enumerate(zip(data_groups, cf_objects)) if cf is not None]
        n_colorbars = len(valid_groups)
        
        if n_colorbars > 0:
            cbar_width = 0.8 / n_colorbars
            
            for bar_idx, (group_idx, group, cf) in enumerate(valid_groups):
                cbar_left = 0.1 + bar_idx * cbar_width
                cbar_ax = fig.add_axes([cbar_left, 0.08, cbar_width * 0.9, 0.02])
                cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal')
                
                # 设置colorbar ticks
                levels = group['levels']
                if isinstance(levels, np.ndarray) and len(levels) > 10:
                    cbar.set_ticks(levels[::2])
                elif isinstance(levels, np.ndarray):
                    cbar.set_ticks(levels)
                
                cbar.set_label(group.get('colorbar_label', ''),
                              fontsize=config['colorbar_fontsize'], labelpad=5)
                cbar.ax.tick_params(labelsize=config['tick_fontsize'])
    
    else:  # vertical
        # 竖向colorbar，在右侧（只使用第一个有效的colorbar）
        for idx, (group, cf) in enumerate(zip(data_groups, cf_objects)):
            if cf is not None:
                cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
                cbar = plt.colorbar(cf, cax=cbar_ax, orientation='vertical')
                
                levels = group['levels']
                if isinstance(levels, np.ndarray) and len(levels) > 10:
                    cbar.set_ticks(levels[::2])
                elif isinstance(levels, np.ndarray):
                    cbar.set_ticks(levels)
                
                cbar.set_label(group.get('colorbar_label', ''),
                              fontsize=config['colorbar_fontsize'], labelpad=15)
                cbar.ax.tick_params(labelsize=config['tick_fontsize'])
                break  # 只使用第一个
    
    # 保存
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

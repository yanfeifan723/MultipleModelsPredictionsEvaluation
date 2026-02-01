#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制 combined_error_analysis.py 和 combined_pearson_analysis.py 中定义的地理分区 (Z1-Z9)
更新：
1. 图例移动至图像下方，呈2排5列显示
2. 右下角增加南海子图 (South China Sea Inset)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import logging
from pathlib import Path
from matplotlib.ticker import FixedLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义区域 (来自 combined_error_analysis.py 和 combined_pearson_analysis.py)
REGIONS = {
    'Z1-Northwest':     {'lat': (39, 49), 'lon': (73, 105)},   # 西北干旱区
    'Z2-InnerMongolia': {'lat': (39, 50), 'lon': (106, 118)},  # 内蒙半干旱区
    'Z3-Northeast':     {'lat': (40, 54), 'lon': (119, 135)},  # 东北湿润区
    'Z4-Tibetan':       {'lat': (27, 39), 'lon': (73, 95)},    # 青藏高寒区
    'Z5-NorthChina':    {'lat': (34, 39), 'lon': (106, 122)},  # 黄土-华北区
    'Z6-Yangtze':       {'lat': (26, 34), 'lon': (109, 123)},  # 长江中下游区
    'Z7-Southwest':     {'lat': (23, 33), 'lon': (96, 108)},   # 四川-西南区
    'Z8-SouthChina':    {'lat': (21, 25), 'lon': (106, 120)},  # 华南湿润区
    'Z9-SouthSea':      {'lat': (18, 21), 'lon': (105, 125)}   # 南海热带区
}

def load_geometries(ax, sub_ax=None):
    """
    加载 Shapefile (国界、省界、河流) 并添加到主图和子图
    """
    bou_paths = [
        Path("/sas12t1/ffyan/boundaries/中国_省1.shp"),
        Path("/sas12t1/ffyan/boundaries/中国_省2.shp")
    ]
    hyd_path = Path("/sas12t1/ffyan/boundaries/河流.shp")
    
    # 1. 加载河流
    hyd_geoms = []
    if hyd_path.exists():
        try:
            reader = shpreader.Reader(str(hyd_path))
            hyd_geoms = list(reader.geometries())
            ax.add_geometries(hyd_geoms, ccrs.PlateCarree(),
                            edgecolor='blue', facecolor='none', 
                            linewidth=0.6, alpha=0.6, zorder=5)
            if sub_ax:
                sub_ax.add_geometries(hyd_geoms, ccrs.PlateCarree(),
                                    edgecolor='blue', facecolor='none', 
                                    linewidth=0.6, alpha=0.6, zorder=5)
        except Exception:
            pass
    
    # 如果没有加载到河流shp，使用 cartopy 自带的 (仅主图，子图通常不加粗糙的自带河流)
    if not hyd_geoms:
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.6, alpha=0.6, zorder=5)

    # 2. 加载国界/省界
    loaded_borders = False
    for bou_path in bou_paths:
        if bou_path.exists():
            try:
                reader = shpreader.Reader(str(bou_path))
                geoms = list(reader.geometries())
                
                # 添加到主图
                ax.add_geometries(geoms, ccrs.PlateCarree(), 
                                edgecolor='black', facecolor='none', 
                                linewidth=0.6, zorder=100)
                # 添加到南海子图
                if sub_ax:
                    sub_ax.add_geometries(geoms, ccrs.PlateCarree(), 
                                        edgecolor='black', facecolor='none', 
                                        linewidth=0.6, zorder=100)
                loaded_borders = True
            except Exception:
                pass
    
    if not loaded_borders:
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, zorder=100)
        if sub_ax:
            sub_ax.add_feature(cfeature.BORDERS, linewidth=1.0, zorder=100)

def setup_map_style(ax):
    """设置主地图基础风格"""
    # 设置范围
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    
    # 基础要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', zorder=50)
    
    # 网格线设置
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = FixedLocator(np.arange(70, 141, 10))
    gl.ylocator = FixedLocator(np.arange(15, 56, 10))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

def setup_scs_inset(ax):
    """
    创建并设置南海子图 (South China Sea Inset)
    参考 combined_pearson_analysis.py 的参数
    """
    # 位置参数：[x, y, width, height] (相对于父ax的坐标 0-1)
    # 贴紧右下角
    sub_ax = ax.inset_axes([0.8052, 0.0, 0.25, 0.30], projection=ccrs.PlateCarree())
    
    # 设置南海范围 [lon_min, lon_max, lat_min, lat_max]
    sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
    
    # 设置白色背景
    sub_ax.patch.set_facecolor('white')
    
    # 基础要素
    sub_ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='gray', zorder=50)
    sub_ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='none', alpha=0.1)
    
    # 移除子图刻度，保留边框
    sub_ax.tick_params(left=False, labelleft=False, 
                       bottom=False, labelbottom=False)
    
    # 设置子图边框
    for spine in sub_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
        
    return sub_ax

def plot_z_regions(output_file="regions_map.png"):
    """主绘图函数"""
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 1. 设置主地图底图
    setup_map_style(ax)
    
    # 2. 创建南海子图
    sub_ax = setup_scs_inset(ax)
    
    # 3. 加载 Shapefile 到主图和子图
    load_geometries(ax, sub_ax)
    
    # 使用 tab10 色板区分不同区域
    cmap = plt.get_cmap('tab10')
    
    # 排序以保持图例顺序一致 (Z1, Z2, ...)
    sorted_regions = sorted(REGIONS.items(), key=lambda x: x[0])
    
    handles = []
    
    for idx, (name, coords) in enumerate(sorted_regions):
        lat_b = coords['lat']
        lon_b = coords['lon']
        
        lat_min, lat_max = min(lat_b), max(lat_b)
        lon_min, lon_max = min(lon_b), max(lon_b)
        
        width = lon_max - lon_min
        height = lat_max - lat_min
        
        color = cmap(idx % 10)
        
        # --- 主图绘制 ---
        # A. 绘制矩形框
        rect = mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=2.5,
            edgecolor=color,
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=200
        )
        ax.add_patch(rect)
        
        # B. 添加区域中心标签 (Z1, Z2...)
        short_name = name.split('-')[0] # 仅显示 Zx
        ax.text(lon_min + width/2, lat_min + height/2, short_name,
                transform=ccrs.PlateCarree(),
                fontsize=11, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                zorder=201)
        
        # --- 南海子图绘制 (仅绘制进入视野的区域，如Z9, Z8) ---
        # 简单起见，全部尝试添加，Matplotlib会自动剪裁超出范围的部分
        rect_sub = mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=2.5,
            edgecolor=color,
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=200
        )
        sub_ax.add_patch(rect_sub)
        
        # 仅在南海子图中为 Z9 (SouthSea) 添加标签，避免拥挤
        if "Z9" in name:
            sub_ax.text(lon_min + width/2, lat_min + height/2, short_name,
                    transform=ccrs.PlateCarree(),
                    fontsize=9, fontweight='bold', color=color,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5),
                    zorder=201)

        # C. 为图例创建 Handle
        handles.append(mpatches.Patch(color=color, label=name, fill=False, linewidth=2.5))

    ax.legend(handles=handles, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), 
              ncol=5, 
              fontsize=11,
              framealpha=0.95,
              borderaxespad=0.)
    
    plt.title('Analysis Regions (Z1-Z9)', fontsize=16, fontweight='bold', pad=15)
    
    # 保存图片
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"图表已保存至: {output_path.absolute()}")
    plt.close()

if __name__ == "__main__":
    plot_z_regions()
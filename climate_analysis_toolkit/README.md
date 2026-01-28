
## 🌟 特性

- **智能数据分析**: 自动识别数据类型、维度、特征
- **自动绘图选择**: 根据数据结构自动选择最佳图表类型
- **智能参数调整**: 自动调整图形大小、颜色、标签等参数
- **专业绘图样式**: 统一的现代化绘图风格
- **多种图表类型**: 热力图、箱线图、散点图、线图、直方图等
- **模块化设计**: 易于扩展和维护

## 📊 支持的图表类型

### 1. 热力图 (Heatmap)
- 相关性矩阵可视化
- 自动颜色范围调整
- 智能注释显示
- 支持多种颜色映射

### 2. 箱线图 (Boxplot)
- 分组数据可视化
- 自动统计信息显示
- 支持多分组对比
- 智能图形尺寸调整

### 3. 散点图 (Scatter Plot)
- 自动坐标轴选择
- 支持多变量数据
- 智能标签显示

### 4. 线图 (Line Plot)
- 时间序列可视化
- 趋势分析
- 自动坐标轴设置

### 5. 直方图 (Histogram)
- 分布分析
- 自动分箱设置
- 密度估计

### 6. 分布图 (Distribution Plot)
- 带核密度估计的直方图
- 概率密度可视化

### 7. 柱状图 (Bar Plot)
- 分类数据可视化
- 自动分组统计

### 8. EOF模态图 (EOF Modes)
- 经验正交函数空间模态可视化
- 支持1D和2D空间坐标
- 自动颜色范围调整
- 智能子图布局

### 9. 主成分时间序列 (PC Time Series)
- 主成分时间序列可视化
- 支持时间坐标轴
- 多模态对比显示

### 10. EOF方差解释 (EOF Variance)
- 方差解释比例可视化
- 累积方差显示
- 自动百分比标注

### 11. 空间场绘图 (Spatial Field)
- 地理投影支持
- 多种地图投影
- 自定义边界线
- 自动颜色范围调整

### 12. 空间对比图 (Spatial Comparison)
- 多数据集空间对比
- 统一颜色范围
- 网格布局自动调整

### 13. 空间异常图 (Spatial Anomaly)
- 异常场计算和可视化
- 自动颜色范围调整
- 发散色标支持

### 14. 空间相关图 (Spatial Correlation)
- 空间相关性计算
- 相关系数可视化
- 统计显著性标注

### 15. Taylor图 (Taylor Diagram)
- 模型性能评估
- 相关性、标准差比、RMSE可视化
- 多模型对比
- 网格布局支持

### 16. 功率谱密度 (Power Spectrum)
- 多种频谱估计方法
- Welch方法、周期图、FFT
- 自动频率分辨率
- 对数坐标显示

### 17. 频谱对比 (Spectrum Comparison)
- 多数据集频谱对比
- 自动颜色分配
- 图例自动生成

### 18. 频谱网格 (Spectrum Grid)
- 多变量频谱网格显示
- 自动布局调整
- 统一坐标轴

### 19. 频谱分析 (Spectrum Analysis)
- 综合频谱分析
- 时间序列、功率谱、周期图、自相关
- 统计信息显示
- 主导频率识别

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 主要依赖：matplotlib, seaborn, pandas, numpy

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd climate_analysis_toolkit

# 安装依赖
pip install -r requirements.txt

# 安装工具包
pip install -e .
```

### 基础使用

#### 1. 完全自动模式
```python
from climate_analysis_toolkit.src.plotting import auto_plot

# 自动选择最佳图表类型
fig, ax = auto_plot(your_data)
```

#### 2. 指定图表类型
```python
# 指定热力图
fig, ax = auto_plot(your_data, plot_type='heatmap')

# 指定箱线图
fig, ax = auto_plot(your_data, plot_type='boxplot')
```

#### 3. 数据分析
```python
from climate_analysis_toolkit.src.plotting import analyze_data_structure

# 分析数据结构
analysis = analyze_data_structure(your_data)
print(analysis['suggested_plots'])  # 查看建议的图表类型
```

## 📁 项目结构

```
climate_analysis_toolkit/
├── src/
│   └── plotting/
│       ├── __init__.py           # 主接口
│       ├── smart_plotter.py      # 智能绘图主模块
│       ├── heatmap.py           # 热力图模块
│       ├── boxplot.py           # 箱线图模块
│       └── statistical_plots.py # 统计绘图模块
├── requirements.txt
├── setup.py
└── README.md
```

## 🎯 智能特性

### 自动参数调整
- **图形尺寸**: 根据数据维度自动调整
- **颜色方案**: 根据数据类型自动选择
- **标签旋转**: 根据标签长度自动调整
- **注释显示**: 根据数据量自动决定是否显示

### 数据识别
- **数据类型**: DataFrame、numpy数组、列表、字典
- **数据维度**: 1D、2D、多维数据
- **变量类型**: 分类变量、数值变量
- **时间序列**: 自动检测时间序列特征

## 📝 使用示例

### 相关性分析
```python
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(0, 1, 100),
    'C': np.random.normal(0, 1, 100)
})

# 自动绘制相关性热力图
fig, ax = auto_plot(data, plot_type='correlation_heatmap')
```

### 分组数据可视化
```python
# 创建分组数据
data = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B', 'C', 'C'] * 20,
    'Value': np.random.normal(0, 1, 120)
})

# 自动绘制箱线图
fig, ax = auto_plot(data, plot_type='boxplot')

### EOF分析可视化
```python
import numpy as np

# 创建EOF模态数据 (n_modes, n_spatial_points)
eof_modes = np.random.randn(4, 100)  # 4个模态，100个空间点

# 绘制EOF模态图
fig, axes = plot_eof_modes(eof_modes, n_modes=4, title="EOF Spatial Modes")

# 创建PC时间序列数据 (n_time, n_modes)
pc_data = np.random.randn(200, 4)  # 200个时间点，4个模态

# 绘制PC时间序列
fig, axes = plot_pc_timeseries(pc_data, title="PC Time Series")

# 创建方差解释数据
variance_ratio = [45.2, 23.1, 12.8, 8.9]  # 各模态方差解释比例

# 绘制方差解释图
fig, ax = plot_eof_variance(variance_ratio, title="EOF Variance Explained")

### 空间绘图
```python
import numpy as np

# 创建空间数据 (lat, lon)
spatial_data = np.random.randn(50, 80)  # 50个纬度点，80个经度点
lats = np.linspace(15, 55, 50)
lons = np.linspace(70, 140, 80)

# 绘制空间场
fig, ax = plot_spatial_field(spatial_data, lats, lons, title="Temperature Field")

# 绘制空间对比
data_dict = {
    'Model A': spatial_data,
    'Model B': spatial_data + np.random.randn(50, 80) * 0.5
}
fig, axes = plot_spatial_comparison(data_dict, lats, lons, title="Model Comparison")
```

### Taylor图
```python
import numpy as np

# 创建观测和模型数据
obs_data = np.random.randn(100)
model_data = {
    'Model A': obs_data + np.random.randn(100) * 0.3,
    'Model B': obs_data + np.random.randn(100) * 0.5,
    'Model C': obs_data + np.random.randn(100) * 0.7
}

# 绘制Taylor图
fig, ax = plot_taylor_diagram(obs_data, model_data, title="Model Performance")
```

### 频谱分析
```python
import numpy as np

# 创建时间序列数据
time_series = np.sin(2 * np.pi * 0.1 * np.arange(1000)) + np.random.randn(1000) * 0.1

# 绘制功率谱密度
fig, ax = plot_power_spectrum(time_series, fs=1.0, title="Power Spectrum")

# 综合频谱分析
fig, axes = plot_spectrum_analysis(time_series, fs=1.0, title="Spectral Analysis")
```

## 📁 输出目录结构

工具包采用统一的输出目录结构，按照计算方法和绘制方法进行分类存放：

### 基础目录结构
```
/sas12t1/ffyan/
├── outputdata/          # 数据文件输出目录
│   ├── eof/            # EOF分析数据
│   ├── common_eof/     # 共同EOF分析数据
│   ├── crpss/          # CRPSS分析数据
│   ├── rmse/           # RMSE分析数据
│   ├── correlation/    # 相关性分析数据
│   ├── spectrum/       # 频谱分析数据
│   └── taylor/         # Taylor分析数据
└── outputplot/         # 图像文件输出目录
    ├── eof/            # EOF分析图像
    ├── common_eof/     # 共同EOF分析图像
    ├── crpss/          # CRPSS分析图像
    ├── rmse/           # RMSE分析图像
    ├── correlation/    # 相关性分析图像
    ├── spectrum/       # 频谱分析图像
    └── taylor/         # Taylor分析图像
```

### 详细目录结构
```
outputdata/
├── eof/
│   ├── temp/
│   │   ├── eof_temp_ECMWF-51-mon_lead1.pkl
│   │   └── eof_temp_NCEP-2_lead2.pkl
│   └── prec/
│       └── eof_prec_ECMWF-51-mon_lead1.pkl
├── rmse/
│   ├── temp/
│   │   ├── rmse_temp_ECMWF-51-mon_lead1.nc
│   │   └── rmse_temp_NCEP-2_lead2.nc
│   └── prec/
│       └── rmse_prec_ECMWF-51-mon_lead1.nc
└── ...

outputplot/
├── eof/
│   ├── temp/
│   │   ├── eof_modes/
│   │   │   ├── eof_temp_eof_modes_ECMWF-51-mon_lead1.png
│   │   │   └── eof_temp_eof_modes_NCEP-2_lead2.png
│   │   ├── pc_timeseries/
│   │   │   ├── eof_temp_pc_timeseries_ECMWF-51-mon_lead1.png
│   │   │   └── eof_temp_pc_timeseries_NCEP-2_lead2.png
│   │   └── variance/
│   │       ├── eof_temp_variance_ECMWF-51-mon_lead1.png
│   │       └── eof_temp_variance_NCEP-2_lead2.png
│   └── prec/
│       └── ...
├── rmse/
│   ├── temp/
│   │   ├── spatial/
│   │   │   ├── rmse_temp_spatial_ECMWF-51-mon_lead1.png
│   │   │   └── rmse_temp_spatial_NCEP-2_lead2.png
│   │   ├── boxplot/
│   │   │   ├── rmse_temp_boxplot_ECMWF-51-mon_lead1.png
│   │   │   └── rmse_temp_boxplot_NCEP-2_lead2.png
│   │   └── comparison/
│   │       └── rmse_temp_comparison_ECMWF-51-mon_lead1.png
│   └── prec/
│       └── ...
└── ...
```

### 文件命名规范

文件命名采用以下格式：
```
{计算方法}_{变量类型}_{绘制方法}_{模型名称}_{预报时效}_{模态编号}.{后缀}
```

示例：
- `eof_temp_eof_modes_ECMWF-51-mon_lead1_mode1.png`
- `rmse_temp_spatial_NCEP-2_lead2.png`
- `crpss_temp_boxplot_UKMO-14_lead0.png`

### 使用输出配置

```python
from src.config.output_config import (
    get_eof_plot_path,
    get_rmse_plot_path,
    get_crpss_plot_path,
    get_standard_filename
)

# 获取EOF图像输出路径
eof_plot_path = get_eof_plot_path("temp", "eof_modes", "ECMWF-51-mon", leadtime=1, mode=1)

# 获取RMSE图像输出路径
rmse_plot_path = get_rmse_plot_path("temp", "spatial", "NCEP-2", leadtime=2)

# 获取标准文件名
filename = get_standard_filename("eof", "temp", "eof_modes", "ECMWF-51-mon", 1, 1)
# 结果: "eof_temp_eof_modes_ECMWF-51-mon_lead1_mode1.png"
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具包。

## �� 许可证

MIT License

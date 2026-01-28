# 统一绘图工具使用指南

## 📋 概述

`plotting_utils.py` 位于 `climate_analysis_toolkit/src/utils/` 中，提供了标准化的空间分布图绘制功能，统一了以下配置：
- **子图间隙**: hspace=0.25, wspace=0.15
- **刻度**: gridlines，字体 12pt
- **绘图方式**: contourf (填色) + contour (轮廓线)
- **显著性标记**: 黑点打点

## 🚀 快速开始

### 导入方式

在MMPE文件夹的脚本中，已经配置了toolkit路径，可以直接导入：

```python
# 方式1: 直接从utils导入（推荐）
from src.utils.plotting_utils import create_spatial_distribution_figure

# 方式2: 完整导入
from src.utils import (
    create_spatial_distribution_figure,
    STANDARD_CONFIG,
    setup_cartopy_axes,
)
```

### 基本用法

```python
from src.utils.plotting_utils import create_spatial_distribution_figure

# 准备数据
data_dict = {
    'ECMWF-51-mon': {
        0: xr.DataArray(...),  # Lead 0数据
        3: xr.DataArray(...)   # Lead 3数据
    },
    'CMCC-35': {...},
    # ... 其他模型
}

# 创建图形
fig = create_spatial_distribution_figure(
    data_dict=data_dict,
    leadtimes=[0, 3],
    vmin=-1,
    vmax=1,
    title='Temporal ACC',
    colorbar_label='ACC',
    output_file='output/acc_maps.png'
)
```

### 添加显著性检验标记

```python
# 准备显著性掩码
significance_dict = {
    'ECMWF-51-mon': {
        0: xr.DataArray(p_values < 0.05),  # 布尔数组
        3: xr.DataArray(p_values < 0.05)
    },
    # ... 其他模型
}

# 绘制带显著性标记的图
fig = create_spatial_distribution_figure(
    data_dict=data_dict,
    leadtimes=[0, 3],
    significance_dict=significance_dict,
    vmin=-1,
    vmax=1,
    title='Temporal ACC (dots: p < 0.05)',
    colorbar_label='ACC',
    output_file='output/acc_maps_significant.png'
)
```

### 自定义配置

```python
from src.utils.plotting_utils import STANDARD_CONFIG, create_spatial_distribution_figure

# 修改配置
custom_config = STANDARD_CONFIG.copy()
custom_config['hspace'] = 0.3  # 增大垂直间隙
custom_config['tick_fontsize'] = 14  # 增大刻度字体

fig = create_spatial_distribution_figure(
    data_dict=data_dict,
    leadtimes=[0, 3],
    config=custom_config,
    ...
)
```

## 📚 核心函数说明

### `create_spatial_distribution_figure()`
创建标准化的空间分布组合图

**参数**:
- `data_dict`: 模型数据字典 `{model: {leadtime: DataArray}}`
- `leadtimes`: 要绘制的leadtime列表（如 `[0, 3]`）
- `lon_range`: 经度范围，默认 `(70, 140)`
- `lat_range`: 纬度范围，默认 `(15, 55)`
- `levels`: 等高线层级（数组或整数）
- `cmap`: colormap名称，默认 `'RdBu_r'`
- `vmin`, `vmax`: 数据范围
- `significance_dict`: 显著性掩码字典（可选）
- `title`: 总标题
- `colorbar_label`: colorbar标签
- `output_file`: 输出文件路径（可选）
- `config`: 自定义配置字典（可选）

### `setup_cartopy_axes()`
设置单个地图轴的标准配置

### `plot_spatial_field_contour()`
绘制填色等高线 + 轮廓线

### `add_significance_stippling()`
添加显著性打点标记

### `create_discrete_colormap_norm()`
创建离散型colormap（用于固定范围）

## 🔧 在现有代码中集成

### 示例1: 修改 `combined_pearson_analysis.py`

**原代码**:
```python
def plot_acc_spatial_maps(self, model_temporal_acc_maps):
    # 大量自定义代码...
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 4, ...)
    # ... 200+ 行代码
```

**新代码**:
```python
# 在文件顶部添加导入（MMPE脚本已有toolkit路径配置）
from src.utils.plotting_utils import create_spatial_distribution_figure

def plot_acc_spatial_maps(self, model_temporal_acc_maps):
    # 准备数据
    data_dict = {}
    significance_dict = {}
    
    for model, leadtime_data in model_temporal_acc_maps.items():
        data_dict[model] = {}
        significance_dict[model] = {}
        for lt, acc_ds in leadtime_data.items():
            data_dict[model][lt] = acc_ds['temporal_acc']
            significance_dict[model][lt] = acc_ds['significant']
    
    # 一行调用完成绘图
    output_file = self.plot_dir / f"acc_spatial_maps_L0_L3_{self.var_type}.png"
    create_spatial_distribution_figure(
        data_dict=data_dict,
        leadtimes=[0, 3],
        significance_dict=significance_dict,
        vmin=-1,
        vmax=1,
        title=f'{self.var_type.upper()} - Temporal ACC (dots: p < 0.05)',
        colorbar_label='Temporal ACC',
        output_file=str(output_file)
    )
```

### 示例2: 修改 `acc_intermember_analysis.py`

```python
# 在文件顶部添加导入
from src.utils.plotting_utils import create_spatial_distribution_figure

def plot_acc_spatial_distribution(self, leadtimes, models):
    # 加载数据
    all_data = self._load_models_data(leadtimes, models)
    
    # 转换为标准格式
    data_dict = {}
    for model in models:
        data_dict[model] = {}
        for lt in leadtimes:
            if lt in all_data and model in all_data[lt]:
                acc = all_data[lt][model]['ACC']
                ic = all_data[lt][model]['inter_member']
                data_dict[model][lt] = acc / ic  # ACC/IC ratio
    
    # 使用工具函数绘图
    output_file = self.output_dir / f"acc_spatial_L{leadtimes}_{self.var_type}.png"
    create_spatial_distribution_figure(
        data_dict=data_dict,
        leadtimes=leadtimes,
        cmap='RdBu_r',
        title=f'ACC/IC Ratio - {self.var_type.upper()}',
        colorbar_label='ACC/IC Ratio',
        output_file=str(output_file)
    )
```

## 🎨 配置选项

所有可配置项都在 `STANDARD_CONFIG` 中定义：

```python
STANDARD_CONFIG = {
    'hspace': 0.25,              # 垂直间隙
    'wspace': 0.15,              # 水平间隙
    'tick_fontsize': 12,         # 刻度字体大小
    'title_fontsize': 18,        # 标题字体大小
    'label_fontsize': 14,        # 轴标签字体大小
    'colorbar_fontsize': 14,     # colorbar字体大小
    'grid_linewidth': 0.5,       # 网格线宽度
    'grid_alpha': 0.5,           # 网格线透明度
    'grid_linestyle': '--',      # 网格线样式
    'contour_linewidth': 0.3,    # 等高线线宽
    'contour_alpha': 0.4,        # 等高线透明度
    'significance_marker_size': 2.0,   # 显著性标记大小
    'significance_marker_alpha': 0.8,  # 显著性标记透明度
}
```

## 📂 文件位置

```
climate_analysis_toolkit/
├── src/
│   └── utils/
│       ├── plotting_utils.py      # 绘图工具模块
│       └── __init__.py             # 已导出绘图函数
└── PLOTTING_UTILS_README.md        # 本文档

MMPE/
├── combined_pearson_analysis.py    # 待集成
├── acc_intermember_analysis.py     # 待集成
├── rmse_spread_analysis.py         # 待集成
├── climatology_analysis.py         # 待集成
└── circulation_analysis.py         # 待集成
```

## ✅ 需要修改的文件清单

- [x] `plotting_utils.py` - 工具模块（已创建并移至toolkit）
- [x] `utils/__init__.py` - 已导出绘图函数
- [ ] `MMPE/combined_pearson_analysis.py` - 使用示例
- [ ] `MMPE/acc_intermember_analysis.py` - ACC成员间分析
- [ ] `MMPE/rmse_spread_analysis.py` - RMSE/Spread分析
- [ ] `MMPE/climatology_analysis.py` - 气候态分析
- [ ] `MMPE/circulation_analysis.py` - 环流分析

## 📝 注意事项

1. **数据格式要求**: DataArray 必须包含 `lon` 和 `lat` 坐标
2. **显著性掩码**: 必须是布尔型 DataArray，True 表示显著
3. **模型名称**: 会自动移除 '-mon' 后缀用于显示
4. **文件路径**: 建议使用 `pathlib.Path` 对象

## 🔍 常见问题

**Q: 如何修改子图布局？**
A: 修改 `config['hspace']` 和 `config['wspace']`

**Q: 如何只绘制填色而不绘制轮廓线？**
A: 使用 `plot_spatial_field_contour(..., add_contour_lines=False)`

**Q: 如何调整显著性打点的大小？**
A: 修改 `config['significance_marker_size']`

**Q: 如何使用固定的离散colormap？**
A: 使用 `create_discrete_colormap_norm()` 创建，传入 `vmin`, `vmax`

## 🎨 多Colorbar支持 (NEW!)

### `create_multi_dataset_spatial_figure()`

适用于需要绘制多个数据集（如气候态+偏差）的场景，支持多个独立的colorbar。

#### 使用场景

1. **气候态 + 偏差图**: 第一列显示观测气候态，其他列显示模型偏差
2. **观测 + 多模型图**: 不同数据源使用不同colorbar
3. **多变量对比图**: 每个变量使用独立的colorbar

#### 基本用法

```python
from src.utils.plotting_utils import create_multi_dataset_spatial_figure

# 示例: 气候态 + 偏差图
data_groups = [
    {
        # 第一组: 观测气候态（只在第一列）
        'data_dict': {
            'OBS': {
                0: obs_clim_L0,  # Lead 0观测
                3: obs_clim_L3,  # Lead 3观测
            }
        },
        'cmap': 'viridis',
        'colorbar_label': 'Climatology (K)',
        'column_indices': [0],  # 只应用于第一列
    },
    {
        # 第二组: 模型偏差（其他所有列）
        'data_dict': {
            'ECMWF-51-mon': {0: bias_L0, 3: bias_L3},
            'CMCC-35': {0: bias_L0, 3: bias_L3},
            'DWD-mon-21': {0: bias_L0, 3: bias_L3},
            # ... 其他模型
        },
        'cmap': 'coolwarm',
        'vmin': -5,
        'vmax': 5,
        'colorbar_label': 'Bias (K)',
        'column_indices': [1, 2, 3, 4, 5, 6],  # 应用于其他列
    }
]

# 创建图形
create_multi_dataset_spatial_figure(
    data_groups=data_groups,
    leadtimes=[0, 3],
    output_file='output/climatology_bias.png',
    colorbar_orientation='horizontal'  # 横向colorbar在底部
)
```

#### 参数说明

**data_groups** (List[Dict]): 数据组列表，每个字典包含:
- `data_dict`: Dict[str, Dict[int, xr.DataArray]] - {model: {leadtime: data}}
- `cmap`: str - colormap名称 (如 'viridis', 'coolwarm')
- `vmin`, `vmax`: Optional[float] - colorbar范围（自动检测）
- `levels`: Optional[np.ndarray] - 等高线层级（自动生成）
- `colorbar_label`: str - colorbar标签
- `significance_dict`: Optional[Dict] - 显著性数据（可选）
- `add_contour_lines`: bool - 是否添加轮廓线（默认True）
- `column_indices`: Optional[List[int]] - 该组应用于哪些列（默认全部）

**colorbar_orientation**: str
- `'horizontal'`: 横向排列在底部（推荐用于多colorbar）
- `'vertical'`: 竖向在右侧（只使用第一个colorbar）

#### 高级示例

```python
# 示例2: 不同变量使用不同colorbar
data_groups = [
    {
        'data_dict': {
            'Model1': {0: temperature_data},
            'Model2': {0: temperature_data},
        },
        'cmap': 'RdYlBu_r',
        'colorbar_label': 'Temperature (°C)',
        'column_indices': [0, 1],
    },
    {
        'data_dict': {
            'Model1': {0: precipitation_data},
            'Model2': {0: precipitation_data},
        },
        'cmap': 'BrBG',
        'colorbar_label': 'Precipitation (mm/day)',
        'column_indices': [2, 3],
    }
]

create_multi_dataset_spatial_figure(
    data_groups=data_groups,
    leadtimes=[0],
    output_file='output/multi_variable.png'
)
```

#### 注意事项

1. **column_indices**: 用于精确控制每个数据组应用于哪些列
   - 如果不指定，数据组会应用于包含对应模型的所有列
   - 使用 `column_indices` 可以实现"第一列观测，其他列偏差"的布局

2. **横向vs竖向colorbar**:
   - 横向colorbar适合多个colorbar场景
   - 竖向colorbar只会显示第一个有效的colorbar

3. **自动范围检测**: 如果未指定 `vmin`/`vmax`，会自动从数据中检测

4. **模型顺序**: 所有数据组的模型会合并并去重，按首次出现顺序排列

# Multiple Models Predictions Evaluation (MMPE)

**多模式预测评估工具集**

这是一个用于气候/数值预报多模式预测性能评估的工具集。该项目采用模块化设计，核心分析逻辑位于 `MMPE/` 目录，底层通用功能由 `climate_analysis_toolkit` 支持。

---

## 📂 目录结构概览

### **1. MMPE/ (核心分析脚本)**

面向特定科学问题的入口脚本，所有脚本均已接入 `common_config.py` 进行统一配置。

* **`circulation_analysis.py`**
* **功能**: 环流场诊断分析。
* **内容**: 分析 850hPa/500hPa 的 u/v 风场、位势高度 (GHT) 以及水汽通量散度。
* **特点**: 基于 ERA5 观测数据，自动计算气候平均态、距平 (Anomaly) 及模式偏差 (Bias)。


* **`combined_error_analysis.py`**
* **功能**: 多模式误差综合分析 (V4版本)。
* **内容**: 计算 RMSE, MAE, Bias 指标。
* **特点**: 支持 **Ensemble Member** (集合成员) 维度的误差计算；绘制带有 Spread (离散度) 阴影的时间序列图与空间分布图。


* **`combined_pearson_analysis.py`**
* **功能**: 综合相关性分析。
* **内容**: 计算多模式预测结果与观测之间的 Pearson 相关系数，评估时空一致性。


* **`climatology_analysis.py`**
* **功能**: 气候态基线分析。
* **内容**: 计算长期平均、季节性特征等基线统计量。


* **`nino34_eawm_index_calculation.ipynb`**
* **功能**: 气候指数计算演示。
* **内容**: 计算 Niño 3.4 指数与东亚冬季风 (EAWM) 指数的 Jupyter Notebook 示例。



### **2. 配置与辅助模块**

* **`common_config.py`**: **核心配置文件**。统一管理模型列表 (`MODEL_LIST`)、文件后缀映射、提前期、季节定义 (`SEASONS`) 及数据路径 (`DATA_PATHS`)。
* `heat_map.py` / `taylor_plot.py` / `plot_regions.py`: 专用的绘图辅助模块。

### **3. climate_analysis_toolkit/ (底层工具包)**

提供数据加载、对齐、并行计算等核心支持。

* **`src/utils/data_loader.py`**: 统一数据加载器，支持加载观测、预报及集合成员数据。
* **`src/core/`**: 包含 EOF、RMSE、相关分析等核心算法实现。
* **`src/plotting/`**: 包含空间图、Taylor图等绘图工具。

---

## ⚙️ 配置指南

在运行任何分析前，请务必检查并修改 `MMPE/common_config.py`：

1. **模型列表 (`MODEL_LIST`)**: 定义需要参与评估的模型名称（如 `CMCC-35`, `ECMWF-51-mon` 等）。
2. **数据路径 (`DATA_PATHS`)**:
* `obs_dir`: 观测数据存放路径。
* `forecast_dir`: 模式预报数据存放路径。
* `output_dir`: 结果输出根目录。


3. **空间范围 (`SPATIAL_BOUNDS`)**: 定义分析的经纬度区域（默认为 lat: 15-55, lon: 70-140）。
4. **异常值处理**: 可配置 `REMOVE_OUTLIERS` 开关及阈值。

---

## 🚀 运行示例

### 1. 环流分析

分析风场与位势高度偏差，并生成 NetCDF 结果与图像。

```bash
cd MMPE
# 运行计算与绘图 (使用 8 个进程并行)
python circulation_analysis.py --models "ECMWF-51-mon,NCEP-2" --leadtimes 0 1 --n_jobs 8

# 仅重新绘图 (需已有计算结果)
python circulation_analysis.py --plot-only

```

### 2. 综合误差分析 (RMSE/MAE/Bias)

计算集合平均与成员误差，并绘制含 Spread 的图表。

```bash
cd MMPE
# 对温度变量进行分析
python combined_error_analysis.py --var temp --models all --leadtimes 0 3 --parallel

```

### 3. 相关性分析

评估预测场与观测场的相关性。

```bash
cd MMPE
python combined_pearson_analysis.py --var prec --models "UKMO-14"

```

---

## 🛠️ 维护说明

* **新增模型**: 请在 `climate_analysis_toolkit/src/utils/data_loader.py` 中添加文件名后缀规则，并在 `common_config.py` 的 `MODEL_LIST` 中注册。
* **依赖管理**: 核心依赖包括 `xarray`, `netCDF4`, `matplotlib`, `cartopy`, `regionmask` 等。
* **已知问题**:
* `NCEP-2` 气压层数据缺失 `q` 变量，环流分析中会自动跳过该模型的水汽通量计算。
* 部分绘图脚本依赖具体的 Shapefile 文件路径，请在 `common_config.py` 或脚本顶部确认 `boundaries` 路径。

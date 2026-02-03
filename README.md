# Multiple Models Predictions Evaluation (MMPE)

**多模式预测评估工具集**

这是一个用于气候/数值预报多模式预测性能评估的工具集和示例脚本集合。该仓库主要由两部分组成：位于 `MMPE/` 目录下的具体分析脚本（面向特定分析任务），以及底层的 `climate_analysis_toolkit` 通用工具包（提供数据加载、对齐、并行计算、绘图核心功能）。

---

## 📂 目录结构概览

- **MMPE/** (主要分析脚本与配置)
  - `circulation_analysis.py`: 环流分析主脚本（u/v风场、位势高度、水汽通量散度），基于 ERA5 观测。
  - `combined_error_analysis.py`: 多模式误差综合分析（RMSE, MAE, Bias），支持 Ensemble Member 维度及 Spread 绘制。
  - `climatology_analysis.py`: 气候态基线分析。
  - `combined_pearson_analysis.py`: 综合 Pearson 相关性分析。
  - `block_bootstrap_score.py`: 评分统计量的块自助（Block Bootstrap）置信区间估计。
  - `rmse_spread_analysis.py`: RMSE 与 Ensemble Spread 对比分析。
  - `acc_intermember_analysis.py`: ACC 技巧评分与成员间相关性分析。
  - `common_config.py`: **[新增]** 统一的配置中心，定义模型列表、文件后缀、提前期、季节映射及空间范围。
  - `MMPE_TOOLKIT_INVENTORY.md`: 模块详细清单与依赖说明。

- **climate_analysis_toolkit/src/** (核心工具包)
  - `utils/`: 基础工具
    - `data_loader.py`: 统一的数据加载器（支持单层、多层、Ensemble 数据）。
    - `alignment.py`: 时空对齐工具。
    - `parallel_utils.py`: 并行计算辅助。
    - `logging_config.py`: 统一日志配置。
    - `data_utils.py`: 通用数据处理（含异常值剔除 `remove_outliers_iqr`）。
  - `core/`: 核心算法（EOF、相关分析、CRPSS、RMSE、谱分析等）。
  - `plotting/`: 绘图模块（空间分布图、Taylor图、Heatmap、Smart Plotter）。
  - `config/`: 工具包内部配置（Outlier, Output, Settings）。

---

## 🚀 快速开始

### 1. 环境准备
建议使用 Python 3.8+，并安装以下核心依赖：
```bash
pip install numpy scipy xarray netCDF4 pandas dask joblib matplotlib cartopy seaborn regionmask

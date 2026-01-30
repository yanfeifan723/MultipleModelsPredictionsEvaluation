## MMPE 模块清单

- `circulation_analysis.py`：环流分析主脚本，依赖 toolkit DataLoader、alignment。
- `climatology_analysis.py`：气候态分析。
- `combined_pearson_analysis.py`：综合 Pearson 分析。
- `TOOLKIT_MAINTENANCE.md`：维护记录。

## Toolkit 目录概览 (`climate_analysis_toolkit/src`)

- `utils/`：data_loader、alignment、aggregation、coord_utils、interpolation、logging_config、parallel_utils 等。
- `core/`：eof、correlation、crpss、rmse、spectrum 等分析实现。
- `plotting/`：空间绘图、EOF、heatmap、Taylor、smart_plotter。
- `analyzer.py`、`cli.py`：统一入口。
- `config/`：outlier、output、settings。

## 重复/潜在冗余点

- 多个 MMPE 脚本内嵌 `remove_outliers_iqr`、`find_valid_data_bounds`、ensemble 数据加载逻辑。
- toolkit `data_loader` 与 MMPE 中的 SimpleDataLoader/SimplifiedRMSE loader 重复。
- 多脚本自定义日志配置、颜色表、常量（MODELS/LEADTIMES）相同。
- `climatology_analysis.py`：
  - 再次维护 `VAR_CONFIG`、`MODELS`、图层样式，可复用 toolkit plotting。
  - 并行策略/日志配置与其他脚本重复。
- `combined_pearson_analysis.py`：
  - 自带时间对齐与数据加载，未充分复用 toolkit 工具。
  - MODELS、LEADTIMES、季节映射单独维护。
 
# Multiple Models Predictions Evaluation (MMPE)

一个用于气候/数值预报多模式预测性能评估的工具集和示例脚本集合。该仓库包含一套“MMPE”分析脚本（面向特定分析任务的入口脚本），以及一个更通用的 `climate_analysis_toolkit` 工具包（提供数据加载、对齐、并行、绘图和常用分析函数）。

---

## 目录概览

- 根目录
  - 多个示例分析脚本（如 `circulation_analysis.py`、`block_bootstrap_score.py` 等）
  - 日志输出目录（运行时生成）
- climate_analysis_toolkit/src
  - utils/ — DataLoader、alignment、aggregation、coord_utils、interpolation、logging_config、parallel_utils 等
  - core/ — eof、correlation、crpss、rmse、spectrum 等分析实现
  - plotting/ — 空间绘图、EOF、heatmap、Taylor、smart_plotter
  - analyzer.py、cli.py — 统一入口与命令行工具
  - config/ — outlier、output、settings 等配置文件

---

## 快速开始

1. 克隆仓库并进入目录
   - git clone <repo-url>
   - cd MultipleModelsPredictionsEvaluation

2. 建议的依赖（示例）
   - Python 3.8+
   - numpy, scipy, xarray, netCDF4, pandas, dask, joblib, matplotlib, cartopy, seaborn
   - 可选：pytorch/numba/CuPy（如果使用 GPU 加速的自定义并行实现）

   示例（使用 pip）：
   - python -m pip install -r requirements.txt
   （如果仓库中无 requirements.txt，请根据上面列出的包创建一个）

3. 示例运行
   - 使用工具包统一入口（若已实现 CLI）：
     - python climate_analysis_toolkit/src/cli.py run --task circulation --config path/to/config.yaml
   - 或直接运行脚本（示例）：
     - python circulation_analysis.py --models "MODEL1,MODEL2" --leadtimes 1,5,10 --var t2m --outdir ./outputs

   注意：各脚本接收不同参数，建议优先使用 toolkit 的 DataLoader 与 CLI 配置来保证一致性。

---

## 主要脚本与模块说明

- circulation_analysis.py
  - 面向环流相关的诊断分析脚本，依赖 toolkit 的 DataLoader 与 alignment 工具进行时空对齐与字段读取。

- block_bootstrap_score.py
  - 使用块自助（block bootstrap）方法对评分统计量进行置信区间估计与差异检验。当前脚本包含自定义 SimpleDataLoader 与 ParallelProcessor（与 toolkit 存在重复实现——见“重构建议”）。

- rmse_spread_analysis.py
  - 计算并绘制 RMSE 与 ensemble spread 的对比图，包含 SimplifiedRMSE、SpreadPlotter。

- acc_intermember_analysis.py
  - 计算预测与观测的 ACC（Anomaly Correlation Coefficient）以及成员间相关，包含异常值剔除函数（如 remove_outliers_iqr）。

- climatology_analysis.py
  - 气候态基线分析（长期平均、季节性/分季统计等）。

- combined_pearson_analysis.py
  - 使用综合 Pearson 相关用于多模式一致性或相关性分析；包含时间对齐与数据加载逻辑。

- TOOLKIT_MAINTENANCE.md
  - 维护记录（说明已知问题和历史决策）

---

## 已发现的重复与潜在冗余（现状总结）

在仓库当前代码中，看到多处实现重复或功能重叠，主要包括：

- 多处脚本中重复实现的数据加载/对齐逻辑（如 `SimpleDataLoader`, `SimplifiedRMSE` 内的加载代码），与 toolkit 的 `data_loader` 功能重合。
- 多脚本各自维护的一致性配置（`VAR_CONFIG`, `MODELS`, `LEADTIMES`, `SPATIAL_BOUNDS` 等）导致维护成本上升。
- 日志配置、并行检测、颜色映射等在多个脚本中重复实现（例如 `PARALLEL_AVAILABLE`、自定义 `ParallelProcessor`），而 toolkit 中已有 `parallel_utils`、`logging_config` 可复用。
- 异常值处理（如 `remove_outliers_iqr`）和有效网格检测散落在多个脚本中，应抽象为公共工具函数。

---

## 推荐的重构与迁移策略（优先级与步骤）

优先级高（建议立即着手）：
1. 统一数据加载
   - 将所有脚本改为使用 toolkit 的 `DataLoader`（或提供兼容的轻量封装），以便集中处理文件名/路径差异与变量名映射。
2. 合并并行与日志模块
   - 移除脚本中散落的 `PARALLEL_AVAILABLE` 兜底实现和自定义 `ParallelProcessor`，改用 `climate_analysis_toolkit/src/utils/parallel_utils.py` 与 `logging_config.py`。
3. 抽取并发布公共配置
   - 将 `VAR_CONFIG`、`MODELS`、`LEADTIMES`、季节映射等集中到 `config/` 下，并在脚本中以相对导入或 CLI 参数引用。

中等优先级（后续优化）：
4. 收敛重复的统计/绘图函数
   - 将 `SimplifiedRMSE`、`SpreadPlotter`、异常值剔除、有效网格检测等函数迁移到 toolkit 的 `core/` 或 `utils/`。
5. 清理遗留代码
   - 将被替换或弃用的实现保存在 `old/` 或 `deprecated/` 中，并编写 `DEPRECATED_CODE.md` 说明迁移理由和位置。

长期改进（可选）：
6. 引入测试与 CI
   - 为关键数据加载、对齐与统计函数编写单元测试，配置 GitHub Actions 以防回归。
7. 文档与示例
   - 在 repo 根目录添加更多示例配置文件与最小可运行数据示例，方便新用户上手。

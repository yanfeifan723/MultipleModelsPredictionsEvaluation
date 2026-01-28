## MMPE 模块清单

- `circulation_analysis.py`：环流分析主脚本，依赖 toolkit DataLoader、alignment。
- `block_bootstrap_score.py`：块自助得分分析，包含 SimpleDataLoader、ParallelProcessor 等重复工具。
- `rmse_spread_analysis.py`：RMSE 与 spread 分析，含 SimplifiedRMSE、SpreadPlotter。
- `acc_intermember_analysis.py`：ACC 与成员间相关分析，自带 remove_outliers_iqr 等工具。
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
- `block_bootstrap_score.py` 仍保留自定义 `ParallelProcessor` 与 toolkit `parallel_utils` 功能重叠，需要在下一步合并。
- 日志目录中 `block_bootstrap_score.log`、`circulation_analysis.log` 反映的缺失变量/文件问题需与数据加载逻辑同步检查。
- `block_bootstrap_score.py`：
  - `SimpleDataLoader.load_obs_data/load_forecast_data` 与 toolkit 重复，可改用 `DataLoader`。
  - `PARALLEL_AVAILABLE` 兜底实现、GPU 检测与日志配置均散落在脚本中，和 toolkit `parallel_utils`、`logging_config` 重合。
  - `VAR_CONFIG`, `MODELS`, `LEADTIMES`, `SPATIAL_BOUNDS` 在脚本内再次定义。
- `rmse_spread_analysis.py`：
  - `SimplifiedRMSE` 和 `SpreadPlotter` 中的观测/预报加载逻辑重复。
  - 输出路径、季节映射与其他脚本一致，考虑集中在配置模块。
- `climatology_analysis.py`：
  - 再次维护 `VAR_CONFIG`、`MODELS`、图层样式，可复用 toolkit plotting。
  - 并行策略/日志配置与其他脚本重复。
- `combined_pearson_analysis.py`：
  - 自带时间对齐与数据加载，未充分复用 toolkit 工具。
  - MODELS、LEADTIMES、季节映射单独维护。

## 后续清理关注点

1. 统一 ensemble 数据加载：所有脚本改用 `DataLoader`/toolkit 助手，便于集中修复文件命名差异。
2. 合并异常值、有效网格、日志配置等通用函数，减少单脚本重复实现。
3. 审核 toolkit `utils.*` 模块，删除未被 MMPE 或 toolkit core 使用的旧函数，必要时迁移到 `old/`.
4. `block_bootstrap_score.py` 优先改造：移除 `SimpleDataLoader`、兜底 `ParallelProcessor`，改用 toolkit 并发/日志模块，并将旧实现记录到 `old/DEPRECATED_CODE.md`。

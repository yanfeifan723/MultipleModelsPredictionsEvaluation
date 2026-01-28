## MMPE / Toolkit 维护记录

更新时间：2025-11-25

### 修复内容
- `load_fcst_data_at_level` 与 `DataLoader.load_pressure_level_data` 逻辑完全对齐，避免两侧重复维护。
- `remove_outliers_iqr`、`find_valid_data_bounds` 合并至 `climate_analysis_toolkit/src/utils/data_utils.py`，所有脚本统一调用。
- 新增 `MMPE/common_config.py` 统一模型列表、文件后缀、提前期与空间范围，`circulation_analysis.py`、`acc_intermember_analysis.py`、`rmse_spread_analysis.py`、`block_bootstrap_score.py`、`climatology_analysis.py`、`combined_pearson_analysis.py` 已切换到共享配置。
- `DataLoader.load_forecast_data_ensemble` 新增，`acc_intermember_analysis.py`、`rmse_spread_analysis.py`、`block_bootstrap_score.py` 共用一套 ensemble 读取逻辑。
- `block_bootstrap_score.py` 移除 `SimpleDataLoader`，直接使用 toolkit `DataLoader` + 本地 resample helper，避免重复代码。

### 已知问题
- `NCEP-2` pressure-level 文件缺少 `q` 变量，导致环流分析在该模式上无法计算 850 hPa 水汽通量散度；脚本目前会跳过并记录警告。

### 后续建议
- 当新增变量或模式时，请先在 `climate_analysis_toolkit/src/utils/data_loader.py` 中扩展 `get_model_suffix` 与 `load_pressure_level_data`，再由 `MMPE` 侧直接调用，保持单一事实源。
- 若脚本需要新的通用工具函数，请首先在 toolkit 中实现后再在 `MMPE` 引用，避免再次出现重复实现。
- 如需扩展模型列表，请在 `common_config.py` 中集中维护，并在各脚本通过 import 获取，确保目录与日志的一致性。


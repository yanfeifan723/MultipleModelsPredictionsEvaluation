# MMPE 数据处理方法文档

本文档整理了MMPE工具包中使用的所有数据处理方法及其数学公式，包含详细的实现步骤、边界情况处理和应用示例。

## 目录

1. [相关系数计算](#1-相关系数计算)
2. [误差指标计算](#2-误差指标计算)
3. [气候态与距平计算](#3-气候态与距平计算)
4. [Bootstrap方法](#4-bootstrap方法)
5. [Brier Score与Brier Skill Score](#5-brier-score与brier-skill-score)
6. [Ensemble Spread计算](#6-ensemble-spread计算)
7. [空间处理方法](#7-空间处理方法)
8. [时间处理方法](#8-时间处理方法)
9. [标准化方法](#9-标准化方法)
10. [指数计算](#10-指数计算)
11. [综合评分方法](#11-综合评分方法)
12. [置信区间与显著性检验](#12-置信区间与显著性检验)
13. [线性回归分析](#13-线性回归分析)
14. [数据验证与质量控制](#14-数据验证与质量控制)
15. [内存优化策略](#15-内存优化策略)
16. [网格匹配与对齐](#16-网格匹配与对齐)

---

## 1. 相关系数计算

### 1.1 Pearson相关系数

Pearson相关系数用于衡量两个变量之间的线性相关程度。

#### 公式

对于两个时间序列 $X(t)$ 和 $Y(t)$，Pearson相关系数定义为：

$$r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2 \sum_{i=1}^{n}(Y_i - \bar{Y})^2}}$$

其中：
- $n$ 为样本数量
- $\bar{X} = \frac{1}{n}\sum_{i=1}^{n}X_i$ 为 $X$ 的均值
- $\bar{Y} = \frac{1}{n}\sum_{i=1}^{n}Y_i$ 为 $Y$ 的均值

#### 沿时间维度的Pearson相关系数

对于空间场数据，沿时间维度计算每个格点的相关系数：

$$r(x,y) = \frac{\sum_{t=1}^{T}(A_t(x,y) - \bar{A}(x,y))(B_t(x,y) - \bar{B}(x,y))}{\sqrt{\sum_{t=1}^{T}(A_t(x,y) - \bar{A}(x,y))^2 \sum_{t=1}^{T}(B_t(x,y) - \bar{B}(x,y))^2}}$$

其中：
- $A_t(x,y)$ 和 $B_t(x,y)$ 分别为时刻 $t$ 在格点 $(x,y)$ 的值
- $\bar{A}(x,y) = \frac{1}{T}\sum_{t=1}^{T}A_t(x,y)$
- $\bar{B}(x,y) = \frac{1}{T}\sum_{t=1}^{T}B_t(x,y)$

#### 实现细节

**向量化计算**：
使用NumPy的向量化操作，避免循环：

```python
# 计算均值
a_mean = np.nanmean(a, axis=0)  # 沿时间维度求平均
b_mean = np.nanmean(b, axis=0)

# 计算偏差
a_dev = a - a_mean
b_dev = b - b_mean

# 计算相关系数
numerator = np.nansum(a_dev * b_dev, axis=0)
denominator = np.sqrt(np.nansum(a_dev**2, axis=0) * np.nansum(b_dev**2, axis=0))
r = np.where(denominator > 0, numerator / denominator, np.nan)
```

**异常值处理**（可选）：
在计算相关系数前，可以使用IQR方法去除异常值（见[14.1节](#141-iqr方法)）。

**最小样本数要求**：
- 至少需要 $n \geq 3$ 个有效样本（推荐 $n \geq 5$）
- 如果有效样本数不足，返回 `NaN`

**NaN值处理**：
- 使用 `np.nanmean()` 和 `np.nansum()` 忽略NaN值
- 检查有效样本数：`valid_count = (~np.isnan(a) & ~np.isnan(b)).sum(axis=0)`
- 如果 `valid_count < threshold`，返回 `NaN`

### 1.2 年际相关系数

计算年际时间序列的相关系数，用于评估模式对年际变化的预测能力。

#### 计算步骤

1. **空间平均**：对观测和预报分别做空间平均，得到逐月区域平均时间序列
   $$\bar{O}_t = \frac{1}{N}\sum_{i,j}O_t(i,j)$$
   $$\bar{F}_t = \frac{1}{N}\sum_{i,j}F_t(i,j)$$

2. **去除气候态**（可选）：按月距平，保留年际信号
   $$O'_t = O_t - \bar{O}_{month(t)}$$
   $$F'_t = F_t - \bar{F}_{month(t)}$$
   其中 $\bar{O}_{month(t)}$ 为对应月份的气候态平均值。

3. **年平均**：对逐月序列按年平均，得到逐年序列
   $$O_y = \frac{1}{12}\sum_{t \in year(y)}O'_t$$
   $$F_y = \frac{1}{12}\sum_{t \in year(y)}F'_t$$

4. **Pearson相关**：对两个逐年序列计算相关系数
   $$r_{interannual} = \frac{\sum_{y}(O_y - \bar{O})(F_y - \bar{F})}{\sqrt{\sum_{y}(O_y - \bar{O})^2 \sum_{y}(F_y - \bar{F})^2}}$$

### 1.3 空间平均后的相关系数

先对观测和预报分别进行空间平均，再计算时间序列的相关系数：

$$\bar{O}_t = \frac{1}{N}\sum_{i,j}O_t(i,j)$$
$$\bar{F}_t = \frac{1}{N}\sum_{i,j}F_t(i,j)$$

$$r_{spatial\_mean} = \frac{\sum_{t}(\bar{O}_t - \bar{\bar{O}})(\bar{F}_t - \bar{\bar{F}})}{\sqrt{\sum_{t}(\bar{O}_t - \bar{\bar{O}})^2 \sum_{t}(\bar{F}_t - \bar{\bar{F}})^2}}$$

---

## 2. 误差指标计算

### 2.1 均方根误差 (RMSE)

RMSE用于衡量预报与观测之间的差异。

#### 空间RMSE

对于每个格点 $(i,j)$，计算时间平均的RMSE：

$$RMSE(i,j) = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(F_t(i,j) - O_t(i,j))^2}$$

#### 实现细节

**有效数据掩码**：
只计算观测和预报都有效的格点：

$$\text{valid}(t,i,j) = \begin{cases}
1 & \text{如果 } O_t(i,j) \text{ 和 } F_t(i,j) \text{ 都有效} \\
0 & \text{否则}
\end{cases}$$

$$RMSE(i,j) = \sqrt{\frac{\sum_{t=1}^{T}\text{valid}(t,i,j) \cdot (F_t(i,j) - O_t(i,j))^2}{\sum_{t=1}^{T}\text{valid}(t,i,j)}}$$

**最小有效样本数**：
- 至少需要 $N_{valid} \geq 12$ 个有效时间点（推荐 $N_{valid} \geq 0.2T$）
- 如果有效样本数不足，返回 `NaN`

**内存优化**：
对于大数组，使用分块计算：

1. 将时间维度分成多个块：$T = \bigcup_{k=1}^{K} T_k$
2. 对每个块计算平方差：$D_k(i,j) = \frac{1}{|T_k|}\sum_{t \in T_k}(F_t(i,j) - O_t(i,j))^2$
3. 合并结果：$RMSE(i,j) = \sqrt{\frac{1}{K}\sum_{k=1}^{K}D_k(i,j)}$

#### 时间序列RMSE

对于每个时刻 $t$，计算空间平均的RMSE：

$$RMSE(t) = \sqrt{\frac{1}{N}\sum_{i,j}(F_t(i,j) - O_t(i,j))^2}$$

#### 总体RMSE

$$RMSE = \sqrt{\frac{1}{NT}\sum_{t=1}^{T}\sum_{i,j}(F_t(i,j) - O_t(i,j))^2}$$

### 2.2 归一化RMSE得分

将RMSE归一化到[0,1]区间，并反向定义为得分（RMSE越小，得分越高）：

$$n_{RMSE} = \frac{RMSE - RMSE_{min}}{RMSE_{max} - RMSE_{min}}$$

$$s_{RMSE} = 1 - \text{clip}(n_{RMSE}, 0, 1)$$

其中 $RMSE_{min}$ 和 $RMSE_{max}$ 分别为所有格点的最小和最大RMSE值。

---

## 3. 气候态与距平计算

### 3.1 月气候态

计算每个月份的气候态平均值（多年平均）：

$$\bar{X}_{month} = \frac{1}{Y}\sum_{y=1}^{Y}X_{y,month}$$

其中 $Y$ 为基期内的年数。

#### 实现步骤

1. **选择基期数据**：
   $$X_{baseline} = X[\text{time} \in [T_{start}, T_{end}]]$$
   其中 $T_{start}$ 和 $T_{end}$ 分别为基期的开始和结束时间。

2. **按月份分组**：
   使用 `groupby('time.month')` 将数据按月份分组。

3. **计算平均值**：
   $$\bar{X}_{m} = \frac{1}{N_m}\sum_{t:month(t)=m}X_t$$
   其中 $N_m$ 为基期内月份 $m$ 的样本数。

#### 输出格式

气候态数据维度为 `(month, lat, lon)` 或 `(month, number, lat, lon)`（如果有ensemble维度）。

#### 应用

- 计算月距平：$X'_t = X_t - \bar{X}_{month(t)}$
- 标准化数据：去除季节循环，保留年际变化信号

### 3.2 季节气候态

计算每个季节的气候态平均值：

$$\bar{X}_{season} = \frac{1}{Y \times M_{season}}\sum_{y=1}^{Y}\sum_{m \in season}X_{y,m}$$

其中 $M_{season}$ 为季节包含的月数（通常为3）。

### 3.3 月距平

从原始数据中去除月气候态：

$$X'_{t} = X_t - \bar{X}_{month(t)}$$

其中 $month(t)$ 为时刻 $t$ 对应的月份。

### 3.4 季节距平

从原始数据中去除季节气候态：

$$X'_{t} = X_t - \bar{X}_{season(t)}$$

其中 $season(t)$ 为时刻 $t$ 对应的季节。

**注意**：对于跨年季节（如DJF），12月归属于下一年：
- 1992年12月 → 1993年DJF
- 1993年1月 → 1993年DJF
- 1993年2月 → 1993年DJF

#### 跨年季节处理实现

对于DJF季节，需要特殊处理：

1. **创建season_year坐标**：
   $$\text{season\_year}(t) = \begin{cases}
   \text{year}(t) + 1 & \text{如果 } \text{month}(t) = 12 \\
   \text{year}(t) & \text{否则}
   \end{cases}$$

2. **按season_year分组**：
   $$X_{season,y} = \frac{1}{M_{season}}\sum_{t: \text{season\_year}(t) = y \land \text{month}(t) \in \text{season}}X_t$$

   其中 $M_{season}$ 为季节包含的月数（通常为3）。

3. **其他季节**（MAM、JJA、SON）：
   直接按年份分组，无需特殊处理。

---

## 4. Bootstrap方法

### 4.1 Block Bootstrap

Block Bootstrap用于评估统计量的置信区间，保持时间序列的自相关性。

#### 方法步骤

1. **划分时间块**：将时间序列划分为 $B$ 个不重叠的时间块，每个块长度为 $L$
   $$Block_i = \{t_{iL+1}, t_{iL+2}, ..., t_{(i+1)L}\}, \quad i = 0, 1, ..., B-1$$
   
   实际上，使用滑动窗口可以创建更多块：
   $$n_{blocks} = T - L + 1$$
   其中 $T$ 为时间序列长度。

2. **Bootstrap重采样**：从 $n_{blocks}$ 个块中有放回地随机抽取 $n_{blocks}$ 个块，组成新的时间序列
   $$S_b = \{Block_{i_1}, Block_{i_2}, ..., Block_{i_{n_{blocks}}}\}$$
   其中 $i_j \sim \text{Uniform}(0, n_{blocks}-1)$，$b = 1, 2, ..., B$

3. **计算统计量**：对每个bootstrap样本计算目标统计量
   $$S_b = f(S_b)$$
   其中 $f$ 为目标统计量函数（如RMSE、相关系数等）

4. **置信区间**：使用所有bootstrap样本的统计量分布计算置信区间
   $$CI_{1-\alpha} = [Q_{\alpha/2}, Q_{1-\alpha/2}]$$
   其中 $Q_p$ 为第 $p$ 分位数。

#### Block大小选择

- **默认值**：$L = 12$ 个月（1年）
- **原理**：保持季节循环的完整性
- **权衡**：
  - $L$ 太小：可能破坏自相关性
  - $L$ 太大：bootstrap样本数减少，统计精度降低

#### Bootstrap次数

- **默认值**：$B = 1000$ 次
- **原理**：平衡计算成本和统计精度
- **建议**：至少 $B \geq 500$ 次，推荐 $B = 1000$ 次

#### 实现细节

**独立随机数生成器**：
每个进程使用独立的随机数生成器，避免线程安全问题：

```python
rng = np.random.RandomState()
selected_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
```

**Fisher z变换**（用于相关系数）：
1. 对相关系数进行Fisher z变换：$z_b = \text{arctanh}(r_b)$
2. 计算z的置信区间：$CI_z = [Q_{z,\alpha/2}, Q_{z,1-\alpha/2}]$
3. 逆变换得到原始尺度的置信区间：$CI_r = [\tanh(Q_{z,\alpha/2}), \tanh(Q_{z,1-\alpha/2})]$

### 4.2 Fisher z变换

Fisher z变换用于将相关系数转换为近似正态分布，便于进行统计推断。

#### 变换公式

$$z = \text{arctanh}(r) = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right)$$

其中 $r$ 为相关系数，$r \in (-1, 1)$。

#### 逆变换

$$r = \tanh(z) = \frac{e^{2z} - 1}{e^{2z} + 1}$$

#### 应用

在Block Bootstrap中，对相关系数进行Fisher z变换后再计算置信区间，最后通过逆变换得到原始尺度的置信区间。

---

## 5. Brier Score与Brier Skill Score

### 5.1 Brier Score (BS)

Brier Score用于评估概率预报的准确性。

#### 公式

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$$

其中：
- $p_i$ 为预报概率（0到1之间）
- $o_i$ 为观测值（0或1，二元事件）
- $N$ 为样本数量

#### 在MMPE中的应用

将连续变量转换为二元事件（如是否超过阈值），计算ensemble预报的概率：

$$p_i = \frac{1}{M}\sum_{m=1}^{M}I(F_{i,m} > threshold)$$

其中 $I(\cdot)$ 为指示函数，$M$ 为ensemble成员数。

### 5.2 Brier Skill Score (BSS)

BSS用于衡量预报相对于参考预报（如气候态预报）的技能。

#### 公式

$$BSS = 1 - \frac{BS}{BS_{ref}}$$

其中：
- $BS$ 为预报的Brier Score
- $BS_{ref}$ 为参考预报的Brier Score（通常使用气候态概率）

#### 解释

- $BSS > 0$：预报优于参考预报
- $BSS = 0$：预报与参考预报相当
- $BSS < 0$：预报劣于参考预报
- $BSS = 1$：完美预报

#### 参考预报

使用气候态命中率作为参考预报概率：

$$p_{ref} = \frac{1}{T}\sum_{t=1}^{T}I(O_t > threshold)$$

---

## 6. Ensemble Spread计算

### 6.1 Ensemble Mean

计算ensemble的平均值：

$$\bar{F}_t(i,j) = \frac{1}{M}\sum_{m=1}^{M}F_{t,m}(i,j)$$

其中 $M$ 为ensemble成员数。

### 6.2 Total Spread

计算所有成员和时间点的平均绝对偏差：

$$Spread_{total}(i,j) = \frac{1}{TM}\sum_{t=1}^{T}\sum_{m=1}^{M}|F_{t,m}(i,j) - \bar{F}_t(i,j)|$$

### 6.3 Spread per Member

计算每个成员的平均绝对偏差：

$$Spread_m(i,j) = \frac{1}{T}\sum_{t=1}^{T}|F_{t,m}(i,j) - \bar{F}_t(i,j)|$$

### 6.4 Temporal Spread

计算空间平均后的时间序列spread：

$$Spread_t = \frac{1}{NM}\sum_{i,j}\sum_{m=1}^{M}|F_{t,m}(i,j) - \bar{F}_t(i,j)|$$

### 6.5 Spread-Error Ratio

计算ensemble spread与RMSE的比值，用于评估ensemble的校准程度。

#### 公式

$$R_{spread/error}(i,j) = \frac{Spread_{total}(i,j)}{RMSE(i,j) + \epsilon}$$

其中 $\epsilon = 10^{-10}$ 为小常数，防止除零。

#### 解释

- $R \approx 1$：ensemble校准良好，spread与误差匹配
- $R < 1$：ensemble欠分散（under-dispersive），spread小于实际误差
- $R > 1$：ensemble过分散（over-dispersive），spread大于实际误差

#### 应用

用于评估ensemble预报系统的可靠性，理想情况下 $R \approx 1$。

---

## 7. 空间处理方法

### 7.1 空间插值

使用线性插值将预报数据插值到观测网格：

$$F_{interp}(i,j) = \text{interp}(F, \text{grid}_{obs})$$

在MMPE中使用xarray的`interp`方法，采用双线性插值。

### 7.2 面积加权平均

使用纬度余弦权重进行面积加权平均：

$$w_i = \cos(\phi_i)$$

$$\bar{X} = \frac{\sum_{i=1}^{N}w_i X_i}{\sum_{i=1}^{N}w_i}$$

其中 $\phi_i$ 为第 $i$ 个格点的纬度（弧度）。

#### 应用

- **Nino3.4指数**：对Nino3.4区域（5°N-5°S, 190°E-240°E）的SST进行面积加权平均
- **EAWM指数**：对南北两个区域的500hPa纬向风进行面积加权平均

### 7.3 空间降采样

使用coarsen方法进行空间平均降采样：

$$X_{coarse}(i,j) = \frac{1}{K^2}\sum_{k=0}^{K-1}\sum_{l=0}^{K-1}X(Ki+k, Kj+l)$$

其中 $K$ 为降采样因子（如10，表示每10个格点合并为1个）。

### 7.4 空间裁剪

裁剪到指定区域：

$$X_{cropped} = X[\text{lat}_{min}:\text{lat}_{max}, \text{lon}_{min}:\text{lon}_{max}]$$

---

## 8. 时间处理方法

### 8.1 时间对齐

将观测和预报数据对齐到共同的时间点：

$$T_{common} = T_{obs} \cap T_{fcst}$$

$$O_{aligned} = O[T_{common}]$$
$$F_{aligned} = F[T_{common}]$$

### 8.2 时间聚合

#### 日值到月值

$$X_{month} = \frac{1}{D_{month}}\sum_{d \in month}X_d$$

其中 $D_{month}$ 为该月的天数。

#### 月值到年值

$$X_{year} = \frac{1}{12}\sum_{m=1}^{12}X_{year,m}$$

#### 月值到季节值

$$X_{season} = \frac{1}{M_{season}}\sum_{m \in season}X_m$$

其中 $M_{season}$ 为季节包含的月数。

### 8.3 时间重采样

使用重采样方法统一时间分辨率：

$$X_{resampled} = \text{resample}(X, \text{freq})$$

例如，将日值数据重采样为月平均值：`resample(time='1MS').mean()`

---

## 9. 标准化方法

### 9.1 Z-score标准化

将数据标准化为均值为0、标准差为1的分布：

$$Z = \frac{X - \mu}{\sigma}$$

其中：
- $\mu = \frac{1}{N}\sum_{i=1}^{N}X_i$ 为均值
- $\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(X_i - \mu)^2}$ 为标准差

#### 应用

- **EAWM指数**：对DJF季节的指数时间序列进行z-score标准化

---

## 10. 指数计算

### 10.1 Nino3.4指数

#### 计算步骤

1. **区域选择**：选择Nino3.4区域（5°N-5°S, 190°E-240°E）的SST数据

2. **面积加权平均**：使用cos(lat)权重计算空间平均
   $$SST_{nino34}(t) = \frac{\sum_{i,j}w_i SST_{t,i,j}}{\sum_{i,j}w_i}$$
   其中 $w_i = \cos(\phi_i)$

3. **计算月距平**：去除月气候态
   $$Nino34(t) = SST_{nino34}(t) - \overline{SST_{nino34}}_{month(t)}$$

### 10.2 EAWM指数

#### 计算步骤

1. **区域选择**：
   - 南区：25°-35°N, 80°-120°E
   - 北区：45°-55°N, 80°-120°E

2. **面积加权平均**：分别计算两个区域的500hPa纬向风空间平均
   $$u_{south}(t) = \frac{\sum_{i,j \in south}w_i u_{500,t,i,j}}{\sum_{i,j \in south}w_i}$$
   $$u_{north}(t) = \frac{\sum_{i,j \in north}w_i u_{500,t,i,j}}{\sum_{i,j \in north}w_i}$$

3. **计算差值**：
   $$I_{EAWM}(t) = u_{south}(t) - u_{north}(t)$$

4. **季节筛选**：仅保留DJF季节（12、1、2月）的数据

5. **Z-score标准化**：
   $$I^*_{EAWM}(t) = \frac{I_{EAWM}(t) - \mu_{EAWM}}{\sigma_{EAWM}}$$

---

## 11. 综合评分方法

### 11.1 Accuracy计算

#### Raw Accuracy

计算预报值在观测值一定误差范围内的比例：

$$Accuracy_{raw} = \frac{1}{N}\sum_{i=1}^{N}I\left(\frac{|F_i - O_i|}{|O_i| + \epsilon} \leq \delta\right)$$

其中：
- $\delta$ 为相对误差阈值（温度：0.05，降水：0.3）
- $\epsilon$ 为小常数，防止除零

**动态epsilon**：
为了处理不同量级的数据，使用动态epsilon：

$$\epsilon_{dynamic} = \max(\epsilon_{base}, \min(0.01 \times |O_i|, 1000 \times \epsilon_{base}))$$

其中 $\epsilon_{base} = 10^{-6}$。

#### Accuracy with Climatology

计算气候态预报的准确性：

$$Accuracy_{clim} = \frac{1}{N}\sum_{i=1}^{N}I\left(\frac{|C - O_i|}{|O_i| + \epsilon} \leq \delta\right)$$

其中 $C$ 为气候态值（通常为长期平均值）。

#### Equitable Accuracy

考虑气候态预报的准确性，计算相对于气候态预报的技能：

$$Accuracy_{equitable} = \frac{Accuracy_{raw} - Accuracy_{clim}}{1 - Accuracy_{clim}}$$

**边界情况处理**：
- 如果 $Accuracy_{clim} = 1$（气候态预报完美），则 $Accuracy_{equitable} = 0$
- 如果 $Accuracy_{clim} = 0$（气候态预报完全错误），则 $Accuracy_{equitable} = Accuracy_{raw}$
- 结果限制在 $[-1, 1]$ 范围内

**解释**：
- $Accuracy_{equitable} > 0$：预报优于气候态预报
- $Accuracy_{equitable} = 0$：预报与气候态预报相当
- $Accuracy_{equitable} < 0$：预报劣于气候态预报
- $Accuracy_{equitable} = 1$：完美预报

### 11.2 归一化得分

#### Accuracy得分

$$s_{accuracy} = Accuracy_{equitable}$$

#### RMSE得分

见[2.2节](#22-归一化rmse得分)

#### PCC得分

将Pearson相关系数从[-1,1]映射到[0,1]：

$$s_{PCC} = \frac{r + 1}{2}$$

### 11.3 综合得分

计算加权平均的综合得分：

$$Score = \frac{w_{acc} \cdot s_{accuracy} + w_{RMSE} \cdot s_{RMSE} + w_{PCC} \cdot s_{PCC}}{w_{acc} + w_{RMSE} + w_{PCC}}$$

默认权重为 $(1, 1, 1)$，即等权重平均。

---

## 12. ACC (Anomaly Correlation Coefficient)

### 12.1 ACC定义

ACC用于评估ensemble mean与观测的异常相关系数。

#### 计算步骤

1. **计算异常**：去除气候态
   $$A_t(i,j) = F_t(i,j) - \bar{F}_{month(t)}(i,j)$$
   $$B_t(i,j) = O_t(i,j) - \bar{O}_{month(t)}(i,j)$$

2. **计算相关系数**：沿时间维度计算每个格点的相关系数
   $$ACC(i,j) = \frac{\sum_{t}(A_t(i,j) - \bar{A}(i,j))(B_t(i,j) - \bar{B}(i,j))}{\sqrt{\sum_{t}(A_t(i,j) - \bar{A}(i,j))^2 \sum_{t}(B_t(i,j) - \bar{B}(i,j))^2}}$$

### 12.2 Inter-member Correlation

计算ensemble成员间的相关性，用于评估ensemble的多样性。

#### 方法1：成员对之间的平均相关系数

$$IC = \frac{2}{M(M-1)}\sum_{i=1}^{M}\sum_{j=i+1}^{M}r(F_i, F_j)$$

其中 $r(F_i, F_j)$ 为成员 $i$ 和成员 $j$ 之间的相关系数。

**计算步骤**：
1. 对每个成员对 $(i,j)$，计算时间序列相关系数
2. 对所有成员对的相关系数求平均

**解释**：
- $IC \approx 1$：成员间高度相关，ensemble多样性低
- $IC \approx 0$：成员间独立，ensemble多样性高
- $IC < 0$：成员间负相关（罕见）

#### 方法2：每个成员与ensemble mean的相关系数

$$IC = \frac{1}{M}\sum_{m=1}^{M}r(F_m, \bar{F})$$

其中 $\bar{F} = \frac{1}{M}\sum_{m=1}^{M}F_m$ 为ensemble mean。

**计算步骤**：
1. 计算ensemble mean：$\bar{F}_t = \frac{1}{M}\sum_{m=1}^{M}F_{m,t}$
2. 对每个成员 $m$，计算与ensemble mean的相关系数：$r_m = r(F_m, \bar{F})$
3. 对所有成员的相关系数求平均

**解释**：
- $IC \approx 1$：每个成员都与ensemble mean高度相关，ensemble一致性高
- $IC$ 较小：成员间差异较大，ensemble多样性高

#### 应用

- **评估ensemble质量**：IC值反映ensemble的多样性和一致性
- **模式比较**：比较不同模式的IC值，评估ensemble设计
- **与ACC的关系**：通常IC与ACC呈负相关（IC高时ACC可能较低）

---

## 13. Multi-Model Mean (MMM)

### 13.1 MMM计算

计算多个模式的ensemble mean的平均值：

$$MMM_t(i,j) = \frac{1}{K}\sum_{k=1}^{K}\bar{F}_{k,t}(i,j)$$

其中：
- $K$ 为模式数量
- $\bar{F}_{k,t}(i,j)$ 为第 $k$ 个模式在时刻 $t$ 的ensemble mean

### 13.2 MMM Spread

计算MMM的标准差（模式间差异）：

$$MMM_{std}(t) = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(\bar{F}_{k,t} - MMM_t)^2}$$

---

## 14. 异常值处理

### 14.1 IQR方法

使用四分位距（IQR）方法检测和去除异常值：

$$Q1 = \text{percentile}(X, 25\%)$$
$$Q3 = \text{percentile}(X, 75\%)$$
$$IQR = Q3 - Q1$$

异常值定义为：
$$X_i < Q1 - k \cdot IQR \quad \text{或} \quad X_i > Q3 + k \cdot IQR$$

其中 $k$ 为阈值倍数（默认1.5）。

---

## 15. 数据单位转换

### 15.1 降水单位转换

#### 观测数据

从 $kg \cdot m^{-2} \cdot s^{-1}$ 转换为 $mm \cdot day^{-1}$：

$$P_{mm/day} = P_{kg/m^2/s} \times 86400$$

#### 预报数据

从 $kg \cdot m^{-2} \cdot s^{-1}$ 转换为 $mm \cdot day^{-1}$：

$$P_{mm/day} = P_{kg/m^2/s} \times 86400 \times 1000$$

---

## 12. 置信区间与显著性检验

### 12.1 Bootstrap置信区间

使用Block Bootstrap方法计算统计量的置信区间。

#### 计算步骤

1. **Bootstrap重采样**：进行 $B$ 次bootstrap重采样，每次得到统计量 $S_b$，$b = 1, 2, ..., B$

2. **分位数法计算置信区间**：
   $$CI_{1-\alpha} = [Q_{\alpha/2}, Q_{1-\alpha/2}]$$
   其中：
   - $Q_p$ 为第 $p$ 分位数
   - $\alpha$ 为显著性水平（通常为0.05，对应95%置信区间）

3. **Fisher z变换后的置信区间**（用于相关系数）：
   - 对相关系数进行Fisher z变换：$z_b = \text{arctanh}(r_b)$
   - 计算z的置信区间：$CI_z = [Q_{z,\alpha/2}, Q_{z,1-\alpha/2}]$
   - 逆变换得到原始尺度的置信区间：$CI_r = [\tanh(Q_{z,\alpha/2}), \tanh(Q_{z,1-\alpha/2})]$

#### 实现细节

- **最小样本数要求**：至少需要 $n \geq 3$ 个有效样本才能计算相关系数
- **Block大小选择**：通常选择 $L = 12$ 个月（1年），保持季节循环
- **Bootstrap次数**：通常 $B = 1000$ 次，平衡计算成本和统计精度

### 12.2 显著性检验

#### Pearson相关系数的显著性检验

使用t检验评估相关系数的统计显著性：

$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}$$

其中 $n$ 为样本数量。t统计量服从自由度为 $n-2$ 的t分布。

**p值计算**：
$$p = 2 \times P(T_{n-2} > |t|)$$

**显著性水平**：
- $p < 0.001$：极显著（***）
- $p < 0.01$：非常显著（**）
- $p < 0.05$：显著（*）
- $p \geq 0.05$：不显著

#### 线性回归的显著性检验

对于线性回归 $y = \alpha + \beta x$，检验斜率 $\beta$ 的显著性：

$$t_\beta = \frac{\beta}{SE(\beta)}$$

其中 $SE(\beta)$ 为斜率的标准误差：

$$SE(\beta) = \frac{\sqrt{\frac{1}{n-2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}}$$

---

## 13. 线性回归分析

### 13.1 简单线性回归

计算两个变量之间的线性关系：$y = \alpha + \beta x + \epsilon$

#### 参数估计

**斜率**：
$$\beta = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$$

**截距**：
$$\alpha = \bar{y} - \beta\bar{x}$$

**相关系数**：
$$r = \frac{\text{Cov}(x,y)}{\sigma_x \sigma_y} = \beta \frac{\sigma_x}{\sigma_y}$$

#### 拟合优度

**决定系数（R²）**：
$$R^2 = r^2 = \frac{\text{Var}(\hat{y})}{\text{Var}(y)} = 1 - \frac{\text{SSE}}{\text{SST}}$$

其中：
- $\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ 为残差平方和
- $\text{SST} = \sum_{i=1}^{n}(y_i - \bar{y})^2$ 为总平方和

### 13.2 稳健回归（Robust Regression）

去除离群值后的线性回归，提高回归的稳健性。

#### 方法

1. **识别离群值**：使用IQR方法或Z-score方法
2. **去除离群值**：从数据中移除离群值
3. **重新拟合**：对剩余数据重新进行线性回归

#### 应用场景

- RMSE vs Spread关系分析
- ACC vs Inter-member Correlation关系分析
- BSS vs Accuracy关系分析

---

## 14. 数据验证与质量控制

### 14.1 有效数据检查

#### 最小样本数要求

对于不同的统计计算，需要不同的最小样本数：

- **相关系数**：至少 $n \geq 3$ 个有效样本（推荐 $n \geq 5$）
- **RMSE**：至少 $n \geq 2$ 个有效样本
- **Bootstrap**：至少 $n \geq 2L$ 个时间点（$L$ 为block大小）

#### 有效覆盖度阈值

对于空间场数据，设置有效覆盖度阈值：

$$\text{Coverage} = \frac{N_{valid}}{N_{total}} \geq \theta$$

其中：
- $N_{valid}$ 为有效格点数
- $N_{total}$ 为总格点数
- $\theta$ 为阈值（通常为0.2，即至少20%的格点有效）

### 14.2 NaN值处理

#### 处理策略

1. **时间对齐**：只保留观测和预报共同有效的时间点
   $$T_{common} = \{t : O_t \text{有效} \land F_t \text{有效}\}$$

2. **空间对齐**：只计算共同有效格点的统计量
   $$N_{valid} = \sum_{i,j} I(O_{i,j} \text{有效} \land F_{i,j} \text{有效})$$

3. **跳过无效计算**：如果有效样本数不足，返回NaN

### 14.3 数据范围检查

#### 物理合理性检查

- **温度**：通常在 $-100°C$ 到 $60°C$ 之间
- **降水**：必须 $\geq 0$，极端值检查（如 $> 1000$ mm/day）
- **相关系数**：必须在 $[-1, 1]$ 范围内
- **RMSE**：必须 $\geq 0$

#### 异常值检测

使用IQR方法检测异常值（见[14.1节](#141-iqr方法)），并在计算前去除。

### 14.4 网格一致性检查

#### 坐标匹配检查

检查观测和预报网格是否匹配：

$$\text{Grids Match} = \begin{cases}
\text{True} & \text{如果 } \text{lat}_{obs} = \text{lat}_{fcst} \text{ 且 } \text{lon}_{obs} = \text{lon}_{fcst} \\
\text{False} & \text{否则}
\end{cases}$$

#### 网格不匹配处理

如果网格不匹配，使用以下方法：

1. **最近邻插值**：`method='nearest'`
2. **线性插值**：`method='linear'`（默认）
3. **坐标选择**：如果预报网格包含观测网格，直接选择对应坐标

---

## 15. 内存优化策略

### 15.1 分块计算（Chunking）

对于大数组，使用分块计算避免内存溢出。

#### 时间分块

将时间维度分成多个块：

$$T = \bigcup_{k=1}^{K} T_k, \quad T_k \cap T_l = \emptyset \text{ for } k \neq l$$

对每个时间块分别计算，然后合并结果：

$$RMSE = \sqrt{\frac{1}{K}\sum_{k=1}^{K} RMSE_k^2}$$

其中 $RMSE_k$ 为第 $k$ 个时间块的RMSE。

#### 空间分块

对于空间场数据，按行或列分块处理：

$$RMSE(i,j) = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(F_t(i,j) - O_t(i,j))^2}$$

按行并行处理，每行独立计算。

### 15.2 延迟加载（Lazy Loading）

使用xarray的延迟加载机制，只在需要时加载数据：

```python
# 延迟加载
ds = xr.open_dataset(file_path)  # 不立即加载数据

# 需要时才加载
data = ds['variable'].load()  # 实际加载数据
```

### 15.3 内存释放

计算完成后及时释放内存：

1. **删除大数组**：`del large_array`
2. **强制垃圾回收**：`gc.collect()`
3. **关闭文件句柄**：`ds.close()`

### 15.4 数据压缩

对于中间结果，使用压缩存储：

- **NetCDF压缩**：`encoding={'zlib': True, 'complevel': 4}`
- **数据类型优化**：使用 `float32` 而非 `float64`（如果精度足够）

---

## 16. 网格匹配与对齐

### 16.1 时间对齐

#### 方法

1. **时间索引交集**：
   $$T_{common} = T_{obs} \cap T_{fcst}$$

2. **时间选择**：
   $$O_{aligned} = O[T_{common}]$$
   $$F_{aligned} = F[T_{common}]$$

3. **最小时间点数检查**：
   $$\text{if } |T_{common}| < N_{min} \text{ then return None}$$

   其中 $N_{min}$ 通常为12（至少1年的月数据）。

#### 实现细节

- 使用 `pandas.Index.intersection()` 或 `xarray` 的 `sel()` 方法
- 处理时区差异（如果存在）
- 处理时间分辨率差异（如日值 vs 月值）

### 16.2 空间对齐

#### 网格匹配判断

检查两个网格是否完全匹配：

$$\text{Match} = \begin{cases}
\text{True} & \text{如果 } |\text{lat}_{obs} - \text{lat}_{fcst}| < \epsilon \text{ 且 } |\text{lon}_{obs} - \text{lon}_{fcst}| < \epsilon \\
\text{False} & \text{否则}
\end{cases}$$

其中 $\epsilon$ 为容差（通常为 $10^{-6}$ 度）。

#### 插值方法

如果网格不匹配，使用插值：

1. **线性插值**（默认）：
   $$F_{interp}(i,j) = \text{interp}(F, \text{grid}_{obs}, \text{method='linear'})$$

2. **最近邻插值**：
   $$F_{interp}(i,j) = F(\arg\min_{(k,l)} \text{dist}((i,j), (k,l)))$$

3. **坐标选择**（如果预报网格包含观测网格）：
   $$F_{interp}(i,j) = F(\text{nearest\_lat}(i), \text{nearest\_lon}(j))$$

#### 实现细节

- 使用 `xarray.interp()` 进行插值
- 对于大数组，考虑使用 `dask` 进行并行插值
- 处理边界情况（如极地地区）

### 16.3 Ensemble数据对齐

对于ensemble数据，需要同时对齐时间和空间：

1. **时间对齐**：与观测数据对齐
2. **空间对齐**：对每个ensemble成员分别进行空间插值
3. **内存优化**：如果ensemble成员数较多，考虑分批处理

---

## 17. 实际应用示例

### 17.1 ACC计算完整流程

#### 步骤1：数据加载

```python
# 加载观测数据
obs_data = load_obs_data(var_type='temp')  # (time, lat, lon)

# 加载预报ensemble数据
fcst_data = load_forecast_data(model='ECMWF-51-mon', 
                               var_type='temp', 
                               leadtime=0)  # (time, number, lat, lon)
```

#### 步骤2：计算气候态

```python
# 计算月气候态（基期：1993-2020）
obs_clim = compute_monthly_climatology(obs_data, baseline='1993-2020')
fcst_clim = compute_monthly_climatology(fcst_data, baseline='1993-2020')
```

#### 步骤3：计算距平

```python
# 去除气候态
obs_anom = remove_climatology(obs_data, obs_clim)
fcst_anom = remove_climatology(fcst_data, fcst_clim)
```

#### 步骤4：计算ensemble mean

```python
# 计算ensemble mean
fcst_mean = fcst_anom.mean(dim='number')  # (time, lat, lon)
```

#### 步骤5：计算ACC

```python
# 对每个格点计算时间序列相关系数
acc = pearson_r_along_time(fcst_mean, obs_anom)  # (lat, lon)
```

### 17.2 Block Bootstrap评分计算流程

#### 步骤1：数据准备

```python
# 对齐时间和空间
obs_aligned, fcst_aligned = align_data(obs_data, fcst_data)
```

#### 步骤2：划分时间块

```python
block_size = 12  # 12个月（1年）
n_blocks = (len(time) - block_size + 1)
```

#### 步骤3：Bootstrap重采样

```python
n_bootstrap = 1000
for b in range(n_bootstrap):
    # 随机选择block
    selected_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)
    
    # 计算每个block的指标
    for block_idx in selected_blocks:
        block_data = data[block_idx:block_idx+block_size]
        metrics = compute_metrics(block_data)
    
    # 平均所有block的指标
    bootstrap_metrics[b] = np.mean(block_metrics)
```

#### 步骤4：计算置信区间

```python
confidence_level = 0.95
alpha = (1 - confidence_level) / 2
ci_lower = np.percentile(bootstrap_metrics, alpha * 100)
ci_upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
```

### 17.3 Nino3.4指数计算流程

#### 步骤1：加载SST数据

```python
sst_data = load_era5_sst_daily(year_range=(1993, 2020))  # (time, lat, lon)
```

#### 步骤2：聚合为月平均

```python
sst_monthly = sst_data.resample(time='1MS').mean()  # (time, lat, lon)
```

#### 步骤3：区域选择

```python
nino34_region = sst_monthly.sel(
    lat=slice(-5, 5),      # 5°N-5°S
    lon=slice(190, 240)   # 190°E-240°E
)
```

#### 步骤4：面积加权平均

```python
# 计算权重
weights = np.cos(np.deg2rad(nino34_region.lat))

# 先对经度求平均
sst_lon_mean = nino34_region.mean(dim='lon')

# 对纬度加权平均
nino34_index = (sst_lon_mean * weights).sum(dim='lat') / weights.sum()
```

#### 步骤5：计算月距平

```python
# 计算月气候态
clim = nino34_index.groupby('time.month').mean('time')

# 计算距平
nino34_anomaly = nino34_index.groupby('time.month') - clim
```

---

## 18. 边界情况处理

### 18.1 数据不足情况

#### 时间序列过短

如果时间序列长度不足：

- **相关系数**：如果 $n < 3$，返回 `NaN`
- **RMSE**：如果 $n < 2$，返回 `NaN`
- **Bootstrap**：如果 $n < 2L$（$L$ 为block大小），返回 `NaN`

#### 空间覆盖不足

如果有效格点数不足：

- **空间平均**：如果有效格点数 $< 10$，返回 `NaN`
- **空间统计**：如果有效覆盖率 $< 20\%$，返回 `NaN`

### 18.2 零方差情况

如果数据方差为零（所有值相同）：

- **相关系数**：返回 `NaN`（无法计算）
- **标准化**：如果标准差 $< 10^{-10}$，返回 `NaN`

### 18.3 除零保护

在所有除法运算中添加小常数防止除零：

- **相对误差**：$\frac{|F - O|}{|O| + \epsilon}$，其中 $\epsilon = 10^{-6}$
- **归一化**：$\frac{X - X_{min}}{X_{max} - X_{min} + \epsilon}$
- **BSS**：$BSS = 1 - \frac{BS}{BS_{ref} + \epsilon}$，其中 $\epsilon = 10^{-4}$

### 18.4 数值稳定性

#### Fisher z变换

在Fisher z变换中，对相关系数进行裁剪：

$$r_{clipped} = \text{clip}(r, -0.9999999, 0.9999999)$$

防止 `arctanh(1)` 或 `arctanh(-1)` 导致无穷大。

#### 平方根计算

在RMSE计算中，确保平方根内的值非负：

$$RMSE = \sqrt{\max(0, \text{mean}((F - O)^2))}$$

---

## 19. 性能优化技巧

### 19.1 向量化计算

使用NumPy的向量化操作替代循环：

```python
# 慢：循环
result = []
for i in range(n):
    result.append(compute(x[i]))

# 快：向量化
result = compute(x)  # 对整个数组操作
```

### 19.2 并行计算

对于独立的任务，使用多进程并行：

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_point, points)
```

### 19.3 缓存中间结果

对于重复计算，缓存中间结果：

```python
# 计算气候态（只需计算一次）
if clim_cache is None:
    clim_cache = compute_climatology(data)
```

### 19.4 数据类型优化

使用合适的数据类型减少内存占用：

- `float32` vs `float64`：如果精度足够，使用 `float32`
- `int32` vs `int64`：根据数据范围选择

---

## 附录：符号说明

### 基本符号

- $O_t(i,j)$：时刻 $t$ 在格点 $(i,j)$ 的观测值
- $F_t(i,j)$：时刻 $t$ 在格点 $(i,j)$ 的预报值
- $F_{t,m}(i,j)$：时刻 $t$ 在格点 $(i,j)$ 的第 $m$ 个ensemble成员值
- $\bar{F}_t(i,j)$：时刻 $t$ 在格点 $(i,j)$ 的ensemble mean
- $T$：时间长度（时间点数）
- $N$：空间格点数
- $M$：ensemble成员数
- $K$：模式数量
- $\phi$：纬度（弧度）
- $\lambda$：经度（弧度）
- $\delta$：相对误差阈值
- $\epsilon$：防止除零的小常数（通常为 $10^{-6}$）
- $I(\cdot)$：指示函数（如果条件为真返回1，否则返回0）

### 统计符号

- $\bar{X}$：变量 $X$ 的均值
- $\sigma_X$：变量 $X$ 的标准差
- $\text{Cov}(X,Y)$：变量 $X$ 和 $Y$ 的协方差
- $\text{Var}(X)$：变量 $X$ 的方差
- $Q_p$：第 $p$ 分位数
- $CI_{1-\alpha}$：$1-\alpha$ 置信区间
- $p$：p值（显著性水平）

### 时间符号

- $t$：时间索引
- $T$：时间序列长度
- $y$：年份
- $m$：月份
- $s$：季节
- $L$：Block大小（时间块长度）
- $B$：Bootstrap次数

### 空间符号

- $(i,j)$：格点索引
- $(x,y)$：空间坐标
- $\text{lat}$：纬度
- $\text{lon}$：经度
- $w_i$：权重（通常为 $\cos(\phi_i)$）

---

## 参考文献

本文档基于MMPE工具包的实际实现整理，主要参考以下文件：

### 核心分析脚本

- `climatology_analysis.py`：气候态分析和偏差计算
- `rmse_spread_analysis.py`：RMSE和ensemble spread分析
- `acc_intermember_analysis.py`：ACC和成员间相关分析
- `block_bootstrap_score.py`：Block Bootstrap评分计算
- `combined_pearson_analysis.py`：Pearson相关分析（年度、季节、月度）
- `circulation_analysis.py`：环流分析和指数计算
- `nino34_eawm_index_calculation.ipynb`：Nino3.4和EAWM指数计算

### 工具函数

- `common_config.py`：通用配置和常量定义
- `nc_downsampling.py`：数据降采样工具

### 相关文档

- `CIRCULATION_ANALYSIS_EXTENSIONS.md`：环流分析扩展说明
- `MMPE_TOOLKIT_INVENTORY.md`：工具包清单
- `TOOLKIT_MAINTENANCE.md`：维护说明

### 理论参考

1. **相关系数**：Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences* (3rd ed.). Academic Press.

2. **Bootstrap方法**：Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall.

3. **Brier Score**：Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.

4. **Ensemble预报**：Palmer, T. N., et al. (2005). Representing model uncertainty in weather and climate prediction. *Annual Review of Earth and Planetary Sciences*, 33, 163-193.

5. **气候指数**：
   - Nino3.4指数：Trenberth, K. E. (1997). The definition of El Niño. *Bulletin of the American Meteorological Society*, 78(12), 2771-2777.
   - EAWM指数：Wang, L., & Chen, W. (2014). An intensity index for the East Asian winter monsoon. *Journal of Climate*, 27(6), 2361-2374.

---

## 版本历史

- **v1.0** (2024)：初始版本，包含基本数据处理方法
- **v1.1** (2024)：添加详细实现步骤、边界情况处理、置信区间计算、线性回归分析、数据验证、内存优化、网格匹配等章节

---

*文档生成日期：2024年*
*最后更新：2024年*
*维护者：MMPE开发团队*


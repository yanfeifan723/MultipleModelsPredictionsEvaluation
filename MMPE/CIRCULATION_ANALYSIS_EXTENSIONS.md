# 环流诊断分析扩展方案

## 数据资源总结

### 观测数据

#### 1. ERA5 再分析数据 (MonthlyPressureLevel)
- **变量**: z (位势), r (相对湿度), q (比湿), t (温度), u (纬向风), v (经向风)
- **气压层**: 12层 [1000, 925, 850, 700, 500, 400, 300, 200, 100, 50, 30, 10] hPa
- **分辨率**: 0.25° × 0.25° (721 lat × 1440 lon)
- **时间范围**: 1980-2022 (从文件名推断)
- **特点**: 
  - 有相对湿度(r)，模式数据没有
  - 完整的多层数据，可用于垂直结构分析

#### 2. CMFD 中国气象强迫数据集(/sas12t1/ffyan/obs/prec_1deg_199301-202012.nc, /sas12t1/ffyan/obs/temp_1deg_199301-202012.nc)
- **变量**: 
  - temp: 近地表气温 (K)
  - prec: 降水 (kg m⁻² s⁻¹)
- **分辨率**: 1° × 1° (40 lat × 70 lon)
- **时间范围**: 1951-2020
- **特点**: 高分辨率中国区域数据，可用于区域精细化分析，但是进行了重采样

### 模式数据

#### 标准模式 (12层，有q变量)
- **CMCC-35**: z, q, t, u, v - 12层 - 40成员
- **ECMWF-51-mon**: z, q, t, u, v - 12层 - 25成员
- **DWD-mon-21**: z, q, t, u, v - 12层 - 30成员

#### 特殊模式
- **NCEP-2**: z, t, u, v (无q!) - 4层 [200, 500, 700, 850] - 28成员
  - ⚠️ **限制**: 缺少比湿，无法进行水汽相关分析
- **Meteo-France-8**: z, q, t, u, v - 11层 [10, 30, 50, 100, 200, 300, 400, 500, 700, 850, 925] - 25成员
- **ECCC-Canada-3**: z, q, t, u, v - 11层 [10, 30, 50, 100, 200, 300, 400, 500, 700, 850, 925] - 10成员

---

## 可进行的分析工作

### 一、基于已有变量的分析（所有模式可用）

#### 1. **垂直结构分析**
**目标**: 分析环流的垂直分布特征

**方案**:
- 计算不同气压层的统计特征（均值、标准差）
- 绘制垂直剖面图（沿特定经度/纬度的u, v, t, GHT垂直分布）
- 分析垂直切变：∂u/∂p, ∂v/∂p
- 识别关键等压面（如850hPa、500hPa、200hPa）的特征差异

**实现要点**:
```python
# 垂直剖面示例
def plot_vertical_section(data, lon_or_lat, value, pressure_levels):
    """沿经度或纬度绘制垂直剖面"""
    # 选择特定经度/纬度
    # 绘制pressure vs lat/lon的contour图
    pass
```

**输出**: 
- 垂直剖面图（pressure-latitude 或 pressure-longitude）
- 各层统计对比表

---

#### 2. **涡度和散度分析**
**目标**: 诊断环流的旋转和辐合/辐散特征

**方案**:
- **相对涡度**: ζ = ∂v/∂x - ∂u/∂y
- **散度**: D = ∂u/∂x + ∂v/∂y
- **绝对涡度**: ζₐ = ζ + f (f为科里奥利参数)
- **涡度平流**: -u·∂ζ/∂x - v·∂ζ/∂y

**实现要点**:
```python
def compute_vorticity(u, v, lat):
    """计算相对涡度"""
    # 使用xarray的梯度函数
    dv_dx = v.differentiate('lon') / (R * np.cos(np.deg2rad(lat)))
    du_dy = u.differentiate('lat') / R
    vorticity = dv_dx - du_dy
    return vorticity

def compute_divergence(u, v, lat):
    """计算散度"""
    du_dx = u.differentiate('lon') / (R * np.cos(np.deg2rad(lat)))
    dv_dy = v.differentiate('lat') / R
    divergence = du_dx + dv_dy
    return divergence
```

**输出**:
- 850hPa和500hPa的涡度、散度场
- 模式与观测的偏差
- 涡度/散度的季节变化

---

#### 3. **急流分析**
**目标**: 诊断副热带急流和极地急流

**方案**:
- 使用200hPa或300hPa的u分量
- 识别急流轴（u > 30 m/s的区域）
- 计算急流强度、位置（纬度）、宽度
- 分析急流的季节变化和年际变率

**实现要点**:
```python
def analyze_jet_stream(u_200, threshold=30):
    """分析急流特征"""
    # 识别急流区域
    jet_mask = u_200 > threshold
    # 计算急流轴位置（最大风速的纬度）
    jet_lat = u_200.argmax(dim='lat').lat
    # 计算急流强度
    jet_strength = u_200.max(dim='lat')
    return jet_lat, jet_strength, jet_mask
```

**输出**:
- 200hPa急流分布图
- 急流强度和位置的时间序列
- 模式对急流的模拟能力评估

---

#### 4. **温度场分析**
**目标**: 分析温度场的空间分布和垂直结构

**方案**:
- 计算各层温度场的气候平均态
- 分析温度梯度（水平、垂直）
- 计算热成风：∂u/∂p = -R/(fp) · ∂T/∂y
- 温度偏差分析（模式 vs ERA5）

**实现要点**:
```python
def compute_thermal_wind(t, pressure, lat):
    """计算热成风"""
    # 热成风关系
    # ∂u/∂p = -R/(fp) · ∂T/∂y
    R = 287.0  # 气体常数
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(lat))  # 科里奥利参数
    dT_dy = t.differentiate('lat') / R  # 温度梯度
    thermal_wind = -R / (f * pressure) * dT_dy
    return thermal_wind
```

**输出**:
- 各层温度场分布
- 温度梯度场
- 热成风分析

---

#### 5. **动能分析**
**目标**: 分析大气的动能分布和变化

**方案**:
- **动能**: KE = 0.5 × (u² + v²)
- 计算各层的动能分布
- 分析动能的季节变化
- 计算动能收支（需要更多变量）

**实现要点**:
```python
def compute_kinetic_energy(u, v):
    """计算动能"""
    ke = 0.5 * (u**2 + v**2)
    ke.attrs = {'long_name': 'Kinetic Energy', 'units': 'm² s⁻²'}
    return ke
```

**输出**:
- 各层动能分布图
- 动能的时间序列和季节变化

---

#### 6. **环流指数扩展**
**目标**: 计算更多大尺度环流指数

**方案**:
- **北极涛动 (AO)**: 使用1000hPa或海平面气压
- **北大西洋涛动 (NAO)**: 500hPa GHT的冰岛-亚速尔群岛差值
- **太平洋-北美型 (PNA)**: 500hPa GHT的关键点组合
- **东亚夏季风指数 (EASM)**: 类似EAWM，但使用JJA数据

**实现要点**:
```python
def compute_nao_index(ght_500):
    """计算NAO指数"""
    # 冰岛区域 (60-70°N, 20-30°W)
    iceland = ght_500.sel(lat=slice(60, 70), lon=slice(330, 340)).mean()
    # 亚速尔群岛区域 (35-40°N, 20-30°W)
    azores = ght_500.sel(lat=slice(35, 40), lon=slice(330, 340)).mean()
    nao = iceland - azores
    # 标准化
    nao_normalized = (nao - nao.mean()) / nao.std()
    return nao_normalized
```

**输出**:
- 各环流指数的时间序列
- 模式对环流指数的模拟能力

---

======================================

---

### 二、基于比湿(q)的分析（NCEP-2除外）

#### 7. **水汽通量分析**（已有函数，需集成）
**目标**: 分析水汽输送特征

**方案**:
- 使用已有的 `compute_moisture_flux_divergence` 函数
- 计算水汽通量矢量：Q = q × (u, v)
- 分析水汽通量散度（辐合/辐散）
- 识别主要水汽通道

**实现要点**:
```python
# 已有函数: compute_moisture_flux_divergence
# 需要集成到主流程中
def analyze_moisture_flux(u, v, q):
    """分析水汽通量"""
    # 计算水汽通量
    qu = q * u
    qv = q * v
    # 计算散度
    divergence = compute_moisture_flux_divergence(u, v, q)
    return qu, qv, divergence
```

**输出**:
- 水汽通量矢量图
- 水汽通量散度场
- 主要水汽通道识别

---

#### 8. **可降水量分析**
**目标**: 计算整层可降水量

**方案**:
- 垂直积分比湿：PW = (1/g) × ∫ q dp
- 分析可降水量的空间分布和季节变化
- 模式与观测对比

**实现要点**:
```python
def compute_precipitable_water(q, pressure):
    """计算可降水量"""
    g = 9.80665
    # 垂直积分
    pw = (1/g) * (q * pressure.diff('level')).sum(dim='level')
    pw.attrs = {'long_name': 'Precipitable Water', 'units': 'mm'}
    return pw
```

**输出**:
- 可降水量分布图
- 可降水量时间序列

---

#### 9. **相对湿度分析**（仅ERA5）
**目标**: 利用ERA5的相对湿度数据

**方案**:
- ERA5有相对湿度(r)，模式数据没有
- 可以从模式数据计算相对湿度：RH = q / qs(T)
- 对比ERA5和模式计算的相对湿度

**实现要点**:
```python
def compute_relative_humidity(q, t, p):
    """从比湿和温度计算相对湿度"""
    # 计算饱和比湿
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))  # 饱和水汽压
    qs = 0.622 * es / (p - 0.378 * es)  # 饱和比湿
    rh = (q / qs) * 100  # 相对湿度(%)
    return rh
```

**输出**:
- 相对湿度分布图
- 模式计算的相对湿度与ERA5的对比

---

### 三、基于CMFD数据的分析

#### 10. **温度-环流关联分析**
**目标**: 分析近地表温度与环流的关系

**方案**:
- 使用CMFD的temp数据（近地表，0.1°分辨率）
- 与850hPa环流场进行相关分析
- 识别温度异常与环流异常的关联

**实现要点**:
```python
def analyze_temp_circulation_correlation(temp_cmfd, u_850, v_850):
    """分析温度与环流的相关性"""
    # 插值到相同网格
    # 计算相关系数
    corr_u = xr.corr(temp_cmfd, u_850, dim='time')
    corr_v = xr.corr(temp_cmfd, v_850, dim='time')
    return corr_u, corr_v
```

**输出**:
- 温度-环流相关图
- 温度异常年份的环流合成分析

---

#### 11. **降水-环流关联分析**
**目标**: 分析降水与环流、水汽输送的关系

**方案**:
- 使用CMFD的prec数据
- 与水汽通量散度进行相关分析
- 分析降水异常与环流异常的关系

**实现要点**:
```python
def analyze_prec_circulation_correlation(prec_cmfd, moisture_div, u_850, v_850):
    """分析降水与环流、水汽的关系"""
    # 计算相关系数
    corr_div = xr.corr(prec_cmfd, moisture_div, dim='time')
    # 合成分析（极端降水年份）
    extreme_years = identify_extreme_years(prec_cmfd)
    composite_circulation = compute_composite(u_850, v_850, extreme_years)
    return corr_div, composite_circulation
```

**输出**:
- 降水-水汽通量散度相关图
- 极端降水年份的环流合成图

---

### 四、高级分析

#### 12. **EOF分析**
**目标**: 识别主要环流模态

**方案**:
- 对风场或GHT场进行EOF分解
- 识别前3-5个主要模态
- 分析各模态的时间系数和物理意义

**实现要点**:
```python
from eofs.xarray import Eof

def perform_eof_analysis(data, neofs=5):
    """进行EOF分析"""
    solver = Eof(data)
    eofs = solver.eofs(neofs=neofs)
    pcs = solver.pcs(npcs=neofs)
    variance = solver.varianceFraction(neigs=neofs)
    return eofs, pcs, variance
```

**输出**:
- EOF空间型图
- 主成分时间序列
- 方差贡献

---

#### 13. **聚类分析**
**目标**: 识别典型环流型

**方案**:
- 对环流场进行K-means或层次聚类
- 识别典型环流型及其出现频率
- 分析各环流型对应的天气特征

**实现要点**:
```python
from sklearn.cluster import KMeans

def cluster_circulation_patterns(data, n_clusters=5):
    """对环流型进行聚类"""
    # 重塑数据
    data_reshaped = data.values.reshape(len(data.time), -1)
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_reshaped)
    # 计算各类型的平均场
    patterns = {}
    for i in range(n_clusters):
        mask = labels == i
        patterns[i] = data.isel(time=mask).mean(dim='time')
    return patterns, labels
```

**输出**:
- 典型环流型分布图
- 各类型的出现频率
- 类型转换分析

---

#### 14. **Hovmöller图分析**
**目标**: 分析环流系统的时间演变和传播

**方案**:
- 沿纬度/经度平均，绘制时间-经度/纬度图
- 识别环流系统的传播特征
- 分析季节内变率

**实现要点**:
```python
def plot_hovmoller(data, lat_or_lon, lat_value=None, lon_value=None):
    """绘制Hovmöller图"""
    if lat_or_lon == 'lat':
        # 沿纬度平均
        data_mean = data.mean(dim='lat')
        x_axis = data_mean.lon
    else:
        # 沿经度平均
        data_mean = data.mean(dim='lon')
        x_axis = data_mean.lat
    
    # 绘制时间-x轴图
    plt.contourf(x_axis, data_mean.time, data_mean.values)
```

**输出**:
- Hovmöller图（时间-经度或时间-纬度）
- 传播速度分析

---

#### 15. **功率谱和小波分析**
**目标**: 分析环流的时间变率特征

**方案**:
- 对环流指数进行功率谱分析
- 使用小波变换分析时间-频率特征
- 识别主要周期（年际、年代际）

**实现要点**:
```python
from scipy import signal
import pywt

def power_spectrum_analysis(time_series):
    """功率谱分析"""
    # 使用Welch方法
    f, Pxx = signal.welch(time_series, fs=1.0, nperseg=len(time_series)//2)
    return f, Pxx

def wavelet_analysis(time_series):
    """小波分析"""
    # 使用Morlet小波
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(time_series, scales, 'morl')
    return coeffs, freqs
```

**输出**:
- 功率谱图
- 小波功率谱图
- 主要周期识别

---

## 实施优先级建议

### 高优先级（易于实现，物理意义明确）
1. ✅ **涡度和散度分析** - 计算简单，诊断价值高
2. ✅ **急流分析** - 使用200hPa数据，诊断意义重要
3. ✅ **水汽通量分析** - 已有函数，只需集成
4. ✅ **温度场分析** - 数据完整，分析价值高

### 中优先级（需要一定开发工作）
5. **垂直结构分析** - 需要新的绘图功能
6. **环流指数扩展** - 需要定义新的指数
7. **可降水量分析** - 需要垂直积分
8. **温度-环流关联** - 需要整合CMFD数据

### 低优先级（复杂但价值高）
9. **EOF分析** - 需要安装eofs库
10. **聚类分析** - 需要机器学习库
11. **Hovmöller图** - 需要新的可视化方法
12. **功率谱分析** - 需要信号处理库

---

## 注意事项

1. **NCEP-2模式限制**: 
   - 缺少比湿(q)，无法进行水汽相关分析
   - 只有4个气压层，垂直分析受限
   - 需要特殊处理或排除

2. **数据分辨率差异**:
   - ERA5: 0.25°
   - 模式: 1°
   - CMFD: 0.1°（仅中国区域）
   - 需要统一插值到相同网格

3. **时间范围**:
   - ERA5: 1980-2022
   - CMFD: 1951-2020
   - 模式: 1993-2020（推测）
   - 需要统一时间范围

4. **Ensemble成员数差异**:
   - 不同模式的成员数不同（10-40）
   - 需要统一处理（ensemble mean或保留所有成员）

---

## 推荐实施顺序

1. **第一阶段**（1-2周）:
   - 涡度和散度分析
   - 急流分析
   - 集成水汽通量分析

2. **第二阶段**（2-3周）:
   - 温度场分析
   - 垂直结构分析
   - 环流指数扩展（EASM, NAO等）

3. **第三阶段**（3-4周）:
   - 可降水量分析
   - 温度-环流关联分析
   - EOF分析

4. **第四阶段**（按需）:
   - 聚类分析
   - Hovmöller图
   - 功率谱分析


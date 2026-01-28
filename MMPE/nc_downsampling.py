import xarray as xr
import numpy as np
import sys

# 支持处理temp和prec两个变量
# 用法: python nc_downsampling.py [temp|prec|all]
# 如果不提供参数，默认处理所有变量
var_type_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
var_name_map = {"temp": "temp", "prec": "prec"}

if var_type_arg not in ["temp", "prec", "all"]:
    raise ValueError(f"不支持的变量类型: {var_type_arg}，支持: temp, prec, all")

# 确定要处理的变量列表
if var_type_arg == "all":
    var_types = ["temp", "prec"]
else:
    var_types = [var_type_arg]

print(f"将处理变量: {', '.join(var_types)}")
print("=" * 60)

# 处理每个变量
for var_type in var_types:
    var_name = var_name_map[var_type]
    
    # 读取原始数据
    input_file = f"CMFD/{var_type}_CMFD_V0200_B-01_01mo_010deg_195101-202012.nc"
    # 输出到obs文件夹，覆盖旧文件
    output_file = f"obs/{var_type}_1deg_199301-202012.nc"
    
    print(f"\n处理变量: {var_type}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    try:
        ds = xr.open_dataset(input_file)
        
        # 处理填充值
        ds[var_name] = ds[var_name].where(ds[var_name] != 1e20, np.nan)
        
        # 时间截取（1993-01至2020-12）
        ds_subset = ds.sel(time=slice("1992-12", "2020-12"))
        
        # 空间降采样到1度×1度（使用coarsen进行空间平均）
        ds_coarse = ds_subset.coarsen(
            lat=10,  # 纬度方向每10个点合并
            lon=10,  # 经度方向每10个点合并
            boundary="trim"  # 自动对齐可整除范围
        ).mean()
        
        # 生成x.5网格坐标（15.5, 16.5, 17.5...）
        # 计算新的网格范围
        lat_min = float(ds_subset.lat.min())
        lat_max = float(ds_subset.lat.max())
        lon_min = float(ds_subset.lon.min())
        lon_max = float(ds_subset.lon.max())
        
        # 确保新坐标数量与coarsen后的数据维度匹配
        n_lat_coarse = len(ds_coarse.lat)
        n_lon_coarse = len(ds_coarse.lon)
        
        # 从lat_min和lon_min开始，每1度生成一个x.5坐标
        lat_start = np.floor(lat_min) + 0.5
        lon_start = np.floor(lon_min) + 0.5
        
        lat_new = np.arange(lat_start, lat_start + n_lat_coarse * 1.0, 1.0)
        lon_new = np.arange(lon_start, lon_start + n_lon_coarse * 1.0, 1.0)
        
        # 确保坐标在合理范围内
        lat_new = lat_new[lat_new <= lat_max]
        lon_new = lon_new[lon_new <= lon_max]
        
        # 如果长度不匹配，调整
        if len(lat_new) > n_lat_coarse:
            lat_new = lat_new[:n_lat_coarse]
        elif len(lat_new) < n_lat_coarse:
            lat_new = np.arange(lat_start, lat_start + n_lat_coarse * 1.0, 1.0)
        
        if len(lon_new) > n_lon_coarse:
            lon_new = lon_new[:n_lon_coarse]
        elif len(lon_new) < n_lon_coarse:
            lon_new = np.arange(lon_start, lon_start + n_lon_coarse * 1.0, 1.0)
        
        # 强制更新坐标为x.5网格
        ds_coarse = ds_coarse.assign_coords({
            "lat": lat_new,
            "lon": lon_new
        })
        
        # 恢复填充值属性
        ds_coarse[var_name].attrs.update({
            "_FillValue": 1e20,
            "missing_value": 1e20
        })
        ds_coarse[var_name] = ds_coarse[var_name].fillna(1e20)
        
        # 保存为新文件
        ds_coarse.to_netcdf(output_file)
        
        print(f"  ✓ 重采样完成:")
        print(f"    原始网格: lat=[{lat_min:.2f}, {lat_max:.2f}], lon=[{lon_min:.2f}, {lon_max:.2f}]")
        print(f"    新网格坐标: lat=[{lat_new[0]:.2f}, {lat_new[-1]:.2f}], lon=[{lon_new[0]:.2f}, {lon_new[-1]:.2f}]")
        print(f"    新网格大小: lat={len(lat_new)}, lon={len(lon_new)}")
        print(f"    输出文件: {output_file}")
        
        ds.close()
        
    except FileNotFoundError:
        print(f"  ✗ 错误: 输入文件不存在: {input_file}")
    except Exception as e:
        print(f"  ✗ 错误: 处理 {var_type} 时出错: {str(e)}")

print("\n" + "=" * 60)
print("处理完成！")

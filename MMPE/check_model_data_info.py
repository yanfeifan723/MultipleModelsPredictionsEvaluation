#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查所有模式数据的信息
使用ncdump和xarray检查每个模式的所有文件的元数据信息
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from collections import defaultdict
import xarray as xr
import numpy as np
from datetime import datetime

# 导入配置
from common_config import MODEL_LIST, MODEL_FILE_MAP, DATA_PATHS

def run_ncdump(file_path, summary=True):
    """运行ncdump命令获取文件信息"""
    try:
        if summary:
            cmd = f"ncdump -h {file_path}"
        else:
            cmd = f"ncdump {file_path}"
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except Exception as e:
        print(f"  Error running ncdump on {file_path}: {e}")
        return None

def get_file_info_xarray(file_path):
    """使用xarray获取文件基本信息"""
    try:
        ds = xr.open_dataset(file_path, decode_times=False)
        info = {
            'dims': dict(ds.dims),
            'coords': list(ds.coords.keys()),
            'data_vars': list(ds.data_vars.keys()),
            'attrs': dict(ds.attrs) if hasattr(ds, 'attrs') else {},
        }
        
        # 获取关键维度信息
        if 'number' in ds.dims:
            info['n_members'] = int(ds.dims['number'])
        
        # 处理纬度维度（可能的名称：lat, latitude）
        lat_dim = None
        lat_coord = None
        for name in ['lat', 'latitude']:
            if name in ds.dims:
                lat_dim = name
                info['n_lat'] = int(ds.dims[name])
            if name in ds.coords:
                lat_coord = name
        
        # 处理经度维度（可能的名称：lon, longitude）
        lon_dim = None
        lon_coord = None
        for name in ['lon', 'longitude']:
            if name in ds.dims:
                lon_dim = name
                info['n_lon'] = int(ds.dims[name])
            if name in ds.coords:
                lon_coord = name
        
        # 处理气压层维度（可能的名称：level, plev, pressure）
        if 'level' in ds.dims:
            info['n_levels'] = int(ds.dims['level'])
            if 'level' in ds.coords:
                info['levels'] = ds.level.values.tolist()
        elif 'plev' in ds.dims:
            info['n_levels'] = int(ds.dims['plev'])
            if 'plev' in ds.coords:
                info['levels'] = ds.plev.values.tolist()
        elif 'pressure' in ds.dims:
            info['n_levels'] = int(ds.dims['pressure'])
            if 'pressure' in ds.coords:
                info['levels'] = ds.pressure.values.tolist()
        
        # 获取坐标范围
        if lat_coord and lat_dim:
            lat_data = ds[lat_coord]
            info['lat_range'] = [float(lat_data.min().values), float(lat_data.max().values)]
            if lat_dim in ds.dims and ds.dims[lat_dim] > 1:
                try:
                    diff = lat_data.diff(lat_dim)
                    if len(diff) > 0:
                        info['lat_resolution'] = float(abs(diff.mean().values))
                except:
                    # 如果diff失败，尝试计算平均间隔
                    if len(lat_data) > 1:
                        vals = lat_data.values
                        info['lat_resolution'] = float(abs((vals[-1] - vals[0]) / (len(vals) - 1)))
        
        if lon_coord and lon_dim:
            lon_data = ds[lon_coord]
            info['lon_range'] = [float(lon_data.min().values), float(lon_data.max().values)]
            if lon_dim in ds.dims and ds.dims[lon_dim] > 1:
                try:
                    diff = lon_data.diff(lon_dim)
                    if len(diff) > 0:
                        info['lon_resolution'] = float(abs(diff.mean().values))
                except:
                    # 如果diff失败，尝试计算平均间隔
                    if len(lon_data) > 1:
                        vals = lon_data.values
                        info['lon_resolution'] = float(abs((vals[-1] - vals[0]) / (len(vals) - 1)))
        
        ds.close()
        return info
    except Exception as e:
        print(f"  Error reading {file_path} with xarray: {e}")
        return None

def check_model_data(model_name, file_type='pl'):
    """检查单个模式的数据信息"""
    print(f"\n{'='*80}")
    print(f"检查模式: {model_name} ({file_type})")
    print(f"{'='*80}")
    
    model_dir = Path(DATA_PATHS["forecast_dir"]) / model_name
    if not model_dir.exists():
        print(f"  错误: 目录不存在: {model_dir}")
        return None
    
    # 获取文件后缀
    suffix = MODEL_FILE_MAP[model_name][file_type]
    
    # 查找所有相关文件
    if file_type == 'sfc':
        pattern = f"*.{suffix}.nc"
    else:
        pattern = f"*.{suffix}.nc"
        # 排除sfc文件
        sfc_suffix = MODEL_FILE_MAP[model_name].get('sfc', '')
        if sfc_suffix:
            pattern = f"*.{suffix}.nc"
    
    files = sorted(list(model_dir.glob(pattern)))
    
    if not files:
        print(f"  未找到文件匹配模式: {pattern}")
        return None
    
    # 过滤掉sfc文件（如果检查pl）
    if file_type == 'pl':
        files = [f for f in files if '.sfc.' not in f.name]
    
    print(f"  找到 {len(files)} 个文件")
    
    if len(files) == 0:
        return None
    
    # 提取年份和月份
    years = set()
    months = set()
    for f in files:
        # 文件名格式: YYYYMM.suffix.nc
        name = f.name
        if len(name) >= 6:
            try:
                year = int(name[:4])
                month = int(name[4:6])
                years.add(year)
                months.add(month)
            except:
                pass
    
    print(f"  年份范围: {min(years)} - {max(years)}")
    print(f"  月份: {sorted(months)}")
    
    # 检查每个年份的成员数和预报时效
    print(f"\n  检查每个年份的成员数和预报时效...")
    year_info = {}
    
    for year in sorted(years):
        year_files = sorted([f for f in files if f.name.startswith(str(year))])
        if not year_files:
            continue
        
        # 检查该年份的几个代表性文件
        sample_year_files = year_files[::max(1, len(year_files)//3)]  # 每季度一个
        if not sample_year_files:
            sample_year_files = year_files[:1]
        
        year_members = []
        year_leadtimes = []
        year_info[year] = {'files_checked': [], 'members': [], 'leadtimes': []}
        
        for file_path in sample_year_files:
            info = get_file_info_xarray(file_path)
            if info:
                if 'n_members' in info:
                    year_members.append(info['n_members'])
                # 预报时效 = time维度大小
                if 'dims' in info and 'time' in info['dims']:
                    year_leadtimes.append(info['dims']['time'])
                year_info[year]['files_checked'].append(file_path.name)
                year_info[year]['members'].append(info.get('n_members'))
                year_info[year]['leadtimes'].append(info.get('dims', {}).get('time'))
        
        if year_members:
            year_info[year]['n_members'] = max(set(year_members), key=year_members.count)  # 最常见的值
            year_info[year]['members_range'] = [min(year_members), max(year_members)] if year_members else None
        if year_leadtimes:
            year_info[year]['n_leadtimes'] = max(set(year_leadtimes), key=year_leadtimes.count)  # 最常见的值
            year_info[year]['leadtimes_range'] = [min(year_leadtimes), max(year_leadtimes)] if year_leadtimes else None
    
    # 检查几个代表性文件用于其他信息
    sample_files = []
    if files:
        # 选择不同年份的文件
        sample_years = sorted(years)[::max(1, len(years)//5)]  # 大约每20%选择一年
        for year in sample_years:
            year_files = [f for f in files if f.name.startswith(str(year))]
            if year_files:
                sample_files.append(year_files[0])
    
    if not sample_files:
        sample_files = files[:min(5, len(files))]  # 至少检查前5个文件
    
    print(f"\n  检查 {len(sample_files)} 个代表性文件获取详细信息...")
    
    all_info = []
    for i, file_path in enumerate(sample_files):
        print(f"    [{i+1}/{len(sample_files)}] {file_path.name}")
        info = get_file_info_xarray(file_path)
        if info:
            info['file'] = str(file_path.name)
            all_info.append(info)
    
    if not all_info:
        return None
    
    # 统计成员数和预报时效的时间变化
    members_by_year = {}
    leadtimes_by_year = {}
    for year, info in year_info.items():
        if 'n_members' in info:
            members_by_year[year] = info['n_members']
        if 'n_leadtimes' in info:
            leadtimes_by_year[year] = info['n_leadtimes']
    
    # 汇总信息
    summary = {
        'model': model_name,
        'file_type': file_type,
        'file_suffix': suffix,
        'n_files': len(files),
        'years': sorted(years),
        'months': sorted(months),
        'year_range': f"{min(years)}-{max(years)}",
        'members_by_year': members_by_year,
        'leadtimes_by_year': leadtimes_by_year,
        'year_info': year_info,
    }
    
    # 检查一致性
    first_info = all_info[0]
    summary.update({
        'n_members': first_info.get('n_members'),
        'n_lat': first_info.get('n_lat'),
        'n_lon': first_info.get('n_lon'),
        'n_levels': first_info.get('n_levels'),
        'levels': first_info.get('levels'),
        'lat_range': first_info.get('lat_range'),
        'lon_range': first_info.get('lon_range'),
        'lat_resolution': first_info.get('lat_resolution'),
        'lon_resolution': first_info.get('lon_resolution'),
        'data_vars': first_info.get('data_vars', []),
        'dims': first_info.get('dims', {}),
        'n_leadtimes': first_info.get('dims', {}).get('time'),  # 预报时效数 = time维度大小
    })
    
    # 检查所有文件是否一致（除了成员数，因为它会变化）
    consistent = True
    for info in all_info[1:]:
        if info.get('n_lat') != summary['n_lat']:
            consistent = False
            print(f"    警告: 纬度维度不一致!")
        if info.get('n_lon') != summary['n_lon']:
            consistent = False
            print(f"    警告: 经度维度不一致!")
    
    summary['consistent'] = consistent
    summary['sample_files'] = [info['file'] for info in all_info]
    
    # 打印摘要
    print(f"\n  摘要信息:")
    if members_by_year:
        unique_members = sorted(set(members_by_year.values()))
        if len(unique_members) == 1:
            print(f"    集合成员数: {unique_members[0]} (所有年份一致)")
        else:
            print(f"    集合成员数: {unique_members[0]} (范围: {min(unique_members)} - {max(unique_members)}, **随时间变化**)")
            # 显示成员数变化的年份
            changes = []
            prev_year = None
            prev_members = None
            for year in sorted(members_by_year.keys()):
                members = members_by_year[year]
                if prev_year and members != prev_members:
                    changes.append(f"{prev_year}->{year}: {prev_members}->{members}")
                prev_year = year
                prev_members = members
            if changes:
                print(f"      变化: {'; '.join(changes[:5])}")
    else:
        print(f"    集合成员数: {summary.get('n_members', 'N/A')}")
    
    if leadtimes_by_year:
        unique_leadtimes = sorted(set(leadtimes_by_year.values()))
        if len(unique_leadtimes) == 1:
            print(f"    预报时效数: {unique_leadtimes[0]}个月 (time维度大小，对应leadtime 0-{unique_leadtimes[0]-1})")
        else:
            print(f"    预报时效数: {unique_leadtimes[0]}个月 (范围: {min(unique_leadtimes)} - {max(unique_leadtimes)}个月, **随时间变化**)")
    elif summary.get('n_leadtimes'):
        print(f"    预报时效数: {summary['n_leadtimes']}个月 (time维度大小，对应leadtime 0-{summary['n_leadtimes']-1})")
    
    print(f"    分辨率: {summary['n_lat']} x {summary['n_lon']}")
    if summary.get('lat_resolution'):
        print(f"    纬度分辨率: {summary['lat_resolution']:.2f}°")
    if summary.get('lon_resolution'):
        print(f"    经度分辨率: {summary['lon_resolution']:.2f}°")
    if summary.get('n_levels'):
        print(f"    气压层数: {summary['n_levels']}")
        if summary.get('levels') and len(summary['levels']) <= 15:
            print(f"    气压层: {summary['levels']}")
    print(f"    变量: {', '.join(summary['data_vars'][:10])}")
    if len(summary['data_vars']) > 10:
        print(f"      ... 共 {len(summary['data_vars'])} 个变量")
    
    return summary

def main():
    """主函数"""
    print("="*80)
    print("MMPE 模式数据信息检查")
    print("="*80)
    print(f"数据目录: {DATA_PATHS['forecast_dir']}")
    print(f"模式列表: {', '.join(MODEL_LIST)}")
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # 检查每个模式
    for model_name in MODEL_LIST:
        model_results = {}
        
        # 检查压力层数据
        print(f"\n检查压力层数据...")
        pl_info = check_model_data(model_name, 'pl')
        if pl_info:
            model_results['pressure_level'] = pl_info
        
        # 检查地表数据
        print(f"\n检查地表数据...")
        sfc_info = check_model_data(model_name, 'sfc')
        if sfc_info:
            model_results['surface'] = sfc_info
        
        if model_results:
            all_results[model_name] = model_results
    
    # 保存结果
    output_file = Path(__file__).parent / "MODEL_DATA_INFO.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\n结果已保存到: {output_file}")
    
    # 生成Markdown文档
    generate_markdown_doc(all_results)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def generate_markdown_doc(results):
    """生成Markdown文档"""
    md_file = Path(__file__).parent / "MODEL_DATA_INVENTORY.md"
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# MMPE 模式数据清单\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("本文档详细列出了MMPE工具包中使用的所有模式数据的详细信息。\n\n")
        f.write("## 目录\n\n")
        
        for model_name in MODEL_LIST:
            f.write(f"- [{model_name}](#{model_name.lower().replace('-', '')})\n")
        
        f.write("\n---\n\n")
        
        # 详细内容
        for model_name in MODEL_LIST:
            if model_name not in results:
                continue
            
            model_results = results[model_name]
            f.write(f"## {model_name}\n\n")
            
            # 压力层数据
            if 'pressure_level' in model_results:
                pl_info = model_results['pressure_level']
                f.write("### 压力层数据 (Pressure Level)\n\n")
                f.write(f"- **文件后缀**: `{pl_info['file_suffix']}`\n")
                f.write(f"- **文件数量**: {pl_info['n_files']}\n")
                f.write(f"- **年份范围**: {pl_info['year_range']}\n")
                f.write(f"- **月份**: {', '.join(map(str, pl_info['months']))}\n")
                # 集合成员数信息
                if 'members_by_year' in pl_info and pl_info['members_by_year']:
                    unique_members = sorted(set(pl_info['members_by_year'].values()))
                    if len(unique_members) == 1:
                        f.write(f"- **集合成员数**: {unique_members[0]} (所有年份一致)\n")
                    else:
                        f.write(f"- **集合成员数**: {unique_members[0]} (范围: {min(unique_members)} - {max(unique_members)}, **随时间变化**)\n")
                        # 详细列出成员数变化
                        f.write(f"  - 成员数变化详情:\n")
                        for year in sorted(pl_info['members_by_year'].keys()):
                            f.write(f"    - {year}: {pl_info['members_by_year'][year]} 成员\n")
                else:
                    f.write(f"- **集合成员数**: {pl_info.get('n_members', 'N/A')}\n")
                
                # 预报时效信息
                if 'leadtimes_by_year' in pl_info and pl_info['leadtimes_by_year']:
                    unique_leadtimes = sorted(set(pl_info['leadtimes_by_year'].values()))
                    leadtime_desc = f"{unique_leadtimes[0]}个月"
                    if len(unique_leadtimes) > 1:
                        leadtime_desc += f" (范围: {min(unique_leadtimes)} - {max(unique_leadtimes)}个月, **随时间变化**)"
                    f.write(f"- **预报时效数**: {leadtime_desc} (time维度大小，通常对应leadtime 0-5)\n")
                elif pl_info.get('n_leadtimes'):
                    f.write(f"- **预报时效数**: {pl_info['n_leadtimes']}个月 (time维度大小，通常对应leadtime 0-5)\n")
                else:
                    f.write(f"- **预报时效数**: N/A\n")
                f.write(f"- **空间分辨率**: {pl_info['n_lat']} × {pl_info['n_lon']}\n")
                if pl_info.get('lat_resolution'):
                    f.write(f"  - 纬度分辨率: {pl_info['lat_resolution']:.2f}°\n")
                    f.write(f"  - 纬度范围: {pl_info['lat_range'][0]:.2f}° - {pl_info['lat_range'][1]:.2f}°\n")
                if pl_info.get('lon_resolution'):
                    f.write(f"  - 经度分辨率: {pl_info['lon_resolution']:.2f}°\n")
                    f.write(f"  - 经度范围: {pl_info['lon_range'][0]:.2f}° - {pl_info['lon_range'][1]:.2f}°\n")
                if pl_info.get('n_levels'):
                    f.write(f"- **气压层数**: {pl_info['n_levels']}\n")
                    if pl_info.get('levels'):
                        levels = pl_info['levels']
                        if len(levels) <= 15:
                            f.write(f"  - 气压层 (hPa): {', '.join(map(str, levels))}\n")
                        else:
                            f.write(f"  - 气压层范围: {min(levels)} - {max(levels)} hPa\n")
                            f.write(f"  - 前10层: {', '.join(map(str, levels[:10]))}\n")
                f.write(f"- **变量数量**: {len(pl_info['data_vars'])}\n")
                f.write(f"- **主要变量**: {', '.join(pl_info['data_vars'][:15])}\n")
                if len(pl_info['data_vars']) > 15:
                    f.write(f"  - ... 共 {len(pl_info['data_vars'])} 个变量\n")
                f.write(f"- **维度信息**: {pl_info['dims']}\n")
                f.write(f"- **数据一致性**: {'是' if pl_info['consistent'] else '否'}\n")
                f.write("\n")
            
            # 地表数据
            if 'surface' in model_results:
                sfc_info = model_results['surface']
                f.write("### 地表数据 (Surface)\n\n")
                f.write(f"- **文件后缀**: `{sfc_info['file_suffix']}`\n")
                f.write(f"- **文件数量**: {sfc_info['n_files']}\n")
                f.write(f"- **年份范围**: {sfc_info['year_range']}\n")
                f.write(f"- **月份**: {', '.join(map(str, sfc_info['months']))}\n")
                # 集合成员数信息
                if 'members_by_year' in sfc_info and sfc_info['members_by_year']:
                    unique_members = sorted(set(sfc_info['members_by_year'].values()))
                    if len(unique_members) == 1:
                        f.write(f"- **集合成员数**: {unique_members[0]} (所有年份一致)\n")
                    else:
                        f.write(f"- **集合成员数**: {unique_members[0]} (范围: {min(unique_members)} - {max(unique_members)}, **随时间变化**)\n")
                        # 详细列出成员数变化
                        f.write(f"  - 成员数变化详情:\n")
                        for year in sorted(sfc_info['members_by_year'].keys()):
                            f.write(f"    - {year}: {sfc_info['members_by_year'][year]} 成员\n")
                else:
                    f.write(f"- **集合成员数**: {sfc_info.get('n_members', 'N/A')}\n")
                
                # 预报时效信息
                if 'leadtimes_by_year' in sfc_info and sfc_info['leadtimes_by_year']:
                    unique_leadtimes = sorted(set(sfc_info['leadtimes_by_year'].values()))
                    leadtime_desc = f"{unique_leadtimes[0]}个月"
                    if len(unique_leadtimes) > 1:
                        leadtime_desc += f" (范围: {min(unique_leadtimes)} - {max(unique_leadtimes)}个月, **随时间变化**)"
                    f.write(f"- **预报时效数**: {leadtime_desc} (time维度大小，通常对应leadtime 0-5)\n")
                elif sfc_info.get('n_leadtimes'):
                    f.write(f"- **预报时效数**: {sfc_info['n_leadtimes']}个月 (time维度大小，通常对应leadtime 0-5)\n")
                else:
                    f.write(f"- **预报时效数**: N/A\n")
                f.write(f"- **空间分辨率**: {sfc_info['n_lat']} × {sfc_info['n_lon']}\n")
                if sfc_info.get('lat_resolution'):
                    f.write(f"  - 纬度分辨率: {sfc_info['lat_resolution']:.2f}°\n")
                    f.write(f"  - 纬度范围: {sfc_info['lat_range'][0]:.2f}° - {sfc_info['lat_range'][1]:.2f}°\n")
                if sfc_info.get('lon_resolution'):
                    f.write(f"  - 经度分辨率: {sfc_info['lon_resolution']:.2f}°\n")
                    f.write(f"  - 经度范围: {sfc_info['lon_range'][0]:.2f}° - {sfc_info['lon_range'][1]:.2f}°\n")
                f.write(f"- **变量数量**: {len(sfc_info['data_vars'])}\n")
                f.write(f"- **主要变量**: {', '.join(sfc_info['data_vars'][:15])}\n")
                if len(sfc_info['data_vars']) > 15:
                    f.write(f"  - ... 共 {len(sfc_info['data_vars'])} 个变量\n")
                f.write(f"- **维度信息**: {sfc_info['dims']}\n")
                f.write(f"- **数据一致性**: {'是' if sfc_info['consistent'] else '否'}\n")
                f.write("\n")
            
            f.write("---\n\n")
        
        # 总结表格
        f.write("## 总结表格\n\n")
        f.write("| 模式 | 类型 | 成员数 | 预报时效 | 分辨率 | 年份范围 | 气压层数 |\n")
        f.write("|------|------|--------|----------|--------|----------|----------|\n")
        
        for model_name in MODEL_LIST:
            if model_name not in results:
                continue
            
            model_results = results[model_name]
            
            # 压力层
            if 'pressure_level' in model_results:
                pl = model_results['pressure_level']
                n_levels = pl.get('n_levels', 'N/A')
                if isinstance(n_levels, int):
                    n_levels = str(n_levels)
                
                # 成员数信息
                if 'members_by_year' in pl and pl['members_by_year']:
                    unique_members = sorted(set(pl['members_by_year'].values()))
                    if len(unique_members) == 1:
                        members_str = str(unique_members[0])
                    else:
                        members_str = f"{min(unique_members)}-{max(unique_members)}"
                else:
                    members_str = str(pl.get('n_members', 'N/A'))
                
                # 预报时效信息
                leadtime_str = str(pl.get('n_leadtimes', pl.get('dims', {}).get('time', 'N/A')))
                if leadtime_str != 'N/A' and leadtime_str.isdigit():
                    leadtime_str = f"{leadtime_str}个月"
                
                f.write(f"| {model_name} | PL | {members_str} | {leadtime_str} | ")
                f.write(f"{pl.get('n_lat', 'N/A')}×{pl.get('n_lon', 'N/A')} | ")
                f.write(f"{pl.get('year_range', 'N/A')} | {n_levels} |\n")
            
            # 地表
            if 'surface' in model_results:
                sfc = model_results['surface']
                
                # 成员数信息
                if 'members_by_year' in sfc and sfc['members_by_year']:
                    unique_members = sorted(set(sfc['members_by_year'].values()))
                    if len(unique_members) == 1:
                        members_str = str(unique_members[0])
                    else:
                        members_str = f"{min(unique_members)}-{max(unique_members)}"
                else:
                    members_str = str(sfc.get('n_members', 'N/A'))
                
                # 预报时效信息
                leadtime_str = str(sfc.get('n_leadtimes', sfc.get('dims', {}).get('time', 'N/A')))
                if leadtime_str != 'N/A' and leadtime_str.isdigit():
                    leadtime_str = f"{leadtime_str}个月"
                
                f.write(f"| {model_name} | SFC | {members_str} | {leadtime_str} | ")
                f.write(f"{sfc.get('n_lat', 'N/A')}×{sfc.get('n_lon', 'N/A')} | ")
                f.write(f"{sfc.get('year_range', 'N/A')} | - |\n")
    
    print(f"Markdown文档已生成: {md_file}")

if __name__ == "__main__":
    main()

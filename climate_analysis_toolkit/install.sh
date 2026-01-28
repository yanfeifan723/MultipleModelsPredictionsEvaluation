#!/bin/bash

# 气候分析工具包安装脚本
# 支持多种环境配置

set -e  # 遇到错误时退出

echo "=========================================="
echo "气候分析工具包安装脚本"
echo "=========================================="

# 检测操作系统
OS=$(uname -s)
echo "检测到操作系统: $OS"

# 检测Python环境
if command -v conda &> /dev/null; then
    echo "检测到Conda环境"
    ENV_MANAGER="conda"
elif command -v mamba &> /dev/null; then
    echo "检测到Mamba环境"
    ENV_MANAGER="mamba"
else
    echo "检测到pip环境"
    ENV_MANAGER="pip"
fi

# 创建虚拟环境（如果不存在）
ENV_NAME="clim"
if [ "$ENV_MANAGER" = "conda" ] || [ "$ENV_MANAGER" = "mamba" ]; then
    if ! $ENV_MANAGER env list | grep -q "^$ENV_NAME "; then
        echo "创建Conda环境: $ENV_NAME"
        $ENV_MANAGER create -n $ENV_NAME python=3.10 -y
    else
        echo "Conda环境 $ENV_NAME 已存在"
    fi
    echo "激活环境: $ENV_NAME"
    source $($ENV_MANAGER info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
else
    echo "使用pip安装依赖"
fi

# 升级pip
echo "升级pip..."
python -m pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
python -m pip install numpy scipy pandas xarray netCDF4

# 安装科学计算依赖
echo "安装科学计算依赖..."
python -m pip install scikit-learn scipy

# 安装可视化依赖
echo "安装可视化依赖..."
python -m pip install matplotlib cartopy cmocean seaborn

# 安装地理数据处理依赖
echo "安装地理数据处理依赖..."
python -m pip install fiona shapely pyproj

# 安装开发工具（可选）
if [ "$1" = "--dev" ]; then
    echo "安装开发工具..."
    python -m pip install pytest pytest-cov black flake8 mypy
fi

# 安装项目本身
echo "安装气候分析工具包..."
python -m pip install -e .

# 创建必要的目录
echo "创建数据目录..."
mkdir -p /sas12t1/ffyan/obs
mkdir -p /sas12t1/ffyan/output/{temp,prec}/{eof_analysis,spatial_corr,spatial_rmse,spectrum_analysis,plots}
mkdir -p /sas12t1/ffyan/temp

# 检查安装
echo "检查安装..."
python -c "
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.analyzer import ClimateAnalyzer
print('✓ 所有依赖安装成功')
print('✓ 工具包导入成功')
"

# 运行测试
echo "运行基础测试..."
python test_toolkit.py

echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 运行测试: python test_toolkit.py"
echo "3. 查看文档: cat README.md"
echo ""
echo "注意事项:"
echo "- 观测数据目录: /sas12t1/ffyan/obs (已自动创建)"
echo "- 预测数据目录: /raid62/EC-C3S/month"
echo "- 输出目录: /sas12t1/ffyan/output (已自动创建)"
echo "- 如需修改数据路径，请编辑 src/config/settings.py"
echo ""

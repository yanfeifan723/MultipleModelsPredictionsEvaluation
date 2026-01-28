#!/usr/bin/env python3
"""
气候分析工具包安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements文件
requirements = []
with open(this_directory / "requirements.txt", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="climate-analysis-toolkit",
    version="1.0.0",
    author="Climate Analysis Team",
    author_email="climate@example.com",
    description="一个通用的智能绘图工具包，提供类似Origin的自动化绘图功能",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/climate-analysis-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "climate-analyzer=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords="plotting visualization smart-plotter origin-like",
    project_urls={
        "Bug Reports": "https://github.com/example/climate-analysis-toolkit/issues",
        "Source": "https://github.com/example/climate-analysis-toolkit",
        "Documentation": "https://climate-analysis-toolkit.readthedocs.io/",
    },
)

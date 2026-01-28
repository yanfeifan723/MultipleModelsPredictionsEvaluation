# 气候分析工具包 Makefile
# 提供便捷的命令来管理工具包

.PHONY: help install install-dev test test-verbose clean docs lint format check-all

# 默认目标
help:
	@echo "气候分析工具包管理命令"
	@echo "========================"
	@echo ""
	@echo "安装和设置:"
	@echo "  install      - 安装工具包和依赖"
	@echo "  install-dev  - 安装开发依赖"
	@echo ""
	@echo "测试:"
	@echo "  test         - 运行基础测试"
	@echo "  test-verbose - 运行详细测试"
	@echo "  test-all     - 运行所有测试"
	@echo ""
	@echo "代码质量:"
	@echo "  lint         - 代码风格检查"
	@echo "  format       - 代码格式化"
	@echo "  check-all    - 运行所有检查"
	@echo ""
	@echo "文档:"
	@echo "  docs         - 生成文档"
	@echo ""
	@echo "清理:"
	@echo "  clean        - 清理临时文件"
	@echo "  clean-all    - 清理所有生成的文件"
	@echo ""
	@echo "使用示例:"
	@echo "  make install"
	@echo "  make test"
	@echo "  make check-all"

# 安装
install:
	@echo "安装气候分析工具包..."
	@bash install.sh

install-dev:
	@echo "安装开发依赖..."
	@bash install.sh --dev

# 测试
test:
	@echo "运行基础测试..."
	@python test_toolkit.py

test-verbose:
	@echo "运行详细测试..."
	@python -m pytest tests/ -v

test-all:
	@echo "运行所有测试..."
	@python test_toolkit.py
	@python tests/test_comprehensive.py

# 代码质量检查
lint:
	@echo "运行代码风格检查..."
	@flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "格式化代码..."
	@black src/ tests/ --line-length=100

check-all: lint format test
	@echo "所有检查完成"

# 文档
docs:
	@echo "生成文档..."
	@if command -v sphinx-build > /dev/null; then \
		cd docs && make html; \
	else \
		echo "警告: sphinx-build 未找到，跳过文档生成"; \
	fi

# 清理
clean:
	@echo "清理临时文件..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	@echo "清理所有生成的文件..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf output/
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/

# 环境检查
check-env:
	@echo "检查环境..."
	@python -c "import sys; print(f'Python版本: {sys.version}')"
	@python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
	@python -c "import xarray; print(f'Xarray版本: {xarray.__version__}')"
	@python -c "import matplotlib; print(f'Matplotlib版本: {matplotlib.__version__}')"
	@python -c "import cartopy; print(f'Cartopy版本: {cartopy.__version__}')"

# 示例运行
example:
	@echo "运行基础示例..."
	@python examples/basic_usage.py

# 命令行接口测试
cli-test:
	@echo "测试命令行接口..."
	@python -m src.cli info --models
	@python -m src.cli info --variables
	@python -m src.cli info --config

# 构建包
build:
	@echo "构建包..."
	@python setup.py sdist bdist_wheel

# 发布准备
release-prep: clean-all check-all build
	@echo "发布准备完成"

# 快速开始
quick-start: install test example
	@echo "快速开始完成！"
	@echo "现在您可以:"
	@echo "1. 查看 examples/basic_usage.py"
	@echo "2. 运行 python -m src.cli --help"
	@echo "3. 查看 README.md 了解更多信息"

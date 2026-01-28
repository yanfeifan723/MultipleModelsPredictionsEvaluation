#!/usr/bin/env python3
"""
气候分析工具包命令行接口
提供便捷的命令行操作
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

from .analyzer import ClimateAnalyzer
from .config.settings import get_model_names, get_variable_names, get_lead_times


def setup_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="气候分析工具包命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行EOF分析
  climate-analyzer eof --var temp --models CMCC-35 ECMWF-51-mon
  
  # 运行相关性分析
  climate-analyzer correlation --var prec --lead-times 0 1 2
  
  # 运行RMSE分析
  climate-analyzer rmse --var temp --all-models
  
  # 运行功率谱分析
  climate-analyzer spectrum --var temp --region global
  
  # 创建所有图表
  climate-analyzer plots --var temp --output-dir ./my_plots
        """
    )
    
    # 全局参数
    parser.add_argument(
        "--obs-dir", 
        type=str, 
        default="/sas12t1/ffyan/obs",
        help="观测数据目录 (默认: /sas12t1/ffyan/obs)"
    )
    parser.add_argument(
        "--forecast-dir", 
        type=str, 
        default="/raid62/EC-C3S/month",
        help="预测数据目录 (默认: /raid62/EC-C3S/month)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/sas12t1/ffyan/output",
        help="输出目录 (默认: /sas12t1/ffyan/output)"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--log-file", 
        type=str,
        help="日志文件路径"
    )
    parser.add_argument(
        "--log-dir", 
        type=str,
        default="/sas12t1/ffyan/log",
        help="日志目录路径 (默认: /sas12t1/ffyan/log)"
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # EOF分析命令
    eof_parser = subparsers.add_parser("eof", help="运行EOF分析")
    eof_parser.add_argument("--var", required=True, choices=get_variable_names(), help="变量类型")
    eof_parser.add_argument("--models", nargs="+", choices=get_model_names(), help="模型列表")
    eof_parser.add_argument("--all-models", action="store_true", help="使用所有模型")
    eof_parser.add_argument("--n-modes", type=int, default=6, help="EOF模态数量")
    eof_parser.add_argument("--standardize", action="store_true", help="标准化数据")
    
    # 相关性分析命令
    corr_parser = subparsers.add_parser("correlation", help="运行相关性分析")
    corr_parser.add_argument("--var", required=True, choices=get_variable_names(), help="变量类型")
    corr_parser.add_argument("--models", nargs="+", choices=get_model_names(), help="模型列表")
    corr_parser.add_argument("--all-models", action="store_true", help="使用所有模型")
    corr_parser.add_argument("--lead-times", nargs="+", type=int, choices=get_lead_times(), help="提前期列表")
    corr_parser.add_argument("--all-lead-times", action="store_true", help="使用所有提前期")
    corr_parser.add_argument("--min-valid-points", type=int, default=3, help="最小有效点数")
    
    # RMSE分析命令
    rmse_parser = subparsers.add_parser("rmse", help="运行RMSE分析")
    rmse_parser.add_argument("--var", required=True, choices=get_variable_names(), help="变量类型")
    rmse_parser.add_argument("--models", nargs="+", choices=get_model_names(), help="模型列表")
    rmse_parser.add_argument("--all-models", action="store_true", help="使用所有模型")
    rmse_parser.add_argument("--lead-times", nargs="+", type=int, choices=get_lead_times(), help="提前期列表")
    rmse_parser.add_argument("--all-lead-times", action="store_true", help="使用所有提前期")
    rmse_parser.add_argument("--high-bias-threshold", type=float, default=15, help="高偏差阈值")
    rmse_parser.add_argument("--high-rmse-threshold", type=float, default=15, help="高RMSE阈值")
    
    # 功率谱分析命令
    spectrum_parser = subparsers.add_parser("spectrum", help="运行功率谱分析")
    spectrum_parser.add_argument("--var", required=True, choices=get_variable_names(), help="变量类型")
    spectrum_parser.add_argument("--models", nargs="+", choices=get_model_names(), help="模型列表")
    spectrum_parser.add_argument("--all-models", action="store_true", help="使用所有模型")
    spectrum_parser.add_argument("--region", default="global", choices=["global", "east_asia", "south_asia"], help="分析区域")
    spectrum_parser.add_argument("--window-type", default="hanning", help="窗函数类型")
    spectrum_parser.add_argument("--detrend", action="store_true", help="去趋势")
    
    # 绘图命令
    plots_parser = subparsers.add_parser("plots", help="创建所有图表")
    plots_parser.add_argument("--var", required=True, choices=get_variable_names(), help="变量类型")
    plots_parser.add_argument("--output-dir", type=str, help="输出目录")
    plots_parser.add_argument("--dpi", type=int, default=300, help="图像DPI")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="运行测试")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    # 信息命令
    info_parser = subparsers.add_parser("info", help="显示工具包信息")
    info_parser.add_argument("--models", action="store_true", help="显示支持的模型")
    info_parser.add_argument("--variables", action="store_true", help="显示支持的变量")
    info_parser.add_argument("--config", action="store_true", help="显示配置信息")
    
    return parser


def run_eof_analysis(args) -> None:
    """运行EOF分析"""
    print(f"运行EOF分析 - 变量: {args.var}")
    
    # 确定模型列表
    if args.all_models:
        models = get_model_names()
    elif args.models:
        models = args.models
    else:
        models = None
    
    # 创建分析器
    analyzer = ClimateAnalyzer(
        obs_dir=args.obs_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # 运行分析
    results = analyzer.run_eof_analysis(args.var, models)
    print(f"EOF分析完成，结果保存在: {args.output_dir}")


def run_correlation_analysis(args) -> None:
    """运行相关性分析"""
    print(f"运行相关性分析 - 变量: {args.var}")
    
    # 确定模型和提前期列表
    if args.all_models:
        models = get_model_names()
    elif args.models:
        models = args.models
    else:
        models = None
    
    if args.all_lead_times:
        lead_times = get_lead_times()
    elif args.lead_times:
        lead_times = args.lead_times
    else:
        lead_times = [0, 1, 2]  # 默认前3个提前期
    
    # 创建分析器
    analyzer = ClimateAnalyzer(
        obs_dir=args.obs_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # 运行分析
    results = analyzer.run_correlation_analysis(args.var, models)
    print(f"相关性分析完成，结果保存在: {args.output_dir}")


def run_rmse_analysis(args) -> None:
    """运行RMSE分析"""
    print(f"运行RMSE分析 - 变量: {args.var}")
    
    # 确定模型列表
    if args.all_models:
        models = get_model_names()
    elif args.models:
        models = args.models
    else:
        models = None
    
    # 创建分析器
    analyzer = ClimateAnalyzer(
        obs_dir=args.obs_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # 运行分析
    results = analyzer.run_rmse_analysis(args.var, models)
    print(f"RMSE分析完成，结果保存在: {args.output_dir}")


def run_spectrum_analysis(args) -> None:
    """运行功率谱分析"""
    print(f"运行功率谱分析 - 变量: {args.var}, 区域: {args.region}")
    
    # 确定模型列表
    if args.all_models:
        models = get_model_names()
    elif args.models:
        models = args.models
    else:
        models = None
    
    # 创建分析器
    analyzer = ClimateAnalyzer(
        obs_dir=args.obs_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # 运行分析
    results = analyzer.run_spectrum_analysis(args.var, args.region, models)
    print(f"功率谱分析完成，结果保存在: {args.output_dir}")


def create_plots(args) -> None:
    """创建所有图表"""
    print(f"创建图表 - 变量: {args.var}")
    
    # 创建分析器
    analyzer = ClimateAnalyzer(
        obs_dir=args.obs_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    
    # 创建图表
    output_dir = args.output_dir if args.output_dir else None
    plot_paths = analyzer.create_all_plots(args.var, output_dir)
    
    print("图表创建完成:")
    for plot_type, path in plot_paths.items():
        print(f"  {plot_type}: {path}")


def run_tests(args) -> None:
    """运行测试"""
    print("运行工具包测试...")
    
    # 导入测试模块
    try:
        from test_toolkit import main as run_test_main
        success = run_test_main()
        if success:
            print("✓ 所有测试通过")
        else:
            print("✗ 部分测试失败")
            sys.exit(1)
    except ImportError as e:
        print(f"✗ 无法导入测试模块: {e}")
        sys.exit(1)


def show_info(args) -> None:
    """显示工具包信息"""
    if args.models:
        print("支持的模型:")
        for model in get_model_names():
            print(f"  - {model}")
        print()
    
    if args.variables:
        print("支持的变量:")
        for var in get_variable_names():
            print(f"  - {var}")
        print()
    
    if args.config:
        print("配置信息:")
        print(f"  观测数据目录: {args.obs_dir}")
        print(f"  预测数据目录: {args.forecast_dir}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  支持的提前期: {get_lead_times()}")
        print()
    
    if not any([args.models, args.variables, args.config]):
        print("气候分析工具包")
        print("=" * 30)
        print("一个综合的气候数据分析工具包")
        print()
        print("支持的分析类型:")
        print("  - EOF分析")
        print("  - 相关性分析")
        print("  - RMSE分析")
        print("  - 功率谱分析")
        print()
        print("使用方法:")
        print("  climate-analyzer --help")
        print("  climate-analyzer info --models")
        print("  climate-analyzer info --variables")


def main():
    """主函数"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    
    # 确保日志目录存在
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志文件路径
    if args.log_file:
        log_file_path = args.log_file
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(args.log_dir, f"climate_analysis_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 执行相应的命令
    try:
        if args.command == "eof":
            run_eof_analysis(args)
        elif args.command == "correlation":
            run_correlation_analysis(args)
        elif args.command == "rmse":
            run_rmse_analysis(args)
        elif args.command == "spectrum":
            run_spectrum_analysis(args)
        elif args.command == "plots":
            create_plots(args)
        elif args.command == "test":
            run_tests(args)
        elif args.command == "info":
            show_info(args)
        else:
            print(f"未知命令: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

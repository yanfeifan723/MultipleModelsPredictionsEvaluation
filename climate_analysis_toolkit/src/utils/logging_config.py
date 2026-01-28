"""
日志配置工具模块
提供统一的日志配置功能
"""

import os
import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "/sas12t1/ffyan/log",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    module_name: str = "climate_analysis",
    use_basic_config: bool = True
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_dir: 日志目录路径
        log_file: 日志文件名（可选）
        log_level: 日志级别
        module_name: 模块名称
        use_basic_config: 是否使用basicConfig（MMPE脚本兼容模式）
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件路径
    if log_file is None:
        log_file = f"{module_name}.log"
    
    log_file_path = os.path.join(log_dir, log_file)
    
    if use_basic_config:
        # MMPE脚本使用的简化格式
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(module_name)
    else:
        # 标准配置方式
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有的处理器
        logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 设置格式化器
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger


def get_logger(module_name: str = "climate_analysis") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        module_name: 模块名称
    
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(module_name)


def configure_logging_from_config(config: dict) -> logging.Logger:
    """
    从配置文件配置日志
    
    Args:
        config: 配置字典
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    log_config = config.get('logging', {})
    
    return setup_logging(
        log_dir=log_config.get('log_dir', '/sas12t1/ffyan/log'),
        log_level=log_config.get('level', 'INFO'),
        module_name='climate_analysis'
    )

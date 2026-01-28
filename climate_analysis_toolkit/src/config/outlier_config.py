#!/usr/bin/env python3
"""
离群值检测配置文件

定义不同场景下的离群值检测参数，包括：
- RMSE分析
- EOF分析
- 其他气候数据分析

支持多种检测方法和阈值配置。
"""

from typing import Dict, Any

# RMSE分析的离群值检测配置
RMSE_OUTLIER_CONFIG = {
    'method': 'iqr',  # 检测方法：'iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof'
    'threshold': 6.0,  # 检测阈值 - 大幅放宽到6.0以最小化离群值检测
    'min_samples': 10,  # 最小样本数
    'group_by': ['model', 'leadtime'],  # 分组维度
    'description': 'RMSE分析的离群值检测配置，使用IQR方法，阈值为6.0（大幅放宽标准）'
}

# EOF分析的离群值检测配置
EOF_OUTLIER_CONFIG = {
    'method': 'modified_zscore',  # 使用修正Z-score方法，对异常值更鲁棒
    'threshold': 3.5,  # 检测阈值
    'min_samples': 20,  # 最小样本数
    'group_by': ['mode', 'leadtime'],  # 分组维度
    'description': 'EOF分析的离群值检测配置，使用修正Z-score方法，阈值为3.5'
}

# 通用气候数据的离群值检测配置
GENERAL_OUTLIER_CONFIG = {
    'method': 'iqr',  # 检测方法
    'threshold': 1.5,  # 检测阈值
    'min_samples': 5,  # 最小样本数
    'group_by': None,  # 不分组
    'description': '通用气候数据的离群值检测配置，使用IQR方法，阈值为1.5'
}

# 严格模式的离群值检测配置
STRICT_OUTLIER_CONFIG = {
    'method': 'zscore',  # 使用Z-score方法
    'threshold': 3.0,  # 严格的检测阈值
    'min_samples': 30,  # 较大的最小样本数
    'group_by': ['model', 'leadtime', 'season'],  # 更细粒度的分组
    'description': '严格模式的离群值检测配置，使用Z-score方法，阈值为3.0'
}

# 宽松模式的离群值检测配置
LENIENT_OUTLIER_CONFIG = {
    'method': 'iqr',  # 使用IQR方法
    'threshold': 3.0,  # 宽松的检测阈值
    'min_samples': 5,  # 较小的最小样本数
    'group_by': ['model'],  # 简单的分组
    'description': '宽松模式的离群值检测配置，使用IQR方法，阈值为3.0'
}

# 机器学习方法的离群值检测配置
ML_OUTLIER_CONFIG = {
    'method': 'isolation_forest',  # 使用隔离森林方法
    'contamination': 0.1,  # 预期的异常值比例
    'min_samples': 50,  # 较大的最小样本数
    'group_by': ['model', 'leadtime'],  # 分组维度
    'description': '机器学习方法的离群值检测配置，使用隔离森林方法'
}

# 配置字典，用于快速选择不同的配置
OUTLIER_CONFIGS = {
    'rmse': RMSE_OUTLIER_CONFIG,
    'eof': EOF_OUTLIER_CONFIG,
    'general': GENERAL_OUTLIER_CONFIG,
    'strict': STRICT_OUTLIER_CONFIG,
    'lenient': LENIENT_OUTLIER_CONFIG,
    'ml': ML_OUTLIER_CONFIG
}

def get_outlier_config(config_name: str = 'rmse') -> Dict[str, Any]:
    """
    获取离群值检测配置
    
    Args:
        config_name: 配置名称，可选值：'rmse', 'eof', 'general', 'strict', 'lenient', 'ml'
    
    Returns:
        离群值检测配置字典
    """
    if config_name not in OUTLIER_CONFIGS:
        raise ValueError(f"不支持的配置名称: {config_name}。支持的配置: {list(OUTLIER_CONFIGS.keys())}")
    
    return OUTLIER_CONFIGS[config_name].copy()

def validate_outlier_config(config: Dict[str, Any]) -> bool:
    """
    验证离群值检测配置
    
    Args:
        config: 离群值检测配置字典
    
    Returns:
        配置是否有效
    """
    required_keys = ['method', 'threshold', 'min_samples']
    
    # 检查必需键
    for key in required_keys:
        if key not in config:
            print(f"错误: 配置缺少必需键 '{key}'")
            return False
    
    # 验证方法
    valid_methods = ['iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof']
    if config['method'] not in valid_methods:
        print(f"错误: 不支持的检测方法 '{config['method']}'。支持的方法: {valid_methods}")
        return False
    
    # 验证阈值
    if config['threshold'] <= 0:
        print(f"错误: 阈值必须大于0，当前值: {config['threshold']}")
        return False
    
    # 验证最小样本数
    if config['min_samples'] < 1:
        print(f"错误: 最小样本数必须大于0，当前值: {config['min_samples']}")
        return False
    
    # 方法特定的验证
    if config['method'] == 'iqr' and config['threshold'] > 5:
        print(f"警告: IQR阈值 {config['threshold']} 可能过大，建议使用1.5-3.0")
    
    if config['method'] == 'zscore' and config['threshold'] < 2:
        print(f"警告: Z-score阈值 {config['threshold']} 可能过小，建议使用2.5-3.5")
    
    if config['method'] == 'modified_zscore' and config['threshold'] < 2:
        print(f"警告: 修正Z-score阈值 {config['threshold']} 可能过小，建议使用3.0-4.0")
    
    return True

def print_outlier_configs():
    """打印所有可用的离群值检测配置"""
    print("可用的离群值检测配置:")
    print("=" * 50)
    
    for name, config in OUTLIER_CONFIGS.items():
        print(f"\n{name.upper()} 配置:")
        print(f"  方法: {config['method']}")
        print(f"  阈值: {config['threshold']}")
        print(f"  最小样本数: {config['min_samples']}")
        print(f"  分组: {config['group_by']}")
        print(f"  描述: {config['description']}")

if __name__ == "__main__":
    # 测试配置
    print_outlier_configs()
    
    # 测试获取配置
    print("\n" + "=" * 50)
    print("测试获取配置:")
    
    for config_name in OUTLIER_CONFIGS.keys():
        config = get_outlier_config(config_name)
        print(f"\n{config_name}: {config['description']}")
        print(f"  验证结果: {'通过' if validate_outlier_config(config) else '失败'}")

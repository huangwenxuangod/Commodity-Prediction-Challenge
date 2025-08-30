#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品预测挑战赛 - 小白学习开始脚本
第一阶段：理解数据

这个脚本将帮助你：
1. 加载和查看数据
2. 理解数据的基本结构
3. 分析目标变量
4. 创建可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据文件"""
    print("🔄 正在加载数据...")
    
    try:
        # 加载训练标签数据
        train_labels = pd.read_csv('../../data/train_labels.csv')
        print("✅ 成功加载 train_labels.csv")
        
        # 加载目标变量定义
        target_pairs = pd.read_csv('../../data/target_pairs.csv')
        print("✅ 成功加载 target_pairs.csv")
        
        return train_labels, target_pairs
    
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到数据文件 - {e}")
        print("请确保你在 hwx/learning/ 目录下运行此脚本")
        return None, None
    except Exception as e:
        print(f"❌ 错误：加载数据时出现问题 - {e}")
        return None, None

def analyze_basic_info(train_labels, target_pairs):
    """分析数据的基本信息"""
    print("\n" + "="*50)
    print("📊 数据基本信息分析")
    print("="*50)
    
    # 训练数据基本信息
    print(f"\n📈 训练数据 (train_labels.csv):")
    print(f"   - 数据形状: {train_labels.shape}")
    print(f"   - 行数: {train_labels.shape[0]:,} 行")
    print(f"   - 列数: {train_labels.shape[1]:,} 列")
    
    # 目标变量信息
    target_columns = [col for col in train_labels.columns if col.startswith('target_')]
    print(f"   - 目标变量数量: {len(target_columns)} 个")
    
    # 目标变量定义信息
    print(f"\n🎯 目标变量定义 (target_pairs.csv):")
    print(f"   - 目标变量数量: {len(target_pairs)} 个")
    print(f"   - 滞后设置: {target_pairs['lag'].unique()}")
    
    # 显示前几个目标变量的定义
    print(f"\n📋 前5个目标变量定义:")
    for i, row in target_pairs.head().iterrows():
        print(f"   {row['target']}: {row['pair']} (滞后{row['lag']}期)")
    
    return target_columns

def analyze_target_variables(train_labels, target_columns):
    """分析目标变量的分布和特征"""
    print("\n" + "="*50)
    print("🎯 目标变量深度分析")
    print("="*50)
    
    # 选择几个目标变量进行详细分析
    sample_targets = target_columns[:5]  # 分析前5个目标变量
    
    print(f"\n📊 分析前5个目标变量的统计特征:")
    
    for target in sample_targets:
        data = train_labels[target].dropna()
        print(f"\n{target}:")
        print(f"   - 数据点数量: {len(data):,}")
        print(f"   - 平均值: {data.mean():.6f}")
        print(f"   - 标准差: {data.std():.6f}")
        print(f"   - 最小值: {data.min():.6f}")
        print(f"   - 最大值: {data.max():.6f}")
        print(f"   - 缺失值数量: {train_labels[target].isna().sum():,}")

def create_visualizations(train_labels, target_columns):
    """创建可视化图表"""
    print("\n" + "="*50)
    print("📊 创建可视化图表")
    print("="*50)
    
    # 选择几个目标变量进行可视化
    sample_targets = target_columns[:6]  # 可视化前6个目标变量
    
    # 创建目标变量分布图
    print("🖼️  正在创建目标变量分布图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('目标变量分布分析', fontsize=16, fontweight='bold')
    
    for i, target in enumerate(sample_targets):
        row = i // 3
        col = i % 3
        
        data = train_labels[target].dropna()
        
        # 创建直方图
        axes[row, col].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col].set_title(f'{target} 分布', fontsize=12)
        axes[row, col].set_xlabel('值')
        axes[row, col].set_ylabel('频次')
        
        # 添加统计信息
        mean_val = data.mean()
        std_val = data.std()
        axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                               label=f'均值: {mean_val:.4f}')
        axes[row, col].axvline(mean_val + std_val, color='orange', linestyle=':', 
                               label=f'均值+标准差: {mean_val + std_val:.4f}')
        axes[row, col].axvline(mean_val - std_val, color='orange', linestyle=':', 
                               label=f'均值-标准差: {mean_val - std_val:.4f}')
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image/target_variables_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 分布图已保存为: image/target_variables_distribution.png")
    
    # 创建相关性热力图
    print("🖼️  正在创建相关性热力图...")
    
    # 计算目标变量之间的相关性
    correlation_data = train_labels[target_columns[:10]].corr()  # 前10个目标变量
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('目标变量相关性热力图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('image/target_correlations.png', dpi=300, bbox_inches='tight')
    print("✅ 相关性热力图已保存为: image/target_correlations.png")
    
    plt.show()

def analyze_time_series(train_labels):
    """分析时间序列特性"""
    print("\n" + "="*50)
    print("⏰ 时间序列特性分析")
    print("="*50)
    
    # 检查是否有时间列
    if 'date_id' in train_labels.columns:
        print("📅 发现时间列: date_id")
        
        # 转换时间格式
        try:
            train_labels['date_id'] = pd.to_datetime(train_labels['date_id'])
            
            # 时间范围分析
            start_date = train_labels['date_id'].min()
            end_date = train_labels['date_id'].max()
            total_days = (end_date - start_date).days
            
            print(f"   - 开始日期: {start_date}")
            print(f"   - 结束日期: {end_date}")
            print(f"   - 总天数: {total_days:,} 天")
            print(f"   - 数据点数量: {len(train_labels):,}")
            
            # 检查时间连续性
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            missing_dates = date_range.difference(train_labels['date_id'])
            
            if len(missing_dates) == 0:
                print("   - 时间连续性: ✅ 完整（无缺失日期）")
            else:
                print(f"   - 时间连续性: ⚠️  有 {len(missing_dates)} 个缺失日期")
                
        except Exception as e:
            print(f"   - 时间格式转换失败: {e}")
    else:
        print("⚠️  未发现时间列，可能数据已经按时间排序")

def create_summary_report(train_labels, target_columns, target_pairs):
    """创建数据概览报告"""
    print("\n" + "="*50)
    print("📋 数据概览报告")
    print("="*50)
    
    # 数据质量报告
    print("\n🔍 数据质量分析:")
    
    # 缺失值分析
    missing_data = train_labels.isnull().sum()
    missing_percentage = (missing_data / len(train_labels)) * 100
    
    print("   - 缺失值统计:")
    for col in target_columns[:10]:  # 显示前10个目标变量
        if missing_data[col] > 0:
            print(f"     {col}: {missing_data[col]:,} 个缺失值 ({missing_percentage[col]:.2f}%)")
    
    # 异常值分析
    print("\n   - 异常值分析 (使用3倍标准差):")
    for col in target_columns[:5]:  # 分析前5个目标变量
        data = train_labels[col].dropna()
        mean_val = data.mean()
        std_val = data.std()
        
        outliers = data[(data < mean_val - 3*std_val) | (data > mean_val + 3*std_val)]
        print(f"     {col}: {len(outliers)} 个异常值 ({len(outliers)/len(data)*100:.2f}%)")
    
    # 目标变量分组分析
    print("\n📊 目标变量分组分析:")
    
    # 按滞后分组
    lag_groups = target_pairs.groupby('lag').size()
    print("   - 按滞后分组:")
    for lag, count in lag_groups.items():
        print(f"     滞后{lag}期: {count} 个目标变量")
    
    # 按资产类型分组（简单分析）
    print("\n   - 按资产类型分组:")
    asset_types = {
        'LME': 0,  # 伦敦金属交易所
        'JPX': 0,  # 日本交易所
        'US_Stock': 0,  # 美股
        'FX': 0,   # 外汇
        'Other': 0
    }
    
    for _, row in target_pairs.iterrows():
        pair = row['pair']
        if 'LME_' in pair:
            asset_types['LME'] += 1
        elif 'JPX_' in pair:
            asset_types['JPX'] += 1
        elif 'US_Stock_' in pair:
            asset_types['US_Stock'] += 1
        elif 'FX_' in pair:
            asset_types['FX'] += 1
        else:
            asset_types['Other'] += 1
    
    for asset_type, count in asset_types.items():
        if count > 0:
            print(f"     {asset_type}: {count} 个目标变量")

def main():
    """主函数"""
    print("🎯 欢迎来到商品预测挑战赛学习之旅！")
    print("🚀 让我们开始第一阶段：理解数据")
    print("="*60)
    
    # 1. 加载数据
    train_labels, target_pairs = load_data()
    if train_labels is None:
        return
    
    # 2. 分析基本信息
    target_columns = analyze_basic_info(train_labels, target_pairs)
    
    # 3. 分析目标变量
    analyze_target_variables(train_labels, target_columns)
    
    # 4. 分析时间序列特性
    analyze_time_series(train_labels)
    
    # 5. 创建可视化
    create_visualizations(train_labels, target_columns)
    
    # 6. 创建总结报告
    create_summary_report(train_labels, target_columns, target_pairs)
    
    print("\n" + "="*60)
    print("🎉 第一阶段学习完成！")
    print("📚 你已经学会了：")
    print("   ✅ 如何加载和查看数据")
    print("   ✅ 理解数据的基本结构")
    print("   ✅ 分析目标变量的分布")
    print("   ✅ 创建可视化图表")
    print("   ✅ 识别数据质量问题")
    print("\n🚀 接下来准备进入第二阶段：特征工程！")
    print("="*60)

if __name__ == "__main__":
    main()

# 🎯 第一阶段：理解数据 (第1-2周)
# 学习目标：知道数据长什么样，理解每个数字的含义，发现数据的规律

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加项目根目录到路径，以便导入数据
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_and_explore_data():
    """任务1：加载和查看数据"""
    print("🚀 开始第一阶段：理解数据")
    print("=" * 50)
    
    try:
        # 加载训练标签
        print("📊 正在加载训练标签数据...")
        train_labels = pd.read_csv('../../data/train_labels.csv')
        
        print("✅ 数据加载成功！")
        print(f"📈 数据形状: {train_labels.shape}")
        print(f"📋 列名: {train_labels.columns.tolist()}")
        
        print("\n🔍 前5行数据:")
        print(train_labels.head())
        
        print("\n📊 数据基本信息:")
        print(train_labels.info())
        
        print("\n📈 数值列统计信息:")
        print(train_labels.describe())
        
        return train_labels
        
    except FileNotFoundError:
        print("❌ 错误：找不到数据文件")
        print("请确保在正确的目录下运行此脚本")
        return None
    except Exception as e:
        print(f"❌ 错误：{e}")
        return None

def analyze_target_variables(train_labels):
    """任务2：分析目标变量"""
    print("\n" + "=" * 50)
    print("🎯 任务2：分析目标变量分布")
    print("=" * 50)
    
    if train_labels is None:
        print("❌ 无法分析目标变量，数据未加载")
        return
    
    # 选择几个目标变量进行可视化
    target_columns = [col for col in train_labels.columns if col.startswith('target_')]
    
    if not target_columns:
        print("❌ 未找到目标变量列")
        return
    
    print(f"🎯 找到 {len(target_columns)} 个目标变量")
    print(f"目标变量列表: {target_columns[:10]}...")  # 只显示前10个
    
    # 选择前6个目标变量进行可视化
    targets_to_plot = target_columns[:6]
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('目标变量分布分析', fontsize=16, fontweight='bold')
        
        for i, target in enumerate(targets_to_plot):
            row = i // 3
            col = i % 3
            
            # 获取非空值
            data = train_labels[target].dropna()
            
            if len(data) > 0:
                # 绘制直方图
                axes[row, col].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[row, col].set_title(f'{target} 分布', fontweight='bold')
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
                axes[row, col].legend()
                
                print(f"📊 {target}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 样本数={len(data)}")
            else:
                axes[row, col].text(0.5, 0.5, f'{target}\n无数据', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                print(f"⚠️ {target}: 无数据")
        
        plt.tight_layout()
        plt.show()
        
        # 保存图表
        plt.savefig('image/target_variables_distribution.png', dpi=300, bbox_inches='tight')
        print("💾 图表已保存为 'image/target_variables_distribution.png'")
        
    except Exception as e:
        print(f"❌ 绘图错误：{e}")
        print("请检查matplotlib是否正确安装")

def analyze_data_quality(train_labels):
    """分析数据质量"""
    print("\n" + "=" * 50)
    print("🔍 数据质量分析")
    print("=" * 50)
    
    if train_labels is None:
        return
    
    # 检查缺失值
    print("📊 缺失值统计:")
    missing_data = train_labels.isnull().sum()
    missing_percentage = (missing_data / len(train_labels)) * 100
    
    missing_df = pd.DataFrame({
        '列名': missing_data.index,
        '缺失值数量': missing_data.values,
        '缺失值百分比': missing_percentage.values
    })
    
    # 只显示有缺失值的列
    missing_df = missing_df[missing_df['缺失值数量'] > 0].sort_values('缺失值数量', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✅ 没有缺失值！")
    
    # 检查重复值
    print(f"\n🔄 重复行数量: {train_labels.duplicated().sum()}")
    
    # 检查数据类型
    print("\n📋 数据类型:")
    print(train_labels.dtypes.value_counts())

def main():
    """主函数"""
    print("🎯 商品预测挑战赛 - 小白学习指南")
    print("📚 第一阶段：理解数据")
    print("=" * 60)
    
    # 任务1：加载和查看数据
    train_labels = load_and_explore_data()
    
    # 任务2：分析目标变量
    analyze_target_variables(train_labels)
    
    # 额外任务：分析数据质量
    analyze_data_quality(train_labels)
    
    print("\n" + "=" * 60)
    print("🎉 第一阶段学习任务完成！")
    print("📝 接下来你可以：")
    print("   1. 仔细查看数据分布图表")
    print("   2. 思考数据的规律和特点")
    print("   3. 准备进入第二阶段：特征工程")
    print("=" * 60)

if __name__ == "__main__":
    main()

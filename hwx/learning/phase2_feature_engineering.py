# 🎯 第二阶段：创建特征 (第3-4周)
# 学习目标：学会从原始数据中提取有用信息，创建能够帮助预测的特征

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_data():
    """加载数据"""
    print("📊 正在加载数据...")
    try:
        train_labels = pd.read_csv('../data/train_labels.csv')
        print(f"✅ 成功加载训练标签数据，形状: {train_labels.shape}")
        return train_labels
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def create_time_features(df):
    """任务1：创建时间特征"""
    print("\n" + "=" * 50)
    print("⏰ 任务1：创建时间特征")
    print("=" * 50)
    
    # 确保date_id是日期类型
    if 'date_id' in df.columns:
        if df['date_id'].dtype == 'object':
            df['date_id'] = pd.to_datetime(df['date_id'])
        elif df['date_id'].dtype == 'int64':
            # 如果是整数，假设是日期ID，转换为日期
            df['date_id'] = pd.to_datetime(df['date_id'], unit='D', origin='1970-01-01')
        
        print("📅 创建基础时间特征...")
        
        # 基础时间特征
        df['year'] = df['date_id'].dt.year
        df['month'] = df['date_id'].dt.month
        df['day_of_week'] = df['date_id'].dt.dayofweek
        df['quarter'] = df['date_id'].dt.quarter
        df['day_of_year'] = df['date_id'].dt.dayofyear
        
        # 季节性特征
        df['is_quarter_start'] = df['date_id'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date_id'].dt.is_quarter_end.astype(int)
        df['is_month_start'] = df['date_id'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date_id'].dt.is_month_end.astype(int)
        
        # 周期性特征
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        print("✅ 时间特征创建完成！")
        print(f"新增特征数量: {len([col for col in df.columns if col in ['year', 'month', 'day_of_week', 'quarter', 'day_of_year', 'is_quarter_start', 'is_quarter_end', 'is_month_start', 'is_month_end', 'sin_month', 'cos_month', 'sin_day_of_week', 'cos_day_of_week']])}")
        
        return df
    else:
        print("⚠️ 未找到date_id列，跳过时间特征创建")
        return df

def create_lag_features(df, target_cols, lags=[1, 2, 3, 5, 10]):
    """任务2：创建滞后特征"""
    print("\n" + "=" * 50)
    print("🔄 任务2：创建滞后特征")
    print("=" * 50)
    
    if not target_cols:
        print("⚠️ 未找到目标变量，跳过滞后特征创建")
        return df
    
    print(f"🎯 为目标变量创建滞后特征，滞后期数: {lags}")
    
    # 按日期排序（如果有date_id列）
    if 'date_id' in df.columns:
        df = df.sort_values('date_id').reset_index(drop=True)
    
    new_features_count = 0
    
    for col in target_cols:
        for lag in lags:
            feature_name = f'{col}_lag_{lag}'
            df[feature_name] = df[col].shift(lag)
            new_features_count += 1
    
    print(f"✅ 滞后特征创建完成！新增特征数量: {new_features_count}")
    
    return df

def create_statistical_features(df, target_cols, windows=[5, 10, 20]):
    """创建统计特征"""
    print("\n" + "=" * 50)
    print("📊 创建统计特征")
    print("=" * 50)
    
    if not target_cols:
        print("⚠️ 未找到目标变量，跳过统计特征创建")
        return df
    
    print(f"📈 创建滑动窗口统计特征，窗口大小: {windows}")
    
    new_features_count = 0
    
    for col in target_cols:
        for window in windows:
            # 移动平均
            df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            
            # 移动标准差
            df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            
            # 移动最大值
            df[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
            
            # 移动最小值
            df[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            
            # 移动中位数
            df[f'{col}_median_{window}'] = df[col].rolling(window=window, min_periods=1).median()
            
            new_features_count += 5
    
    print(f"✅ 统计特征创建完成！新增特征数量: {new_features_count}")
    
    return df

def analyze_feature_correlations(df, target_cols, max_features=20):
    """分析特征相关性"""
    print("\n" + "=" * 50)
    print("🔗 分析特征相关性")
    print("=" * 50)
    
    if not target_cols:
        print("⚠️ 未找到目标变量，跳过相关性分析")
        return
    
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 限制特征数量以避免内存问题
    if len(numeric_cols) > max_features:
        # 优先选择目标变量和重要特征
        important_cols = target_cols + [col for col in numeric_cols if 'lag_' in col or 'ma_' in col or 'std_' in col]
        selected_cols = list(set(important_cols))[:max_features]
        print(f"⚠️ 特征数量过多，选择前{max_features}个重要特征进行分析")
    else:
        selected_cols = numeric_cols
    
    try:
        # 计算相关性矩阵
        correlation_matrix = df[selected_cols].corr()
        
        # 创建热力图
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        
        plt.title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 保存图表
        plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
        print("💾 相关性热力图已保存为 'feature_correlations.png'")
        
        # 分析目标变量的相关性
        print("\n🎯 目标变量相关性分析:")
        for target in target_cols[:5]:  # 只分析前5个目标变量
            if target in correlation_matrix.columns:
                correlations = correlation_matrix[target].abs().sort_values(ascending=False)
                print(f"\n{target} 的相关性排序 (前10个):")
                print(correlations.head(10))
        
    except Exception as e:
        print(f"❌ 相关性分析失败: {e}")

def evaluate_feature_quality(df, target_cols):
    """评估特征质量"""
    print("\n" + "=" * 50)
    print("📊 特征质量评估")
    print("=" * 50)
    
    if not target_cols:
        print("⚠️ 未找到目标变量，跳过特征质量评估")
        return
    
    print("🔍 特征质量报告:")
    
    # 统计信息
    total_features = len(df.columns)
    target_features = len(target_cols)
    engineered_features = total_features - target_features
    
    print(f"📈 总特征数量: {total_features}")
    print(f"🎯 目标变量数量: {target_features}")
    print(f"🔧 工程特征数量: {engineered_features}")
    
    # 缺失值统计
    missing_stats = df.isnull().sum()
    features_with_missing = missing_stats[missing_stats > 0]
    
    if len(features_with_missing) > 0:
        print(f"\n⚠️ 有缺失值的特征数量: {len(features_with_missing)}")
        print("缺失值最多的特征:")
        print(features_with_missing.head().to_dict())
    else:
        print("\n✅ 所有特征都没有缺失值")
    
    # 特征类型分布
    print(f"\n📋 特征类型分布:")
    print(df.dtypes.value_counts())

def main():
    """主函数"""
    print("🎯 商品预测挑战赛 - 小白学习指南")
    print("📚 第二阶段：特征工程")
    print("=" * 60)
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 获取目标变量列
    target_columns = [col for col in df.columns if col.startswith('target_')]
    print(f"🎯 找到 {len(target_columns)} 个目标变量")
    
    # 任务1：创建时间特征
    df = create_time_features(df)
    
    # 任务2：创建滞后特征
    df = create_lag_features(df, target_columns)
    
    # 额外任务：创建统计特征
    df = create_statistical_features(df, target_columns)
    
    # 分析特征相关性
    analyze_feature_correlations(df, target_columns)
    
    # 评估特征质量
    evaluate_feature_quality(df, target_columns)
    
    # 保存处理后的数据
    output_file = 'train_labels_with_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 处理后的数据已保存为 '{output_file}'")
    
    print("\n" + "=" * 60)
    print("🎉 第二阶段学习任务完成！")
    print("📝 接下来你可以：")
    print("   1. 查看特征相关性热力图")
    print("   2. 分析哪些特征最有价值")
    print("   3. 准备进入第三阶段：模型训练")
    print("=" * 60)

if __name__ == "__main__":
    main()

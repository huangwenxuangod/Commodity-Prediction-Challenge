# 🎯 第三阶段：训练模型 (第5-6周)
# 学习目标：学会训练机器学习模型，能够进行预测

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_data_with_features():
    """加载带有特征的数据"""
    print("📊 正在加载带有特征的数据...")
    
    # 尝试加载已经处理过的数据
    feature_file = 'image/train_labels_with_features.csv'
    if os.path.exists(feature_file):
        print(f"✅ 找到特征文件: {feature_file}")
        df = pd.read_csv(feature_file)
    else:
        print("⚠️ 未找到特征文件，尝试加载原始数据...")
        try:
            df = pd.read_csv('../../data/train_labels.csv')
            print("✅ 成功加载原始数据")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, None
    
    # 获取目标变量列
    target_columns = [col for col in df.columns if col.startswith('target_')]
    print(f"🎯 找到 {len(target_columns)} 个目标变量")
    
    return df, target_columns

def prepare_features_and_targets(df, target_columns):
    """准备特征和目标变量"""
    print("\n" + "=" * 50)
    print("🔧 准备特征和目标变量")
    print("=" * 50)
    
    # 选择特征列（排除目标变量和日期列）
    feature_columns = [col for col in df.columns 
                      if col not in target_columns and col != 'date_id']
    
    print(f"📊 特征列数量: {len(feature_columns)}")
    print(f"🎯 目标变量数量: {len(target_columns)}")
    
    # 处理缺失值
    print("\n🔍 处理缺失值...")
    initial_missing = df[feature_columns + target_columns].isnull().sum().sum()
    print(f"初始缺失值总数: {initial_missing}")
    
    # 对于特征列，用0填充缺失值
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # 对于目标变量，删除有缺失值的行
    df_clean = df.dropna(subset=target_columns)
    
    final_missing = df_clean[feature_columns + target_columns].isnull().sum().sum()
    print(f"处理后缺失值总数: {final_missing}")
    print(f"清理后数据形状: {df_clean.shape}")
    
    # 准备特征和目标变量
    X = df_clean[feature_columns]
    y = df_clean[target_columns]
    
    print(f"✅ 特征矩阵形状: {X.shape}")
    print(f"✅ 目标变量形状: {y.shape}")
    
    return X, y, feature_columns

def train_simple_model(X, y, target_name):
    """任务1：训练简单模型"""
    print(f"\n🎯 训练简单模型: {target_name}")
    print("-" * 40)
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 时间序列数据不打乱
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    # 创建模型
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=0
    )
    
    print("🚀 开始训练模型...")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("✅ 模型训练完成！")
    
    # 预测
    y_pred = model.predict(X_val)
    
    # 评估
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"📊 模型性能评估:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 前10个重要特征:")
    print(feature_importance.head(10))
    
    return model, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_importance': feature_importance
    }

def train_multi_output_model(X, y):
    """任务2：多目标预测"""
    print("\n" + "=" * 50)
    print("🎯 任务2：多目标预测")
    print("=" * 50)
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    # 创建多输出随机森林
    base_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    multi_model = MultiOutputRegressor(base_model)
    
    print("🚀 开始训练多输出模型...")
    
    # 训练模型
    multi_model.fit(X_train, y_train)
    
    print("✅ 多输出模型训练完成！")
    
    # 预测
    y_pred = multi_model.predict(X_val)
    
    # 评估每个目标变量
    print("\n📊 多输出模型性能评估:")
    
    results = {}
    for i, target_name in enumerate(y.columns):
        target_actual = y_val.iloc[:, i]
        target_pred = y_pred[:, i]
        
        mse = mean_squared_error(target_actual, target_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_actual, target_pred)
        r2 = r2_score(target_actual, target_pred)
        
        results[target_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"\n{target_name}:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
    
    # 计算平均性能
    avg_rmse = np.mean([results[target]['rmse'] for target in results])
    avg_r2 = np.mean([results[target]['r2'] for target in results])
    
    print(f"\n🏆 平均性能:")
    print(f"  平均RMSE: {avg_rmse:.6f}")
    print(f"  平均R²: {avg_r2:.6f}")
    
    return multi_model, results

def compare_models(single_results, multi_results):
    """比较单目标和多目标模型"""
    print("\n" + "=" * 50)
    print("🔍 模型比较分析")
    print("=" * 50)
    
    try:
        # 检查数据结构并安全地提取数据
        if isinstance(single_results, dict) and len(single_results) > 0:
            first_target = list(single_results.keys())[0]
            
            # 安全地提取RMSE值
            if isinstance(single_results[first_target], dict) and 'rmse' in single_results[first_target]:
                single_rmse = single_results[first_target]['rmse']
            else:
                single_rmse = single_results[first_target] if isinstance(single_results[first_target], (int, float)) else 0
                
            if isinstance(multi_results, dict) and len(multi_results) > 0:
                if isinstance(multi_results[first_target], dict) and 'rmse' in multi_results[first_target]:
                    multi_rmse = multi_results[first_target]['rmse']
                else:
                    multi_rmse = multi_results[first_target] if isinstance(multi_results[first_target], (int, float)) else 0
            else:
                multi_rmse = multi_results if isinstance(multi_results, (int, float)) else 0
            
            print(f"📊 比较 {first_target} 的预测性能:")
            print(f"单目标模型 RMSE: {single_rmse:.6f}")
            print(f"多目标模型 RMSE: {multi_rmse:.6f}")
            
            if single_rmse > 0 and multi_rmse > 0:
                if single_rmse < multi_rmse:
                    print("🏆 单目标模型表现更好")
                else:
                    print("🏆 多目标模型表现更好")
                    
                # 计算性能提升
                improvement = ((single_rmse - multi_rmse) / single_rmse) * 100
                print(f"性能提升: {improvement:.2f}%")
            else:
                print("⚠️  无法比较模型性能（RMSE值异常）")
        else:
            print("⚠️  单目标模型结果为空或格式异常")
            return
            
        # 可视化比较
        try:
            # 准备比较数据
            if isinstance(single_results, dict) and isinstance(multi_results, dict):
                targets = list(single_results.keys())[:5]  # 只比较前5个
                
                single_rmse_list = []
                multi_rmse_list = []
                
                for target in targets:
                    # 安全地提取RMSE值
                    if isinstance(single_results[target], dict) and 'rmse' in single_results[target]:
                        single_rmse_list.append(single_results[target]['rmse'])
                    else:
                        single_rmse_list.append(single_results[target] if isinstance(single_results[target], (int, float)) else 0)
                        
                    if isinstance(multi_results[target], dict) and 'rmse' in multi_results[target]:
                        multi_rmse_list.append(multi_results[target]['rmse'])
                    else:
                        multi_rmse_list.append(multi_results[target] if isinstance(multi_results[target], (int, float)) else 0)
            else:
                # 如果结果不是字典格式，使用简单的比较
                targets = ['模型1', '模型2']
                single_rmse_list = [single_rmse]
                multi_rmse_list = [multi_rmse]
            
            # 创建比较图表
            x = np.arange(len(targets))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            rects1 = ax.bar(x - width/2, single_rmse_list, width, label='单目标模型', alpha=0.8, color='skyblue')
            rects2 = ax.bar(x + width/2, multi_rmse_list, width, label='多目标模型', alpha=0.8, color='lightcoral')
            
            ax.set_xlabel('目标变量')
            ax.set_ylabel('RMSE')
            ax.set_title('单目标 vs 多目标模型性能比较')
            ax.set_xticks(x)
            ax.set_xticklabels(targets, rotation=45)
            ax.legend()
            
            # 添加数值标签
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.4f}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            plt.show()
            
            # 保存图表
            plt.savefig('image/model_comparison.png', dpi=300, bbox_inches='tight')
            print("💾 模型比较图表已保存为 'image/model_comparison.png'")
            
        except Exception as e:
            print(f"❌ 可视化比较失败: {e}")
            
    except Exception as e:
        print(f"❌ 模型比较分析失败: {e}")
        print("继续执行其他功能...")

def cross_validation_analysis(X, y, n_splits=5):
    """交叉验证分析"""
    print("\n" + "=" * 50)
    print("🔄 交叉验证分析")
    print("=" * 50)
    
    # 使用随机森林进行交叉验证
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    print("🚀 开始交叉验证...")
    
    # 对每个目标变量进行交叉验证
    cv_scores = {}
    for i, target_name in enumerate(y.columns[:5]):  # 只验证前5个目标变量
        target_values = y.iloc[:, i]
        
        # 删除目标变量中的缺失值
        valid_indices = ~target_values.isnull()
        X_valid = X[valid_indices]
        y_valid = target_values[valid_indices]
        
        if len(X_valid) > 0:
            scores = cross_val_score(model, X_valid, y_valid, 
                                   cv=n_splits, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            
            cv_scores[target_name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std(),
                'scores': rmse_scores
            }
            
            print(f"{target_name}: 平均RMSE = {rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")
    
    return cv_scores

def main():
    """主函数"""
    print("🎯 商品预测挑战赛 - 小白学习指南")
    print("📚 第三阶段：模型训练")
    print("=" * 60)
    
    # 加载数据
    df, target_columns = load_data_with_features()
    if df is None:
        return
    
    # 准备特征和目标变量
    X, y, feature_columns = prepare_features_and_targets(df, target_columns)
    if X is None:
        return
    
    # 任务1：训练简单模型（选择第一个目标变量）
    print(f"\n🎯 选择第一个目标变量进行单目标训练: {target_columns[0]}")
    single_model, single_results = train_simple_model(X, y.iloc[:, 0], target_columns[0])
    
    # 任务2：多目标预测
    multi_model, multi_results = train_multi_output_model(X, y)
    
    # 比较模型性能 - 确保数据结构一致
    if single_results is not None and multi_results is not None:
        compare_models(single_results, multi_results)
    else:
        print("⚠️  跳过模型比较（模型训练结果为空）")
    
    # 交叉验证分析
    cv_scores = cross_validation_analysis(X, y)
    
    # 保存模型
    import joblib
    
    # 保存单目标模型
    joblib.dump(single_model, 'image/single_target_model.pkl')
    print("\n💾 单目标模型已保存为 'image/single_target_model.pkl'")
    
    # 保存多目标模型
    joblib.dump(multi_model, 'image/multi_target_model.pkl')
    print("💾 多目标模型已保存为 'image/multi_target_model.pkl'")
    
    print("\n" + "=" * 60)
    print("🎉 第三阶段学习任务完成！")
    print("📝 接下来你可以：")
    print("   1. 分析模型性能结果")
    print("   2. 查看特征重要性")
    print("   3. 准备进入第四阶段：深度学习")
    print("=" * 60)

if __name__ == "__main__":
    main()

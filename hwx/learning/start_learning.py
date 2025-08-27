#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 商品预测挑战赛 - 小白学习快速启动脚本
帮助你快速开始学习之旅
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """打印欢迎横幅"""
    print("=" * 80)
    print("🎯 商品预测挑战赛 - 小白完全指南")
    print("🚀 欢迎开始你的学习之旅！")
    print("=" * 80)
    print()

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python环境...")
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 警告: 建议使用Python 3.8或更高版本")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("\n方法1：使用conda（推荐）")
        print("conda env create -f environment.yml")
        print("conda activate hwx-learning")
        print("\n方法2：使用pip")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖包都已安装")
        return True

def install_dependencies():
    """安装依赖包"""
    print("\n📦 正在安装依赖包...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ 依赖包安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def show_learning_path():
    """显示学习路径"""
    print("\n📚 学习路径总览:")
    print("=" * 50)
    print("第1-2周：理解数据 → 第3-4周：创建特征 → 第5-6周：训练模型")
    print("    ↓                    ↓                    ↓")
    print("  知道数据长什么样     学会提取有用信息      学会预测未来值")
    print()
    print("第7-9周：深度学习 → 第10-11周：优化模型 → 第12周：冲刺金牌")
    print("    ↓                    ↓                    ↓")
    print("  使用高级算法          提升预测精度         最终优化部署")
    print("=" * 50)

def show_available_scripts():
    """显示可用的学习脚本"""
    print("\n📖 可用的学习脚本:")
    print("-" * 40)
    
    scripts = [
        ("phase1_data_understanding.py", "第一阶段：理解数据", "适合初学者，从数据基础开始"),
        ("phase2_feature_engineering.py", "第二阶段：特征工程", "学习创建有用的特征"),
        ("phase3_model_training.py", "第三阶段：模型训练", "训练机器学习模型")
    ]
    
    for i, (script, title, description) in enumerate(scripts, 1):
        print(f"{i}. {title}")
        print(f"   文件: {script}")
        print(f"   说明: {description}")
        print()
    
    print("💡 建议按顺序学习，每个阶段都要完全理解后再进入下一阶段")

def run_script(script_name):
    """运行指定的脚本"""
    if not os.path.exists(script_name):
        print(f"❌ 脚本文件不存在: {script_name}")
        return False
    
    print(f"\n🚀 开始运行: {script_name}")
    print("=" * 50)
    
    try:
        # 运行脚本
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ 脚本 {script_name} 运行完成！")
            return True
        else:
            print(f"\n❌ 脚本 {script_name} 运行失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行脚本时出错: {e}")
        return False

def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 请选择操作:")
        print("1. 开始第一阶段学习 (理解数据)")
        print("2. 开始第二阶段学习 (特征工程)")
        print("3. 开始第三阶段学习 (模型训练)")
        print("4. 查看学习路径")
        print("5. 检查环境")
        print("6. 安装依赖包")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        if choice == '0':
            print("\n👋 再见！祝你学习顺利！")
            break
        elif choice == '1':
            run_script('phase1_data_understanding.py')
        elif choice == '2':
            run_script('phase2_feature_engineering.py')
        elif choice == '3':
            run_script('phase3_model_training.py')
        elif choice == '4':
            show_learning_path()
        elif choice == '5':
            check_python_version()
            check_dependencies()
        elif choice == '6':
            install_dependencies()
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️ 检测到缺少依赖包")
        install_choice = input("是否现在安装依赖包？(y/n): ").strip().lower()
        if install_choice == 'y':
            if install_dependencies():
                deps_ok = True
            else:
                print("❌ 依赖包安装失败，请手动安装")
                return
    
    if not python_ok or not deps_ok:
        print("\n❌ 环境检查未通过，请解决问题后重试")
        return
    
    # 显示学习路径
    show_learning_path()
    
    # 显示可用脚本
    show_available_scripts()
    
    # 交互式菜单
    print("\n🎉 环境检查完成！现在可以开始学习了！")
    interactive_menu()

if __name__ == "__main__":
    main()

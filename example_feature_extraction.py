#!/usr/bin/env python3
"""
示例脚本：展示如何使用修改后的MCR类进行CLIP特征提取和存储

本脚本展示了三种使用模式：
1. 提取特征并保存到磁盘（内存效率模式）
2. 加载预提取的特征（快速启动模式）
3. 运行完整的检索流水线
"""

import os
import sys

# 添加正确的路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
clip_modules_path = os.path.join(current_dir, 'clip', 'modules')
sys.path.insert(0, clip_modules_path)

try:
    from retriever import MCR
except ImportError as e:
    print(f"导入错误：{e}")
    print("请确保您在正确的目录中运行此脚本")
    print(f"当前目录：{current_dir}")
    print(f"预期的retriever.py路径：{os.path.join(clip_modules_path, 'retriever.py')}")
    sys.exit(1)


def extract_and_save_features():
    """模式1：提取CLIP特征并保存到磁盘（内存效率模式）"""
    print("=" * 60)
    print("模式1：提取CLIP特征并保存（内存效率模式）")
    print("=" * 60)
    
    # 创建MCR实例，启用特征提取和内存效率模式
    mcr = MCR(
        dataset_name='mmimdb',
        data_dir='./datasets/mmimdb',  # 根据您的数据路径调整
        extract_features=True,         # 提取特征
        memory_efficient=True,         # 内存效率模式：提取后立即清理内存中的特征张量
        feature_dir='./features/mmimdb'  # 特征保存目录
    )
    
    print("✓ 特征提取完成，已保存到磁盘")
    print("✓ 内存中的大型张量已清理")


def load_precomputed_features():
    """模式2：加载预提取的特征（快速启动）"""
    print("=" * 60)
    print("模式2：加载预提取的特征（快速启动）")
    print("=" * 60)
    
    # 创建MCR实例，加载预提取的特征
    mcr = MCR(
        dataset_name='mmimdb',
        data_dir='./datasets/mmimdb',
        extract_features=False,        # 不提取特征，直接加载
        feature_dir='./features/mmimdb'
    )
    
    print("✓ 预提取特征加载完成，启动速度大大提升")
    return mcr


def run_full_pipeline():
    """模式3：运行完整的检索流水线"""
    print("=" * 60)
    print("模式3：运行完整的检索流水线")
    print("=" * 60)
    
    # 首先尝试加载预提取的特征，如果不存在则自动提取
    mcr = MCR(
        dataset_name='mmimdb',
        data_dir='./datasets/mmimdb',
        extract_features=False,  # 先尝试加载
        feature_dir='./features/mmimdb'
    )
    
    # 运行完整的检索流水线
    results = mcr.run(
        save_results=True,
        output_dir='./results/mmimdb'
    )
    
    print("✓ 检索流水线完成")
    print(f"✓ 结果已保存到 ./results/mmimdb")
    
    # 显示一些统计信息
    for split_name, df in results.items():
        print(f"  - {split_name} split: {len(df)} samples")
        if 't2t_id_list' in df.columns:
            print(f"    * Text-to-text retrieval: ✓")
        if 'i2i_id_list' in df.columns:
            print(f"    * Image-to-image retrieval: ✓")


def compare_memory_usage():
    """比较内存效率模式和正常模式的内存使用情况"""
    print("=" * 60)
    print("内存使用比较")
    print("=" * 60)
    
    import psutil
    import gc
    
    # 获取初始内存使用
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 测试内存效率模式
    print("\n测试内存效率模式...")
    mcr_efficient = MCR(
        dataset_name='mmimdb',
        data_dir='./datasets/mmimdb',
        extract_features=True,
        memory_efficient=True,
        feature_dir='./features/mmimdb_efficient'
    )
    
    efficient_memory = process.memory_info().rss / 1024 / 1024
    print(f"内存效率模式内存使用: {efficient_memory:.1f} MB")
    
    # 清理
    del mcr_efficient
    gc.collect()
    
    # 测试正常模式（注释掉以避免内存不足）
    # print("\n测试正常模式...")
    # mcr_normal = MCR(
    #     dataset_name='mmimdb',
    #     data_dir='./datasets/mmimdb',
    #     extract_features=True,
    #     memory_efficient=False,
    #     feature_dir='./features/mmimdb_normal'
    # )
    # 
    # normal_memory = process.memory_info().rss / 1024 / 1024
    # print(f"正常模式内存使用: {normal_memory:.1f} MB")
    
    print(f"\n内存效率模式相比初始状态增加: {efficient_memory - initial_memory:.1f} MB")


if __name__ == '__main__':
    print("CLIP特征提取和存储示例")
    print("作者：WTY")
    print()
    
    # 检查数据目录是否存在
    if not os.path.exists('./datasets/mmimdb'):
        print("警告：数据目录 './datasets/mmimdb' 不存在")
        print("请根据您的实际数据路径修改脚本中的路径")
        print()
    
    try:
        # 运行示例
        print("选择要运行的示例：")
        print("1. 提取特征并保存（内存效率模式）")
        print("2. 加载预提取特征")
        print("3. 运行完整检索流水线")
        print("4. 内存使用比较")
        print("5. 运行所有示例")
        
        choice = input("请输入选择 (1-5) 或按回车键运行示例1: ").strip()
        
        if choice == '' or choice == '1':
            extract_and_save_features()
        elif choice == '2':
            load_precomputed_features()
        elif choice == '3':
            run_full_pipeline()
        elif choice == '4':
            compare_memory_usage()
        elif choice == '5':
            extract_and_save_features()
            print("\n" + "="*60 + "\n")
            load_precomputed_features()
            print("\n" + "="*60 + "\n")
            run_full_pipeline()
        else:
            print("无效选择，运行默认示例...")
            extract_and_save_features()
            
    except Exception as e:
        print(f"错误：{e}")
        print("请检查数据路径和依赖项是否正确安装")

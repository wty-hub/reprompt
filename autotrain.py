import os
import signal
import sys
import subprocess
from pathlib import Path

# 数据集及其任务名
datasets = [
    # ("mmimdb", "task_finetune_mmimdb"),
    ("Food101", "task_finetune_food101"),
    # ("Hatefull_Memes", "task_finetune_hatememes"),
]

# 缺失率
ratios = [
    0.5, 
    0.7, 
    0.9,
    ]
# 缺失类型
types = [
    "image", 
    "text", 
    "both"
    ]
# 随机种子
seeds = [520, 12351]
# 训练轮数
max_epochs = 10

# 数据集根目录
data_roots = {
    "mmimdb": "../arrow_datasets/mmimdb",
    "Food101": "../arrow_datasets/Food101",
    "Hatefull_Memes": "../arrow_datasets/hateful_memes",
}

# 训练命令模板
# cmd_template = (
#     "python run.py with "
#     "data_root={data_root} "
#     "num_gpus=1 num_nodes=1 per_gpu_batchsize=4 "
#     "{task} "
#     "exp_name=exp_{dataset}_{missing_type}_{ratio}"
#     " train_ratio={ratio} train_type={missing_type}"
# )

# # 信号处理函数
# def signal_handler(sig, frame):
#     print('\n收到中断信号，正在退出训练程序...')
#     sys.exit(0)

# # 注册信号处理器
# signal.signal(signal.SIGINT, signal_handler)

# 胜利的场景
# beat = {
#     "mmimdb": [
#         # ("both", 0.5),
#         # ("text", 0.7),
#         # ("both", 0.9),
        
#     ],
#     "Hatefull_Memes": [],
#     "Food101": []
# }

# from autotrain import beat

if __name__ == "__main__":
    # 获取当前目录名
    current_dir_name = Path.cwd().name

    for dataset, task in datasets:
        data_root = data_roots[dataset]
        for ratio in ratios:
            for missing_type in types:
                for seed in seeds:
                    exp_name = f"{current_dir_name}-{dataset}-{ratio}-{missing_type}"
                    cmd = (
                        f"python run.py with "
                        f"data_root={data_root} "
                        f"num_gpus=1 num_nodes=1 per_gpu_batchsize=32 "
                        f"{task} "
                        f"exp_name={exp_name} "
                        f"test_ratio={ratio} "
                        f"test_type={missing_type} "
                        f"seed={seed} "
                        f"max_epoch={max_epochs} "
                        f"with_food_vocab "
                    )
                    print(f"Running: {cmd}")
                    
                    result = os.system(cmd)

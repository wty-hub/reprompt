import math
import os
from pathlib import Path
import argparse
import random

tasks = {
    "mmimdb": "task_finetune_mmimdb",
    "food101": "task_finetune_food101",
    "hatememes": "task_finetune_hatememes",
}

# 缺失率
ratios = [
    0.5, 
    0.7, 
    0.9,
    ]
# 缺失类型
types = [
    "text", 
    "image", 
    "both"
]

# 数据集根目录
data_roots = {
    "mmimdb": "../arrow_datasets/mmimdb",
    "food101": "../arrow_datasets/Food101",
    "hatememes": "../arrow_datasets/hateful_memes",
}

iccv_values = [
    56.16, 58.55, 56.81, 53.34, 56.89, 55.22, 53.47, 56.69, 53.06,
    83.47, 89.81, 86.33, 80.09, 88.34, 82.95, 76.46, 87.82, 81.26,
    66.41, 64.93, 68.09, 63.70, 66.79, 65.57, 66.23, 64.66, 66.74
]

best = {}
i = 0
for dataset in tasks:
    best[dataset] = {}
    for ratio in ratios:
        best[dataset][ratio] = {}
        for type in types:
            best[dataset][ratio][type] = (iccv_values[i], None)

current_dir_name = Path.cwd().name

# Parse command line arguments to get dataset parameter
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['mmimdb', 'food101', 'hatememes'])
parser.add_argument('--ratio', type=float, required=False, choices=[0.5, 0.7, 0.9])
parser.add_argument('-type', type=str, required=False, choices=['text', 'image', 'both'])

args = parser.parse_args()
current_dataset = args.dataset

while True:
    # for r in ratios:
        # for t in types:
    r = random.choice(ratios)
    t = random.choice(types)
    prompt_length = random.randint(16, 32)
    seed = random.randint(0, 10086)
    augmented_length = random.randint(1, 4)
    data_root = data_roots[current_dataset]
    task = tasks[current_dataset]
    batch_size = random.randint(12, 30)
    max_epochs = 15 if dataset == 'food101' else 7
    exp_name = f'{current_dir_name}-{current_dataset}-{r}-{t}_pLen={prompt_length}_aLen={augmented_length}-batch_size={batch_size}'
    cmd = (
        f"python run.py with "
        f"data_root={data_root} "
        f"dataset={current_dataset} "
        f"num_gpus=1 num_nodes=1 per_gpu_batchsize={batch_size} "
        f"{task} "
        f"exp_name={exp_name} "
        f"test_ratio={r} "
        f"test_type={t} "
        f"seed={seed} "
        f"max_epoch={max_epochs} "
    )
    print(f'Running {cmd}')
    result = os.system(cmd)
    if result != 0:
        exit(result)
# # import os
# # import csv

# # def find_max_value(path, column):
# #     with open(path, 'r') as f:
# #         reader = csv.DictReader(f)
# #         max_value = None
# #         for row in reader:
# #             try:
# #                 value = float(row[column])
# #                 if max_value is None or max_value < value:
# #                     max_value = value

# #             except (ValueError, TypeError, KeyError):
# #                 # print("问题列：", row)
# #                 continue
# #         return max_value
    

# # max_value = {}

# # for root, dirs, files in os.walk('result'):
# #     for file in files:
# #         if "metrics.csv" == file:
# #             # print(f"loading {root}/{file}")
# #             max_value[root] = find_max_value(os.path.join(root, file), 'val/the_metric')
# # for key, value in sorted(list(max_value.items())):
# #     print(f"{key}:\t{value}")

import csv
from pathlib import Path


def find_max_value(path, column):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        max_value = None
        for row in reader:
            try:
                value = float(row[column])
                if max_value is None or max_value < value:
                    max_value = value

            except (ValueError, TypeError, KeyError):
                # print("问题列：", row)
                continue
        return max_value


# # 数据集s
# datasets = [
#     "mmimdb",
#     "Food101",
#     "Hatefull_Memes",
# ]
# # 缺失率s
# ratios = [0.5, 0.7, 0.9]
# # 缺失类型
# types = ["image", "text", "both"]

# max_value = {}
# import math

# for dataset in datasets:
#     max_value[dataset] = {}
#     for type in types:
#         max_value[dataset][type] = {}
#         for ratio in ratios:
#             max_value[dataset][type][ratio] = -math.inf

# import re

# pattern = "train_(mmimdb|Food101|Hatefull_Memes)_([^_]+)_(-?\d+\.\d+)_.*"

# for root, dirs, files in os.walk("result"):
#     mtch = re.search(pattern, root)
#     if mtch:
#         dataset = mtch.group(1)
#         type = mtch.group(2)
#         ratio = float(mtch.group(3))
#         for file in files:
#             if "metrics.csv" == file:
#                 # print(dataset, type, ratio)
#                 # print(root, file)
#                 cur_max = find_max_value(os.path.join(root, file), "val/the_metric")
#                 max_value[dataset][type][ratio] = max(cur_max, max_value[dataset][type][ratio])
# for key, value in sorted(list(max_value.items())):
#     print(f"{key}:\t{value}")

# maxv = find_max_value('result/new-prompt-mmimdb-0.5-image_seed520/version_1/metrics.csv', 'val/the_metric')
# maxv = find_max_value('result/new-prompt-mmimdb-0.5-both_seed520/version_2/metrics.csv', 'val/the_metric')
# print(maxv)

import math
import os
import re

pattern = rf'{Path.cwd().name}-(mmimdb|Food101|Hatefull_Memes)-(-?\d+\.\d+)-(both|image|text).*'

# 数据集s
datasets = [
    # "mmimdb",
    "Food101",
    # "Hatefull_Memes",
]
# 缺失率s
ratios = [
    0.5, 
    0.7, 
    0.9
    ]
# 缺失类型
types = ["image", "text", "both"]

max_value = {}

for dataset in datasets:
    max_value[dataset] = {}
    for ratio in ratios:
        max_value[dataset][ratio] = {}
        for type in types:
            max_value[dataset][ratio][type] = -math.inf


for root, dirs, files in os.walk("result"):
    mtch = re.search(pattern, root)
    if mtch:
        dataaset = mtch.group(1)
        ratio = float(mtch.group(2))
        type = mtch.group(3)
        for file in files:
            if 'metrics.csv' == file:
                cur_max = find_max_value(os.path.join(root, file), 'val/the_metric')
                if cur_max is not None:
                    max_value[dataset][ratio][type] = max(cur_max, max_value[dataset][ratio][type])
for key, value in max_value.items():
    print(f'{key}\t{value}')
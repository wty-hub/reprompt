# 在文本prompt中插入固定词汇embedding的使用示例

"""
使用说明：
1. 在配置文件中添加 'fixed_vocab_list' 参数
2. 这些词汇的embedding将从预训练CLIP模型中提取
3. 这些embedding作为buffer注册，不会被训练更新
4. 在forward过程中，这些固定embedding会被插入到文本prompt中
"""

# 示例配置
config_example = {
    'vit': 'ViT-B/32',
    'prompt_length': 12,
    'prompt_depth': 2,
    'fixed_vocab_list': ['photo', 'image', 'picture', 'scene'],  # 添加这个参数
    # ... 其他配置参数
}

# 使用方法:
# 1. 在config中添加fixed_vocab_list参数，包含你想要插入的固定词汇
# 2. 这些词汇的embedding会从预训练的CLIP模型中提取
# 3. 在训练过程中，这些embedding不会被更新

# 固定词汇embedding的特点：
# - 从预训练CLIP模型的token_embedding层提取
# - 使用register_buffer注册，不参与梯度更新
# - 在每个batch的文本prompt中都会插入
# - 插入位置在可训练prompt和common prompt之间

# 注意事项：
# 1. 固定词汇会增加prompt的长度
# 2. 确保prompt长度不超过模型的最大序列长度限制
# 3. 选择的固定词汇应该与你的任务相关

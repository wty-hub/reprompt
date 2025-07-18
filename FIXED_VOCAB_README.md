# 固定词汇Embedding插入功能说明

## 功能概述
这个功能允许在文本prompt中插入几个固定词汇的embedding，这些embedding不会在训练过程中被更新。

## 实现原理

### 1. 初始化阶段
- 在`MultiModalPromptLearner`的`__init__`方法中，接收固定词汇列表`fixed_vocab_list`
- 使用预训练CLIP模型的`token_embedding`层提取这些词汇的embedding
- 通过`register_buffer`将这些embedding注册为模型的缓冲区，确保它们不参与梯度更新

### 2. Forward阶段
- 在生成文本prompt时，将固定的vocabulary embedding插入到可训练的prompt中
- 插入顺序：可训练prompt → 固定vocabulary embedding → common prompt

## 使用方法

### 配置示例
```python
config = {
    'vit': 'ViT-B/32',
    'prompt_length': 12,
    'prompt_depth': 2,
    'fixed_vocab_list': ['photo', 'image', 'picture', 'scene'],  # 添加固定词汇
    # ... 其他配置
}
```

### 代码调用
```python
# 创建模型时会自动使用固定词汇
model = CLIPransformerSS(config)
```

## 技术细节

### 1. Embedding提取
```python
# 从预训练模型提取embedding
fixed_tokens = torch.cat([clip.tokenize(word, context_length=77, truncate=True) for word in self.fixed_vocab_list])
with torch.no_grad():
    fixed_embeddings = clip_model.token_embedding(fixed_tokens).type(dtype)
    # 提取实际词汇embedding（跳过[SOS] token）
    fixed_embeddings = fixed_embeddings[:, 1, :].squeeze(1)
```

### 2. Buffer注册
```python
# 注册为buffer（不参与训练）
self.register_buffer('fixed_text_embeddings', fixed_embeddings)
```

### 3. Prompt组合
```python
# 在forward中组合prompt
prompt_components = [all_prompts_text[0][i]]  # 可训练prompt

if self.num_fixed_tokens > 0:
    # 添加固定词汇embedding
    batch_fixed_embeddings = self.fixed_text_embeddings.unsqueeze(0).expand(1, -1, -1)
    prompt_components.append(batch_fixed_embeddings.squeeze(0))

# 添加common prompt
prompt_components.append(self.common_prompt_projection_text(common_prompt))

all_prompts_text[0][i] = torch.cat(prompt_components, 0)
```

## 优势

1. **固定性**：固定词汇的embedding不会在训练中改变，保持原始的语义信息
2. **灵活性**：可以通过配置文件轻松指定固定词汇
3. **效率**：固定embedding只在初始化时计算一次
4. **兼容性**：与现有的prompt学习框架完全兼容

## 注意事项

1. **序列长度**：固定词汇会增加prompt长度，需要确保不超过模型限制
2. **词汇选择**：选择与任务相关的词汇能获得更好的效果
3. **数量控制**：固定词汇数量不宜过多，以免影响可训练prompt的学习

## 应用场景

- **领域适应**：插入领域相关的固定词汇
- **任务指导**：添加任务特定的关键词
- **知识注入**：引入外部知识的关键概念

这个功能为prompt learning提供了更多的控制能力，允许在保持某些语义信息不变的同时，学习任务特定的prompt表示。

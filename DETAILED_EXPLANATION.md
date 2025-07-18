# Fixed Vocabulary Embedding Detailed Working Principle

## Data Flow Diagram

```
Input: fixed_vocab_list = ["photo", "image", "scene"]

Initialization Phase:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Vocabulary       │ -> │ Pre-trained      │ -> │ Register as         │
│ Tokenization     │    │ Embedding        │    │ Buffer              │
│ "photo" -> 1125  │    │ 1125 -> [512dim] │    │ register_buffer()   │
│ "image" -> 2158  │    │ 2158 -> [512dim] │    │ (Not trainable)     │
│ "scene" -> 3268  │    │ 3268 -> [512dim] │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘

Forward Phase:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Trainable       │    │ Fixed           │    │ Common Prompt   │
│ Prompt          │    │ Vocabulary      │    │ [L2, 512]       │
│ [L1, 512]       │    │ [3, 512]        │    │ (After          │
│ (Learnable)     │    │ (Fixed)         │    │ Projection)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                          ┌─────▼─────┐
                          │ torch.cat │
                          │ (dim=0)   │
                          └─────┬─────┘
                                │
                        ┌───────▼───────┐
                        │ Final Text     │
                        │ Prompt         │
                        │ [L1+3+L2, 512] │
                        └───────────────┘
```

## Key Feature Analysis

### 1. Fixed Property Guarantee
- Uses `register_buffer()` to ensure no gradient updates
- Uses `torch.no_grad()` during initialization to avoid computation graph construction
- Uses the same embedding values in every forward pass

### 2. Semantic Preservation
- Extracted from pre-trained CLIP model, maintaining original semantics
- Skips special tokens ([SOS], [EOS]), only takes actual vocabulary embeddings
- Maintains compatibility with pre-trained models

### 3. Flexible Integration
- Controls fixed vocabulary list through configuration files
- Seamlessly integrates with existing prompt learning frameworks
- Supports dynamic adjustment of fixed vocabulary quantity
```

## Specific Example

Assume configuration: `fixed_vocab_list = ["photo", "image"]`

### State after initialization:
```python
self.fixed_text_embeddings.shape = [2, 512]
# Contains pre-trained embeddings for "photo" and "image"
```

### Prompt composition during forward:
```
Original trainable prompt: [4, 512]  # prompt_length_half = 4
Fixed vocabulary embedding: [2, 512]  # "photo", "image"
Common prompt:             [4, 512]  # Projected common prompt

Final combined prompt:     [10, 512] # 4 + 2 + 4 = 10
```

### Processing in Transformer:
```
Text input: "A cat sitting on a chair" -> embedding [77, 512]
Prompt:     Fixed + trainable combined prompt -> [10, 512]

Combined sequence: [10+77, 512] = [87, 512]
```

## Advantage 

1. **Stable Semantic Anchor**: Fixed vocabulary provides invariant semantic reference
2. **Reduced Parameter Search Space**: Some prompts don't need to be learned
3. **Domain Knowledge Injection**: Can insert domain-specific keywords
4. **Enhanced Interpretability**: Clear knowledge of which vocabularies are used
5. **Training Efficiency**: Reduces the number of parameters to optimize

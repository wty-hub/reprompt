# MCR Retriever Documentation

This document explains how to use the MCR (Multi-modal Cross-modal Retrieval) functionality that has been ported to this project.

## Overview

The MCR retriever provides functionality for:
- Encoding images and texts using CLIP
- Computing similarity-based retrieval between samples
- Text-to-text (T2T) and Image-to-image (I2I) retrieval within datasets
- Saving retrieval results for downstream tasks

## Key Adaptations

The original MCR code has been adapted to work with this project's structure:

1. **CLIP Model**: Uses the project's existing CLIP implementation instead of Hugging Face transformers
2. **Data Format**: Works with PyArrow files (.arrow) instead of pickle files (.pkl)
3. **Configuration**: Integrates with the Sacred configuration system
4. **Device Management**: Proper GPU/CPU device handling
5. **Dataset Structure**: Adapted to work with the project's dataset organization

## Usage

### 1. Basic Usage

```python
from clip.modules.retriever import create_retriever
from clip.config import ex

# Simple configuration
class Config:
    per_gpu_batchsize = 32
    vit = 'ViT-B/16'
    data_root = './datasets'

config = Config()

# Create retriever
retriever = create_retriever(
    config=config,
    dataset_name='food101',  # or 'mmimdb', 'hatememes'
    data_dir='../arrow_datasets/Food101',
    device='cuda'
)

# Run retrieval
results = retriever.run(save_results=True)
```

### 2. Using Sacred Configuration

Run with the integrated script:

```bash
# For Food101 dataset
python run_retrieval.py with task_finetune_food101

# For MMIMDB dataset  
python run_retrieval.py with task_finetune_mmimdb

# For Hateful Memes dataset
python run_retrieval.py with task_finetune_hatememes
```

### 3. Standalone Example

```bash
python run_retrieval_example.py
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `per_gpu_batchsize` | Batch size for processing | 32 |
| `vit` | CLIP model variant | 'ViT-B/16' |
| `data_root` | Root directory for datasets | './datasets' |
| `top_k` | Number of retrieved samples | 20 |

## Dataset Support

The retriever supports three datasets:

### Food101
- **Data directory**: `../arrow_datasets/Food101`
- **Splits**: train, test
- **Text column**: 'text'
- **Arrow files**: `food101_train.arrow`, `food101_test.arrow`

### MMIMDB
- **Data directory**: `../arrow_datasets/mmimdb`
- **Splits**: train, dev, test
- **Text column**: 'plots'
- **Arrow files**: `mmimdb_train.arrow`, `mmimdb_dev.arrow`, `mmimdb_test.arrow`

### Hateful Memes
- **Data directory**: `../arrow_datasets/hateful_memes`
- **Splits**: train, dev, test
- **Text column**: 'text'
- **Arrow files**: `hatememes_train.arrow`, `hatememes_dev.arrow`, `hatememes_test.arrow`

## Output Format

The retriever adds the following columns to each dataset split:

- `image_features`: CLIP image embeddings
- `text_features`: CLIP text embeddings
- `t2t_id_list`: List of retrieved sample IDs for text-to-text retrieval
- `t2t_sims_list`: List of similarity scores for T2T retrieval
- `t2t_label_list`: List of labels for retrieved T2T samples
- `i2i_id_list`: List of retrieved sample IDs for image-to-image retrieval
- `i2i_sims_list`: List of similarity scores for I2I retrieval
- `i2i_label_list`: List of labels for retrieved I2I samples

## Output Files

Results are saved as pickle files:
```
{output_dir}/
├── train_with_retrieval.pkl
├── dev_with_retrieval.pkl (if applicable)
└── test_with_retrieval.pkl
```

## Memory and Performance

- **Batch Processing**: Images and texts are processed in batches to manage memory
- **GPU Support**: Automatically uses CUDA if available
- **Progress Tracking**: Uses tqdm for progress bars
- **Memory Bank**: Training/validation data serves as retrieval memory bank

## Integration with Existing Code

The MCR retriever is designed to complement the existing missing-modality aware prompt learning:

1. **Pre-processing**: Run MCR to generate retrieval results
2. **Training**: Use retrieval results as additional context for prompt learning
3. **Evaluation**: Leverage retrieved samples for improved missing modality handling

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure arrow files exist in the specified data directory
2. **Memory Issues**: Reduce batch size if encountering OOM errors
3. **Device Errors**: Check CUDA availability and device compatibility

### Memory Optimization

For large datasets:
```python
# Reduce batch size
config.per_gpu_batchsize = 16

# Use CPU if GPU memory is insufficient
device = 'cpu'
```

## Example Output

```
Using device: cuda
Loaded train split with 75750 samples
Loaded test split with 25250 samples
Generating retrieval vectors...
Processing train split...
Encoding images...
Encoding images (train): 100%|████████████| 2368/2368 [02:15<00:00, 17.42it/s]
Encoding texts...
Encoding texts (train): 100%|████████████| 2368/2368 [00:45<00:00, 52.17it/s]
...
Within-dataset retrieval completed!
Saved train results to ./retrieval_results/food101/train_with_retrieval.pkl
Saved test results to ./retrieval_results/food101/test_with_retrieval.pkl

=== Summary ===
train: 75750 samples
  - Avg T2T retrieved: 20.0
  - Avg I2I retrieved: 20.0
test: 25250 samples  
  - Avg T2T retrieved: 20.0
  - Avg I2I retrieved: 20.0
```

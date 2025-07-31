#!/usr/bin/env python3
"""
Standalone retrieval runner that avoids importing problematic modules.
"""
import os
import torch

from clip.modules.retriever import FeatureExtractor, MultiChannelRetriever, FeatureBank

def main():
    """Main function to run retrieval without importing problematic modules."""
    # Check if datasets exist
    datasets_info = {
        'mmimdb': '../arrow_datasets/mmimdb',
        'food101': '../arrow_datasets/Food101',
        'hatememes': '../arrow_datasets/hateful_memes'
    }
    
    ratios = [0.5, 0.7, 0.9]
    scenarios = [
        'both',
        'image',
        'text'
    ]
    splits = ['train', 'val', 'test']

    print("Available datasets:")
    for name, path in datasets_info.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (not found)")
    
    for dataset_name in datasets_info.keys():
        # for scenario in scenarios:
            # for ratio in ratios:
            if dataset_name not in datasets_info:
                print(f"Unknown dataset: {dataset_name}")
                return
            
            data_dir = datasets_info[dataset_name]
            if not os.path.exists(data_dir):
                print(f"Dataset directory not found: {data_dir}")
                return
            
            bnk = FeatureBank(dataset_name, data_dir, 'both', 0.7, remake=True)
            for split in splits:
                text_tokens = bnk.get_text_tokens(torch.randint(0, len(bnk.get_tokens(split, 'text')), (100,)), split)
                print(f'text_tokens.shape: {text_tokens.shape}')
                image_tokens = bnk.get_image_tokens(torch.randint(0, len(bnk.get_tokens(split, 'image')), (100,)), split)
                print(f'image_tokens.shape: {image_tokens.shape}')

if __name__ == '__main__':
    main()

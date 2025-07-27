#!/usr/bin/env python3
"""
Standalone retrieval runner that avoids importing problematic modules.
"""
import os
import torch

from clip.modules.retriever import FeatureExtractor

def main():
    """Main function to run retrieval without importing problematic modules."""
    # Check if datasets exist
    datasets_info = {
        'mmimdb': '../arrow_datasets/mmimdb',
        'food101': '../arrow_datasets/Food101',
        'hatememes': '../arrow_datasets/hateful_memes'
    }
    
    print("Available datasets:")
    for name, path in datasets_info.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (not found)")
    
    for dataset_name in datasets_info.keys():
        if dataset_name not in datasets_info:
            print(f"Unknown dataset: {dataset_name}")
            return
        
        data_dir = datasets_info[dataset_name]
        if not os.path.exists(data_dir):
            print(f"Dataset directory not found: {data_dir}")
            return
        
        try:
            # Create and run retriever
            print(f"Creating MCR retriever for {dataset_name}...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            print(f'load dataset: {dataset_name}')
            featureExtractor = FeatureExtractor(dataset_name, data_dir)
            print(f'feature extracting done')
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()

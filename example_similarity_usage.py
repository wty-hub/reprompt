"""
Example usage of MultiChannelRetriever with similarity saving functionality.

This example demonstrates how to:
1. Initialize the retriever with similarity saving
2. Access saved similarity scores
3. Get retrieval results with both indices and similarity values
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clip.modules.retriever import MultiChannelRetriever

def main():
    # Example configuration
    dataset_name = "food101"  # or "mmimdb", "hatememes"
    data_dir = "./datasets/Food101"  # Update this path to your data directory
    top_k = 20
    
    # Initialize retriever with similarity saving
    print("Initializing MultiChannelRetriever...")
    retriever = MultiChannelRetriever(
        dataset_name=dataset_name,
        data_dir=data_dir,
        top_k=top_k,
        remake=False  # Set to True to recompute everything
    )
    
    # Example: Get image similarities for a specific query
    split = "test"
    query_idx = 0
    
    print(f"\nExample 1: Get image similarities for {split} split, query {query_idx}")
    image_similarities = retriever.get_image_similarities(split, query_idx)
    print(f"Image similarities: {image_similarities[:5]}...")  # Show first 5
    
    print(f"\nExample 2: Get text similarities for {split} split, query {query_idx}")
    text_similarities = retriever.get_text_similarities(split, query_idx)
    print(f"Text similarities: {text_similarities[:5]}...")  # Show first 5
    
    print(f"\nExample 3: Get retrieval results with similarities")
    # Get image retrieval results with similarities
    img_indices, img_similarities = retriever.get_retrieval_results_with_similarities(
        split, query_idx, modality='image'
    )
    print(f"Top-5 image retrieval results:")
    for i in range(min(5, len(img_indices))):
        print(f"  Rank {i+1}: Index {img_indices[i]}, Similarity {img_similarities[i]:.4f}")
    
    # Get text retrieval results with similarities
    txt_indices, txt_similarities = retriever.get_retrieval_results_with_similarities(
        split, query_idx, modality='text'
    )
    print(f"Top-5 text retrieval results:")
    for i in range(min(5, len(txt_indices))):
        print(f"  Rank {i+1}: Index {txt_indices[i]}, Similarity {txt_similarities[i]:.4f}")
    
    print(f"\nExample 4: Get all similarities for a split")
    all_image_sims = retriever.get_image_similarities(split)
    print(f"Number of queries in {split} split: {len(all_image_sims)}")
    
    print(f"\nExample 5: Get all similarities across all splits")
    all_splits_sims = retriever.get_image_similarities()
    print(f"Available splits: {list(all_splits_sims.keys())}")

if __name__ == "__main__":
    main()

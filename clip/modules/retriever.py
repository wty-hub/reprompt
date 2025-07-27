"""
Multi-modal Cross-modal Retrieval (MCR) module.

This module provides efficient CLIP-based feature extraction and retrieval capabilities.

Key Features:
- Automatic CLIP feature extraction from images and texts
- Persistent feature storage/loading for faster subsequent runs
- Memory-efficient processing for large datasets
- Batch processing for optimal GPU utilization
- Support for multiple datasets (MMIMDB, Food101, HateMemes)
"""
import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import warnings

from clip.modules.clip_missing_aware_prompt_module import TextEncoder, load_clip_to_cpu
from clip.modules.vision_transformer_prompts import VisionTransformer
from clip.gadgets.cache import HashedLinkedList

from .clip import tokenize, _transform

# Suppress PIL DecompressionBomb warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class FeatureExtractor:
    def __init__(self, dataset_name, data_dir, remake=False):
        self.dataset_name = dataset_name.lower()
        print(f'dataset: {self.dataset_name}')
        # self.batch_size = getattr(config, 'per_gpu_batchsize', 32)
        self.batch_size = 64

        self.top_k = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set data directory
        self.data_dir = data_dir
        self.feature_dir = os.path.join(data_dir, 'features')
        self.model = load_clip_to_cpu('ViT-B/16', 0, 0).to(self.device)
        self.text_encoder = TextEncoder(self.model)
        self.preprocess = _transform(224)  # ViT-B/16 input size is 224
        self.remake = remake
        self.extract_features()
        
    def extract_features(self):
        """Load dataset splits and either extract CLIP features or load pre-extracted features."""
        self.image_features = {}
        self.text_features = {}
        
        # Define split names based on dataset
        split_names = ['train', 'val', 'test']
        text_column_name = ""
        if self.dataset_name == 'mmimdb':
            arrow_names = ['mmimdb_train', 'mmimdb_dev', 'mmimdb_test']
            text_column_name = "plots"
        elif self.dataset_name == 'food101':
            arrow_names = ['food101_train', 'food101_val', 'food101_test']
            text_column_name = "text"
        elif self.dataset_name in ['hatememes', 'hatefull_memes']:
            arrow_names = ['hatememes_train', 'hatememes_dev', 'hatememes_test']
            text_column_name = "text"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Load each split
        for split, arrow_name in zip(split_names, arrow_names):
            arrow_path = os.path.join(self.data_dir, f"{arrow_name}.arrow")
            if not os.path.exists(arrow_path):
                print(f"Warning: {arrow_path} not found")
                continue
                
            # Load basic data
            table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(arrow_path, "r")
            ).read_all().to_pandas()
            
            # print(f"Processing {split} split with {len(table)} samples...")
            
            # Check if pre-extracted features exist
            feature_path = os.path.join(self.feature_dir, f"{split}_features.pt")
            
            if os.path.exists(feature_path) and not self.remake:
                # Load pre-extracted features
                print(f"Loading pre-extracted features from {feature_path}")
                features_data = torch.load(feature_path, map_location='cpu')
                
                self.image_features[split] = features_data['image_features']
                self.text_features[split] = features_data['text_features']
                
                print(f"Loaded image features shape: {self.image_features[split].shape}")
                print(f"Loaded text features shape: {self.text_features[split].shape}")
            else:

                # Extract text features 
                print("Extracting text features...")
                text_features_list = []
                all_texts = table[text_column_name]
                
                # Process texts
                raw_texts = []
                for text in all_texts:
                    if isinstance(text, list):
                        raw_texts.append(' '.join(text))  # Join multiple texts
                    else:
                        raw_texts.append(str(text))
                
                for i in tqdm(range(0, len(raw_texts), self.batch_size), desc=f"Processing texts ({split})"):
                    batch_end = min(i + self.batch_size, len(raw_texts))
                    batch_texts = raw_texts[i:batch_end]
                    
                    # Tokenize and encode batch
                    batch_features = self._encode_text(batch_texts)
                    text_features_list.extend(batch_features.cpu().tolist())
                
                self.text_features[split] = torch.tensor(text_features_list)
                print(f"Text features shape: {self.text_features[split].shape}")


                # extract image features 
                print("Extracting image features...")
                image_features_list = []
                
                for i in tqdm(range(0, len(table), self.batch_size), desc=f"Processing images ({split})"):
                    batch_end = min(i + self.batch_size, len(table))
                    batch_images = []
                    
                    # Process batch of images
                    for idx in range(i, batch_end):
                        img_bytes = table['image'].iloc[idx]
                        # Convert bytes to PIL Image
                        image_io = io.BytesIO(img_bytes)
                        image_io.seek(0)
                        pil_image = Image.open(image_io).convert("RGB")
                        # Apply CLIP preprocessing
                        tensor_image = self.preprocess(pil_image)
                        batch_images.append(tensor_image)
                    
                    # Stack and encode batch
                    batch_tensor = torch.stack(batch_images)
                    batch_features = self._encode_image(batch_tensor)
                    image_features_list.extend(batch_features.cpu().tolist())
                
                self.image_features[split] = torch.tensor(image_features_list)
                print(f"Image features shape: {self.image_features[split].shape}")
                
                
                # Save extracted features
                self.save_features(split)
                
                print(f"Completed feature extraction for {split} split")
                
    def save_features(self, split):
        """Save extracted features to disk."""
        os.makedirs(self.feature_dir, exist_ok=True)
        feature_path = os.path.join(self.feature_dir, f"{split}_features.pt")
        
        features_data = {
            'image_features': self.image_features[split],
            'text_features': self.text_features[split]
        }
        
        torch.save(features_data, feature_path)
        print(f"Saved features to {feature_path}")

                
    def _encode_text(self, texts):
        """Encode text tokens using CLIP text encoder."""
        assert isinstance(texts, list)
        assert isinstance(texts[0], str)

        # text_tokens = tokenize(texts, truncate=True).to(self.device)
        text_tokens = torch.stack([tokenize(tx, context_length=77, truncate=True) for tx in texts]).to(self.device).squeeze(1)
        missing_type = torch.zeros(len(texts))
        # Get text features
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens, [], missing_type)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        return text_features
    
    def _encode_image(self, image_tensors):
        """Encode preprocessed image tensors using CLIP image encoder."""
        # Move to device and get image features
        image_batch = image_tensors.to(self.device)
        
        with torch.no_grad():
            # Create missing_type tensor with proper type and device
            missing_type = torch.zeros(image_batch.size(0), dtype=self.model.dtype, device=self.device)
            
            # For the prompt-aware CLIP model, we need to pass empty prompts
            all_prompts_image = []  # Empty list for prompts
            
            # Call visual encoder with all required parameters
            image_features = self.model.visual(
                image_batch.type(self.model.dtype), 
                all_prompts_image, 
                missing_type
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
        
    def get_image_features(self):
        return self.image_features
    
    def get_text_features(self):
        return self.text_features


class MultiChannelRetriever:
    def __init__(self, dataset_name, data_dir, top_k=20, remake=False):
        self.featureExtractor = FeatureExtractor(dataset_name, data_dir, remake)
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_features = self.featureExtractor.get_text_features()['train']
        self.image_features = self.featureExtractor.get_image_features()['train']
        self.batch_size = 64
        self.top_k = top_k
        self.retrieval_indices_path = os.path.join(self.data_dir, 'retrieval_indices.pt')
        if os.path.exists(self.retrieval_indices_path) and not remake:
            print(f"Loading pre-computed retrieval indices from {self.retrieval_indices_path}")
            retrieval_data = torch.load(self.retrieval_indices_path, map_location='cpu')
            self.image_list = retrieval_data['image_list']
            self.text_list = retrieval_data['text_list']
            # Load similarity scores if available
            if 'image_similarities' in retrieval_data:
                self.image_similarities = retrieval_data['image_similarities']
                self.text_similarities = retrieval_data['text_similarities']
            else:
                self.image_similarities = {}
                self.text_similarities = {}
            print("Loaded existing retrieval indices")
        else:
            self.create_id_list()
        
    def compute_similarity(self, querys, values):
        """
        get similarity value between querys(vectors) and values(vectors)
        return shape: (len(querys), len(values))
        """
        querys = querys.to(self.device)
        values = values.to(self.device)
        
        similarities = []
        
        for i in range(0, len(querys), self.batch_size):
            batch_end = min(i + self.batch_size, len(querys))
            batch_queries = querys[i:batch_end]
            
            batch_similarity = F.cosine_similarity(
                batch_queries.unsqueeze(1),  # (batch_size, 1, feature_dim)
                values.unsqueeze(0),         # (1, num_values, feature_dim)
                dim=2
            )
            
            similarities.append(batch_similarity.cpu())
        
        return torch.cat(similarities, dim=0)
    
    def create_id_list(self):
        self.image_list = {}
        self.text_list = {}
        self.image_similarities = {}
        self.text_similarities = {}
        split_names = ['train', 'val', 'test']
        train_image_features = self.featureExtractor.image_features['train']
        train_text_features = self.featureExtractor.text_features['train']
        for split in split_names:
            print(f'create id_list for {split}')
            self.image_list[split] = {}
            self.text_list[split] = {}
            self.image_similarities[split] = {}
            self.text_similarities[split] = {}
            count = len(self.featureExtractor.image_features[split])
            for i in tqdm(range(0, count, self.batch_size)):
                batch_end = min(i + self.batch_size, count)
                batch_image_querys = self.featureExtractor.image_features[split][i:batch_end]
                image_similarities = self.compute_similarity(batch_image_querys, train_image_features)
                batch_text_querys = self.featureExtractor.text_features[split][i:batch_end]
                text_similarities = self.compute_similarity(batch_text_querys, train_text_features)
                
                # For each query in the batch
                for j, (img_sim, txt_sim) in enumerate(zip(image_similarities, text_similarities)):
                    query_idx = i + j
                    
                    # For image similarities - exclude self if from train split
                    if split == 'train':
                        img_sim[query_idx] = -float('inf')  # Exclude self
                    
                    # Get top_k indices and similarities for image
                    top_img_similarities, top_img_indices = torch.topk(img_sim, self.top_k)
                    self.image_list[split][query_idx] = top_img_indices.tolist()
                    self.image_similarities[split][query_idx] = top_img_similarities.tolist()
                    
                    # For text similarities - exclude self if from train split  
                    if split == 'train':
                        txt_sim[query_idx] = -float('inf')  # Exclude self
                    
                    # Get top_k indices and similarities for text
                    top_txt_similarities, top_txt_indices = torch.topk(txt_sim, self.top_k)
                    self.text_list[split][query_idx] = top_txt_indices.tolist()
                    self.text_similarities[split][query_idx] = top_txt_similarities.tolist()
        # Save image_list, text_list, and similarity scores to dataset directory
        retrieval_data = {
            'image_list': self.image_list,
            'text_list': self.text_list,
            'image_similarities': self.image_similarities,
            'text_similarities': self.text_similarities
        }
        torch.save(retrieval_data, self.retrieval_indices_path)
        print(f"Saved retrieval indices and similarities to {self.retrieval_indices_path}")
    
    def get_image_similarities(self, split, query_idx):
        return self.image_similarities.get(split, {}).get(query_idx, [])
    
    def get_text_similarities(self, split, query_idx):
        return self.text_similarities.get(split, {}).get(query_idx, [])
    
    def get_retrieval_results_with_similarities(self, split, query_idx, modality='image'):
        if modality == 'image':
            indices = self.image_list.get(split, {}).get(query_idx, [])
            similarities = self.image_similarities.get(split, {}).get(query_idx, [])
        elif modality == 'text':
            indices = self.text_list.get(split, {}).get(query_idx, [])
            similarities = self.text_similarities.get(split, {}).get(query_idx, [])
        else:
            raise ValueError("modality must be 'image' or 'text'")
        
        return indices, similarities

class RetrieverBank:
    
    def __init__(self, dataset_name, data_dir, cache_size=20000, remake=False):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.remake = remake
        self.num_samples = 0
        self.arrow_path = ""
        self.cache_path = os.path.join(data_dir, 'bank')
        os.makedirs(self.cache_path, exist_ok=True)
        self.cache_size = cache_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load CLIP model and text encoder
        self.model = load_clip_to_cpu('ViT-B/16', 0, 0).to(self.device)
        self.text_encoder = TextEncoder(self.model)
        self.vision_transformer: VisionTransformer = self.model.visual
        self.conv1 = self.vision_transformer.conv1
        self.class_embedding = self.vision_transformer.class_embedding
        self.positional_embedding = self.vision_transformer.positional_embedding
        self.ln_pre = self.vision_transformer.ln_pre
        self.batch_size = 64
        # self.block_size = cache_size // self.batch_size
        self.cache_size = cache_size
        self.text_tokens = []
        self.image_cache = HashedLinkedList(self.cache_size)
        self.prepare_cache()

    def __len__(self):
        return self.num_samples

    def get_image_token(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
        return x
    
    def prepare_cache(self):
        # Try to load cached metadata first
        if os.path.exists(self.cache_path) and not self.remake:
            print('cache file already exists')
            self.load_text_tokens()
            return
        print('creating caches:')
        # Determine arrow file name and text column based on dataset
        if self.dataset_name == 'mmimdb':
            arrow_name = 'mmimdb_train'
            self.text_column = 'plots'
        elif self.dataset_name == 'food101':
            arrow_name = 'food101_train'
            self.text_column = 'text'
        elif self.dataset_name in ['hatememes', 'hateful_memes']:
            arrow_name = 'hatememes_train'
            self.text_column = 'text'
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Set arrow file path
        self.arrow_path = os.path.join(self.data_dir, f"{arrow_name}.arrow")
        if not os.path.exists(self.arrow_path):
            raise FileNotFoundError(f"data file not found: {self.arrow_path}")
        
        table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(self.arrow_path, "r")
        ).read_all().to_pandas()
        
        self.num_samples = len(table)

        all_texts = table[self.text_column].to_list()
        raw_texts = []
        for text in all_texts:
            if isinstance(text, list):
                raw_texts.append(' '.join(text))  # Join multiple texts
            else:
                raw_texts.append(str(text))
        self.text_tokens = torch.stack([tokenize(tx, context_length=77, truncate=True) for tx in raw_texts]).squeeze(1)
        self.save_text_tokens(self.text_tokens)
        print(f'save text tokens')

        all_images = table['image']

        preprocess = _transform(224)

        print("Preparing image prompts...")
        for i in tqdm(range(0, len(all_images), self.batch_size), desc="Processing images in batches"):
            batch_end = min(i + self.batch_size, len(all_images))
            batch_imgs = []
            for img_bytes in all_images[i:batch_end]:
                image_io = io.BytesIO(img_bytes)
                pil_image = Image.open(image_io).convert("RGB")
                tensor_image = preprocess(pil_image)
                batch_imgs.append(tensor_image)
            batch_tensor = torch.stack(batch_imgs)
            batch_tensor = batch_tensor.to(self.device)
            with torch.no_grad():
                batch_tensor = self.get_image_token(batch_tensor)
            self.save_image_tokens(batch_tensor, i // self.batch_size)
        print('preparing done')

    def save_text_tokens(self, text_tokens):
        filename = os.path.join(self.cache_path, 'text_tokens.pt')
        torch.save(text_tokens, filename)

    def save_image_tokens(self, image_tokens, block_idx):
        filename = os.path.join(self.cache_path, f'image_tokens_{block_idx}.pt')
        torch.save(image_tokens, filename)
        # print(f'save image tokens to {filename}')

    def load_text_tokens(self):
        filename = os.path.join(self.cache_path, 'text_tokens.pt')
        self.text_tokens = torch.load(filename)

    def load_image_tokens(self, block_idx):
        filename = os.path.join(self.cache_path, f'image_tokens_{block_idx}.pt')
        image_tokens = torch.load(filename)
        return image_tokens

    def load_image_cache(self, idx):
        block_idx = idx // self.batch_size
        # block_offst = idx % self.batch_size
        block = self.load_image_tokens(block_idx)
        for i in range(block_idx * self.batch_size, block_idx * self.batch_size + len(block)):
            self.image_cache.put(i, block[i - block_idx * self.batch_size])

    def get_text_tokens(self, id) -> torch.tensor:
        return self.text_tokens[id]

    def get_image_tokens(self, id) -> torch.tensor:
        if self.image_cache.get(id) is None:
            self.load_image_cache()
        return self.image_cache.get(id)
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
import math
import os
import io
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import warnings

from clip.modules.utils import TextEncoder, load_clip_to_cpu
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
        self.batch_size = 64
        self.top_k = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set data directory
        self.data_dir = data_dir
        self.feature_dir = os.path.join(data_dir, 'features')
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
            
            # Check if pre-extracted features exist
            feature_path = os.path.join(self.feature_dir, f"{split}_features.pt")
            
            if os.path.exists(feature_path) and not self.remake:
                # Load pre-extracted features
                print(f"Loading pre-extracted features from {feature_path}")
                features_data = torch.load(feature_path, map_location='cpu')
                
                self.image_features[split] = features_data['image_features']
                self.text_features[split] = features_data['text_features']
                
                # print(f"Loaded image features shape: {self.image_features[split].shape}")
                # print(f"Loaded text features shape: {self.text_features[split].shape}")
            else:
                self.model = load_clip_to_cpu('ViT-B/16', 0, 0).to(self.device)
                print("FeatureExtractor: Loading CLIP model")
                self.text_encoder = TextEncoder(self.model)
                self.preprocess = _transform(224)  # ViT-B/16 input size is 224
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
                del self.model
                
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
            self.image_similarities = retrieval_data['image_similarities']
            self.text_similarities = retrieval_data['text_similarities']
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
    
    # def get_image_similarities(self, split, query_idx):
    #     return self.image_similarities.get(split, {}).get(query_idx, [])
    
    # def get_text_similarities(self, split, query_idx):
    #     return self.text_similarities.get(split, {}).get(query_idx, [])
    
    def get_retrieval_results_with_similarities(self, split, query_idx: int, modality):
        if modality == 'image':
            indices = self.image_list[split][query_idx]
            # indices = [self.image_list.get(split).get(int(i)) for i in query_idx]
            similarities = self.image_similarities[split][query_idx]
            # similarities = [self.image_similarities.get(split).get(int(i)) for i in query_idx]
        else:
            indices = self.text_list[split][query_idx]
            # indices = [self.text_list.get(split).get(int(i)) for i in query_idx]
            similarities = self.text_similarities[split][query_idx]
            # similarities = [self.text_similarities.get(split).get(int(i)) for i in query_idx]
        return indices, similarities

    # def get_retrieval_index(self, split, query_idx: int, modality):
    #     if modality == 'image':
    #         indices = self.image_list.get(split).get(query_idx)
    #     else:
    #         indices = self.text_list.get(split).get(query_idx)
    #     return indices

    # def get_similarities(self, split, query_idx: int, modality):
    #     if modality == 'image':
    #         similarities = self.image_similarities.get(split).get(query_idx)
    #     else:
    #         similarities = self.text_similarities.get(split).get(query_idx)
    #     return similarities
    

class FeatureBank:
    def __init__(self, dataset_name, data_dir, missing_scenario, missing_ratio, remake=False):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.remake = remake
        self.cache_path = os.path.join(data_dir, 'bank')
        os.makedirs(self.cache_path, exist_ok=True)
        self.cache_file = os.path.join(self.cache_path, f'{dataset_name}_features.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.missing_screnario = missing_scenario
        self.missing_ratio = missing_ratio
        self.missing_table_root = './datasets/missing_tables/'
        self.missing_tables = {}
        missing_ratio_str = f"{int(missing_ratio * 10):02d}"

        self.missing_tables['train'] = torch.load(os.path.join(self.missing_table_root,
                                                                f'{dataset_name}_train_missing_{missing_scenario}_07.pt'))
        if self.dataset_name == 'mmimdb':
            self.missing_tables['val'] = torch.load(
                os.path.join(self.missing_table_root, f'mmimdb_dev_missing_{missing_scenario}_{missing_ratio_str}.pt'))
            self.missing_tables['test'] = torch.load(
                os.path.join(self.missing_table_root, f'mmimdb_test_missing_{missing_scenario}_{missing_ratio_str}.pt'))
        elif self.dataset_name == 'food101':
            self.missing_tables['val'] = torch.load(
                os.path.join(self.missing_table_root, f'food101_val_missing_{missing_scenario}_{missing_ratio_str}.pt'))
            self.missing_tables['test'] = torch.load(
                os.path.join(self.missing_table_root, f'food101_test_missing_{missing_scenario}_{missing_ratio_str}.pt'))
        elif self.dataset_name in ['hatememes', 'hateful_memes']:
            self.missing_tables['val'] = torch.load(
                os.path.join(self.missing_table_root, f'hatememes_dev_missing_{missing_scenario}_{missing_ratio_str}.pt'))
            self.missing_tables['test'] = torch.load(
                os.path.join(self.missing_table_root, f'hatememes_test_missing_{missing_scenario}_{missing_ratio_str}.pt'))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.prepare_cache()


    def make_image_token(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.model.visual(x, [], torch.zeros((x.shape[0],)))
        return x
    
    def prepare_cache(self):
        # Determine arrow file name and text column based on dataset
        self.arrow_name = {}
        if self.dataset_name == 'mmimdb':
            self.arrow_name["train"] = 'mmimdb_train.arrow'
            self.arrow_name["val"] = 'mmimdb_dev.arrow'
            self.arrow_name["test"] = 'mmimdb_test.arrow'
            self.text_column = 'plots'
        elif self.dataset_name == 'food101':
            self.arrow_name["train"] = 'food101_train.arrow'
            self.arrow_name["val"] = 'food101_val.arrow'
            self.arrow_name["test"] = 'food101_test.arrow'
            self.text_column = 'text'
        elif self.dataset_name in ['hatememes', 'hateful_memes']:
            self.arrow_name["train"] = 'hatememes_train.arrow'
            self.arrow_name["val"] = 'hatememes_dev.arrow'
            self.arrow_name["test"] = 'hatememes_test.arrow'
            self.text_column = 'text'
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Try to load cached first
        if os.path.exists(self.cache_file) and not self.remake:
            print('cache file already exists, loading.')
            self.tokens = torch.load(self.cache_file)
            return
        
        # Load CLIP model and text encoder
        self.model = load_clip_to_cpu('ViT-B/16', 0, 0).to(self.device)
        print("RetrieverBank: Loading new CLIP model")
        self.text_encoder = TextEncoder(self.model)
        self.vision_transformer: VisionTransformer = self.model.visual
        self.conv1 = self.vision_transformer.conv1
        self.class_embedding = self.vision_transformer.class_embedding
        self.positional_embedding = self.vision_transformer.positional_embedding
        self.ln_pre = self.vision_transformer.ln_pre
        print('creating caches:')

        self.tokens = {}
        
        for split in ['train', 'val', 'test']:
            self.tokens[split] = {}
            table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(os.path.join(self.data_dir, self.arrow_name[split]), "r")
            ).read_all().to_pandas()
            
            all_texts = table[self.text_column].to_list()
            raw_texts = []
            for text in all_texts:
                if isinstance(text, list):
                    raw_texts.append(' '.join(text))  # Join multiple texts
                else:
                    raw_texts.append(str(text))
            
            print("Preparing text tokens...")
            text_tokens = []
            for i in tqdm(range(0, len(raw_texts), self.batch_size), desc="Processing texts in batches"):
                batch_end = min(i + self.batch_size, len(raw_texts))
                batch_texts = raw_texts[i:batch_end]
                batch_tokens = torch.stack([tokenize(tx, context_length=77, truncate=True) for tx in batch_texts]).squeeze(1)
                with torch.no_grad():
                    batch_tokens = self.make_text_tokens(batch_tokens)
                text_tokens.append(batch_tokens.cpu())
            
            # Concatenate all text tokens into one tensor
            text_tokens = torch.cat(text_tokens, dim=0)
            # self.save_text_tokens(self.text_tokens)
            self.tokens[split]['text'] = text_tokens
            all_images = table['image']
            preprocess = _transform(224)
            print("Preparing image prompts...")
            image_tokens = []
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
                    batch_tensor = self.make_image_token(batch_tensor)
                image_tokens.append(batch_tensor.cpu())
            # Concatenate all image tokens into one tensor
            image_tokens = torch.cat(image_tokens, dim=0)
            self.tokens[split]['image'] = image_tokens
        # Save all image tokens at once
        torch.save(self.tokens, self.cache_file)
        print('preparing done')
        del self.text_encoder
        del self.model

    def get_tokens(self, split, modality) -> Tensor:
        return self.tokens[split][modality]

    # def save_text_tokens(self, text_tokens):
    #     filename = os.path.join(self.cache_path, 'text_tokens.pt')
    #     torch.save(text_tokens, filename)
    #     print(f'save text tokens to {filename}')

    # def save_image_tokens(self, image_tokens):
    #     filename = os.path.join(self.cache_path, f'image_tokens.pt')
    #     torch.save(image_tokens, filename)
    #     print(f'save image tokens to {filename}')

    # def load_text_tokens(self):
    #     for split in ['train', 'val', 'test']:
            
    # def load_image_tokens(self):
    #     filename = os.path.join(self.cache_path, f'image_tokens.pt')
    #     self.image_tokens = torch.load(filename)

    def get_text_tokens(self, id, split) -> torch.tensor:
        """
        Get text tokens for one ID or a list of IDs
        
        Args:
            id: int or list of ints - the ID(s) to retrieve tokens for
            
        Returns:
            torch.tensor: text tokens for the given ID(s)
        """
        if isinstance(id, (list, tuple)) or isinstance(id, torch.Tensor) and id.dim() > 0:
            return torch.stack([self.tokens[split]['text'][i] for i in id])
        else:
            return self.tokens[split]['text'][id]

    def get_image_tokens(self, id, split) -> torch.tensor:
        """
        Get image tokens for one ID or a list of IDs
        
        Args:
            id: int or list of ints - the ID(s) to retrieve tokens for
            
        Returns:
            torch.tensor: image tokens for the given ID(s)
        """
        if isinstance(id, (list, tuple)) or isinstance(id, torch.Tensor) and id.dim() > 0:
            return torch.stack([self.tokens[split]['image'][i] for i in id])
        else:
            return self.tokens[split]['image'][id]

    def get_missing_type(self, id, split):
        """
        Get missing type for one ID or a list of IDs
        
        Args:
            id: int or list of ints - the ID(s) to check
            split: 'train', 'val' or 'test'

        Returns:
            int or list of ints: missing type(s) for the given ID(s)
        """
        if isinstance(id, (list, tuple)) or isinstance(id, torch.Tensor) and id.dim() > 0:
            return [int(self.missing_tables[split][i]) for i in id]
        else:
            return int(self.missing_tables[split][id])
        
    def make_text_tokens(self, text_tokens) -> torch.tensor:
        """
        Batch convert multiple text_tokens into CLIP text feature vector representations
        using the full text encoder (not just token embeddings)
        
        Returns:
            torch.tensor: Batch text feature vectors, shape=[batch_size, 512]
        """
        # Ensure the model is loaded
        
        # Use the full TextEncoder to encode text
        with torch.no_grad():
            missing_type = torch.zeros(text_tokens.size(0))
            text_features = self.text_encoder(text_tokens.to(self.device), [], missing_type)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    

COMPLETE = 0
MISSING_TEXT = 1
MISSING_IMAGE = 2
TEXT_DIM = 512
IMAGE_DIM = 768
TOKEN_DIM = 512

class RetrievalPromptLearner(nn.Module):

    def __init__(self, dataset_name, data_dir, prompt_length, missing_scenario, missing_ratio, remake=False, 
                 dropout_rate=0.3, use_layer_norm=True, temperature=1.0, top_k_candidates=5):
        super().__init__()
        self.retriever = MultiChannelRetriever(dataset_name, data_dir, remake=remake)
        # Convert missing_scenario to proper format if it's a string
        self.missing_scenario = missing_scenario
        self.bank = FeatureBank(dataset_name, data_dir, missing_scenario, missing_ratio, remake=False)
        self.prompt_length = prompt_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Regularization parameters
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.temperature = temperature  # For softmax sampling
        self.top_k_candidates = top_k_candidates  # Number of candidates to sample from
        
        # Text prompt generator with regularization
        text_layers = [
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.LayerNorm(TEXT_DIM) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.LayerNorm(TEXT_DIM) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(TEXT_DIM, prompt_length * TEXT_DIM)
        ]
        self.text_prompt_generator = nn.Sequential(*[layer for layer in text_layers if layer is not None])

        # Image prompt generator with regularization
        image_layers = [
            nn.Linear(TEXT_DIM, TEXT_DIM), # image_token from bank is actually image feature, which is 512-dim
            nn.LayerNorm(TEXT_DIM) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.LayerNorm(TEXT_DIM) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(TEXT_DIM, prompt_length * IMAGE_DIM)
        ]
        self.image_prompt_generator = nn.Sequential(*[layer for layer in image_layers if layer is not None])

    def get_available_tokens(self, idx, split, modality):
        with torch.no_grad():
            if modality == 'image':
                return self.bank.get_image_tokens(idx, split)
            elif modality == 'text':
                return self.bank.get_text_tokens(idx, split)
            else:
                raise ValueError(f'invalid modality: {modality}')


    def get_retrieved_sampled_token(self, idx, split, with_modality):
        """
        Get available train data id with modality using probabilistic sampling
        from top-k candidates to reduce overfitting
        """
        with torch.no_grad():
            most_close_idx, most_close_similarities = self.retriever.get_retrieval_results_with_similarities(
                'train', int(idx), with_modality)
            missing_modality = MISSING_TEXT if with_modality == 'text' else MISSING_IMAGE
            
            # Collect valid candidates
            valid_candidates = []
            valid_similarities = []
            
            for ret_idx, sim in zip(most_close_idx, most_close_similarities):
                if self.bank.get_missing_type(ret_idx, 'train') != missing_modality:
                    valid_candidates.append(ret_idx)
                    valid_similarities.append(sim)
            
            if len(valid_candidates) == 0:
                return None
            
            # Sample from top-k candidates during training
            if self.training and len(valid_candidates) > 1:
                # Use only top-k candidates
                k = min(self.top_k_candidates, len(valid_candidates))
                top_indices = torch.topk(torch.tensor(valid_similarities), k=k).indices
                
                # Convert similarities to probabilities with temperature scaling
                top_similarities = torch.tensor([valid_similarities[i] for i in top_indices])
                probabilities = F.softmax(top_similarities / self.temperature, dim=0)
                
                # Sample based on probabilities
                sampled_idx = torch.multinomial(probabilities, 1).item()
                best_idx = valid_candidates[top_indices[sampled_idx].item()]
            else:
                # During evaluation, use the most similar candidate
                best_sim_idx = torch.argmax(torch.tensor(valid_similarities)).item()
                best_idx = valid_candidates[best_sim_idx]
            
            if with_modality == 'image':
                return self.bank.get_text_tokens(best_idx, split)
            elif with_modality == 'text':
                return self.bank.get_image_tokens(best_idx, split)
            else:
                raise TypeError(f'illegal modality {with_modality}')

    def get_highest_available_token(self, idx, split, with_modality):
        """get available train data id with modality at highest similarity (deprecated, use get_sampled_available_token instead)"""
        with torch.no_grad():
            most_close_idx, most_close_similarities = self.retriever.get_retrieval_results_with_similarities(
                split, int(idx), with_modality)
            missing_modality = MISSING_TEXT if with_modality == 'text' else MISSING_IMAGE
            best_idx, best_sim = -1, -math.inf
            for ret_idx, sim in zip(most_close_idx, most_close_similarities):
                if self.bank.get_missing_type(ret_idx, split) != missing_modality:
                    if sim > best_sim:
                        best_idx = ret_idx
                        best_sim = sim
            if best_idx == -1:
                return None
            if with_modality == 'image':
                return self.bank.get_text_tokens(best_idx, split)
            elif with_modality == 'text':
                return self.bank.get_image_tokens(best_idx, split)
            else:
                raise TypeError(f'illegal modality {with_modality}')
    
    def generate_text_prompt(self, text_tokens: Tensor):
        """
        Args:
            text_tokens: Tensor, whose shape is [num_of_samples, 512] (text feature vector)
        """
        if not isinstance(text_tokens, torch.Tensor):
            if len(text_tokens) == 0:
                return torch.tensor([], device=self.device)
            text_tokens = torch.stack(text_tokens)
        
        if text_tokens.numel() == 0:
            return text_tokens.new_empty(0)

        text_tokens = text_tokens.to(self.device)
        
        # Add input noise during training for better generalization
        if self.training:
            noise = torch.randn_like(text_tokens) * 0.01
            text_tokens = text_tokens + noise
            
        prompts = self.text_prompt_generator(text_tokens)
        prompts = prompts.view(prompts.shape[0], self.prompt_length, TEXT_DIM)
        
        # Apply L2 normalization to prevent exploding gradients
        prompts = F.normalize(prompts, p=2, dim=-1)
        
        return prompts
    
    def generate_image_prompt(self, image_tokens: Tensor):
        """
        Args:
            image_tokens: Tensor, whose shape is [num_of_samples, 512]
        """
        if not isinstance(image_tokens, torch.Tensor):
            if len(image_tokens) == 0:
                return torch.tensor([], device=self.device)
            image_tokens = torch.stack(image_tokens)

        if image_tokens.numel() == 0:
            return image_tokens.new_empty(0)

        image_tokens = image_tokens.to(self.device)
        
        # Add input noise during training for better generalization
        if self.training:
            noise = torch.randn_like(image_tokens) * 0.01
            image_tokens = image_tokens + noise
            
        prompts = self.image_prompt_generator(image_tokens)
        prompts = prompts.view(prompts.shape[0], self.prompt_length, IMAGE_DIM)
        
        # Apply L2 normalization to prevent exploding gradients
        prompts = F.normalize(prompts, p=2, dim=-1)
        
        return prompts

        
    def generate_prompts(self, batch_idx, split, missing_type):
        """
        generate residual prompts for every sample
        """
        assert len(batch_idx) == len(missing_type)
        
        text_tokens = []
        image_tokens = []
        # prompt_types = []
        
        # Collect needed tokens
        for i, idx in enumerate(batch_idx):
            if missing_type[i] == 0: # no missing
                # prompt_types.append(0)
                text_tokens.append(self.get_available_tokens(idx, split, 'text'))
                image_tokens.append(self.get_available_tokens(idx, split, 'image'))
            elif missing_type[i] == 1: # missing text, retrieve text by image
                token = self.get_retrieved_sampled_token(idx, split, 'image')
                if token is not None:
                    # prompt_types.append(1)
                    text_tokens.append(token)
                else:
                    # prompt_types.append(3)
                    text_tokens.append(torch.zeros((TOKEN_DIM,)))
                image_tokens.append(self.get_available_tokens(idx, split, 'image'))
            elif missing_type[i] == 2: # missing image, retrieve image by text
                token = self.get_retrieved_sampled_token(idx, split, 'text')
                if token is not None:
                    # prompt_types.append(2)
                    image_tokens.append(token)
                else:
                    # prompt_types.append(3)
                    image_tokens.append(torch.zeros((TOKEN_DIM,)))
                text_tokens.append(self.get_available_tokens(idx, split, 'text'))
        # Generate prompts
        text_prompt = self.generate_text_prompt(text_tokens)
        image_prompt = self.generate_image_prompt(image_tokens)
        
        # Organize return results
        # i, j, k = 0, 0, 0
        # all_prompts = [None for _ in range(len(batch_idx))]
        # while k < len(batch_idx):
        #     if prompt_types[k] == 1:
        #         all_prompts[k] = text_prompt[i]
        #         i += 1
        #     elif prompt_types[k] == 2:
        #         all_prompts[k] = image_prompt[j]
        #         j += 1
        #     k += 1
        
        # return all_prompts
        return text_prompt, image_prompt
    
    def set_temperature(self, temperature):
        """Set temperature for sampling during training"""
        self.temperature = temperature
    
    def set_top_k(self, top_k):
        """Set number of top candidates to sample from"""
        self.top_k_candidates = top_k

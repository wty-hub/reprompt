import torch
import torch.nn as nn
import pytorch_lightning as pl
from clip.modules.retriever import RetrievalPromptLearner
from clip.modules import clip_utils, objectives, clip
import copy

from clip.modules.utils import TextEncoder, load_clip_to_cpu

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, residual_length):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length//2 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts

        # Fixed vocabulary embeddings will be integrated into text_prompt_missing
        trainable_prompt_length = prompt_length_half - residual_length
        
        self.visual_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 512, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(nn.init.normal_(torch.empty(trainable_prompt_length, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_text),
                )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_image),
                )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

    def forward(self, missing_type, residual_prompts):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        all_prompts_text = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i]==0:  # modality complete
                initial_prompt_text = torch.cat((residual_prompts[0][i], self.text_prompt_complete), dim=0)
                initial_prompt_image = torch.cat((residual_prompts[1][i], self.visual_prompt_complete), dim=0)
            elif missing_type[i]==1:  # missing text 
                initial_prompt_text = torch.cat((residual_prompts[0][i], self.text_prompt_missing), dim=0)
                initial_prompt_image = torch.cat((residual_prompts[1][i], self.visual_prompt_complete), dim=0)
            elif missing_type[i]==2:  # missing image 
                initial_prompt_text = torch.cat((residual_prompts[0][i], self.text_prompt_complete), dim=0)
                initial_prompt_image = torch.cat((residual_prompts[1][i], self.visual_prompt_missing), dim=0)
            
            # Add retrieved prompt if available (replacing part of initial prompts)
            # retrieved = retrieved_prompt[i]
            # if retrieved is not None:
                # Replace part of initial prompts with retrieved prompts for missing modalities
                # if missing_type[i] == 1:  # missing text
                    # replace_length = min(retrieved.shape[0], initial_prompt_text.shape[0])
                    # initial_prompt_text = torch.cat((retrieved[:replace_length], initial_prompt_text[replace_length:]), dim=0)
                # elif missing_type[i] == 2:  # missing image
                    # replace_length = min(retrieved.shape[0], initial_prompt_image.shape[0])
                    # initial_prompt_image = torch.cat((retrieved[:replace_length], initial_prompt_image[replace_length:]), dim=0)

            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
        # generate the prompts in each layer as a tensor [B, L, C]
        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]
        return all_prompts_image, all_prompts_text   

class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, augmenter: RetrievalPromptLearner, residual_length):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(prompt_length, prompt_depth, clip_model, residual_length)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.augmenter = augmenter

    def forward(self, image, text, missing_type, idx, phase):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        #logit_scale = self.logit_scale.exp()
        retrieved_prompt = self.augmenter.generate_prompts(idx, phase, missing_type)
        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type, retrieved_prompt)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        return torch.cat([image_features, text_features], -1)

class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])

        print("Building custom CLIP")
        hidden_size = 512*2

        # ===================== Retrieval augmenter ============ #
        self.augmenter = RetrievalPromptLearner(
            dataset_name=config['dataset'],
            data_dir=config['data_root'],
            prompt_length=config.get('augmented_length', 3),
            missing_scenario=config['missing_type']['val'],
            missing_ratio=config['test_ratio'],
            remake=config.get('remake_retriever', False), # Optional: allow forcing remake from config
        )

        self.model = CustomCLIP(
            config['prompt_length'], config['prompt_depth'], clip_model, self.augmenter, 
            config.get('augmented_length', 3))

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
# 
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
                    param.requires_grad_(False)

            # # Double check
            # enabled = set()
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         enabled.add(name)
            # print(f"Parameters to be updated: {enabled}")

        clip_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.records = {}

    def infer(
        self,
        batch,
        phase: str = "train"
    ):
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        idx = batch['arrow_index']
        if self.hparams.config["test_only"]:
            self.model.eval()
            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()
            
            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()            
            
            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()
        both_feats = self.model(img, text, batch["missing_type"], idx, phase)  
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:
                both_feats[idx, :feature_dim].zero_()
            
        ret = {
            "cls_feats": both_feats,
        }

        return ret

    def forward(self, batch, phase="train"):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch, phase))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch, "train")
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch, "val")

    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch, "test")
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)

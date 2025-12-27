import contextlib
import logging
from typing import List, Literal, Optional, Union

import einops
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, LlamaForCausalLM, Pix2StructVisionModel

from models.dinov2_models.vision_transformer import vit_large as dino_vit_large
from models.eva_vit import create_eva_vit_g
from models.moe_adapters.mole_adapter import set_moe_mlp_adapter_llama, set_router_embedding_mlp, set_adapter_llama
from models.moe_adapters.move_modules import ADT, MoEAggregator
from common.registry import registry

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_module(module):
    for _, param in module.named_parameters():
        param.requires_grad = False
    module.eval()
    module.train = disabled_train


@registry.register_model("mome_model")
class MoMEModel(nn.Module):
    """
    MoME model
    
    Attributes:
        clip_path (str): Path to the EVA_VIT_G model.
        dino_path (str): Path to the DINO model.
        pix_path (str): Path to the Bert-base-uncased model.
        llm_path (str): Path to the Vicuna-v1.5 model.
        sentencebert_path (str): Path to the Sentence-BERT model.
        max_txt_len (int): Maximum length of the input sequence. Default is 1024.
        adapter_llm_finetune (bool): Whether to finetune the LLM adapter. Default is True.
        using_ADT (bool): Whether to use a ADT for feature transformation. Default is True.
        router_method (Literal["mlp", "add"]): Method for routing, either "mlp" or "add". Default is "mlp".
        visual_backbone (Literal["clip", "dino", "pix2struct", "all"]): Type of visual backbone to use. Default is "clip".
        enable_mole (bool): Whether to enable MoLE adapters in LLM. Default is True.
    """

    def __init__(
        self,
        clip_path: str,
        dino_path: str,
        pix_path: str,
        llm_path: str,
        sentencebert_path: str,
        
        max_txt_len: int = 1024,
        adapter_llm_finetune: bool = True,
        using_selector: bool = True,
        router_method: Literal["mlp", "add"] = "mlp",
        visual_backbone: Literal["clip", "dino", "pix2struct", "all"] = "clip",
        enable_mole: bool = True,
    ):
        super().__init__()
        
        self.max_txt_len = max_txt_len
        self.adapter_llm_finetune = adapter_llm_finetune
        self.using_selector = using_selector
        self.visual_backbone = visual_backbone
        self.boost_lr_scale = 0.2

        logging.info(f"visual_backbone: {visual_backbone}")
        logging.info(f"using_selector: {using_selector}")
        
        if self.visual_backbone in ["clip","all"]:
            print('Loading CLIP')
            self.visual_encoder = create_eva_vit_g(path=clip_path, img_size=224)
            freeze_module(self.visual_encoder)
            print('Loading CLIP Done')
        
        if self.visual_backbone in ["dino","all"]:
            print('Loading DINO')
            self.dino_encoder = dino_vit_large(patch_size=14,img_size=518, init_values=1.0, block_chunks=0)
            self.dino_encoder.load_state_dict(torch.load(dino_path))
            freeze_module(self.dino_encoder)
            print('Loading DINO Done')
        
        if self.visual_backbone in ["pix2struct","all"]:
            print('Loading Pix2Struct')
            self.pix2struct_encoder = Pix2StructVisionModel.from_pretrained(pix_path)
            freeze_module(self.pix2struct_encoder)
            print('Loading Pix2Struct Done')
        
        self._init_llm(llm_path, adapter_llm_finetune, enable_mole=enable_mole) 
        
        # Init Sentence Bert
        self.bert_model = SentenceTransformer(sentencebert_path)
        freeze_module(self.bert_model)
        
        if self.using_selector:
            if self.visual_backbone in ["dino","all"]:
                self.dino_selector = ADT(dim=self.dino_encoder.num_features, num_layers=6, origin_hw=16, num_queries=256)
                self.dino_llm_proj = nn.Linear(self.dino_encoder.num_features, self.llm_model.config.hidden_size)
            if self.visual_backbone in ["pix2struct","all"]:
                self.pix_selector = ADT(dim=self.pix2struct_encoder.config.hidden_size, num_layers=6, origin_hw=16, num_queries=256)
                self.pix_llm_proj = nn.Linear(self.pix2struct_encoder.config.hidden_size, self.llm_model.config.hidden_size)
        else:
            if self.visual_backbone in ["dino","all"]:
                self.dino_proj = nn.Linear(self.dino_encoder.num_features, self.llm_model.config.hidden_size)
            if self.visual_backbone in ["pix2struct","all"]:
                self.pix2struct_proj = nn.Linear(self.pix2struct_encoder.config.hidden_size, self.llm_model.config.hidden_size)
                
        if self.visual_backbone in ["clip","all"]:
            self.clip_ln = LayerNorm(self.visual_encoder.num_features)
            self.clip_proj = nn.Linear(self.visual_encoder.num_features, self.llm_model.config.hidden_size)
                
        self.downsampler = nn.AdaptiveAvgPool2d((16, 16))
            
        if self.visual_backbone == "all":
            self.move_router = MoEAggregator(
                num_experts=3,
                seq_dim=self.bert_model.get_sentence_embedding_dimension(),
                router_method=router_method
            )
        
        # Pre-compute the embeddings for the prompt
        single_prompt = "USER: <Img><ImageHere></Img>"
        p_before, p_after = single_prompt.split('<ImageHere>')
        llm_device = list(self.llm_model.parameters())[0].device
        p_before_tokens = self.llm_tokenizer(p_before, return_tensors="pt").to(llm_device)
        p_after_tokens = self.llm_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(llm_device)
        self.p_before_embeds = self.llm_model.get_input_embeddings()(p_before_tokens.input_ids)
        self.p_after_embeds = self.llm_model.get_input_embeddings()(p_after_tokens.input_ids)
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def _init_llm(self, llm_path:str, adapter_llm_finetune:bool, enable_mole:bool=True):
        print('Loading Vicuna')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16,
        )
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        for name, param in self.llm_model.named_buffers():
            param.requires_grad = False
        
        if adapter_llm_finetune:
            if enable_mole:
                logging.info("using mole adapter for llm finetune")
                set_moe_mlp_adapter_llama(self.llm_model, self.llm_model.config.hidden_size, bottleneck=64, num_adapters=4)
            else:
                logging.info("using normal adapter for llm finetune")
                set_adapter_llama(self.llm_model, self.llm_model.config.hidden_size, bottleneck=64)
        
        print('Loading Vicuna Done')

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def select_pix2srtuct(self, image_embeds, pix2struct_embeds, image_shape):
        pix_embeds = self.pix_selector(image_embeds, pix2struct_embeds, image_shape)
        return self.pix_llm_proj(pix_embeds)

    def select_dino(self, image_embeds, dino_embeds, image_shape):
        dino_embeds = self.dino_selector(image_embeds, dino_embeds, image_shape)
        return self.dino_llm_proj(dino_embeds)
    
    def downsample_dino(self, dino_embeds):
        dino_embeds = einops.rearrange(dino_embeds, 'b (h w) d -> b d h w', h=32, w=32)
        dino_embeds = self.downsampler(dino_embeds)
        dino_embeds = einops.rearrange(dino_embeds, 'b d h w -> b (h w) d')
        return dino_embeds
    
    def downsample_pix2struct(self, pix2struct_embeds, image_shape):
        pix2struct_embeds_l = []
        for img_feat, shape in zip(pix2struct_embeds, image_shape):
            h, w = shape[0].item(), shape[1].item()
            img_feat = img_feat[:h*w]
            img_feat = einops.rearrange(img_feat, '(h w) d -> d h w', h=h, w=w)
            img_feat = self.downsampler(img_feat)
            img_feat = einops.rearrange(img_feat, 'd h w -> (h w) d')
            pix2struct_embeds_l.append(img_feat)
        return torch.stack(pix2struct_embeds_l)
    
    def encode_img(self, image_samples, question_embedding:torch.Tensor):
        """
        Encodes the input image.

        Args:
            image (torch.Tensor): The input image tensor of shape (B, C, H, W).
            text (List[str], optional): The input text list with length B. Defaults to None.

        Returns:
            inputs_llm (torch.Tensor): The encoded image tensor of shape (B, Q, H).
            atts_llm (torch.Tensor): The attention tensor of shape (B, Q).
        """
        image = image_samples["image"]
        assert image.device == self.device, f"image.device:{image.device} should be equal to self.device:{self.device}"
        set_router_embedding_mlp(self.llm_model, question_embedding)

        with self.maybe_autocast():
            if self.using_selector:
                if self.visual_backbone in ["dino", "all"]:
                    dino_output = self.dino_encoder(image_samples["dino_image"], is_training=True)
                    dino_embeds = dino_output["x_norm_patchtokens"]
                    dino_embeds_transformed = self.downsample_dino(dino_embeds)
                    dino_embeds = self.select_dino(dino_embeds_transformed, dino_embeds, torch.tensor([[32, 32]], device=image.device))
                    inputs_llm = dino_embeds
                if self.visual_backbone in ["pix2struct", "all"]:
                    pix2struct_embeds = self.pix2struct_encoder(
                        flattened_patches=image_samples["flattened_patches"],
                        attention_mask=image_samples["attention_mask"],
                    ).last_hidden_state
                    pix2struct_embeds_transformed = self.downsample_pix2struct(pix2struct_embeds, image_samples["image_shape"])
                    pix2struct_embeds = self.select_pix2srtuct(pix2struct_embeds_transformed, pix2struct_embeds, image_samples["image_shape"])
                    inputs_llm = pix2struct_embeds
                if self.visual_backbone in ["clip", "all"]:
                    image_embeds = self.visual_encoder(image, return_intermediate=False)
                    image_embeds = self.clip_ln(image_embeds)
                    clip_embeds = self.clip_proj(image_embeds)
                    inputs_llm = clip_embeds

                if self.visual_backbone == "all":
                    inputs_llm = self.move_router([clip_embeds[:,1:,:], dino_embeds, pix2struct_embeds], question_embedding)
                    inputs_llm = torch.cat([clip_embeds[:,0,:].unsqueeze(1), inputs_llm], dim=1) # add cls
                
            else:
                if self.visual_backbone in ["pix2struct", "all"]:
                    pix2struct_embeds = self.pix2struct_proj(self.pix2struct_encoder(
                        flattened_patches=image_samples["flattened_patches"],
                        attention_mask=image_samples["attention_mask"],
                    ).last_hidden_state)
                    pix2struct_embeds = self.downsample_pix2struct(pix2struct_embeds, image_samples["image_shape"])
                    inputs_llm = pix2struct_embeds
                if self.visual_backbone in ["dino", "all"]:
                    dino_output = self.dino_encoder(image_samples["dino_image"], is_training=True)
                    dino_embeds = self.downsample_dino(dino_output["x_norm_patchtokens"])
                    dino_embeds = self.dino_proj(dino_embeds)
                    inputs_llm = dino_embeds
                if self.visual_backbone in ["clip", "all"]:
                    image_embeds = self.visual_encoder(image, return_intermediate=False)
                    image_embeds = self.clip_ln(image_embeds)
                    clip_embeds = self.clip_proj(image_embeds)
                    inputs_llm = clip_embeds
                
                if self.visual_backbone == "all":
                    inputs_llm = self.move_router([clip_embeds[:,1:,:], dino_embeds, pix2struct_embeds], question_embedding)
                    inputs_llm = torch.cat([clip_embeds[:,0,:].unsqueeze(1), inputs_llm], dim=1) # add cls
            
        atts_llm = torch.ones(inputs_llm.size()[:-1],dtype=torch.long).to(image.device)
        
        return inputs_llm, atts_llm

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """
        concat input_ids with [gMASK] <sop> , output_ids without special tokens, and add eos token
        """
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def prepare_inputs_multimodal(
        self,
        questions,
        images:Union[torch.Tensor, list],
        answer:Optional[List[str]]=None,
        truncate=False,
        padding_side="right",
        check=True,
        question_embedding=None,
    ):
        self.llm_tokenizer.padding_side = padding_side
        # 1. encode image
        image_embeds, _ = self.encode_img(images, question_embedding)
        # 2. prompt wrap image embeddings
        batch_size = len(questions)
        p_before_embeds = self.p_before_embeds.expand(batch_size, -1, -1).to(self.device)
        p_after_embeds = self.p_after_embeds.expand(batch_size, -1, -1).to(self.device)
        input_embeds = torch.cat([p_before_embeds, image_embeds, p_after_embeds], dim=1)
        # 3. encode questions
        questions = [q+" ASSISTANT: " for q in questions]
        question_tokens = self.llm_tokenizer(
            questions, return_tensors="pt", padding="longest", add_special_tokens=False, max_length=self.max_txt_len, truncation=truncate
        ).to(self.device)
        
        if answer is not None:
            answer = [a+self.llm_tokenizer.eos_token for a in answer]
            answer_tokens = self.llm_tokenizer(
                answer, return_tensors="pt", padding="longest", add_special_tokens=False, max_length=self.max_txt_len, truncation=truncate
            ).to(self.device)
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                question_tokens.input_ids,
                question_tokens.attention_mask,
                answer_tokens.input_ids,
                answer_tokens.attention_mask
            )
            
            labels = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids']==self.llm_tokenizer.pad_token_id, -100
            )
            
            for i,l in enumerate(input_part_targets_len):
                labels[i,:l] = -100
            labels = torch.cat([
                torch.ones(input_embeds.shape[:-1], dtype=torch.long).to(self.device).fill_(-100),
                labels
            ], dim=1)
        else:
            llm_tokens = {"input_ids": question_tokens.input_ids, "attention_mask": question_tokens.attention_mask}
            labels = None
        
        attention_mask = torch.cat([
            torch.ones(input_embeds.shape[:-1], dtype=torch.long).to(self.device), llm_tokens['attention_mask']
        ], dim=1)
        input_embeds = torch.cat([
            input_embeds, self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        ], dim=1)
        # 4. truncate input_embeds
        if truncate:
            input_embeds = input_embeds[:,:self.max_txt_len]
            attention_mask = attention_mask[:,:self.max_txt_len]
            if labels is not None:
                labels = labels[:,:self.max_txt_len]
        
        return input_embeds, attention_mask, labels
    
    def get_sentence_embeddings(self, questions):
        """
        Get sentence embeddings from questions.

        Args:
            questions (List[str]): List of questions.

        Returns:
            torch.Tensor: The sentence embeddings of shape (B, H).
        """
        question_embedding = self.bert_model.encode(questions, convert_to_tensor=True, show_progress_bar=False)
        return question_embedding
    
    def forward(self, samples:dict):
        """
        Args:
            samples: Dict
                image: Optional[torch.Tensor] (B, C, H, W) or (B, Q, C, H, W) or List[torch.Tensor] B x (Q, C, H, W)
                question: List[str]
                answer: List[str]
        Returns:
            Dict
                loss: torch.Tensor
        """
        
        question_embedding = self.get_sentence_embeddings(samples["question"])
        inputs_embeds, attention_mask, targets = self.prepare_inputs_multimodal(
            samples["question"], samples, samples["answer"], truncate=True, question_embedding=question_embedding
        )
        
        # Generate outputs with LLM 
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        

        lm_loss = outputs.loss

        return {"loss": lm_loss}
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        p_boost, p_boost_non_wd = [], []
        boost_name = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            elif "pix_selector" in n or "dino_selector" in n or "vit_selector" in n:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_boost_non_wd.append(p)
                else:
                    p_boost.append(p)
                boost_name.append(n)
            else:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
            {"params": p_boost, "weight_decay": weight_decay, "lr_scale": lr_scale*self.boost_lr_scale},
            {"params": p_boost_non_wd, "weight_decay": 0, "lr_scale": lr_scale*self.boost_lr_scale},
        ]
        return optim_params
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=0.0,
    ):
        """
        Args:
            samples:Dict
                image: Optional[torch.Tensor] (B, C, H, W)
                question: List[str]
        Returns:
            output_text: List[str]
        """
        question_embedding = self.get_sentence_embeddings(samples["question"])
        # question_embedding = None
        inputs_embeds, attention_mask, targets = self.prepare_inputs_multimodal(
            samples["question"], samples, answer=None, truncate=False, padding_side="left",question_embedding=question_embedding
        )

        # generate output with LLM
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        
        return output_text
    
    @classmethod
    def from_config(cls, cfg: Union[str, OmegaConf]):
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg).model
        clip_path = cfg.get("clip_path")
        dino_path = cfg.get("dino_path")
        pix_path = cfg.get("pix_path")
        llm_path = cfg.get("llm_path")
        sentencebert_path = cfg.get("sentencebert_path")
        
        max_txt_len = cfg.get("max_txt_len", 1024)
        adapter_llm_finetune = cfg.get("adapter_llm_finetune", True)
        using_selector = cfg.get("using_selector", True)
        router_method = cfg.get("router_method", "mlp")
        visual_backbone = cfg.get("visual_backbone", "all")
        enable_mole = cfg.get("enable_mole", True)

        model = cls(
            clip_path=clip_path,
            dino_path=dino_path,
            pix_path=pix_path,
            llm_path=llm_path,
            sentencebert_path=sentencebert_path,
            
            max_txt_len=max_txt_len,
            adapter_llm_finetune=adapter_llm_finetune,
            using_selector=using_selector,
            router_method=router_method,
            visual_backbone=visual_backbone,
            enable_mole=enable_mole,
        )

        if cfg.get("ckpt") is not None:
            ckpt_path = cfg.get("ckpt")
            ckpt = torch.load(cfg.get("ckpt"), map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
                
            if "stage1" in ckpt_path:
                print(f"Loading stage1 checkpoint from {ckpt_path} and duplicating weights for MoLE adapters")
                new_ckpt = {}
                for k in ckpt.keys():
                    if ".adapter." in k:
                        # adapter adapter weights
                        for idx in range(4):
                            new_k = k.replace(".adapter.", f".adapter.adapters.{idx}.")
                            new_ckpt[new_k] = ckpt[k].clone()
                    else:
                        new_ckpt[k] = ckpt[k]
                result = model.load_state_dict(new_ckpt, strict=False)
            else:
                print(f"Loading checkpoint from {ckpt_path}")
                result = model.load_state_dict(ckpt, strict=False)
            print(f"Unexpected keys: {result.unexpected_keys}")

        return model
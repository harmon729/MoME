import os

import requests
import torch
from huggingface_hub import snapshot_download

CKPT_DIR = "./checkpoints"

CLIP_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
PIX_REPO = "google/pix2struct-base"
SENTENCE_BERT_REPO = "sentence-transformers/all-MiniLM-L6-v2"
VICUNA_REPO = "lmsys/vicuna-7b-v1.5"
MOME_REPO = "daybreaksly/MoME"


def download_file(url):
    destination = os.path.join(CKPT_DIR, url.split("/")[-1])
    if os.path.exists(destination):
        print(f"File {destination} already exists. Skipping download.")
        return
    print(f"Downloading file from {url} to {destination}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"File downloaded successfully and saved to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        
def download_huggingface(repo_id):
    print(f"Downloading model from Huggingface Hub: {repo_id}")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir_use_symlinks=False, 
                      local_dir=os.path.join(CKPT_DIR, repo_id.split("/")[-1]))

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Download ckpt for CLIP
    download_file(CLIP_URL)
    # Download ckpt for DINO
    download_file(DINO_URL)
    
    # Download ckpt for Pix2Struct and convert it to vision ckpt
    if os.path.exists(os.path.join(CKPT_DIR, "pix2struct-encoder-base")):
        print("Model pix2struct-encoder-base already exists. Skipping download.")
    else:
        download_huggingface(PIX_REPO)
        old_ckpt = torch.load(os.path.join(CKPT_DIR, "pix2struct-base", "pytorch_model.bin"))
        new_ckpt = {}
        for key, value in old_ckpt.items():
            if "encoder." in key:
                new_ckpt[key[len("encoder."):]] = value
        os.remove(os.path.join(CKPT_DIR, "pix2struct-base", "model.safetensors"))
        os.remove(os.path.join(CKPT_DIR, "pix2struct-base", "pytorch_model.bin"))
        torch.save(new_ckpt, os.path.join(CKPT_DIR, "pix2struct-base", "pytorch_model.bin"))
        os.rename(os.path.join(CKPT_DIR, "pix2struct-base"), os.path.join(CKPT_DIR, "pix2struct-encoder-base"))
    
    # Download ckpt for Sentence Bert
    download_huggingface(SENTENCE_BERT_REPO)
    # Download ckpt for Vicuna
    download_huggingface(VICUNA_REPO)
    # Download ckpt for MoME
    download_huggingface(MOME_REPO)
    
    
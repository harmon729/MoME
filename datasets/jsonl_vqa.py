import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from PIL import Image
import torch
from torch.utils.data import Dataset

from common.registry import registry
from preprocessors.multi_image_processor import MultiImageProcessor


class JSONLVQADataset(Dataset):
    def __init__(self, ann_path: Union[str, List[str]], vis_root: Optional[str]) -> None:
        # Support a single file or a list of files in JSON format.
        self.vis_root = vis_root
        self.samples: List[Dict[str, Any]] = []

        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        else:
            ann_paths = list(ann_path)

        for p in ann_paths:
            assert os.path.exists(p), f"Annotation file not found: {p}"
            self._load_ann_file(p)

        self.processor = MultiImageProcessor()

    def _load_ann_file(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, list), f"Expected a list of records in {path}"
            self.samples.extend(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        # Accept Combined_VQA style keys: image_path OR image_id (fallback), and our own "image"
        img_path = rec.get("image_path").replace("Datasets/", "")
        assert img_path, f"Missing image path for record {idx}"
        if self.vis_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.vis_root, img_path)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            prepared = self.processor.prepare_multi_image_input(im)

        sample = {
            "question": rec.get("question"),
            "answer": rec.get("answer", ""),
        }
        sample.update(prepared)
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch])
        dino_images = torch.stack([b["dino_image"] for b in batch])
        flattened_patches = torch.stack([b["flattened_patches"] for b in batch])
        attention_masks = torch.stack([b["attention_mask"] for b in batch])
        image_shapes = torch.stack([b["image_shape"] for b in batch])
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        # Router supports a single route index across the module; use the first
        out = {
            "image": images,
            "dino_image": dino_images,
            "flattened_patches": flattened_patches,
            "attention_mask": attention_masks,
            "image_shape": image_shapes,
            "question": questions,
            "answer": answers,
        }
        return out

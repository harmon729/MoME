import math
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import Pix2StructImageProcessor

from preprocessors.square_processor import SquareImageProcessor


def get_image_shape(image):
    patch_height, patch_width = 16,16
    image_height, image_width = image.size
    max_patches = 2048

    # maximize scale s.t.
    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)

    return torch.tensor([num_feasible_cols, num_feasible_rows])

class MultiImageProcessor:
    def __init__(self) -> None:
        self.clip_preprocessor = SquareImageProcessor()
        self.dino_preprocessor = SquareImageProcessor(image_size=448)
        self.pix_preprocessor = Pix2StructImageProcessor()

    def prepare_multi_image_input(self, image: Image.Image) -> Dict:
        dino_image = self.dino_preprocessor(image)
        pix2struct_inputs = self.pix_preprocessor(image, return_tensors="pt")
        clip_image = self.clip_preprocessor(image)
        image_shape = get_image_shape(image)
        
        return {
            "image": clip_image,
            "dino_image": dino_image,
            "flattened_patches": pix2struct_inputs["flattened_patches"].squeeze(0),
            "attention_mask": pix2struct_inputs["attention_mask"].squeeze(0),
            "image_shape": image_shape,
        }
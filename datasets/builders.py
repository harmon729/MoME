import os
from omegaconf import OmegaConf
from common.registry import registry
from .jsonl_vqa import JSONLVQADataset


@registry.register_builder("jsonl_vqa")
def build_jsonl_vqa_dataset(name):
    # expects a config file path passed directly via name, or a named entry under configs/datasets if extended later
    if os.path.exists(name):
        cfg = OmegaConf.load(name)
    else:
        # simple inline OmegaConf dict usage
        raise AssertionError(f"Please provide ann config file path for dataset builder, got: {name}")

    ann_path = cfg.get("ann_path")
    vis_root = cfg.get("vis_root", None)
    return JSONLVQADataset(ann_path=ann_path, vis_root=vis_root)


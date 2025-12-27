import argparse
import os
import logging
from datetime import datetime

from omegaconf import OmegaConf
from accelerate import Accelerator

from common.registry import registry
from models.mome_model import MoMEModel  # register
from datasets import builders  # register
from trainer import Trainer


def build_model(cfg):
    model_cfg = cfg.model
    model_cls = registry.get_model_class("mome_model")
    return model_cls.from_config(model_cfg)


def build_dataset(ds_cfg):
    from datasets.jsonl_vqa import JSONLVQADataset
    ds = JSONLVQADataset(
        ann_path=ds_cfg.ann_path,
        vis_root=ds_cfg.get("vis_root", None),
    )
    # optional sampling ratio for multi-dataset mixing
    ratio = float(ds_cfg.get("sample_ratio", 1.0))
    setattr(ds, "sample_ratio", ratio)
    return ds


def build_train_datasets(cfg):
    # Support either a list under cfg.train_datasets or single cfg.train_dataset
    if cfg.get("train_datasets", None) is not None:
        return [build_dataset(ds_cfg) for ds_cfg in cfg.train_datasets]
    elif cfg.get("train_dataset", None) is not None:
        return [build_dataset(cfg.train_dataset)]
    else:
        raise ValueError("Config must define train_datasets (list) or train_dataset (single)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-path", required=True, help="Path to training config YAML")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    out_base = cfg.run.get("output_dir", "outputs/mome_train")
    job_id = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(out_base, job_id)
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(log_with=["tensorboard"], project_dir=out_dir, mixed_precision=cfg.run.get("mixed_precision", "bf16"))
    accelerator.init_trackers(project_name="mome_train")

    model = build_model(cfg)
    train_datasets = build_train_datasets(cfg)

    trainer = Trainer(cfg=cfg, accelerator=accelerator, model=model, train_datasets=train_datasets)
    trainer.train(out_dir)


if __name__ == "__main__":
    main()

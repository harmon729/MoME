import os
import time
import json
import logging
from typing import List, Dict, Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, cfg, accelerator: Accelerator, model, train_datasets: List[Dataset]) -> None:
        self.cfg = cfg
        self.accelerator = accelerator
        self._model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # run config
        # batch_size_train is total effective batch size, divide by world size for per-GPU batch size
        total_batch_size = int(cfg.run.batch_size_train)
        world_size = accelerator.num_processes
        self.batch_size = total_batch_size // world_size
        if self.accelerator.is_main_process:
            logging.info(f"Total batch size: {total_batch_size}, World size: {world_size}, Per-GPU batch size: {self.batch_size}")
        self.num_workers = int(cfg.run.get("num_workers", 8))
        self.max_epoch = int(cfg.run.max_epoch)
        self.iters_per_epoch = int(cfg.run.iters_per_epoch)
        self.print_freq = int(cfg.run.get("print_freq", 100))

        # support multiple training datasets
        assert isinstance(train_datasets, list) and len(train_datasets) > 0, "train_datasets must be a non-empty list"
        self.train_datasets = train_datasets
        self.train_loaders: List[DataLoader] = []
        self.ratios: List[float] = []
        for ds in self.train_datasets:
            dl = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=getattr(ds, "collate_fn", None),
            )
            self.train_loaders.append(dl)
            ratio = float(getattr(ds, "sample_ratio", 1.0))
            self.ratios.append(max(ratio, 0.0))

        # optimizer
        weight_decay = float(cfg.run.get("weight_decay", 0.05))
        init_lr = float(cfg.run.init_lr)
        lr_layer_decay = float(cfg.run.get("lr_layer_decay", 1.0))
        self.optimizer = torch.optim.AdamW(
            self._model.get_optimizer_params(weight_decay, lr_layer_decay),
            lr=init_lr,
            betas=(0.9, 0.999),
        )

        # lr scheduler (linear warmup + cosine), matching root trainer behavior
        warmup_lr = float(cfg.run.get("warmup_lr", -1))
        warmup_steps = int(cfg.run.get("warmup_steps", 0))
        decay_rate = cfg.run.get("lr_decay_rate", None)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_epoch*self.iters_per_epoch
        )

        # prepare with accelerator (model, optimizer, and all loaders)
        prepared = self.accelerator.prepare(self._model, self.optimizer, *self.train_loaders)
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.train_loaders = list(prepared[2:])

        self.start_epoch = 0

    def _save_checkpoint(self, out_dir: str, cur_epoch: int):
        os.makedirs(out_dir, exist_ok=True)
        state_dict = self.accelerator.get_state_dict(self.model)
        # strip frozen params
        param_grad_dic = {k: v.requires_grad for (k, v) in self._model.named_parameters()}
        for k in list(state_dict.keys()):
            if k in param_grad_dic and not param_grad_dic[k]:
                del state_dict[k]
        ckpt = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "epoch": cur_epoch,
        }
        save_to = os.path.join(out_dir, f"checkpoint_{cur_epoch}.pth")
        if self.accelerator.is_main_process:
            logging.info(f"Saving checkpoint to {save_to}")
        torch.save(ckpt, save_to)

    def _build_schedule(self, iters: int) -> List[int]:
        # Build a deterministic per-epoch sampling schedule based on ratios
        if not any(self.ratios):
            # default uniform
            counts = [iters // len(self.train_loaders)] * len(self.train_loaders)
            for i in range(iters - sum(counts)):
                counts[i % len(counts)] += 1
        else:
            total = sum(self.ratios)
            raw = [iters * r / total for r in self.ratios]
            counts = [int(x) for x in raw]
            rem = iters - sum(counts)
            if rem > 0:
                fracs = sorted([(i, raw[i] - counts[i]) for i in range(len(raw))], key=lambda t: t[1], reverse=True)
                for i in range(rem):
                    counts[fracs[i % len(fracs)][0]] += 1
        schedule: List[int] = []
        for i, c in enumerate(counts):
            schedule.extend([i] * c)
        # small shuffle-like interleave to avoid long streaks: round-robin merge
        schedule_sorted: List[int] = []
        buckets = {i: c for i, c in enumerate(counts)}
        cur = 0
        for _ in range(iters):
            # find next available bucket
            for _try in range(len(buckets)):
                idx = (cur + _try) % len(self.train_loaders)
                if buckets.get(idx, 0) > 0:
                    schedule_sorted.append(idx)
                    buckets[idx] -= 1
                    if buckets[idx] == 0:
                        del buckets[idx]
                    cur = (idx + 1) % len(self.train_loaders)
                    break
        return schedule_sorted if len(schedule_sorted) == iters else schedule[:iters]

    def train(self, out_dir: str):
        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()

            # fresh iterators per epoch
            iters = [iter(dl) for dl in self.train_loaders]
            schedule = self._build_schedule(self.iters_per_epoch)
            running = 0.0

            for it, li in enumerate(schedule):
                # fetch batch from selected loader
                try:
                    samples = next(iters[li])
                except StopIteration:
                    iters[li] = iter(self.train_loaders[li])
                    samples = next(iters[li])

                loss_dict = self.model(samples)
                loss = loss_dict["loss"]
                self.accelerator.backward(loss)
                # step lr schedule per inner step
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                running += loss.item()

                if (it + 1) % self.print_freq == 0 and self.accelerator.is_main_process:
                    avg = running / self.print_freq
                    running = 0.0
                    logging.info(
                        f"Epoch {epoch} Iter {it+1}/{self.iters_per_epoch} loss {avg:.4f} (loader {li})"
                    )

            # save after each epoch
            self._save_checkpoint(out_dir, epoch)

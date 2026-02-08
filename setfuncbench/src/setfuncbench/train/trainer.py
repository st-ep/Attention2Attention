from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from typing import Dict, Optional

import torch
import torch.nn as nn

from setfuncbench.config import DatasetConfig, ModelConfig, TrainConfig
from setfuncbench.data.registry import sample_batch
from setfuncbench.utils.seed import SeedSequence, set_global_seed


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    seed_seq: SeedSequence,
    num_batches: int,
    seed_offset: int = 10_000,
) -> Dict[str, float]:
    """
    Evaluate model MSE on qry_y over num_batches synthetic batches.

    Important: preserves and restores the incoming model mode (train/eval) so that
    training does not accidentally continue in eval mode after evaluation.
    """
    was_training = model.training
    model.eval()
    try:
        losses = []
        for i in range(num_batches):
            cfg_i = replace(dataset_cfg, seed=seed_seq.seed_for_step(i, offset=seed_offset))
            batch = sample_batch(cfg_i, device=device)
            pred = model(batch)
            loss = mse_loss(pred, batch["qry_y"]).item()
            losses.append(loss)
        return {"mse": float(sum(losses) / max(1, len(losses)))}
    finally:
        if was_training:
            model.train()


class Trainer:
    """
    Minimal trainer:
      - Samples synthetic batches on-the-fly by varying cfg.seed
      - MSE loss on qry_y
      - Saves checkpoints to runs/<exp_name>/
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr)
        self.seed_seq = SeedSequence(train_cfg.seed)

        self.run_path = os.path.join(train_cfg.run_dir, train_cfg.exp_name)
        os.makedirs(self.run_path, exist_ok=True)

        # Save config metadata once
        self._save_config()

    def _save_config(self) -> None:
        cfg_blob = {
            "dataset_cfg": self.dataset_cfg.to_dict(),
            "model_cfg": self.model_cfg.to_dict(),
            "train_cfg": asdict(self.train_cfg),
        }
        with open(os.path.join(self.run_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_blob, f, indent=2, sort_keys=True)

    def save_checkpoint(self, step: int, filename: str) -> str:
        path = os.path.join(self.run_path, filename)
        payload = {
            "step": step,
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
            "dataset_cfg": self.dataset_cfg.to_dict(),
            "model_cfg": self.model_cfg.to_dict(),
            "train_cfg": asdict(self.train_cfg),
        }
        torch.save(payload, path)
        return path

    def train(self) -> str:
        # Global seed affects model init randomness and any global-RNG usage.
        # Determinism is an opt-in debug mode.
        set_global_seed(self.train_cfg.seed, deterministic=self.train_cfg.deterministic)

        self.model.train()

        last_ckpt: Optional[str] = None
        for step in range(self.train_cfg.steps):
            # Deterministic but changing batches: seed = base + step
            cfg_step = replace(self.dataset_cfg, seed=self.seed_seq.seed_for_step(step))
            batch = sample_batch(cfg_step, device=self.device)

            pred = self.model(batch)
            loss = mse_loss(pred, batch["qry_y"])

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            if (step + 1) % self.train_cfg.log_every == 0 or step == 0:
                print(f"[step {step+1:>6}/{self.train_cfg.steps}] train_mse={loss.item():.6f}")

            if self.train_cfg.eval_every > 0 and (step + 1) % self.train_cfg.eval_every == 0:
                metrics = evaluate(
                    self.model,
                    self.dataset_cfg,
                    device=self.device,
                    seed_seq=self.seed_seq,
                    num_batches=self.train_cfg.eval_batches,
                )
                print(f"           eval_mse={metrics['mse']:.6f}")
                # Extra safety: ensure we resume training mode even if evaluate() changes in the future.
                self.model.train()

            if self.train_cfg.save_every and (step + 1) % self.train_cfg.save_every == 0:
                last_ckpt = self.save_checkpoint(step=step + 1, filename=f"checkpoint_step_{step+1}.pt")
                # Also refresh "last"
                self.save_checkpoint(step=step + 1, filename="checkpoint_last.pt")

        # Always save last checkpoint
        last_ckpt = self.save_checkpoint(step=self.train_cfg.steps, filename="checkpoint_last.pt")
        return last_ckpt

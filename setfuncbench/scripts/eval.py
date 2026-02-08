from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, ModelConfig, TrainConfig
from setfuncbench.data.registry import sample_batch
from setfuncbench.models.registry import create_model
from setfuncbench.train.trainer import mse_loss
from setfuncbench.utils.device import default_device
from setfuncbench.utils.seed import SeedSequence, set_global_seed


def _cfg_from_dicts(payload: Dict[str, Any]) -> tuple[DatasetConfig, ModelConfig, TrainConfig]:
    d = payload["dataset_cfg"]
    m = payload["model_cfg"]
    t = payload["train_cfg"]
    dataset_cfg = DatasetConfig(**d)
    model_cfg = ModelConfig(**m)
    train_cfg = TrainConfig(**t)
    return dataset_cfg, model_cfg, train_cfg


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint (SetFuncBench).")
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default=default_device())

    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed_offset", type=int, default=50_000)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms (debug mode; may slow down GPU runs).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    payload = torch.load(args.ckpt, map_location=device)
    dataset_cfg, model_cfg, train_cfg = _cfg_from_dicts(payload)
    train_cfg.device = args.device  # override device for eval

    # Determinism is opt-in (debug mode).
    set_global_seed(train_cfg.seed, deterministic=args.deterministic)

    model = create_model(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    seed_seq = SeedSequence(train_cfg.seed)
    losses = []
    for i in range(args.eval_batches):
        cfg_i = replace(dataset_cfg, seed=seed_seq.seed_for_step(i, offset=args.seed_offset))
        batch = sample_batch(cfg_i, device=device)
        pred = model(batch)
        losses.append(float(mse_loss(pred, batch["qry_y"]).item()))

    mse = sum(losses) / max(1, len(losses))
    print(f"ckpt={args.ckpt}")
    print(f"dataset={dataset_cfg.name} model={model_cfg.name}")
    print(f"eval_batches={args.eval_batches} mse={mse:.6f}")


if __name__ == "__main__":
    main()

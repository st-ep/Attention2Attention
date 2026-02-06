from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from setfuncbench.config import DatasetConfig, ModelConfig, TrainConfig
from setfuncbench.models.registry import create_model, list_models
from setfuncbench.data.registry import list_datasets
from setfuncbench.train.trainer import Trainer
from setfuncbench.utils.seed import set_global_seed


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    """
    Parse repeated --dataset_param k=v into a dict with basic type inference.
    """
    out: Dict[str, Any] = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"Expected key=value, got: {s}")
        k, v = s.split("=", 1)
        v_strip = v.strip().lower()
        if v_strip in ("true", "false"):
            out[k] = (v_strip == "true")
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model on a dataset (SetFuncBench).")

    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--model", type=str, required=True, choices=list_models())

    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--Q", type=int, default=64)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=0)

    parser.add_argument("--dataset_param", action="append", default=[], help="Dataset param override: key=value")
    parser.add_argument("--model_param", action="append", default=[], help="Model param override: key=value")

    args = parser.parse_args()

    dataset_params = _parse_kv(args.dataset_param)
    model_params = _parse_kv(args.model_param)

    dataset_cfg = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        K=args.K,
        M=args.M,
        Q=args.Q,
        seed=args.seed,
        params=dataset_params,
    )
    model_cfg = ModelConfig(name=args.model, params=model_params)
    train_cfg = TrainConfig(
        exp_name=args.exp_name,
        run_dir=args.run_dir,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
    )

    os.makedirs(os.path.join(train_cfg.run_dir, train_cfg.exp_name), exist_ok=True)

    set_global_seed(train_cfg.seed, deterministic=True)
    model = create_model(model_cfg)

    trainer = Trainer(model=model, dataset_cfg=dataset_cfg, model_cfg=model_cfg, train_cfg=train_cfg)
    ckpt_path = trainer.train()
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

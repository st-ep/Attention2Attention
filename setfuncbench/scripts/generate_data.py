from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Any, Dict, List

import torch

from setfuncbench.config import DatasetConfig
from setfuncbench.data.registry import list_datasets, sample_batch
from setfuncbench.utils.device import default_device


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    """Parse repeated key=value arguments with basic type inference."""
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


def _to_cpu_tree(x: Any) -> Any:
    """Recursively move any tensors in nested dict/list/tuple structures to CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_tree(v) for v in x)
    return x


def _save_one(path: str, cfg: DatasetConfig, batch: Dict[str, Any], save_device: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "dataset_cfg": cfg.to_dict(),
        "batch": _to_cpu_tree(batch) if save_device == "cpu" else batch,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save synthetic dataset batches (SetFuncBench).")

    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--device", type=str, default=default_device())

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--Q", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--seed_stride", type=int, default=1)

    parser.add_argument("--dataset_param", action="append", default=[], help="Dataset param override: key=value")

    parser.add_argument("--save_dir", type=str, default="generated")
    parser.add_argument("--save_name", type=str, default="sample")
    parser.add_argument(
        "--save_device",
        type=str,
        choices=["cpu", "as_is"],
        default="cpu",
        help="Store tensors on CPU for portability (default) or keep original device tensors.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Explicit output .pt file path (allowed only when --num_batches 1).",
    )

    args = parser.parse_args()

    if args.num_batches < 1:
        raise ValueError("--num_batches must be >= 1")
    if args.seed_stride < 1:
        raise ValueError("--seed_stride must be >= 1")
    if args.out and args.num_batches != 1:
        raise ValueError("--out is only supported when --num_batches=1")

    dataset_params = _parse_kv(args.dataset_param)
    base_cfg = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        K=args.K,
        M=args.M,
        Q=args.Q,
        seed=args.seed,
        params=dataset_params,
    )

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "dataset": args.dataset,
        "device": args.device,
        "save_device": args.save_device,
        "num_batches": args.num_batches,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "shape": {
            "batch_size": args.batch_size,
            "K": args.K,
            "M": args.M,
            "Q": args.Q,
        },
        "dataset_params": dataset_params,
        "files": [],
    }

    first_shapes: Dict[str, Any] = {}

    for i in range(args.num_batches):
        seed_i = args.seed + i * args.seed_stride
        cfg_i = replace(base_cfg, seed=seed_i)
        batch = sample_batch(cfg_i, device=device)

        if i == 0:
            for key in ("ctx_x", "ctx_y", "qry_x", "qry_y"):
                first_shapes[key] = list(batch[key].shape)

        if args.num_batches == 1:
            out_path = args.out if args.out else os.path.join(args.save_dir, f"{args.save_name}.pt")
        else:
            out_path = os.path.join(args.save_dir, f"{args.save_name}_{i:06d}.pt")

        _save_one(out_path, cfg_i, batch, save_device=args.save_device)
        manifest["files"].append({"index": i, "seed": seed_i, "path": out_path})

    manifest["sample_shapes"] = first_shapes

    # Write manifest for reproducibility/tracking.
    manifest_path = os.path.join(args.save_dir, f"{args.save_name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"dataset={args.dataset} device={args.device}")
    print(f"num_batches={args.num_batches} saved_to={args.save_dir}")
    print(f"sample_shapes={first_shapes}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()

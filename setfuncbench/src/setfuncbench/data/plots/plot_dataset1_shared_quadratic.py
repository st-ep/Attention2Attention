from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch


Batch = Dict[str, torch.Tensor]
DATASET_NAME = "dataset1_shared_quadratic"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), DATASET_NAME, "preview.png")


def load_batch_from_pt(path: str) -> Batch:
    """
    Load a generated .pt file and return the batch dict.

    Supports either of:
      - payload format: {"dataset_cfg": ..., "batch": {...}}
      - raw batch dict: {"ctx_x": ..., "ctx_y": ..., ...}
    """
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "batch" in payload and isinstance(payload["batch"], dict):
        batch = payload["batch"]
    elif isinstance(payload, dict):
        batch = payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)}")

    for key in ("ctx_x", "ctx_y", "qry_x", "qry_y"):
        if key not in batch:
            raise KeyError(f"Missing required key '{key}' in loaded batch")
    return batch  # type: ignore[return-value]


def plot_dataset1_shared_quadratic_batch(
    batch: Batch,
    out_path: str,
    sample_index: int = 0,
    num_functions: int = 6,
    cols: int = 3,
    dpi: int = 180,
) -> str:
    """Plot Dataset 1 curves for one sample and save a PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    ctx_x = batch["ctx_x"].detach().cpu().squeeze(-1)  # (B,K,M)
    ctx_y = batch["ctx_y"].detach().cpu().squeeze(-1)  # (B,K,M)
    qry_x = batch["qry_x"].detach().cpu().squeeze(-1)  # (B,K,Q)
    qry_y = batch["qry_y"].detach().cpu().squeeze(-1)  # (B,K,Q)

    if ctx_x.ndim != 3:
        raise ValueError(f"Expected ctx_x shape (B,K,M,1) before squeeze, got {batch['ctx_x'].shape}")

    B, K, _ = ctx_x.shape
    if not (0 <= sample_index < B):
        raise IndexError(f"sample_index={sample_index} out of range for batch size {B}")

    n = max(1, min(int(num_functions), int(K)))
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), sharex=True, sharey=True)
    if hasattr(axes, "flatten"):
        axes_list = list(axes.flatten())
    elif isinstance(axes, list):
        axes_list = axes
    else:
        axes_list = [axes]

    for k in range(n):
        ax = axes_list[k]
        xq = qry_x[sample_index, k]
        yq = qry_y[sample_index, k]
        order = torch.argsort(xq)

        ax.plot(xq[order].numpy(), yq[order].numpy(), lw=1.5, label="query")
        ax.scatter(
            ctx_x[sample_index, k].numpy(),
            ctx_y[sample_index, k].numpy(),
            s=24,
            c="red",
            label="context",
        )
        ax.set_title(f"sample={sample_index} func={k}")

    for i in range(n, len(axes_list)):
        axes_list[i].axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("dataset1_shared_quadratic", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset1_shared_quadratic batch from a .pt file.")
    parser.add_argument("--input", type=str, required=True, help="Path to generated .pt batch file")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_functions", type=int, default=6)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    batch = load_batch_from_pt(args.input)
    out = plot_dataset1_shared_quadratic_batch(
        batch=batch,
        out_path=args.out,
        sample_index=args.sample_index,
        num_functions=args.num_functions,
        cols=args.cols,
        dpi=args.dpi,
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

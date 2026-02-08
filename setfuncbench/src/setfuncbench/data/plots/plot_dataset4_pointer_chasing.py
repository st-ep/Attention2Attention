from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch

from setfuncbench.data.plots.plot_dataset1_shared_quadratic import load_batch_from_pt


Batch = Dict[str, Any]
DATASET_NAME = "dataset4_pointer_chasing"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), DATASET_NAME, "preview.png")


def _get_latent_2d(batch: Batch, key: str) -> torch.Tensor | None:
    latents = batch.get("latents", None)
    if not isinstance(latents, dict):
        return None
    v = latents.get(key, None)
    if not isinstance(v, torch.Tensor):
        return None
    v = v.detach().cpu()
    if v.ndim != 2:
        return None
    return v


def _get_latent_scalar(batch: Batch, key: str) -> Any:
    latents = batch.get("latents", None)
    if not isinstance(latents, dict):
        return None
    v = latents.get(key, None)
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return None
    return v


def plot_dataset4_pointer_chasing_batch(
    batch: Batch,
    out_path: str,
    sample_index: int = 0,
    num_functions: int = 6,
    cols: int = 3,
    dpi: int = 180,
) -> str:
    """Plot Dataset 4 curves for one sample and save a PNG."""
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

    next_index = _get_latent_2d(batch, "next_index")
    t_index = _get_latent_2d(batch, "t_index")
    h_hops = _get_latent_scalar(batch, "H")

    B, K, M = ctx_x.shape
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

    # Context token colors correspond to sentinel x slots:
    # 0:-3 (key), 1:-2 (value), 2:-1 (ptr key), 3:0 (intercept)
    token_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    token_labels = ["ctx:key", "ctx:value", "ctx:ptr_key", "ctx:intercept"]
    cmap = plt.get_cmap("tab10")

    for k in range(n):
        ax = axes_list[k]
        xq = qry_x[sample_index, k]
        yq = qry_y[sample_index, k]
        order = torch.argsort(xq)

        color = "C0"
        next_text = ""
        t_text = ""
        if t_index is not None:
            tid = int(t_index[sample_index, k].item())
            color = cmap(tid % 10)
            t_text = f" t={tid}"
        if next_index is not None:
            nid = int(next_index[sample_index, k].item())
            next_text = f" nxt={nid}"

        ax.plot(xq[order].numpy(), yq[order].numpy(), lw=1.5, c=color, label="query")

        mk = min(M, 4)
        for m in range(mk):
            label = token_labels[m] if k == 0 else None
            ax.scatter(
                [float(ctx_x[sample_index, k, m].item())],
                [float(ctx_y[sample_index, k, m].item())],
                s=30,
                c=[token_colors[m]],
                label=label,
            )

        ax.set_title(f"sample={sample_index} func={k}{next_text}{t_text}")

    for i in range(n, len(axes_list)):
        axes_list[i].axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    title = "dataset4_pointer_chasing"
    if h_hops is not None:
        title += f" (H={h_hops})"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset4_pointer_chasing batch from a .pt file.")
    parser.add_argument("--input", type=str, required=True, help="Path to generated .pt batch file")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_functions", type=int, default=6)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    batch = load_batch_from_pt(args.input)
    out = plot_dataset4_pointer_chasing_batch(
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

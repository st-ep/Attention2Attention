# Dataset 2: Mixture of Curvatures

## Purpose

Dataset 2 introduces hidden clusters of functions with shared curvature plus controlled outliers.
It is designed to stress:
- selective communication (identify who belongs with whom),
- extrapolation (context on left, queries on right),
- robustness (optional outliers that break cluster assumptions).

## Tensor Interface

Standard SetFuncBench interface:
- `ctx_x`, `ctx_y`: `(B, K, M, 1)`
- `qry_x`, `qry_y`: `(B, K, Q, 1)`

## Key Constraints

- `M >= 2` is required (first two context x-values are fixed).
- `K >= 2 * G` is required (at least two functions per group).
- `min_non_outliers_per_group >= 2` is required for group identifiability.
- Domain split must satisfy `0 <= x_ctx_max < x_qry_min <= 1`.

## Generative Process

For each sample:

1. Assign each function to one hidden group `g in {0, ..., G-1}`:
- Balanced mode (`group_imbalance = 0`): roughly equal group counts.
- Imbalanced mode (`group_imbalance > 0`): one group can dominate, but each group still has at least 2 members.

2. Sample outlier mask with probability `p_out` per function, then resample failing samples until each group has enough non-outliers.
- If resampling still fails after `max_outlier_resample_tries`, outliers are disabled for failing samples (safe fallback).

3. Sample group curvatures:
- `a_g ~ N(0, sigma_a^2)` for each group.
- Non-outlier function curvature `a_k = a_{g_k}`.
- Outlier function curvature `a_k ~ N(0, sigma_a_out^2)` independently.

4. Sample raw slopes and enforce within-group zero-sum on non-outliers:
- `b_raw ~ N(0, sigma_b^2)`.
- For each group, subtract group mean over non-outlier members.
- Result: sum of non-outlier `b_k` inside each group is `0`.

5. Sample function intercepts:
- `c_k ~ N(0, sigma_c^2)`.

6. Build context/query x-values with left-right split:
- Context:
  - `ctx_x[..., 0] = 0`
  - `ctx_x[..., 1] = x_ctx_max`
  - remaining context points (if any) uniform in `[0, x_ctx_max]`
- Queries:
  - `qry_x ~ Uniform[x_qry_min, 1]`

7. Evaluate quadratics:
- `y = a_k * x^2 + b_k * x + c_k`
- Add context noise `sigma_y` (and optional query noise `sigma_q`).

8. Randomly permute functions along `K` to present an unordered set.

## Defaults and Parameters

Main parameters:
- `G` (default `3`): number of hidden groups.
- `x_ctx_max` (default `0.2`): right edge of context region.
- `x_qry_min` (default `0.6`): left edge of query region.
- `sigma_y` (default `0.02`): context noise.
- `sigma_q` (default `0.0`): optional query noise.
- `sigma_a` (default `1.0`): group curvature scale.
- `sigma_b` (default `1.0`): slope scale.
- `sigma_c` (default `0.5`): intercept scale.
- `p_out` (default `0.1`): outlier probability.
- `sigma_a_out` (default `sigma_a`): outlier curvature scale.

Robustness/structure controls:
- `min_non_outliers_per_group` (default `2`).
- `max_outlier_resample_tries` (default `50`).
- `group_imbalance` (default `0.0`).

## Returned Latents

The batch includes:
- `g`: `(B, K)` aligned hidden group ids.
- `outlier`: `(B, K)` aligned outlier flags.
- `a_g`: `(B, G)` group curvatures.
- `a_k`: `(B, K)` aligned per-function curvature.
- `b_k`: `(B, K)` aligned per-function slope.
- `c_k`: `(B, K)` aligned per-function intercept.
- `x_ctx_max`, `x_qry_min`: scalar tensors.

Per-function latents are aligned to the final permuted function order.

## Why This Dataset Is Useful

- Forces models to infer latent cluster structure from sparse noisy context.
- Distinguishes non-selective communication from selective routing.
- Adds extrapolation pressure by separating context and query x-ranges.
- Supports outlier stress testing without breaking batch generation.

## Example

```bash
python scripts/train.py \
  --dataset dataset2_mixture_curvatures \
  --model baseline_c_comm_transformer \
  --dataset_param G=4 \
  --dataset_param p_out=0.15 \
  --dataset_param x_ctx_max=0.15 \
  --dataset_param x_qry_min=0.7
```

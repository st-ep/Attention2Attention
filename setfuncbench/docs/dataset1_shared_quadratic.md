# Dataset 1: Shared Quadratic Core

## Purpose

Dataset 1 is the simplest communication benchmark in SetFuncBench.  
Every function in a set shares the same global quadratic shape, but each function has its own intercept offset.

This means:
- Some structure is globally shared across the set (`a`, `b`).
- Some structure is function-specific (`c_k`).
- Good performance benefits from aggregating information across functions, not only fitting each function independently.

## Tensor Interface

The sampled batch follows the standard interface:
- `ctx_x`, `ctx_y`: `(B, K, M, 1)`
- `qry_x`, `qry_y`: `(B, K, Q, 1)`

`B` is batch size, `K` is number of functions per set, `M` is number of context points, `Q` is number of query points.

## Generative Process

For each sample `b`:

1. Sample shared global coefficients:
- `a_b ~ N(0, sigma_a^2)`
- `b_b ~ N(0, sigma_b^2)`

2. For each function `k`, sample function-specific intercept:
- `c_{b,k} ~ N(0, sigma_c^2)`

3. Sample context and query x-values independently:
- `ctx_x ~ Uniform[-1, 1]`
- `qry_x ~ Uniform[-1, 1]`

4. Compute outputs:
- `ctx_y = a * ctx_x^2 + b * ctx_x + c + eps_ctx`
- `qry_y = a * qry_x^2 + b * qry_x + c`
- `eps_ctx ~ N(0, sigma_y^2)` (if `sigma_y > 0`)

5. Randomly permute the `K` functions within each sample to enforce set structure.

## Defaults and Parameters

Dataset parameters are passed through `DatasetConfig.params`.

- `sigma_y` (default `0.01`): context observation noise.
- `sigma_a` (default `1.0`): scale of shared quadratic term.
- `sigma_b` (default `1.0`): scale of shared linear term.
- `sigma_c` (default `0.5`): scale of per-function intercepts.

## Returned Latents

The batch includes `latents` for debugging and analysis:
- `a`: shape `(B,)`, shared quadratic coefficient per sample.
- `b`: shape `(B,)`, shared linear coefficient per sample.
- `c`: shape `(B, K)`, per-function intercepts.

Per-function latents are aligned with the final permuted function order returned in tensors.

## Why This Dataset Is Useful

- Tests whether models exploit shared structure across functions.
- Keeps mechanics simple while still rewarding communication.
- Good first sanity benchmark before moving to selective routing datasets.

## Example

```bash
python scripts/train.py \
  --dataset dataset1_shared_quadratic \
  --model baseline_b_global_pool \
  --dataset_param sigma_y=0.02
```

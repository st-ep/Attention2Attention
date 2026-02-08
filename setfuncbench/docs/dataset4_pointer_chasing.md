# Dataset 4: Key-Value Pointer Chasing

## Purpose

Dataset 4 is a structured retrieval task packaged in the same `(ctx, qry)` function interface.
Each function encodes key/value/pointer/intercept tokens in context.
The target query slope is obtained by following pointers for `H` hops through the set.

This stresses:
- selective retrieval,
- multi-hop message passing,
- robust parsing of typed context tokens.

## Tensor Interface

Standard SetFuncBench interface:
- `ctx_x`, `ctx_y`: `(B, K, M, 1)`
- `qry_x`, `qry_y`: `(B, K, Q, 1)`

## Key Constraint

- `M` must be exactly `4`.
- Context x-values are fixed sentinel positions:
  - `x = -3`: own key token
  - `x = -2`: own value token
  - `x = -1`: pointer token (successor key)
  - `x = 0`: intercept token

## Generative Process

For each sample:

1. Sample per-function latent variables in original index order:
- key `u_k`
- value `v_k`
- intercept `b_k`

2. Construct a random single-cycle successor mapping `next(k)`:
- build random permutation `pi`
- define successor by cycle order in `pi`.

3. Compute `H`-hop target index:
- `t(k) = next^H(k)`.

4. Define query slope:
- `slope_k = v_{t(k)}`.

5. Build context tensors with sentinel x-values and noisy token observations:
- at `x=-3`: observed key (`u_k + noise`)
- at `x=-2`: observed value (`v_k + noise`)
- at `x=-1`: observed successor key (`u_{next(k)} + noise`)
- at `x=0`: observed intercept (`b_k + noise`)

6. Sample query x-values:
- `qry_x ~ Uniform[0, 1]`.

7. Build query targets:
- `qry_y = slope_k * qry_x + b_k`.

8. Randomly permute functions over `K` to present as an unordered set.

## Defaults and Parameters

Core:
- `H` (default `2`): number of pointer hops.
- `sigma_u` (default `1.0`): key scale.
- `sigma_v` (default `1.0`): value scale.
- `sigma_b` (default `0.5`): intercept scale.

Observation noise:
- `sigma_u_obs` (default `0.01`)
- `sigma_v_obs` (default `0.01`)
- `sigma_ptr` (default `0.02`)
- `sigma_b_obs` (default `0.01`)

Key generation:
- `keys_mode` (default `"linspace"`): one of `linspace`, `gaussian`, `bucket`.
- `key_jitter` (default `1e-3`): jitter amount.
- `num_key_buckets` (default `max(2, K//2)`) for `bucket` mode.

## Key Modes

- `linspace` (clean default): well-separated keys, shuffled per sample.
- `gaussian`: keys sampled from Gaussian, more near-collision risk.
- `bucket`: discrete key buckets, intentional collisions.

## Returned Latents

The batch includes aligned debug latents:
- `u`, `v`, `b`, `slope`: `(B, K)`
- `next_index`: `(B, K)` successor index in returned (permuted) order.
- `t_index`: `(B, K)` `H`-hop target index in returned order.
- `H`: scalar integer.
- `keys_mode`: string.

Per-function latents are aligned with the final returned function order.

## Why This Dataset Is Useful

- Strongly differentiates shallow pooling from true multi-hop communication.
- Exposes whether a model can decode token roles from sentinel x-values.
- Allows controlled difficulty via `H`, key mode, and observation noise.

## Example

```bash
python scripts/train.py \
  --dataset dataset4_pointer_chasing \
  --model baseline_c_comm_transformer \
  --M 4 \
  --dataset_param H=3 \
  --dataset_param keys_mode=linspace
```

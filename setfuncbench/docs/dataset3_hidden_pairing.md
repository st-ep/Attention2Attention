# Dataset 3: Hidden Pairing

## Purpose

Dataset 3 creates hidden function pairs where each pair shares the same quadratic shape.
It is designed to test selective "who-to-talk-to" behavior:
- every function has exactly one informative partner,
- non-partner functions are mostly distractors.

## Tensor Interface

Standard SetFuncBench interface:
- `ctx_x`, `ctx_y`: `(B, K, M, 1)`
- `qry_x`, `qry_y`: `(B, K, Q, 1)`

## Key Constraint

- `K` must be even (`K % 2 == 0`) because functions are partitioned into disjoint pairs.

## Generative Process

For each sample:

1. Build a random hidden pairing:
- Draw a random permutation of indices.
- Pair adjacent positions in permutation space: `(0,1), (2,3), ...`.
- Scatter pair ids back to original index positions.

2. Sample pair-level shared coefficients:
- `a_pair ~ N(0, sigma_a^2)`
- `b_pair ~ N(0, sigma_b^2)`

3. Sample function-level intercepts:
- `c_k ~ N(0, sigma_c^2)`

4. For each function `k`, gather pair-shared `a_k`, `b_k` from its pair id.

5. Sample inputs:
- `ctx_x ~ Uniform[-1, 1]`
- `qry_x ~ Uniform[-1, 1]`

6. Evaluate:
- `ctx_y = a_k * ctx_x^2 + b_k * ctx_x + c_k + eps_ctx`
- `qry_y = a_k * qry_x^2 + b_k * qry_x + c_k + eps_qry(optional)`

7. Apply a final random permutation over `K` to hide pair order and enforce set structure.

## Defaults and Parameters

- `sigma_y` (default `0.02`): context noise.
- `sigma_a` (default `1.0`): pair-level quadratic scale.
- `sigma_b` (default `1.0`): pair-level linear scale.
- `sigma_c` (default `0.5`): function-level intercept scale.
- `sigma_q` (default `0.0`): optional query noise.

## Returned Latents

The batch includes:
- `pair_id`: `(B, K)` aligned pair id per function.
- `a_k`: `(B, K)` aligned quadratic coefficient.
- `b_k`: `(B, K)` aligned linear coefficient.
- `c_k`: `(B, K)` aligned intercept.

Per-function latents are aligned with the final permuted function order.

## Why This Dataset Is Useful

- Cleanly separates selective communication from global pooling.
- Rewards attention mechanisms that can discover a single relevant partner.
- Keeps base function family (quadratic) simple while making routing the hard part.

## Example

```bash
python scripts/train.py \
  --dataset dataset3_hidden_pairing \
  --model baseline_c_comm_transformer \
  --batch_size 16 --K 16 --M 2 --Q 64
```

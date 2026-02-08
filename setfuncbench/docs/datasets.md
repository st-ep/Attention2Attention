# Datasets

SetFuncBench studies communication and attention over sets of partially observed functions, where each function can be underdetermined from only its own context points.

All datasets share the same core tensor interface:
- `ctx_x, ctx_y`: `(B, K, M, 1)`
- `qry_x, qry_y`: `(B, K, Q, 1)`

## Detailed Specs

- Dataset 1: [Shared quadratic core](./dataset1_shared_quadratic.md)
- Dataset 2: [Mixture of curvatures](./dataset2_mixture_curvatures.md)
- Dataset 3: [Hidden pairing](./dataset3_hidden_pairing.md)
- Dataset 4: [Key-value pointer chasing](./dataset4_pointer_chasing.md)

## Quick Comparison

### Dataset 1 - Shared quadratic core

Functions in the same set share global quadratic terms while keeping function-specific offsets. This tests whether models can exploit global shared structure.

### Dataset 2 - Mixture of curvatures

Functions belong to hidden groups with shared curvature and constrained slopes, plus optional outliers. Context and query ranges are separated to stress extrapolation and selective grouping.

### Dataset 3 - Hidden pairing

Each function has exactly one partner with shared parameters. The main challenge is selective communication: discover the right partner, ignore others.

### Dataset 4 - Key-value pointer chasing

Context uses sentinel tokens to encode key/value/pointer/intercept fields. Query slope is obtained via `H` pointer hops, stressing retrieval and multi-hop reasoning.

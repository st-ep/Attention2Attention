# Datasets (high-level)

This benchmark studies **communication / attention** over **sets of functions** where each function is individually underdetermined from its own small context set.

All datasets share the same tensor interface:
- `ctx_x, ctx_y`: (B, K, M, 1)
- `qry_x, qry_y`: (B, K, Q, 1)

## Dataset 1 — Shared quadratic core
A set of functions share a global latent structure; each function also has its own offset. Solving well requires aggregating information across functions.

## Dataset 2 — Mixture-of-curvatures quadratics
Functions belong to hidden clusters that share curvature, with an additional within-cluster constraint that makes the cluster jointly identifiable. Context lives on the left of the domain while queries live on the right to stress extrapolation. Outliers can be included.

## Dataset 3 — Hidden pairing
Each function has exactly one partner that shares parameters; only that partner contains the information needed to disambiguate. This stresses selective “who-to-talk-to” behavior.

## Dataset 4 — Key–value pointer chasing
Context contains special sentinel tokens encoding keys, values, pointers, and intercepts. The slope for each function is the value obtained after H pointer-chasing hops through the set. This stresses selective retrieval and multi-hop message passing.

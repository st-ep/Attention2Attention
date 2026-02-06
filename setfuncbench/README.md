# SetFuncBench

A lightweight research codebase for benchmarking **attention/communication** on **sets of partially observed functions**.

This repo provides:
- A unified dataset API: `sample_batch(cfg, device) -> batch dict`
- A unified model API: `model(batch) -> pred_qry_y`
- A minimal trainer with MSE loss
- CLI scripts for train/eval
- Smoke tests for datasets + models

## Install

Recommended (editable install):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

> Note: install PyTorch according to your platform (CPU/CUDA). If `pip install torch` doesn’t match your setup,
> follow the official PyTorch install instructions and then run `pip install -e .[dev]`.

## Quickstart

Train Baseline A on Dataset 1 (end-to-end runnable):

```bash
python scripts/train.py \
  --dataset dataset1_shared_quadratic \
  --model baseline_a_no_talk \
  --exp_name debug_d1_a \
  --device cpu \
  --steps 200 \
  --batch_size 16 --K 16 --M 2 --Q 64 \
  --seed 0
```

Evaluate a checkpoint:

```bash
python scripts/eval.py \
  --ckpt runs/debug_d1_a/checkpoint_last.pt \
  --device cpu \
  --eval_batches 10
```

## Repo philosophy

* Minimal dependencies: `torch`, `numpy`, `pytest`
* Simple, explicit configuration via `argparse` + dataclasses
* Deterministic data generation controlled by `cfg.seed`
* Easy extension via registries:

  * Datasets: `src/setfuncbench/data/registry.py`
  * Models: `src/setfuncbench/models/registry.py`

## Docs

* High-level dataset overview: `docs/datasets.md`

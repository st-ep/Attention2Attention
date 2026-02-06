from __future__ import annotations

from typing import Dict, List, Type

import torch.nn as nn

from setfuncbench.config import ModelConfig
from setfuncbench.models.baseline_a_no_talk import BaselineANoTalk
from setfuncbench.models.baseline_b_global_pool import BaselineBGlobalPool
from setfuncbench.models.baseline_c_comm_transformer import BaselineCCommTransformer


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline_a_no_talk": BaselineANoTalk,
    "baseline_b_global_pool": BaselineBGlobalPool,
    "baseline_c_comm_transformer": BaselineCCommTransformer,
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{cfg.name}'. Available: {list_models()}")
    return MODEL_REGISTRY[cfg.name](cfg)

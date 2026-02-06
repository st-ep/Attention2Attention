from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class DatasetConfig:
    """
    Configuration for sampling a synthetic batch.

    Required tensor shapes:
      ctx_x, ctx_y: (B, K, M, 1)
      qry_x, qry_y: (B, K, Q, 1)
    """
    name: str
    batch_size: int
    K: int
    M: int
    Q: int
    seed: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for constructing a model."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Configuration for the training loop."""
    exp_name: str = "debug"
    run_dir: str = "runs"
    device: str = "cpu"
    seed: int = 0

    steps: int = 1000
    lr: float = 1e-3

    log_every: int = 50
    eval_every: int = 200
    eval_batches: int = 5
    save_every: int = 0  # 0 => only save last


def get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    """Small helper to read typed params with defaults."""
    return params[key] if key in params else default

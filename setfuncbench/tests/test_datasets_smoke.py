import torch

from setfuncbench.config import DatasetConfig
from setfuncbench.data.registry import list_datasets, sample_batch


def test_dataset_shapes_and_determinism_cpu():
    device = torch.device("cpu")

    for name in list_datasets():
        # dataset-specific minimal M
        M = 4 if name == "dataset4_pointer_chasing" else 2

        cfg = DatasetConfig(
            name=name,
            batch_size=2,
            K=6,
            M=M,
            Q=5,
            seed=123,
            params={},  # keep defaults
        )

        b1 = sample_batch(cfg, device=device)
        b2 = sample_batch(cfg, device=device)

        # Determinism under fixed seed: exact match
        for key in ["ctx_x", "ctx_y", "qry_x", "qry_y"]:
            assert torch.equal(b1[key], b2[key]), f"{name}: {key} is not deterministic"
            assert b1[key].dtype == torch.float32
            assert b1[key].device.type == "cpu"

        # Shape checks (redundant with registry asserts, but nice as a smoke test)
        assert b1["ctx_x"].shape == (cfg.batch_size, cfg.K, cfg.M, 1)
        assert b1["ctx_y"].shape == (cfg.batch_size, cfg.K, cfg.M, 1)
        assert b1["qry_x"].shape == (cfg.batch_size, cfg.K, cfg.Q, 1)
        assert b1["qry_y"].shape == (cfg.batch_size, cfg.K, cfg.Q, 1)

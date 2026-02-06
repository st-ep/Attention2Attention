import torch

from setfuncbench.config import DatasetConfig, ModelConfig
from setfuncbench.data.registry import sample_batch
from setfuncbench.models.registry import list_models, create_model


def test_models_forward_backward_smoke():
    device = torch.device("cpu")

    # Use Dataset 1 for a consistent small batch
    dataset_cfg = DatasetConfig(
        name="dataset1_shared_quadratic",
        batch_size=2,
        K=4,
        M=2,
        Q=3,
        seed=0,
        params={},
    )
    batch = sample_batch(dataset_cfg, device=device)

    for model_name in list_models():
        model_cfg = ModelConfig(
            name=model_name,
            params={
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 4,
                "enc_depth": 2,
                "dec_depth": 2,
            },
        )

        model = create_model(model_cfg).to(device)
        pred = model(batch)

        assert pred.shape == batch["qry_y"].shape, f"{model_name} output shape mismatch"

        loss = ((pred - batch["qry_y"]) ** 2).mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), f"{model_name} has no gradients"

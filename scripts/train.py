"""Launch training from a TOML config file."""
import sys
import tomllib

from sgnm.config import DataConfig, TrainConfig
from sgnm.training import train


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_model(name: str, cfg: dict):
    if name == "gnm":
        from sgnm.models import SGNM
        from sgnm.config import ModelConfig
        return SGNM(ModelConfig(**cfg))
    elif name == "equivariant":
        from sgnm.equivariant import EquivariantReactivityModel
        return EquivariantReactivityModel(**cfg)
    else:
        raise ValueError(f"Unknown model: {name}")


def run(config_path: str):
    cfg = load_config(config_path)

    data_config = DataConfig(**cfg.get("data", {}))
    train_config = TrainConfig(**cfg.get("train", {}))

    models = {}
    for name in ("gnm", "equivariant"):
        if name in cfg:
            models[name] = build_model(name, cfg[name])

    if not models:
        print("Error: config must contain [gnm] and/or [equivariant] section")
        sys.exit(1)

    results = train(models=models, data_config=data_config, train_config=train_config)

    for name, r in results.items():
        print(f"\n[{name}] Best val loss: {r.best_val_loss:.4f}, final epoch: {r.final_epoch}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    run(config_path)
else:
    run(sys.argv[1] if len(sys.argv) > 1 else "config.toml")

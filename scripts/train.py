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
        return SGNM(**cfg)
    elif name == "equivariant":
        from sgnm.equivariant import EquivariantReactivityModel
        return EquivariantReactivityModel(**cfg)
    else:
        raise ValueError(f"Unknown model: {name}")


def run(config_path: str):
    cfg = load_config(config_path)

    data_config = DataConfig(**cfg.get("data", {}))
    train_config = TrainConfig(**cfg.get("train", {}))

    for name in ("gnm", "equivariant"):
        if name in cfg:
            model = build_model(name, cfg[name])
            result = train(name, model, data_config, train_config)
            print(f"\n[{name}] Best val loss: {result.best_val_loss:.4f}, "
                  f"final epoch: {result.final_epoch}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    run(config_path)

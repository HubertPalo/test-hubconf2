from pathlib import Path

import torch
import yaml

dependencies = ["torch", "pyyaml"]

CONFIG_PATH = Path(__file__).parent / "repositories.yaml"


def _load_models():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    models = {}

    for repo in config.get("repositories", []):
        for model in repo.get("models", []):
            name = model["hub_name"]

            if name in models:
                raise ValueError(
                    f"Identificador duplicado: {name}"
                )

            models[name] = {
                **model,
                "repository_id": repo["repository_id"],
            }

    return models


def __getattr__(name):
    models = _load_models()

    if name not in models:
        raise AttributeError(
            f"Modelo no registrado en el hub: {name}"
        )

    config = models[name]

    def load(pretrained=True, **kwargs):
        return torch.hub.load(
            config["repository_id"],
            config["entrypoint"],
            pretrained=pretrained,
            **kwargs,
        )

    load.__name__ = name
    load.__doc__ = (
        f"Carga {name} desde {config['repository_id']}."
    )

    return load
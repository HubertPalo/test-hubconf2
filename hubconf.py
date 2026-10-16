from pathlib import Path as _Path

import torch as _torch
import yaml as _yaml

dependencies = ["torch", "pyyaml"]

_CONFIG_PATH = _Path(__file__).parent / "repositories.yaml"


def _load_config():
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


def _make_entrypoint(model_config, repository_id):
    def entrypoint(pretrained=True, **kwargs):
        return _torch.hub.load(
            repository_id,
            model_config["entrypoint"],
            pretrained=pretrained,
            **kwargs,
        )

    entrypoint.__doc__ = (
        f"Load {model_config['hub_name']} "
        f"from {repository_id}."
    )
    return entrypoint


def _register_models():
    config = _load_config()

    for repo in config.get("repositories", []):
        repository_id = repo["repository_id"]
        publisher = repo["publisher"]
        publication_year = repo["publication_year"]
        author = repo["author_in_charge"]
        author_initials = author[author.index("(")+1:author.index(")")]
        base_prefix = publisher + str(publication_year)[-2:] + author_initials

        for model in repo.get("models", []):
            name = f"{base_prefix}_{model['hub_alias']}"

            if not name.isidentifier():
                raise ValueError(
                    f"El nombre {name!r} no es un identificador "
                    "Python válido. Usa guiones bajos en lugar "
                    "de guiones."
                )

            globals()[name] = _make_entrypoint(
                model, repository_id
            )


_register_models()
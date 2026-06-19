# Training utils

from flax import nnx
import orbax.checkpoint as ocp

import os, json
from pathlib import Path
from typing import Any


def save_model(model: nnx.Module, out_dir: os.PathLike):
    """
    orbax model saving boilerplate

    https://flax.readthedocs.io/en/stable/guides/checkpointing.html
    """
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(out_dir, state)


def load_model(
    model_type: type[nnx.Module],
    model_config: tuple[tuple, dict[str, Any]],
    read_dir: os.PathLike
):
    """
    orbax model loading boilerplate

    https://flax.readthedocs.io/en/stable/guides/checkpointing.html
    """

    config_args, config_kwargs = model_config

    # duct tape fix. need mutable state inside lambda, otherwise:
    # flax.errors.TraceContextError: Cannot mutate RngCount from a different trace level
    if "rngs" in config_kwargs:
        config_kwargs["rngs"] = nnx.Rngs(0)

    abstract_model = nnx.eval_shape(
        lambda: model_type(*config_args, **config_kwargs)
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(read_dir, abstract_state)
    model = nnx.merge(graphdef, state)

    return model


def save_job(model: nnx.Module, metrics: dict, job_dir: os.PathLike):
    job_dir = Path(job_dir)

    save_model(model, job_dir/"model")
    with open(job_dir/"metrics.json", "w") as f:
        json.dump(metrics, f)


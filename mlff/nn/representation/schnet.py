import flax.linen as nn
import logging

from typing import Any, Sequence

from mlff.nn.base.sub_module import BaseSubModule
from mlff.nn.stacknet import StackNet
from mlff.nn.layer.schnet_layer import SchNetLayer
from mlff.nn.observable import Energy
from mlff.nn.embed import AtomTypeEmbed, GeometryEmbed

# logging.basicConfig(level=logging.INFO)


def init_schnet(
    prop_keys: dict[str, str],
    F: int = 128,
    n_layer: int = 6,
    embeddings: Sequence[BaseSubModule] | None = None,
    obs: Sequence[BaseSubModule] | None = None,
    schnet_layer_kwargs: dict[str, Any] | None = None,
    geometry_embed_kwargs: dict[str, Any] | None = None,
) -> StackNet:

    layer_arguments = _default_layer_arguments(F=F)
    if schnet_layer_kwargs is not None:
        layer_arguments.update(schnet_layer_kwargs)

    geometry_embed_arguments = _default_geometry_embed_arguments()
    if geometry_embed_kwargs is not None:
        geometry_embed_arguments.update(geometry_embed_kwargs)

    if embeddings is None:
        embeddings = [
            AtomTypeEmbed(num_embeddings=100, features=F, prop_keys=prop_keys)
        ]

    geometry_embedding = GeometryEmbed(prop_keys=prop_keys, **geometry_embed_arguments)

    schnet_layer = [SchNetLayer(**layer_arguments) for _ in range(n_layer)]
    if obs is None:
        obs = [Energy(prop_keys=prop_keys)]

    net = StackNet(
        geometry_embeddings=[geometry_embedding],
        feature_embeddings=embeddings,
        layers=schnet_layer,
        observables=obs,
        prop_keys=prop_keys,
    )
    return net


def _default_geometry_embed_arguments() -> dict[str, list[int] | str | int | float]:
    return {
        "degrees": [0],
        "radial_basis_function": "rbf",
        "n_rbf": 20,
        "r_cut": 5.0,
        "radial_cutoff_fn": "cosine_cutoff_fn",
        "sphc": False,
    }


def _default_layer_arguments(F: int) -> dict[str, list[int] | str | int | float]:
    return {
        "F": F,
        "in2f_features": [F],
        "filter_features": [F, F],
        "f2out_features": [F, F],
    }

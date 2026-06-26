import jax.numpy as jnp
import flax.linen as nn
import e3x

from typing import Any, Optional

from jraph import segment_sum

from mlff.masking.mask import safe_scale
from mlff.nn.base.sub_module import BaseSubModule


class PermanentDipoleVecSparse(BaseSubModule):
    prop_keys: dict | None
    partial_charges: Optional[Any] = None
    module_name: str = "permanent_dipole_vec"

    @nn.compact
    def __call__(self, inputs: dict, *args, **kwargs) -> dict[str, jnp.ndarray]:

        batch_segments = inputs["batch_segments"]  # (num_nodes)
        graph_mask = inputs["graph_mask"]  # (num_graphs)
        positions = inputs["positions"]  # (num_nodes, 3)

        num_graphs = len(graph_mask)

        # Calculate partial charges
        partial_charges = self.partial_charges(inputs)["partial_charges"]

        if positions is None:
            # TODO: do not calculate DipoleVecSparse if there is no positions
            mu_i = 1 * partial_charges[:, None]
        else:
            mu_i = positions * partial_charges[:, None]

        dipole = segment_sum(
            mu_i, segment_ids=batch_segments, num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask[:, None])

        return dict(dipole_vec=dipole_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class TransitionDipoleVecSparse(BaseSubModule):
    prop_keys: dict | None
    partial_charges: Optional[Any] = None
    module_name: str = "transition_dipole_vec"

    @nn.compact
    def __call__(self, inputs: dict, *args, **kwargs) -> dict[str, jnp.ndarray]:

        batch_segments = inputs["batch_segments"]  # (num_nodes)
        graph_mask = inputs["graph_mask"]  # (num_graphs)
        positions = inputs["positions"]  # (num_nodes, 3)

        num_graphs = len(graph_mask)

        # Calculate partial charges
        partial_charges = self.partial_charges(inputs)["partial_charges"]

        if positions is None:
            # TODO: do not calculate DipoleVecSparse if there is no positions
            mu_i = 1 * partial_charges[:, None]
        else:
            mu_i = positions * partial_charges[:, None]

        dipole = segment_sum(
            mu_i, segment_ids=batch_segments, num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask[:, None])

        return dict(dipole_vec=dipole_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class NACsSparse(BaseSubModule):
    prop_keys: dict | None
    partial_charges: Optional[Any] = None
    module_name: str = "nacs_vec"
    output_feature_size: int = 1
    output_degree_max: int = 5

    @nn.compact
    def __call__(self, inputs: dict, *args, **kwargs) -> dict[str, jnp.ndarray]:

        batch_segments = inputs["batch_segments"]  # (num_nodes)
        graph_mask = inputs["graph_mask"]  # (num_graphs)
        positions = inputs["positions"]  # (num_nodes, 3)
        states = inputs["state_descriptors"]  # ????
        features = inputs["x"]  # (num_nodes, num_states, num_features)
        equiv_features = inputs["ev"]  # (num_nodes, num_states, num_tot_m)

        x1, x2 = ((), ())  # equivariant features of state 1, and state 2 respectively

        # TODO: FIXME: Consider calculating a new equivariant feature vector for NACs. The last one in the final layer may, however, have sufficient channel capacity left
        # TODO: efficiently evaluate for each state combination

        nacs_prediction_irreps = e3x.nn.TensorDense(
            features=self.output_feature_size, max_degree=self.output_degree_max
        )(x1, x2)

        # TODO: FIXME: Mix in the representations of the different features of the states

        # TODO: Convert back into vector representation from irreps
        nacs_prediction_vec = e3x.so3.irreps.irreps_to_tensor(
            nacs_prediction_irreps, degree=self.output_degree_max
        )

        # num_graphs = len(graph_mask)

        # # Calculate partial charges
        # partial_charges = self.partial_charges(inputs)["partial_charges"]

        # if positions is None:
        #     # TODO: do not calculate DipoleVecSparse if there is no positions
        #     mu_i = 1 * partial_charges[:, None]
        # else:
        #     mu_i = positions * partial_charges[:, None]

        # dipole = segment_sum(
        #     mu_i, segment_ids=batch_segments, num_segments=num_graphs
        # )  # (num_graphs, 3)

        # dipole_vec = safe_scale(dipole, graph_mask[:, None])

        # return dict(dipole_vec=dipole_vec)
        return dict(nacs=nacs_prediction_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

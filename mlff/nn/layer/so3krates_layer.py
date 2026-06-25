import jax.numpy as jnp
import flax.linen as nn
import jax
from jaxtyping import Float, Int, Bool

from jax.ops import segment_sum
from functools import partial
from typing import Any, Callable, Dict, Sequence
from itertools import chain

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale, safe_mask
from mlff.nn.mlp import MLP, ResidualMLP
from mlff.nn.activation_function.activation_function import silu
from mlff.sph_ops import make_l0_contraction_fn


class So3kratesLayer(BaseSubModule):
    fb_rad_filter_features: Sequence[int]
    gb_rad_filter_features: Sequence[int]
    fb_sph_filter_features: Sequence[int]
    gb_sph_filter_features: Sequence[int]

    degrees: Sequence[int]

    fb_attention: str = "conv_att"
    gb_attention: str = "conv_att"
    fb_filter: str = "radial_spherical"
    gb_filter: str = "radial_spherical"

    residual_mlp_1: bool = False
    residual_mlp_2: bool = False

    num_heads: int = 4

    final_layer: bool = False
    non_local_sphc: bool = False
    non_local_feature: bool = False
    fast_attention_kwargs: Dict | None = None
    chi_cut: float | None = None
    chi_cut_dynamic: bool = False
    parity: bool = True
    layer_normalization: bool = False
    sphc_normalization: bool = False
    neighborhood_normalization: bool = False
    module_name: str = "so3krates_layer"

    def setup(self):
        self.chi_cut_fn = lambda y, *args, **kwargs: jnp.zeros(1)

        if self.chi_cut is not None or self.chi_cut_dynamic is True:
            raise NotImplementedError(
                "Improved version of non-local corrections will come soon. Stay tuned!"
            )

        if self.neighborhood_normalization is True:
            raise DeprecationWarning("Neighborhood normalization is deprecated.")

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        ev: Float[jnp.ndarray, "node sphc_feature"],
        rbf_ij: Float[jnp.ndarray, "pair K"],
        ylm_ij: Float[jnp.ndarray, "pair sphc_feature"],
        cut: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        point_mask: Bool[jnp.ndarray, "node"],
        *args,
        **kwargs,
    ):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            ev (Array): Spherical harmonic coordinates, shape: (n,m_tot)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            ylm_ij (Array): Spherical harmonics from i to j, shape: (n_pairs,m_tot)
            cut (Array): Output of the cutoff function feature block, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            point_mask (Array): index based mask to exclude nodes that come from padding, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """

        num_features = x.shape[-1]
        assert num_features % self.num_heads == 0, (
            "The number of invariant features must be divisible by the number of attention heads to comply with euclidean self-attention requirements"
        )

        # tot_num_heads = self.num_heads + len(self.degrees)
        assert num_features % len(self.degrees) == 0, (
            "The number of invariant features must be divisible by the spherical harmonics degree to comply with spherical self-attention requirements"
        )

        self.sow("record", "ev_in", ev)

        ev_ij = safe_scale(
            jax.vmap(lambda i, j: ev[j] - ev[i])(idx_i, idx_j),
            scale=pair_mask[:, None],
        )  # shape: (P,m_tot)

        contraction_fn = make_l0_contraction_fn(self.degrees, dtype=ev.dtype)
        m_ev_ij = contraction_fn(ev_ij)  # shape: (P,|l|)

        if self.ev_cut_dynamic:
            raise RuntimeError(
                "You should not end up here. Please report to "
                "https://github.com/thorben-frank/mlff/issues"
            )
        else:
            phi_ev_cut = jnp.zeros_like(cut, dtype=cut.dtype)

        # pre layer-normalization
        if self.layer_normalization:
            x_pre_1 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x)
        else:
            x_pre_1 = x

        x_local = FeatureBlock(
            filter=self.fb_filter,
            rad_filter_features=self.fb_rad_filter_features,
            sph_filter_features=self.fb_sph_filter_features,
            attention=self.fb_attention,
            num_heads=self.num_heads,
        )(
            x=x_pre_1,
            rbf_ij=rbf_ij,
            d_ev_ij_l=m_ev_ij,
            cut=cut,
            idx_i=idx_i,
            idx_j=idx_j,
            pair_mask=pair_mask,
        )  # shape: (n,F)

        ev_local = GeometricBlock(
            filter=self.gb_filter,
            rad_filter_features=self.gb_rad_filter_features,
            sph_filter_features=self.gb_sph_filter_features,
            attention=self.gb_attention,
            degrees=self.degrees,
        )(
            ev=ev,
            ylm_ij=ylm_ij,
            x=x_pre_1,
            rbf_ij=rbf_ij,
            d_ev_ij_l=m_ev_ij,
            cut=cut,
            phi_ev_cut=phi_ev_cut,
            idx_i=idx_i,
            idx_j=idx_j,
            pair_mask=pair_mask,
        )  # shape: (n,m_tot)

        if self.non_local_feature:
            raise NotImplementedError
        else:
            x_non_local = jnp.float32(0.0)

        if self.non_local_sphc:
            raise NotImplementedError
        else:
            ev_non_local = jnp.float32(0.0)

        # add local and potential non local features and sphc, respectively and first skip connection
        x_skip_1 = x + x_local + x_non_local
        ev_skip_1 = ev + ev_local + ev_non_local

        if self.residual_mlp_1:
            x_skip_1 = ResidualMLP()(x_skip_1)

        # second pre layer-normalization
        if self.layer_normalization:
            x_pre_2 = safe_mask(
                point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_1
            )
        else:
            x_pre_2 = x_skip_1

        # feature <-> sphc interaction layer
        delta_x, delta_ev = InteractionBlock(self.degrees, parity=self.parity)(
            x_pre_2, ev_skip_1, point_mask
        )

        # second skip connection
        x_skip_2 = x_skip_1 + delta_x
        ev_skip_2 = ev_skip_1 + delta_ev

        if self.residual_mlp_2:
            x_skip_2 = ResidualMLP()(x_skip_2)

        # in the final layer apply post layer-normalization
        if self.final_layer:
            if self.layer_normalization:
                x_skip_2 = safe_mask(
                    point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_2
                )
            else:
                x_skip_2 = x_skip_2

        self.sow("record", "ev_out", ev_skip_2)

        return {"x": x_skip_2, "ev": ev_skip_2}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.module_name: {
                "fb_filter": self.fb_filter,
                "fb_rad_filter_features": self.fb_rad_filter_features,
                "fb_sph_filter_features": self.fb_sph_filter_features,
                "fb_attention": self.fb_attention,
                "gb_filter": self.gb_filter,
                "gb_rad_filter_features": self.gb_rad_filter_features,
                "gb_sph_filter_features": self.gb_sph_filter_features,
                "gb_attention": self.gb_attention,
                "num_heads": self.num_heads,
                "residual_mlp_1": self.residual_mlp_1,
                "residual_mlp_2": self.residual_mlp_2,
                "non_local_sphc": self.non_local_sphc,
                "non_local_feature": self.non_local_feature,
                "fast_attention_kwargs": self.fast_attention_kwargs,
                "ev_cut": self.ev_cut,
                "ev_cut_dynamic": self.ev_cut_dynamic,
                "degrees": self.degrees,
                "parity": self.parity,
                "layer_normalization": self.layer_normalization,
                "sphc_normalization": self.sphc_normalization,
                "neighborhood_normalization": self.neighborhood_normalization,
                "final_layer": self.final_layer,
            }
        }


class FeatureBlock(nn.Module):
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str  # TODO: depracated
    num_heads: int

    def setup(self):
        if self.filter == "radial":
            self.filter_fn = InvariantFilter(
                num_heads=1, features=self.rad_filter_features, activation_fn=silu
            )
        elif self.filter == "radial_spherical":
            self.filter_fn = RadialSphericalFilter(
                rad_num_heads=1,
                rad_features=self.rad_filter_features,
                sph_num_heads=1,
                sph_features=self.sph_filter_features,
                activation_fn=silu,
            )
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        self.attention_fn = ConvAttention(num_heads=self.num_heads)

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        rbf_ij: Float[jnp.ndarray, "pair K"],
        d_ev_ij_l: Float[jnp.ndarray, "pair L"],
        cut: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        *args,
        **kwargs,
    ):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            d_ev_ij_l (Array): Per degree distances of SPHCs, shape: (n_all_pairs,|L|)
            cut (Array): Output of the cutoff function, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = self.filter_fn(rbf=rbf_ij, d_gamma=d_ev_ij_l)  # shape: (n_pairs,F)
        x_ = self.attention_fn(
            x=x,
            w_ij=w_ij,
            cut=cut,
            idx_i=idx_i,
            idx_j=idx_j,
            pair_mask=pair_mask,
        )  # shape: (n,F)
        return x_


class GeometricBlock(nn.Module):
    degrees: Sequence[int]
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str  # TODO: depracated

    def setup(self):
        if self.filter == "radial":
            self.filter_fn = InvariantFilter(
                num_heads=1, features=self.rad_filter_features, activation_fn=silu
            )
        elif self.filter == "radial_spherical":
            self.filter_fn = RadialSphericalFilter(
                rad_num_heads=1,
                rad_features=self.rad_filter_features,
                sph_num_heads=1,
                sph_features=self.sph_filter_features,
                activation_fn=silu,
            )
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        self.attention_fn = SphConvAttention(
            num_heads=len(self.degrees), harmonic_orders=self.degrees
        )

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        ev: Float[jnp.ndarray, "node sphc_feature"],
        rbf_ij: Float[jnp.ndarray, "pair K"],
        ylm_ij: Float[jnp.ndarray, "pair sphc_feature"],
        d_ev_ij_l: Float[jnp.ndarray, "pair L"],
        phi_ev_cut: Float[jnp.ndarray, "pair L"],
        cut: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        *args,
        **kwargs,
    ):
        """

        Args:
            ev (array): spherical coordinates for all orders l, shape: (n,m_tot)
            ylm_ij (array): spherical harmonics for all orders l, shape: (n_all_pairs,n,m_tot)
            x (array): atomic embeddings, shape: (n,F)
            rbf_ij (array): radial basis expansion of distances, shape: (n_pairs,K)
            d_ev_ij_l (array): pairwise distance between spherical coordinates, shape: (n_all_pairs,|L|)
            cut (array): filter cutoff, shape: (n_pairs,L)
            phi_ev_cut (array): cutoff that scales filter values based on distance in Spherical space,
                shape: (n_all_pairs,|L|)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = safe_scale(
            self.filter_fn(rbf=rbf_ij, d_gamma=d_ev_ij_l), scale=pair_mask[:, None]
        )  # shape: (P,F)
        ev_ = self.attention_fn(
            ev=ev,
            ylm_ij=ylm_ij,
            x=x,
            w_ij=w_ij,
            cut=cut,
            phi_ev_cut=phi_ev_cut,
            idx_i=idx_i,
            idx_j=idx_j,
            pair_mask=pair_mask,
        )  # shape: (n,m_tot)
        return ev_  # shape: (n,m_tot)


class InteractionBlock(nn.Module):
    degrees: Sequence[int]
    parity: bool

    def setup(self):
        segment_ids = jnp.array(
            [
                y
                for y in chain(
                    *[[n] * (2 * self.degrees[n] + 1) for n in range(len(self.degrees))]
                )
            ]
        )
        num_segments = len(self.degrees)
        self.v_segment_sum = jax.vmap(
            partial(segment_sum, segment_ids=segment_ids, num_segments=num_segments)
        )

        _repeats = [2 * y + 1 for y in self.degrees]
        self.repeat_fn = partial(
            jnp.repeat,
            repeats=jnp.array(_repeats),
            axis=-1,
            total_repeat_length=sum(_repeats),
        )

        self.contraction_fn = make_l0_contraction_fn(degrees=self.degrees)

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        ev: Float[jnp.ndarray, "node sphc_feature"],
        point_mask: Bool[jnp.ndarray, "node"],
        *args,
        **kwargs,
    ):
        """

        Args:
            x (Array): shape: (n,F)
            ev (Array): shape: (n,m_tot)
            point_mask (Array) shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        F = x.shape[-1]
        nl = len(self.degrees)

        d_ev = self.contraction_fn(ev)  # shape: (n,|l|)

        y = jnp.concatenate([x, d_ev], axis=-1)  # shape: (n,F+|l|)
        a1, b1 = jnp.split(
            MLP(features=[int(F + nl)], activation_fn=silu)(y),
            indices_or_sections=[F],
            axis=-1,
        )
        # shape: (n,F) / shape: (n,n_l) / shape: (n,n_l)
        return a1, self.repeat_fn(b1) * ev


class InvariantFilter(nn.Module):
    num_heads: int
    features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.features[-1] % self.num_heads == 0, (
            f"The number of invariant features ({self.features[-1]}) must be divisible by the number of attention heads ({self.num_heads})"
        )

        f_out = int(self.features[-1] / self.num_heads)
        self._features = [*self.features[:-1], f_out]
        self.filter_fn = nn.vmap(
            MLP,
            in_axes=None,
            out_axes=-2,
            axis_size=self.num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        ev: Float[jnp.ndarray, "node sphc_feature"],
        rbf_ij: Float[jnp.ndarray, "pair K"],
        ylm_ij: Float[jnp.ndarray, "pair order"],
        cut: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        point_mask: Bool[jnp.ndarray, "node"],
        rbf,
        *args,
        **kwargs,
    ):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise geometric features, shape: (...,K)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.filter_fn(self._features, self.activation_fn)(
            rbf
        )  # shape: (...,num_heads,F_head)
        w = w.reshape(*rbf.shape[:-1], -1)  # shape: (...,n,F)
        return w


class RadialSphericalFilter(nn.Module):
    rad_num_heads: int
    rad_features: Sequence[int]
    sph_num_heads: int
    sph_features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.rad_features[-1] % self.rad_num_heads == 0, (
            f"The number of radial features ({self.rad_features[-1]}) must be divisible by the number of radial attention heads ({self.rad_num_heads})"
        )
        assert self.sph_features[-1] % self.sph_num_heads == 0, (
            f"The number of spherical features ({self.sph_features[-1]}) must be divisible by the number of spherical attention heads ({self.sph_num_heads})"
        )

        f_out_rad = int(self.rad_features[-1] / self.rad_num_heads)
        f_out_sph = int(self.sph_features[-1] / self.sph_num_heads)

        self._rad_features = [*self.rad_features[:-1], f_out_rad]
        self._sph_features = [*self.sph_features[:-1], f_out_sph]

        self.rad_filter_fn = nn.vmap(
            MLP,
            in_axes=None,
            out_axes=-2,
            axis_size=self.rad_num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        self.sph_filter_fn = nn.vmap(
            MLP,
            in_axes=None,
            out_axes=-2,
            axis_size=self.sph_num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

    @nn.compact
    def __call__(
        self,
        rbf_ij: Float[jnp.ndarray, "pair K"],
        d_ev_ij_l: Float[jnp.ndarray, "pair L"],
        *args,
        **kwargs,
    ):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise, radial basis expansion, shape: (...,K)
            d_gamma (Array): pairwise distance of spherical coordinates, shape: (...,n_l)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.rad_filter_fn(self._rad_features, self.activation_fn)(
            rbf_ij
        )  # shape: (...,num_heads,F_head)
        w += self.sph_filter_fn(self._sph_features, self.activation_fn)(
            d_ev_ij_l
        )  # shape: (...,num_heads,F_head)
        w = w.reshape(*rbf_ij.shape[:-1], -1)  # shape: (...,n,n,F)
        return w


class ConvAttention(nn.Module):
    num_heads: int

    def setup(self):
        self.coeff_fn = nn.vmap(
            ConvAttentionCoefficients,
            in_axes=(-2, -2, None, None),
            out_axes=-1,
            axis_size=self.num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        self.aggregate_fn = nn.vmap(
            AttentionAggregation,
            in_axes=(-2, -1, None, None),
            out_axes=-2,
            axis_size=self.num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        w_ij: Float[jnp.ndarray, "pair feature"],
        cut: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        *args,
        **kwargs,
    ):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)

        Returns:

        """
        inv_x_head_split, x_heads = equal_head_split(
            x, num_heads=self.num_heads
        )  # shape: (n,num_heads,F_head)
        _, w_heads = equal_head_split(
            w_ij, num_heads=self.num_heads
        )  # shape: (n_pairs,num_heads,F_head)
        alpha = self.coeff_fn()(
            x_heads, w_heads, idx_i, idx_j, pair_mask=pair_mask
        )  # shape: (n_pairs,num_heads)
        alpha = safe_scale(
            alpha, scale=pair_mask[:, None] * cut[:, None]
        )  # shape: (n_pairs,num_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow("record", "alpha", alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_x_head_split(
            self.aggregate_fn()(x_heads, alpha, idx_i, idx_j, pair_mask=pair_mask)
        )  # shape: (n,F)
        return x_


class SphConvAttention(nn.Module):
    num_heads: int
    harmonic_orders: Sequence[int]

    def setup(self):
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(
            jnp.repeat,
            repeats=jnp.array(_repeats),
            axis=-1,
            total_repeat_length=sum(_repeats),
        )
        self.coeff_fn = nn.vmap(
            ConvAttentionCoefficients,
            in_axes=(-2, -2, None, None),
            out_axes=-1,
            axis_size=self.num_heads,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        ev: Float[jnp.ndarray, "node sphc_feature"],
        ylm_ij: Float[jnp.ndarray, "pair sphc_feature"],
        w_ij: Float[jnp.ndarray, "pair feature"],
        cut: Float[jnp.ndarray, "pair"],
        phi_ev_cut: Float[jnp.ndarray, "pair L"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
        *args,
        **kwargs,
    ):
        """

        Args:
            ev (Array): spherical coordinates for all degrees l, shape: (n,m_tot)
            ylm_ij (Array): spherical harmonics for all degrees l, shape: (n_pairs,m_tot)
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            phi_ev_cut (Array): cutoff that scales filter values based on distance in spherical space,
                shape: (n_pairs,n_l)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            args:
            kwargs:

        Returns:

        """

        # number of heads equals number of harmonics, i.e. num_heads = n_l
        inv_x_head_split, x_heads = equal_head_split(
            x, num_heads=self.num_heads
        )  # shape: (n,num_heads,F_head)
        _, w_ij_heads = equal_head_split(
            w_ij, num_heads=self.num_heads
        )  # shape: (n_pairs,num_heads,F_head)
        alpha_ij = self.coeff_fn()(
            x_heads, w_ij_heads, idx_i, idx_j, pair_mask=pair_mask
        )  # shape: (n_pairs,num_heads)
        alpha_r_ij = safe_scale(
            alpha_ij, scale=pair_mask[:, None] * cut[:, None]
        )  # shape: (n_pairs,num_heads)
        alpha_s_ij = safe_scale(
            alpha_ij, scale=pair_mask[:, None] * phi_ev_cut[:, None]
        )  # shape: (n_pairs,num_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,num_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow("record", "alpha_r", alpha_r_ij)
        self.sow("record", "alpha_s", alpha_s_ij)
        self.sow("record", "alpha", alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        ev_ = segment_sum(
            alpha_ij * ylm_ij, segment_ids=idx_i, num_segments=x.shape[0]
        )  # shape: (n,m_tot)
        return ev_


class ConvAttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        w_ij: Float[jnp.ndarray, "pair feature"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
    ):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)

        return (q_i * w_ij * k_j).sum(axis=-1) / jnp.sqrt(x.shape[-1])


class AttentionAggregation(nn.Module):
    @nn.compact
    def __call__(
        self,
        x: Float[jnp.ndarray, "node feature"],
        alpha_ij: Float[jnp.ndarray, "pair"],
        idx_i: Int[jnp.ndarray, "pair"],
        idx_j: Int[jnp.ndarray, "pair"],
        pair_mask: Bool[jnp.ndarray, "pair"],
    ) -> jnp.ndarray:
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            alpha_ij (Array): attention coefficients, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns:

        """

        v_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)
        return segment_sum(
            alpha_ij[:, None] * v_j, segment_ids=idx_i, num_segments=x.shape[0]
        )  # shape: (n,F)


def equal_head_split(x: jnp.ndarray, num_heads: int) -> tuple[Callable, jnp.ndarray]:
    def inv_split(inputs):
        return inputs.reshape(*x.shape[:-1], -1)

    return inv_split, x.reshape(*x.shape[:-1], num_heads, -1)

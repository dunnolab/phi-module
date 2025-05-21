# Adapted from https://github.com/arthurkosmala/EwaldMP

import itertools
import logging
import math
import numpy as np
from scipy.special import binom
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Optional, TypedDict, Union

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

from fairchem.core.common.utils import (
    get_pbc_distances,
    compute_neighbors,
    radius_graph_pbc
)


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str):
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\n"
            f"Actual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor

    name: Optional[str] = None
    index_fn: Optional[IndexFn] = None
    stats: Optional[_Stats] = None

    def __init__(
        self,
        name: Optional[str] = None,
        enforce_consistency: bool = True,
    ):
        super().__init__()

        self.name = name
        self.index_fn = None
        self.stats = None

        self.scale_factor = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        if enforce_consistency:
            self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if not self.fitted:
            return

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue

            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self):
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self):
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: Union[float, torch.Tensor]):
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, float) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: Optional[IndexFn] = None):
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"

        self.stats["variance_in"] = (
            self.stats["variance_in"] / self.stats["n_samples"]
        )
        self.stats["variance_out"] = (
            self.stats["variance_out"] / self.stats["n_samples"]
        )

        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)

        self.set_(value)

        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return

        n_samples = x.shape[0]
        self.stats["variance_out"] += (
            torch.mean(torch.var(x, dim=0)).item() * n_samples
        )

        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += (
                torch.mean(torch.var(ref, dim=0)).item() * n_samples
            )
        self.stats["n_samples"] += n_samples

    def forward(
        self,
        x: torch.Tensor,
        *,
        ref: Optional[torch.Tensor] = None,
    ):
        if self.index_fn is not None:
            self.index_fn()

        if self.fitted:
            x = x * self.scale_factor

        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        return x
    

def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data = tensor.data * (1 / fan_in) ** 0.5

    return tensor
    

class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)
    

class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = torch.nn.SiLU()
            # self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x
    

class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs
    ):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs
                )
                for _ in range(nLayers)
            ]
        )
        # self.inv_sqrt_2 = 1 / math.sqrt(2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.dense_mlp:
            layer.reset_parameters()

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        # x = x * self.inv_sqrt_2
        return x
    

class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(
            -(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled))
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(
                np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return (
            self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)
        )


class RadialBasis(torch.nn.Module):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)  # (nEdges, num_radial)


class EwaldBlock(torch.nn.Module):
    """
    Long-range block from the Ewald message passing method

    Parameters
    ----------
        shared_downprojection: Dense,
            Downprojection block in Ewald block update function,
            shared between subsequent Ewald Blocks.
        emb_size_atom: int
            Embedding size of the atoms.
        downprojection_size: int
            Dimension of the downprojection bottleneck
        num_hidden: int
            Number of residual blocks in Ewald block update function.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
        use_pbc: bool
            Set to True if periodic boundary conditions are applied.
        delta_k: float
            Structure factor voxel resolution
            (only relevant if use_pbc == False).
        k_rbf_values: torch.Tensor
            Pre-evaluated values of Fourier space RBF
            (only relevant if use_pbc == False).
        return_k_params: bool = True,
            Whether to return k,x dot product and damping function values.
    """

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        activation=None,
        name=None,  # identifier in case a ScalingFactor is applied to Ewald output
        use_pbc: bool = True,
        delta_k: float = None,
        k_rbf_values: torch.Tensor = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.return_k_params = return_k_params

        self.delta_k = delta_k
        self.k_rbf_values = k_rbf_values

        self.down = shared_downprojection
        self.up = Dense(
            downprojection_size, emb_size_atom, activation=None, bias=False
        )
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.ewald_layers = self.get_mlp(
            emb_size_atom, emb_size_atom, num_hidden, activation
        )
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        k: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        # Dot products k^Tx and damping values: need to be computed only once per structure
        # Ewald block in first interaction block gets None as input, therefore computes these
        # values and then passes them on to Ewald blocks in later interaction blocks
        dot: torch.Tensor = None,
        sinc_damping: torch.Tensor = None,
    ):
        hres = self.pre_residual(h)
        # Compute dot products and damping values if not already done so by an Ewald block
        # in a previous interaction block
        if dot == None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)
        if sinc_damping == None:
            if self.use_pbc == False:
                sinc_damping = (
                    torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
                )
                sinc_damping = sinc_damping.expand(-1, k.shape[-2])
            else:
                sinc_damping = 1

        # Compute Fourier space filter from weights
        if self.use_pbc:
            self.kfilter = (
                torch.matmul(self.up.linear.weight, self.down.linear.weight)
                .T.unsqueeze(0)
                .expand(num_batch, -1, -1)
            )
        else:
            self.k_rbf_values = self.k_rbf_values.to(x.device)
            self.kfilter = (
                self.up(self.down(self.k_rbf_values))
                .unsqueeze(0)
                .expand(num_batch, -1, -1)
            )

        # Compute real and imaginary parts of structure factor
        sf_real = hres.new_zeros(
            num_batch, dot.shape[-1], hres.shape[-1]
        ).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )
        sf_imag = hres.new_zeros(
            num_batch, dot.shape[-1], hres.shape[-1]
        ).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )

        # Apply Fourier space filter; scatter back to position space
        h_update = 0.01 * torch.sum(
            torch.index_select(sf_real * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1])
            + torch.index_select(sf_imag * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
            dim=1,
        )

        if self.ewald_scale_sum is not None:
            h_update = self.ewald_scale_sum(h_update, ref=h)

        # Apply update function
        for layer in self.ewald_layers:
            h_update = layer(h_update)

        if self.return_k_params:
            return h_update, dot, sinc_damping
        else:
            return h_update


# Atom-to-atom continuous-filter convolution
class HadamardBlock(torch.nn.Module):
    """
    Aggregate atom-to-atom messages by Hadamard (i.e., component-wise)
    product of embeddings and radial basis functions

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_bf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "hadamard_atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_bf = Dense(
            emb_size_bf, emb_size_atom, activation=None, bias=False
        )
        self.scale_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum"
        )
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.layers = self.get_mlp(
            emb_size_atom, emb_size_atom, nHidden, activation
        )

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, bf, idx_s, idx_t):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]
        h_res = self.pre_residual(h)

        mlp_bf = self.dense_bf(bf)

        x = torch.index_select(h_res, 0, idx_s) * mlp_bf

        x2 = scatter(x, idx_t, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x = self.scale_sum(h, x2)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x


def get_k_index_product_set(num_k_x, num_k_y, num_k_z):
    # Get a box of k-lattice indices around (0,0,0)
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1, dtype=torch.float),
        torch.arange(-num_k_y, num_k_y + 1, dtype=torch.float),
        torch.arange(-num_k_z, num_k_z + 1, dtype=torch.float),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    # Cut the box in half (we will always assume point symmetry)
    k_index_product_set = k_index_product_set[
        k_index_product_set.shape[0] // 2 + 1 :
    ]

    # Amount of k-points
    num_k_degrees_of_freedom = k_index_product_set.shape[0]

    return k_index_product_set, num_k_degrees_of_freedom


def get_k_voxel_grid(k_cutoff, delta_k, num_k_rbf):

    # Get indices for a cube of k-lattice sites containing the cutoff sphere
    num_k = k_cutoff / delta_k
    k_index_product_set, _ = get_k_index_product_set(num_k, num_k, num_k)

    # Orthogonal k-space basis, norm delta_k
    k_cell = torch.tensor(
        [[delta_k, 0, 0], [0, delta_k, 0], [0, 0, delta_k]], dtype=torch.float
    )

    # Translate lattice indices into k-vectors
    k_grid = torch.matmul(k_index_product_set, k_cell)

    # Prune all k-vectors outside the cutoff sphere
    k_grid = k_grid[torch.sum(k_grid**2, dim=-1) <= k_cutoff**2]

    # Probably quite arbitrary, for backwards compatibility with scaling
    # yaml files produced with old Ewald Message Passing code
    k_offset = 0.1 if num_k_rbf <= 48 else 0.25

    # Evaluate a basis of Gaussian RBF on the k-vectors
    k_rbf_values = RadialBasis(
        num_radial=num_k_rbf,
        # Avoids zero or extremely small RBF values (there are k-points until
        # right at the cutoff, where all RBF would otherwise drop to 0)
        cutoff=k_cutoff + k_offset,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
    )(
        torch.linalg.norm(k_grid, dim=-1)
    )  # Tensor of shape (N_k, N_RBF)

    num_k_degrees_of_freedom = k_rbf_values.shape[-1]

    return k_grid, k_rbf_values, num_k_degrees_of_freedom


def pos_svd_frame(data):
    pos = data.pos
    batch = torch.zeros_like(data.z) if data.batch is None else data.batch
    batch_size = int(batch.max()) + 1

    with torch.cuda.amp.autocast(False):
        rotated_pos_list = []
        for i in range(batch_size):
            # Center each structure around mean position
            pos_batch = pos[batch == i]
            pos_batch = pos_batch - pos_batch.mean(0)

            # Rotate each structure into its SVD frame
            # (only can do this if structure has at least 3 atoms,
            # i.e., the position matrix has full rank)
            if pos_batch.shape[0] > 2:
                U, S, V = torch.svd(pos_batch)
                rotated_pos_batch = torch.matmul(pos_batch, V)

            else:
                rotated_pos_batch = pos_batch

            rotated_pos_list.append(rotated_pos_batch)

        pos = torch.cat(rotated_pos_list)

    return pos


def x_to_k_cell(cells):

    cross_a2a3 = torch.cross(cells[:, 1], cells[:, 2], dim=-1)
    cross_a3a1 = torch.cross(cells[:, 2], cells[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cells[:, 0], cells[:, 1], dim=-1)
    vol = torch.sum(cells[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    b1 = 2 * np.pi * cross_a2a3 / vol
    b2 = 2 * np.pi * cross_a3a1 / vol
    b3 = 2 * np.pi * cross_a1a2 / vol

    bcells = torch.stack((b1, b2, b3), dim=1)

    return (bcells, vol[:, 0])


@dataclass
class GraphData:
    """Class to keep graph attributes nicely packaged."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_distance_vec: torch.Tensor
    cell_offsets: torch.Tensor
    offset_distances: torch.Tensor
    neighbors: torch.Tensor
    batch_full: torch.Tensor  # used for GP functionality
    z_full: torch.Tensor  # used for GP functionality
    node_offset: int = 0  # used for GP functionality


class GraphModelMixin:
    """Mixin Model class implementing some general convenience properties and methods."""

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
        use_pbc_single=False,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        use_pbc_single = use_pbc_single or self.use_pbc_single
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is None:
            enforce_max_neighbors_strictly = getattr(
                self, "enforce_max_neighbors_strictly", True
            )

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                if use_pbc_single:
                    (
                        edge_index_per_system,
                        cell_offsets_per_system,
                        neighbors_per_system,
                    ) = list(
                        zip(
                            *[
                                radius_graph_pbc(
                                    data[idx],
                                    cutoff,
                                    max_neighbors,
                                    enforce_max_neighbors_strictly,
                                )
                                for idx in range(len(data))
                            ]
                        )
                    )

                    # atom indexs in the edge_index need to be offset
                    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
                    atom_index_offset[0] = 0
                    edge_index = torch.hstack(
                        [
                            edge_index_per_system[idx] + atom_index_offset[idx]
                            for idx in range(len(data))
                        ]
                    )
                    cell_offsets = torch.vstack(cell_offsets_per_system)
                    neighbors = torch.hstack(neighbors_per_system)
                else:
                    ## TODO this is the original call, but blows up with memory
                    ## using two different samples
                    ## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
                    ## sid='75396' (OC22)
                    edge_index, cell_offsets, neighbors = radius_graph_pbc(
                        data,
                        cutoff,
                        max_neighbors,
                        enforce_max_neighbors_strictly,
                    )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return GraphData(
            edge_index=edge_index,
            edge_distance=edge_dist,
            edge_distance_vec=distance_vec,
            cell_offsets=cell_offsets,
            offset_distances=cell_offset_distances,
            neighbors=neighbors,
            node_offset=0,
            batch_full=data.batch,
            z_full=data.z.long(),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
    
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import logging
from dataclasses import asdict, fields

import torch
import torch.nn as nn 
from torch_geometric.nn import SchNet
from torch_geometric.nn import radius_graph
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_scatter import scatter, scatter_add

# from fairchem.core.models.base import GraphModelMixin
from fairchem.core.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
    compute_neighbors
)

from .ewald_utils import (
    EwaldBlock,
    get_k_voxel_grid,
    get_k_index_product_set,
    pos_svd_frame,
    x_to_k_cell,
    GraphModelMixin,
    # GraphData
)
from .model_utils import ModelOutput
from .gemnet.layers.base_layers import Dense
from .phi_module_utils import AlphaNet, laplacian_matvec, block_diag_sparse

    
class SchNetEwald(SchNet, GraphModelMixin):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.use_phi_module = config.model.use_phi_module
        self.use_pbc = self.config.model.use_pbc
        self.regress_forces = self.config.training.predict_forces

        self.max_neighbors = config.model.max_num_neighbors
        self.use_pbc_single = False
        self.otf_graph = True

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'
 
        super().__init__(hidden_channels=config.model.hidden_features,
                         num_filters=config.model.num_filters,
                         num_interactions=config.model.num_interactions,
                         num_gaussians=config.model.num_gaussians,
                         cutoff=config.model.radius_cutoff,
                         max_num_neighbors=config.model.max_num_neighbors, 
                         *args,
                         **kwargs)
        
        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=self.hidden_channels, k=self.config.training.k_eigenvalues)

        # Parse Ewald hyperparams
        if self.use_pbc:
            # Integer values to define box of k-lattice indices
            self.num_k_x = self.config.model.num_k_x 
            self.num_k_y = self.config.model.num_k_y
            self.num_k_z = self.config.model.num_k_z
            self.delta_k = None
        else:
            self.k_cutoff = self.config.model.k_cutoff
            # Voxel grid resolution
            self.delta_k = self.config.model.delta_k
            # Radial k-filter basis size
            self.num_k_rbf = self.config.model.num_k_rbf
        self.downprojection_size = self.config.model.downprojection_size
        # Number of residuals in update function
        self.num_hidden = self.config.model.num_hidden

        # Initialize k-space structure
        if self.use_pbc:
            # Get the reciprocal lattice indices of included k-vectors
            (
                self.k_index_product_set,
                self.num_k_degrees_of_freedom,
            ) = get_k_index_product_set(
                self.num_k_x,
                self.num_k_y,
                self.num_k_z,
            )
            self.k_rbf_values = None
            self.delta_k = None

        else:
            # Get the k-space voxel and evaluate Gaussian RBF (can be done at
            # initialization time as voxel grid stays fixed for all structures)
            (
                self.k_grid,
                self.k_rbf_values,
                self.num_k_degrees_of_freedom,
            ) = get_k_voxel_grid(
                self.k_cutoff,
                self.delta_k,
                self.num_k_rbf,
            )

        # Downprojection layer, weights are shared among all interaction blocks
        self.down = Dense(
            self.num_k_degrees_of_freedom,
            self.downprojection_size,
            activation=None,
            bias=False,
        )

        self.ewald_blocks = torch.nn.ModuleList(
            [
                EwaldBlock(
                    self.down,
                    self.hidden_channels,  # Embedding size of short-range GNN
                    self.downprojection_size,
                    self.num_hidden,  # Number of residuals in update function
                    activation="silu",
                    use_pbc=self.use_pbc,
                    delta_k=self.delta_k,
                    k_rbf_values=self.k_rbf_values,
                )
                for i in range(self.num_interactions)
            ]
        )

        self.skip_connection_factor = (2.0 + 1.0) ** (-0.5)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.z.long()
        pos = (
            pos_svd_frame(data)
            if not self.use_pbc
            else data.pos
        )
        batch = torch.zeros_like(z) if data.batch is None else data.batch
        batch_size = int(batch.max()) + 1

        data.natoms = torch.bincount(batch)
        data_graph = self.generate_graph(data)
        edge_index, edge_weight, distance_vec, cell_offsets, _, neighbors, _, _, _ = [getattr(data_graph, f.name) for f in fields(data_graph)]

        if self.use_pbc:
            # Compute reciprocal lattice basis of structure
            k_cell, _ = x_to_k_cell(data.cell)
            # Translate lattice indices to k-vectors
            k_grid = torch.matmul(
                self.k_index_product_set.to(batch.device), k_cell
            )
        else:
            k_grid = (
                self.k_grid.to(batch.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

        assert z.dim() == 1 and z.dtype == torch.long

        edge_attr = self.distance_expansion(edge_weight)

        h = self.embedding(z)

        if self.use_phi_module:
            # Eigenbasis projection 
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=edge_weight, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, data.batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        dot = None  # These will be computed in first Ewald block and then passed
        sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
        for i in range(self.num_interactions):
            h_ewald, dot, sinc_damping = self.ewald_blocks[i](
                h,
                pos,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            )

            h_at = 0

            h = self.skip_connection_factor * (
                h
                + self.interactions[i](
                    h, edge_index, edge_weight, edge_attr
                )
                + h_ewald
                + h_at
            )

            if self.use_phi_module:
                alpha = self.alpha_model(h) 

                # Perform spectral projection to accumulate potential and charges
                if i == 0:
                    phi = U @ alpha
                    rho = (U * evals) @ alpha
                else:
                    phi_step = U @ alpha
                    rho_step = (U * evals) @ alpha

                    phi += phi_step
                    rho += rho_step

        if self.use_phi_module:
            # Compute PDE residual
            L_phi = laplacian_matvec(Ls, phi, edge_index_L)
            pde_res = (L_phi - rho).pow(2).mean()

            # Apply optional constraint on net zero charge
            net_charge = torch.abs(scatter_add(rho, data.batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        batch = torch.zeros_like(z) if batch is None else batch
        energy = scatter(h, batch, dim=0, reduce='sum')

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                energy += self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)
       
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return ModelOutput(out=energy, forces=forces)
        else:
            return ModelOutput(out=energy) 

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



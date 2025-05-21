import math
from dataclasses import fields
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from torch_geometric.utils import scatter, get_laplacian
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets

from fairchem.core.common.utils import conditional_grad

from ..ewald_utils import (
    pos_svd_frame,
    get_k_index_product_set,
    get_k_voxel_grid,
    EwaldBlock,
    x_to_k_cell,
    GraphModelMixin
)

from ..gemnet.layers.base_layers import Dense
from ..gemnet.layers.embedding_block import AtomEmbedding

from ..model_utils import ModelOutput
from ..phi_module_utils import laplacian_matvec, block_diag_sparse, AlphaNet


class DimeNetPlusPlusEwald(DimeNetPlusPlus, GraphModelMixin):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.use_pbc = self.config.model.use_pbc
        self.use_phi_module = self.config.model.use_phi_module
        self.regress_forces = self.config.training.predict_forces

        self.max_neighbors = config.model.max_num_neighbors
        self.use_pbc_single = False
        self.otf_graph = True

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'

        super().__init__(hidden_channels=config.model.hidden_channels,
                         out_channels=1,
                         num_blocks=config.model.num_blocks,
                         int_emb_size=config.model.int_emb_size,
                         basis_emb_size=config.model.basis_emb_size,
                         out_emb_channels=config.model.out_emb_channels,
                         num_spherical=config.model.num_spherical,
                         num_radial=config.model.num_radial,
                         cutoff=config.model.cutoff,
                         max_num_neighbors=config.model.max_num_neighbors,
                         envelope_exponent=config.model.envelope_exponent,
                         num_before_skip=config.model.num_before_skip,
                         num_after_skip=config.model.num_after_skip,
                         num_output_layers=config.model.num_output_layers,
                         act=config.model.act,
                         output_initializer=config.model.output_initializer,
                         *args, 
                         **kwargs)

        self.hidden_channels = config.model.hidden_channels

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

        # Initialize atom embedding block
        self.atom_emb = AtomEmbedding(self.hidden_channels, num_elements=83)

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
                for i in range(self.num_blocks)
            ]
        )

        self.skip_connection_factor = (
            1.0 + 1.0
        ) ** (-0.5)

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            # In case of DimeNet++ hidden channels equal to 1 due to summation of features in the output blocks
            self.alpha_model = AlphaNet(in_channels=1, k=self.config.training.k_eigenvalues)

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos = (
            pos_svd_frame(data)
            if self.use_pbc
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1

        data.natoms = torch.bincount(batch)
        data_graph = self.generate_graph(data)
        edge_index, dist, _, cell_offsets, offsets, neighbors, _, _, _ = [getattr(data_graph, f.name) for f in fields(data_graph)]

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.neighbors = neighbors
        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data.cell_offsets,
            num_nodes=data.z.size(0),
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

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

        # Embedding block.
        x = self.emb(data.z.long(), rbf, i, j)
      
        # If Ewald MP is used, we have to create atom embeddings borrowing
        # the atomic embedding block from the GemNet architecture
        h = self.atom_emb(data.z.long())
        dot = None  # These will be computed in first Ewald block and then passed
        sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
        pos_detach = pos.detach()

        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=dist, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        # Interaction blocks.
        for block_ind in range(self.num_blocks):
            x = self.interaction_blocks[block_ind](
                x, rbf, sbf, idx_kj, idx_ji
            )

            h_ewald, dot, sinc_damping = self.ewald_blocks[block_ind](
                h,
                pos_detach,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            )

            h_at = 0

            h = self.skip_connection_factor * (h + h_ewald + h_at)
            P = P + self.output_blocks[block_ind + 1](
                x, rbf, i, num_nodes=pos.size(0)
            )

            if self.use_phi_module:
                # Compute eigenbasis coefficients "alpha"
                alpha = self.alpha_model(P) 
                
                # Perform spectral projection to accumulate potential and charges
                if block_ind == 0:
                    self.phi = U @ alpha
                    self.rho = (U * evals) @ alpha
                else:
                    phi_step = U @ alpha
                    rho_step = (U * evals) @ alpha

                    self.phi = self.phi + phi_step
                    self.rho = self.rho + rho_step

        if self.use_phi_module:
            # Compute PDE residual
            L_phi = laplacian_matvec(Ls, self.phi, edge_index_L)
            self.pde_res = (L_phi - self.rho).pow(2).mean()

            # Apply optional constraint on net zero charge
            net_charge = torch.abs(scatter_add(self.rho, batch, dim=0)).sum()
            self.pde_res = self.pde_res + self.config.training.net_charge_lambda * net_charge
        else:
            self.pde_res = None

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (self.phi * self.rho).sum()
                energy = energy + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return ModelOutput(out=energy, forces=forces, pde_residual=self.pde_res)
        else:
            return ModelOutput(out=energy, pde_residual=self.pde_res)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torch_geometric.utils import scatter, get_laplacian
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets

from ..model_utils import ModelOutput
from ..phi_module_utils import laplacian_matvec, block_diag_sparse, AlphaNet


class DimeNetPlusPlusBase(DimeNetPlusPlus):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.use_phi_module = self.config.model.use_phi_module

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

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            # In case of DimeNet++ hidden channels equal to 1 due to summation of features in the output blocks
            self.alpha_model = AlphaNet(in_channels=1, k=self.config.training.k_eigenvalues)

    def forward(self, batch):
        z = batch.z.long()
        pos = batch.pos
        batch = batch.batch

        if self.config.training.predict_forces:
            pos.requires_grad_(True)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk).norm(dim=-1)
        
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=dist, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        # Interaction blocks.
        for block_idx, (interaction_block, output_block) in enumerate(zip(self.interaction_blocks,
                                                   self.output_blocks[1:])):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

            if self.use_phi_module:
                # Compute eigenbasis coefficients "alpha"
                alpha = self.alpha_model(P) 
                
                # Perform spectral projection to accumulate potential and charges
                if block_idx == 0:
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
            net_charge = torch.abs(scatter_add(rho, batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None

        if batch is None:
            out = P.sum(dim=0)
        else:
            out = scatter(P, batch, dim=0, reduce='sum')

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                out = out + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)
        
        if self.config.training.predict_forces:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            forces = torch.autograd.grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]

            if forces is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction."
                )
            return ModelOutput(out=out, forces=-forces, pde_residual=pde_res)
        
        return ModelOutput(out=out, pde_residual=pde_res)
    

import math
import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.utils import get_laplacian
from torch_scatter import scatter, segment_coo, scatter_add

from ..ewald_utils import (
    get_k_index_product_set,
    get_k_voxel_grid,
    get_pbc_distances,
    Dense,
    EwaldBlock,
    pos_svd_frame,
    x_to_k_cell
)

from .utils import get_edge_id, repeat_blocks
from .layers import PaiNNMessage, PaiNNUpdate, PaiNNOutput
from ..gemnet.layers.embedding_block import AtomEmbedding
from ..gemnet.layers.radial_basis import RadialBasis
from ..gemnet.layers.base_layers import ScaledSiLU
from ..scaling import ScaleFactor, load_scales_compat
from ..model_utils import ModelOutput, get_pbc_distances, compute_neighbors, radius_graph_pbc, conditional_grad
from ..phi_module_utils import laplacian_matvec, block_diag_sparse, AlphaNet


class PaiNNEwald(nn.Module):
    def __init__(self, config):
        super(PaiNNEwald, self).__init__()

        self.config = config
        self.use_phi_module = self.config.model.use_phi_module

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'

        self.use_pbc = self.config.model.use_pbc
        self.otf_graph = True
        rbf = {"name": "gaussian"}
        envelope = {"name": "polynomial", "exponent": 5}

        self.hidden_channels = config.model.hidden_channels
        self.num_layers = config.model.num_layers
        self.num_rbf = config.model.num_rbf
        self.cutoff = config.model.cutoff
        self.max_neighbors = config.model.max_neighbors
        self.regress_forces = config.training.predict_forces
        self.direct_forces = config.model.direct_forces
        scale_file = config.model.scale_file
        num_elements = config.model.num_elements

        self.skip_connection_factor = 2.0 ** -0.5

        # Borrowed from GemNet.
        self.symmetric_edge_symmetrization = False

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
                for i in range(self.num_layers)
            ]
            )
        
        self.skip_connection_factor = (
            2.0 + 1.0
        ) ** (-0.5)

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(self.hidden_channels, num_elements)
        
        self.radial_basis = RadialBasis(
            num_radial=self.num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.message_layers.append(
                PaiNNMessage(self.hidden_channels, self.num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(self.hidden_channels))
            setattr(self, "upd_out_scalar_scale_%d" % i, ScaleFactor())

        self.out_energy = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(self.hidden_channels // 2, 1),
        )

        if self.regress_forces is True and self.direct_forces is True:
            self.out_forces = PaiNNOutput(self.hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=self.hidden_channels, k=self.config.training.k_eigenvalues)

        self.reset_parameters()

        load_scales_compat(self, scale_file)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)

    # Borrowed from GemNet.
    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    # Borrowed from GemNet.
    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        batch_idx,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        If `symmetric_edge_symmetrization` is False,
        we only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        If `symmetric_edge_symmetrization` is True,
        we always use both directions.
        """
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat(
                [edge_index, edge_index.flip(0)],
                dim=1,
            )
            cell_offsets_bothdir = torch.cat(
                [cell_offsets, -cell_offsets],
                dim=0,
            )

            # Filter for unique edges
            edge_ids = get_edge_id(
                edge_index_bothdir, cell_offsets_bothdir, num_atoms
            )
            unique_ids, unique_inv = torch.unique(
                edge_ids, return_inverse=True
            )
            perm = torch.arange(
                unique_inv.size(0),
                dtype=unique_inv.dtype,
                device=unique_inv.device,
            )
            unique_idx = scatter(
                perm,
                unique_inv,
                dim=0,
                dim_size=unique_ids.shape[0],
                reduce="min",
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]

            # Order by target index
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]

            # Subindex remaining tensors
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]

            # Count edges per image
            # segment_coo assumes sorted edge_index_new[1] and batch_idx
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(
                ones, edge_index_new[1], dim_size=num_atoms
            )
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            # Generate mask
            mask_sep_atoms = edge_index[0] < edge_index[1]
            # Distinguish edges between the same (periodic) atom by ordering the cells
            cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | (
                    (cell_offsets[:, 0] == 0)
                    & (cell_offsets[:, 1] == 0)
                    & (cell_offsets[:, 2] < 0)
                )
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            # Mask out counter-edges
            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(
                2, -1
            )

            # Concatenate counter-edges after normal edges
            edge_index_cat = torch.cat(
                [edge_index_new, edge_index_new.flip(0)],
                dim=1,
            )

            # Count remaining edges per image
            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device),
                neighbors,
            )
            batch_edge = batch_edge[mask]
            # segment_coo assumes sorted batch_edge
            # Factor 2 since this is only one half of the edges
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(
                ones, batch_edge, dim_size=neighbors.size(0)
            )

            # Create indexing array
            edge_reorder_idx = repeat_blocks(
                torch.div(neighbors_per_image, 2, rounding_mode="floor"),
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            # Reorder everything so the edges of every image are consecutive
            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, False
                )
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, True
                )
                for tensor in reorder_tensors_invneg
            ]

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            edge_index_new.flip(0), -cell_offsets_new, num_atoms
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )
    
    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

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
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, cutoff, max_neighbors
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
            edge_dist = edge_dist.clone() # To fix in-place error

            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    def generate_graph_values(self, data):
        (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(edge_dist, torch.tensor(0.0), atol=1e-6)
        edge_dist[mask_zero] = 1.0e-6
        edge_vector = distance_vec / edge_dist[:, None]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [edge_dist],
            [edge_vector],
            id_swap,
        ) = self.symmetrize_edges(
            edge_index,
            cell_offsets,
            neighbors,
            data.batch,
            [edge_dist],
            [edge_vector],
        )
        
        return (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        )
    
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = (
            pos_svd_frame(data)
            if not self.use_pbc
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1
        z = data.z.long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

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

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=edge_dist, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        #### Interaction blocks ###############################################

        dot = None  # These will be computed in first Ewald block and then passed
        sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
        for i in range(self.num_layers):
            dx_ewald, dot, sinc_damping = self.ewald_blocks[i](
                x, pos, k_grid, batch_size, batch, dot, sinc_damping
            )
           
            dx_at = 0
            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_rbf, edge_vector
            )

            x = x + dx + dx_ewald + dx_at
            vec = vec + dvec
            x = x * self.skip_connection_factor

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

            if self.use_phi_module:
                # Compute eigenbasis coefficients "alpha"
                alpha = self.alpha_model(x) 
                
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
            net_charge = torch.abs(scatter_add(rho, batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0)

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                energy = energy + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)

        if self.regress_forces:
            if self.direct_forces:
                forces = self.out_forces(x, vec)
                return energy, forces
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        x,
                        pos,
                        grad_outputs=torch.ones_like(x),
                        create_graph=True,
                    )[0]
                )
                return ModelOutput(out=energy, forces=forces, pde_residual=pde_res)
        else:
            return ModelOutput(out=energy, pde_residual=pde_res)
    
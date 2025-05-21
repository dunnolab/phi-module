import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

from fairchem.core.common.utils import conditional_grad

from .layers.base_layers import Dense
from .layers.atom_update_block import OutputBlock
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.radial_basis import RadialBasis
from .layers.efficient import EfficientInteractionDownProjection
from .layers.ewald_specific_layers import InteractionBlockTripletsOnly
# from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.spherical_basis import CircularBasisLayer
from .utils import (
    inner_product_normalized, mask_neighbors, ragged_range, repeat_blocks
)

from ..ewald_utils import (
    GraphModelMixin,
    get_k_index_product_set,
    get_pbc_distances,
    get_k_voxel_grid,
    pos_svd_frame,
    x_to_k_cell
)

from ..p3m_utils import Scalar
from ..scaling import ScaleFactor, load_scales_compat
from ..model_utils import ModelOutput, get_pbc_distances, compute_neighbors, radius_graph_pbc
from ..phi_module_utils import laplacian_matvec, block_diag_sparse, AlphaNet


class GemNetTEwald(nn.Module, GraphModelMixin):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.use_pbc = self.config.model.use_pbc
        self.use_phi_module = self.config.model.use_phi_module
        self.regress_forces = self.config.training.predict_forces

        self.use_pbc_single = False
        self.otf_graph = True

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'

        # self.num_targets = num_targets
        assert config.model.num_blocks > 0
        self.num_blocks = config.model.num_blocks
        self.extensive = config.model.extensive

        self.cutoff = config.model.cutoff
        self.max_neighbors = config.model.max_neighbors

        self.regress_forces = config.training.predict_forces
        self.max_neighbors_at = None
        self.distance_expansion_at = None

        # GemNet variants
        self.direct_forces = config.model.direct_forces

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=config.model.num_radial,
            cutoff=config.model.cutoff,
            rbf=config.model.rbf,
            envelope=config.model.envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=config.model.num_radial,
            cutoff=config.model.cutoff,
            rbf=config.model.rbf,
            envelope=config.model.envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            config.model.num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=config.model.cbf,
            efficient=True,
        )

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

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            config.model.num_radial,
            config.model.emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            config.model.num_spherical, config.model.num_radial, config.model.emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            config.model.num_radial,
            config.model.emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            config.model.num_radial,
            config.model.emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(config.model.emb_size_atom, config.model.num_elements)
        self.edge_emb = EdgeEmbedding(
            config.model.emb_size_atom, config.model.num_radial, config.model.emb_size_edge, activation=config.model.activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(config.model.num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=config.model.emb_size_atom,
                    emb_size_edge=config.model.emb_size_edge,
                    emb_size_trip=config.model.emb_size_trip,
                    emb_size_rbf=config.model.emb_size_rbf,
                    emb_size_cbf=config.model.emb_size_cbf,
                    emb_size_bil_trip=config.model.emb_size_bil_trip,
                    num_before_skip=config.model.num_before_skip,
                    num_after_skip=config.model.num_after_skip,
                    num_concat=config.model.num_concat,
                    num_atom=config.model.num_atom,
                    activation=config.model.activation,
                    name=f"IntBlock_{i+1}",
                    use_pbc=self.use_pbc,
                    ewald_downprojection=self.down,
                    downprojection_size=self.downprojection_size,
                    delta_k=self.delta_k,
                    k_rbf_values=self.k_rbf_values,
                )
            )

        for i in range(config.model.num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=config.model.emb_size_atom,
                    emb_size_edge=config.model.emb_size_edge,
                    emb_size_rbf=config.model.emb_size_rbf,
                    nHidden=config.model.num_atom,
                    num_targets=1, # num_targets=num_targets,
                    activation=config.model.activation,
                    output_init=config.model.output_init,
                    direct_forces=self.direct_forces,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_blocks),
            (self.mlp_cbf3.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=config.model.emb_size_atom, k=self.config.training.k_eigenvalues)

        scale_file = config.model.scale_file
        load_scales_compat(self, scale_file)

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

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
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
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
        use_pbc = False
        otf_graph = True

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

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(self, data):
        num_atoms = data.z.size(0)

        (
            edge_index,
            D_st,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -distance_vec / D_st[:, None]

        # Mask interaction edges if required
        select_cutoff = None

        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = (
            pos_svd_frame(data)
            if self.use_pbc
            else data.pos
        )
        batch = data.batch
        batch_size = int(batch.max()) + 1
        atomic_numbers = data.z.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        edge_index_at = None
        edge_weight_at = None
        distance_vec_at = None
        cell_offsets_at = None
        neighbors_at = None
        edge_attr_at = None

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

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

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=D_st, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        dot = (
            None  # These will be computed in first Ewald block and then passed
        )
        sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
        for i in range(self.num_blocks):
            # Interaction block
            h, m, dot, sinc_damping = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                pos=pos,
                k_grid=k_grid,
                batch_size=batch_size,
                batch=batch,
                dot=dot,
                sinc_damping=sinc_damping,
                edge_index_at=edge_index_at,
                edge_weight_at=edge_weight_at,
                edge_attr_at=edge_attr_at,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

            if self.use_phi_module:
                # Compute eigenbasis coefficients "alpha"
                alpha = self.alpha_model(h) 
                
                # Perform spectral projection to accumulate potential and charges
                if i == 0:
                    phi = U @ alpha
                    rho = (U * evals) @ alpha
                else:
                    phi_step = U @ alpha
                    rho_step = (U * evals) @ alpha

                    phi = phi + phi_step
                    rho = rho + rho_step 

        if self.use_phi_module:
            # Compute PDE residual
            L_phi = laplacian_matvec(Ls, phi, edge_index_L)
            pde_res = (L_phi - rho).pow(2).mean()

            # Apply optional constraint on net zero charge
            net_charge = torch.abs(scatter_add(rho, batch, dim=0)).sum()
            pde_res = pde_res + self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                E_t = E_t + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=data.z.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                F_t = F_t.squeeze(1)  # (nAtoms, 3)
            else:
                if self.num_targets > 1:
                    forces = []
                    for i in range(self.num_targets):
                        # maybe this can be solved differently
                        forces += [
                            -torch.autograd.grad(
                                E_t[:, i].sum(), pos, create_graph=True
                            )[0]
                        ]
                    F_t = torch.stack(forces, dim=1)
                    # (nAtoms, num_targets, 3)
                else:
                    F_t = -torch.autograd.grad(
                        E_t.sum(), pos, create_graph=True
                    )[0]
                    # (nAtoms, 3)

            return ModelOutput(out=E_t, forces=forces, pde_residual=pde_res)  # (nMolecules, num_targets), (nAtoms, 3)
        else:
            return ModelOutput(out=E_t, pde_residual=pde_res)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
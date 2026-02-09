#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from data.bioparse import VOCAB

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean

from .diffusion.dpm_full import FullDPM
from ..IterVAE.model import CondIterAutoEncoder
from ..modules.nn import GINEConv, MLP


@R.register('LDMMolDesign')
class LDMMolDesign(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoder = torch.load(autoencoder_ckpt, map_location='cpu', weights_only=False)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        if not hasattr(self.autoencoder, 'training_mode'):
            print(f"[LDM] Warning: Loaded autoencoder does not have 'training_mode'. Defaulting to 'pretrain'.")
            self.autoencoder.training_mode = 'pretrain'
        print(f"[LDM] Loaded autoencoder with training_mode: {self.autoencoder.training_mode}")
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        # Note: Decoder for manifold is set dynamically in forward() with batch context
        
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            atom_positions,     # [Natom], atom order in each block
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        ):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''
        # Check if we need log_var for VAE metric
        need_log_var = self.diffusion.use_semantic_noise and self.diffusion.use_vae_metric

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding based on autoencoder's training_mode
            if need_log_var:
                Zh, Zx, Zh_log_var, Zx_log_var = self._encode_with_autoencoder(
                    X, S, A, atom_positions, bonds, chain_ids, generate_mask, block_lengths, lengths,
                    return_log_var=True
                )
            else:
                Zh, Zx = self._encode_with_autoencoder(
                    X, S, A, atom_positions, bonds, chain_ids, generate_mask, block_lengths, lengths
                )
                Zh_log_var, Zx_log_var = None, None

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # Update decoder for manifold with current batch context (full decoder, not just linear layer)
        if self.diffusion.use_semantic_noise:
            # Create closure that captures current batch context
            def decoder_h_with_context(zh):
                # Use the full decode_block_type decoder
                # zh: [N, latent_size] -> logits: [N, n_block_type]
                return self.autoencoder.decode_block_type(zh, Zx, chain_ids, lengths)
            
            # Pre-compute contexts for X-decoder
            atom_block_ids = length_to_batch_id(block_lengths)
            topo_edges, bond_type = self.autoencoder._get_topo_edges(bonds, atom_block_ids, generate_mask)
            topo_edge_attr = self.autoencoder.atom_edge_embedding(bond_type)
            t_zeros = torch.zeros(len(A), device=X.device)

            def decoder_h_with_context(zh):
                # Use the full decode_block_type decoder
                # zh: [N, latent_size] -> logits: [N, n_block_type]
                return self.autoencoder.decode_block_type(zh, Zx, chain_ids, lengths)
            
            def decoder_x_with_context(zx):
                # For X, use the real decode_structure logic
                # 1. zx (block centers) -> broadcast to atoms
                x_init = zx[atom_block_ids]
                
                # 2. decode structure
                _, x_next = self.autoencoder.decode_structure(
                    Zh, x_init, A, position_ids, 
                    topo_edges, topo_edge_attr, 
                    chain_ids, batch_ids, atom_block_ids, t_zeros
                )
                return x_next
            
            # Aggregator for X (Atoms -> Blocks)
            def aggregator_x(sq_grads):
                # sq_grads: [Natom] -> [Nblock]
                # Sum the squared gradients of all atoms in a block
                from torch_scatter import scatter_sum
                return scatter_sum(sq_grads, atom_block_ids, dim=0, dim_size=Zx.shape[0])
            
            # Set decoders and update metric for the current batch
            self.diffusion.set_decoder_for_manifold(decoder_h_with_context, decoder_x_with_context)
            
            if self.diffusion.manifold_x is not None:
                self.diffusion.manifold_x.set_aggregator(aggregator_x)
                
            if self.diffusion.manifold_h is not None:
                self.diffusion.manifold_h.update_metric(Zh)
            if self.diffusion.manifold_x is not None:
                self.diffusion.manifold_x.update_metric(Zx)

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            Zh_log_var=Zh_log_var,
            Zx_log_var=Zx_log_var,
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']
        
        # Clear local closures to avoid pickling errors
        if self.diffusion.use_semantic_noise:
            self.diffusion.clear_context()

        return loss_dict

    def _encode_with_autoencoder(self, X, S, A, atom_positions, bonds, chain_ids, generate_mask, block_lengths, lengths, return_log_var=False):
        '''
        Encode using the appropriate method based on autoencoder's training_mode.
        This mirrors the logic in IterVAE.generate() to ensure consistency.
        
        Args:
            return_log_var: If True, also return Zh_log_var and Zx_log_var for VAE metric
        '''
        training_mode = self.autoencoder.training_mode
        block_ids = length_to_batch_id(block_lengths)
        Zh_log_var, Zx_log_var = None, None
        
        if training_mode in ('pretrain', 'gan_loss'):
            # Use EPTEncoder only
            if return_log_var:
                Zh_raw, Zx_raw = self.autoencoder.EPTencode(
                    X, S, A, atom_positions, block_lengths, lengths, chain_ids, generate_mask, return_raw=True
                )
                Zx_prior_mu = scatter_mean(X, block_ids, dim=0)
                Zh, Zx, _, _ = self.autoencoder.rsample(Zh_raw, Zx_raw, generate_mask, Zx_prior_mu, deterministic=self.latent_deterministic)
                Zh_log_var = -torch.abs(self.autoencoder.Wh_log_var(Zh_raw))
                Zx_log_var = -torch.abs(self.autoencoder.Wx_log_var(Zh_raw))
                if Zx_log_var.dim() == 2 and Zx_log_var.shape[-1] == 1:
                    Zx_log_var = Zx_log_var.expand(-1, 3)
            else:
                Zh, Zx, _, _, _, _, _, _ = self.autoencoder.EPTencode(
                    X, S, A, atom_positions, block_lengths, lengths, chain_ids, generate_mask, 
                    deterministic=self.latent_deterministic
                )
        elif training_mode in ('feature_align', 'align_gan'):
            # Dual-stream: run both encoders, concatenate and project
            with torch.no_grad():
                Zh_ept_raw, Zx_ept_raw = self.autoencoder.EPTencode(
                    X, S, A, atom_positions, block_lengths, lengths, chain_ids, generate_mask, return_raw=True
                )
            Zh_enc_raw, Zx_enc_raw, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, return_raw=True
            )
            # Dual-stream merge
            Zh_raw, Zx_raw = self.autoencoder._dual_stream_merge(Zh_ept_raw, Zx_ept_raw, Zh_enc_raw, Zx_enc_raw)
            # rsample (deterministic for LDM)
            Zx_prior_mu = scatter_mean(X, block_ids, dim=0)
            Zh, Zx, _, _ = self.autoencoder.rsample(Zh_raw, Zx_raw, generate_mask, Zx_prior_mu, deterministic=self.latent_deterministic)
            if return_log_var:
                Zh_log_var = -torch.abs(self.autoencoder.Wh_log_var(Zh_raw))
                Zx_log_var = -torch.abs(self.autoencoder.Wx_log_var(Zh_raw))
                if Zx_log_var.dim() == 2 and Zx_log_var.shape[-1] == 1:
                    Zx_log_var = Zx_log_var.expand(-1, 3)
        else:  # 'finetune' or default
            if return_log_var:
                Zh_enc_raw, Zx_enc_raw, _, _ = self.autoencoder.encode(
                    X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, return_raw=True
                )
                Zx_prior_mu = scatter_mean(X, block_ids, dim=0)
                Zh, Zx, _, _ = self.autoencoder.rsample(Zh_enc_raw, Zx_enc_raw, generate_mask, Zx_prior_mu, deterministic=self.latent_deterministic)
                Zh_log_var = -torch.abs(self.autoencoder.Wh_log_var(Zh_enc_raw))
                Zx_log_var = -torch.abs(self.autoencoder.Wx_log_var(Zh_enc_raw))
                if Zx_log_var.dim() == 2 and Zx_log_var.shape[-1] == 1:
                    Zx_log_var = Zx_log_var.expand(-1, 3)
            else:
                Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                    X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
                )
        
        if return_log_var:
            return Zh, Zx, Zh_log_var, Zx_log_var
        return Zh, Zx

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            atom_positions,     # [Natom], atom order in each block
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            return_tensor=False,
        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        # ensure there is no data leakage
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # IMPORTANT: During sampling/inference, we DO NOT have the ground truth X for the generation region.
        # If we try to encode the masked X (which is 0), we get garbage log_var.
        # Therefore, we must pass log_var=None to force the diffusion model to use the learned global 'running_stats'.
        # This ensures we use the "Average Semantic Geometry" learned during training.
        
        # NEW: Training-Free Anisotropic - get log_var from VAE encoder for inference
        training_free_anisotropic = sample_opt.pop('training_free_anisotropic', False)
        
        # encoding context based on autoencoder's training_mode
        self.autoencoder.eval()
        if training_free_anisotropic:
            # Get log_var from VAE encoder for training-free anisotropic inference
            Zh, Zx, Zh_log_var, Zx_log_var = self._encode_with_autoencoder(
                X, S, A, atom_positions, bonds, chain_ids, generate_mask, block_lengths, lengths,
                return_log_var=True
            )
            print(f"[Sample] Training-Free Anisotropic enabled: Zh_log_var mean={Zh_log_var.mean():.3f}")
        else:
            Zh, Zx = self._encode_with_autoencoder(
                X, S, A, atom_positions, bonds, chain_ids, generate_mask, block_lengths, lengths
            )
            Zh_log_var, Zx_log_var = None, None

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # MC-CADS: Create decoder functions for both type and structure
        if sample_opt.get('use_mc_cads', False):
            # Pre-compute topo edges from bonds (needed for decode_structure)
            atom_block_ids = length_to_batch_id(block_lengths)
            topo_edges, bond_type = self.autoencoder._get_topo_edges(bonds, atom_block_ids, generate_mask)
            topo_edge_attr = self.autoencoder.atom_edge_embedding(bond_type)
            
            def mc_cads_decoder_factory(autoencoder, A, position_ids, topo_edges, topo_edge_attr, 
                                         chain_ids, batch_ids, atom_block_ids, lengths):
                def decoder_fn(H_t, X_t):
                    """
                    Decode both type logits and structure from current latent states.
                    Returns:
                        type_logits: [Nblock, n_block_type] - block type prediction logits
                        X_next: [Natom, 3] - predicted atom coordinates
                    """
                    # 1. Block type decoding
                    type_logits = autoencoder.decode_block_type(H_t, X_t, chain_ids, lengths)
                    
                    # 2. Structure decoding (atoms from block centers)
                    X_init = X_t[atom_block_ids]  # [Natom, 3]
                    t_zeros = torch.zeros(len(A), device=X_t.device, dtype=torch.long)
                    _, X_next = autoencoder.decode_structure(
                        H_t, X_init, A, position_ids, 
                        topo_edges, topo_edge_attr,
                        chain_ids, batch_ids, atom_block_ids, t_zeros
                    )
                    return type_logits, X_next, atom_block_ids
                return decoder_fn
            
            sample_opt['mc_cads_decoder'] = mc_cads_decoder_factory(
                self.autoencoder, A, position_ids, topo_edges, topo_edge_attr,
                chain_ids, batch_ids, block_ids, lengths
            )
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            Zh_log_var=Zh_log_var,
            Zx_log_var=Zx_log_var,
            training_free_anisotropic=training_free_anisotropic,
            is_aa=is_aa,  # Pass is_aa for MC-CADS structure decomposition
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, atom_positions=atom_positions, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=generate_mask
        )
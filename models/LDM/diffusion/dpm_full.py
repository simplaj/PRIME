import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from tqdm.auto import tqdm

from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

from utils.gnn_utils import variadic_meshgrid, length_to_batch_id
from utils.nn_utils import SinusoidalTimeEmbeddings

from .transition import construct_transition
from .riemannian import create_semantic_manifold, create_semantic_metric
from ...modules.create_net import create_net
from ...modules.nn import MLP


def low_trianguler_inv(L):
    # L: [bs, 3, 3]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(3).unsqueeze(0).expand_as(L).to(L.device), upper=False)
    return L_inv


class EpsilonNet(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            encoder_type='EPT',
            opt={ 'n_layers': 3 }
        ):
        super().__init__()
        
        edge_embed_size = hidden_size // 4
        self.input_mlp = MLP(
            input_size + hidden_size * 2, # latent variable, cond embedding, time embedding
            hidden_size, hidden_size, 3
        )
        self.encoder = create_net(encoder_type, hidden_size, edge_embed_size, opt)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)

    def forward(
            self,
            H_noisy,
            X_noisy,
            cond_embedding,
            edges,
            edge_types,
            generate_mask,
            batch_ids,
            beta,
        ):
        """
        Args:
            H_noisy: (N, hidden_size)
            X_noisy: (N, 3)
            generate_mask: (N)
            batch_ids: (N)
            beta: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 3)
        """
        t_embed = self.time_embedding(beta)
        in_feat = torch.cat([H_noisy, cond_embedding, t_embed], dim=-1)
        in_feat = self.input_mlp(in_feat)
        edge_embed = self.edge_embedding(edge_types)
        block_ids = torch.arange(in_feat.shape[0], device=in_feat.device)
        
        next_H, next_X = self.encoder(in_feat, X_noisy, block_ids, batch_ids, edges, edge_embed)

        # equivariant vector features changes
        eps_X = next_X - X_noisy
        eps_X = torch.where(generate_mask[:, None].expand_as(eps_X), eps_X, torch.zeros_like(eps_X)) 

        # invariant scalar features changes
        next_H = self.hidden2input(next_H)
        eps_H = next_H - H_noisy
        eps_H = torch.where(generate_mask[:, None].expand_as(eps_H), eps_H, torch.zeros_like(eps_H))

        return eps_H, eps_X


class FullDPM(nn.Module):

    def __init__(
        self, 
        latent_size,
        hidden_size,
        num_steps, 
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        encoder_type='EPT',
        trans_pos_opt={}, 
        trans_seq_opt={},
        encoder_opt={},
        # Optional: Semantic-aware noise
        use_semantic_noise=False,
        semantic_noise_opt={},
        # NEW: Use VAE posterior variance instead of Jacobian
        use_vae_metric=False,
    ):
        super().__init__()
        self.eps_net = EpsilonNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.num_steps = num_steps
        self.use_semantic_noise = use_semantic_noise
        self.use_vae_metric = use_vae_metric
        
        # Create transitions (unchanged API)
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)
        
        # Optional: Semantic manifold for H and X
        self.manifold_h = None
        self.manifold_x = None
        self.metric_h = None
        self.metric_x = None
        
        if use_semantic_noise:
            if use_vae_metric:
                # VAE-based semantic metric (stable, no Jacobian)
                # Only apply to H (sequence/type latent), X uses isotropic noise
                self.metric_h = create_semantic_metric(latent_size, semantic_noise_opt, name="H")
                # self.metric_x = create_semantic_metric(3, semantic_noise_opt, name="X")
                self.trans_h.set_semantic_metric(self.metric_h)
                # self.trans_x.set_semantic_metric(self.metric_x)
                print(f"[FullDPM] Using VAE-based semantic metric for noise scaling")
            else:
                # Jacobian-based Riemannian manifold
                self.manifold_h = create_semantic_manifold(latent_size, semantic_noise_opt, name="H")
                self.trans_h.set_manifold(self.manifold_h)
                # self.manifold_x = create_semantic_manifold(3, semantic_noise_opt, name="X")
                # self.trans_x.set_manifold(self.manifold_x)
    
    def set_decoder_for_manifold(self, decoder_fn_h, decoder_fn_x=None):
        """Optional: Set decoder for Jacobian computation.
        
        Args:
            decoder_fn_h: Decoder for H (latent -> block types)
            decoder_fn_x: Decoder for X (if None, X uses isotropic noise)
        """
        if self.manifold_h is not None:
            self.manifold_h.set_decoder(decoder_fn_h)
        if self.manifold_x is not None and decoder_fn_x is not None:
            self.manifold_x.set_decoder(decoder_fn_x)

    def clear_context(self):
        if self.manifold_h:
            self.manifold_h.clear_context()
        if self.manifold_x:
            self.manifold_x.clear_context()

    @torch.no_grad()
    def _get_edges(self, chain_ids, batch_ids, lengths):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = chain_ids[row] == chain_ids[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_types = torch.cat([torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])], dim=0)
        return edges, edge_types
    
    def forward(
            self,
            H_0,                # [Nblock, latent size]
            X_0,                # [Nblock, 3]
            cond_embedding,     # [Nblock, hidden size], conditional embedding
            chain_ids,          # [Nblock]
            generate_mask,      # [Nblock]
            lengths,            # [batch size]
            t=None,
            Zh_log_var=None,    # [Nblock, latent_size] VAE log variance for H
            Zx_log_var=None,    # [Nblock, 3] VAE log variance for X
        ):
        batch_ids = length_to_batch_id(lengths)
        batch_size = batch_ids.max() + 1
        if t == None: # sample time step
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)

        # Update VAE metric running stats during training (H only, X uses isotropic noise)
        if self.use_vae_metric:
            if self.metric_h is not None and Zh_log_var is not None:
                self.metric_h.update_running_stats(Zh_log_var, generate_mask)

        # X: isotropic noise (log_var=None), H: anisotropic noise
        X_noisy, eps_X = self.trans_x.add_noise(X_0, generate_mask, batch_ids, t)
        H_noisy, eps_H = self.trans_h.add_noise(H_0, generate_mask, batch_ids, t, log_var=Zh_log_var)

        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        eps_H_pred, eps_X_pred = self.eps_net(
            H_noisy, X_noisy, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta
        )

        loss_dict = {}

        # equivariant vector feature loss
        loss_X = F.mse_loss(eps_X_pred[generate_mask], eps_X[generate_mask], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
        loss_X = loss_X.sum() / (generate_mask.sum().float() + 1e-8)
        loss_dict['X'] = loss_X

        # invariant scalar feature loss
        loss_H = F.mse_loss(eps_H_pred[generate_mask], eps_H[generate_mask], reduction='none').sum(dim=-1)  # [N]
        loss_H = loss_H.sum() / (generate_mask.sum().float() + 1e-8)
        loss_dict['H'] = loss_H

        return loss_dict

    @torch.no_grad()
    def sample(
            self,
            H,
            X,
            cond_embedding,
            chain_ids,
            generate_mask,
            lengths,
            pbar=False,
            Zh_log_var=None,    # VAE log variance for H
            Zx_log_var=None,    # VAE log variance for X
            training_free_anisotropic=False,  # Enable anisotropic inference without semantic training
            # CADS: Condition-Annealed Diffusion Sampler (minimal params)
            use_cads=False,         # Enable CADS for diversity
            cads_tau1=0.6,          # Start annealing (normalized time)
            cads_tau2=0.9,          # Stop affecting (normalized time)
            cads_noise_scale=0.25,  # Noise magnitude (recommended <= 0.3)
            # Colored Noise: β=0 white, β=1 pink (recommended), β=2 red
            noise_beta=0.0,         # 0=white, 1=pink for sequence correlation
            # Graph Laplacian Noise: structure-aware noise from graph spectrum
            use_graph_laplacian=False,  # Use graph laplacian instead of FFT for colored noise
            # MC-CADS: Multi-Channel Structure-Aware Condition Annealing
            use_mc_cads=False,          # Enable MC-CADS (takes precedence over CADS)
            mc_cads_decoder=None,       # Decoder function: (H_t, X_t) -> logits for stability
            # Thresholds (tau_low, tau_high): gamma = (stability - tau_low) / (tau_high - tau_low)
            # Low thresholds = easy to preserve, high thresholds = needs high stability to preserve
            mc_cads_tau_scaf=(0.3, 0.6),  # Scaffold (context): low -> easy to preserve
            mc_cads_tau_back=(0.4, 0.7),  # Backbone (is_aa=True): medium
            mc_cads_tau_sub=(0.6, 0.9),   # Substituent (is_aa=False): high -> more exploration
            mc_cads_noise_scale=0.25,   # Noise scale s for MC-CADS
            mc_cads_rescale=True,       # Rescale corrupted condition to original mean/std (formula 3)
            mc_cads_mixing_factor=1.0,  # Mixing factor ψ ∈ [0,1]: final = ψ*rescaled + (1-ψ)*corrupted (formula 4)
            is_aa=None,                  # [N] amino acid mask for structure decomposition
        ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 3)
        """
        batch_ids = length_to_batch_id(lengths)

        # Initialize with semantic-aware noise
        # NEW: Training-Free Anisotropic - use log_var directly without trained metrics
        if training_free_anisotropic and Zh_log_var is not None:
            # Normalize by median, then scale
            median_h = torch.median(Zh_log_var, dim=-1, keepdim=True).values if Zh_log_var.dim() > 1 else torch.median(Zh_log_var)
            normalized_h = Zh_log_var - median_h
            std_h = torch.exp(0.5 * normalized_h).clamp(min=0.1, max=3.0)
            # Use pink noise as base if noise_beta > 0
            if noise_beta > 0:
                from .transition import pink_noise_1d
                base_H = pink_noise_1d(H.shape, beta=noise_beta, device=H.device, dtype=H.dtype)
                base_X = pink_noise_1d(X.shape, beta=noise_beta, device=X.device, dtype=X.dtype)
                noise_type = {1.0: 'pink', 2.0: 'red'}.get(noise_beta, f'β={noise_beta}')
                print(f"[Sampling] Training-Free Anisotropic + {noise_type} noise: H scale [{std_h.min():.3f}, {std_h.max():.3f}]")
            else:
                base_H = torch.randn_like(H)
                base_X = torch.randn_like(X)
                print(f"[Sampling] Training-Free Anisotropic: H (scale: [{std_h.min():.3f}, {std_h.max():.3f}], mean: {std_h.mean():.3f})")
            H_rand = base_H * std_h
            X_rand = base_X  # X uses same noise type but no semantic scaling
        elif self.use_semantic_noise:
            if self.use_vae_metric and self.metric_x is not None and self.metric_x.is_initialized:
                print(f"[Sampling] Initializing X with VAE metric (scale mean: {self.metric_x.get_noise_scale().mean():.3f})")
                X_rand = self.metric_x.sample_noise(X, Zx_log_var, generate_mask)
            elif self.manifold_x is not None:
                print(f"[Sampling] Initializing X with Riemannian Prior (G_diag mean: {self.manifold_x.G_diag.mean():.2f})")
                X_rand = self.manifold_x.sample_noise(X)
            else:
                print("[Sampling] Initializing X with Standard Normal")
                X_rand = torch.randn_like(X)
                
            if self.use_vae_metric and self.metric_h is not None and self.metric_h.is_initialized:
                print(f"[Sampling] Initializing H with VAE metric (scale mean: {self.metric_h.get_noise_scale().mean():.3f})")
                H_rand = self.metric_h.sample_noise(H, Zh_log_var, generate_mask)
            elif self.manifold_h is not None:
                print(f"[Sampling] Initializing H with Riemannian Prior (G_diag mean: {self.manifold_h.G_diag.mean():.2f})")
                H_rand = self.manifold_h.sample_noise(H)
            else:
                H_rand = torch.randn_like(H)
        else:
            # Default: use colored noise if noise_beta > 0
            if noise_beta > 0:
                if use_graph_laplacian:
                    from .transition import graph_laplacian_noise
                    noise_type = {1.0: 'pink', 2.0: 'red'}.get(noise_beta, f'β={noise_beta}')
                    print(f"[Sampling] Initializing with Graph-Laplacian {noise_type} noise (β={noise_beta})")
                    X_rand = graph_laplacian_noise(X.shape, batch_ids, beta=noise_beta, device=X.device, dtype=X.dtype)
                    H_rand = graph_laplacian_noise(H.shape, batch_ids, beta=noise_beta, device=H.device, dtype=H.dtype)
                else:
                    from .transition import pink_noise_1d
                    noise_type = {1.0: 'pink', 2.0: 'red'}.get(noise_beta, f'β={noise_beta}')
                    print(f"[Sampling] Initializing with FFT {noise_type} noise (β={noise_beta})")
                    X_rand = pink_noise_1d(X.shape, beta=noise_beta, device=X.device, dtype=X.dtype)
                    H_rand = pink_noise_1d(H.shape, beta=noise_beta, device=H.device, dtype=H.dtype)
            else:
                print("[Sampling] Initializing with Standard Normal")
                X_rand = torch.randn_like(X)
                H_rand = torch.randn_like(H)
        X_init = torch.where(generate_mask[:, None].expand_as(X), X_rand, X)
        H_init = torch.where(generate_mask[:, None].expand_as(H), H_rand, H)

        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

        # CADS logging
        if use_cads:
            print(f"[CADS] Enabled: tau1={cads_tau1}, tau2={cads_tau2}, noise_scale={cads_noise_scale}")
        
        # MC-CADS logging and validation
        if use_mc_cads:
            if mc_cads_decoder is None:
                print("[MC-CADS] Warning: mc_cads_decoder not provided, falling back to CADS")
                use_mc_cads = False
            elif is_aa is None:
                print("[MC-CADS] Warning: is_aa not provided, falling back to CADS")
                use_mc_cads = False
            else:
                print(f"[MC-CADS] Enabled: tau_scaf={mc_cads_tau_scaf}, tau_back={mc_cads_tau_back}, tau_sub={mc_cads_tau_sub}")
        
        # MC-CADS: Store previous decoder output for consecutive step comparison
        prev_X_next = None

        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            
            # Update manifold metric dynamically during sampling
            if self.manifold_h is not None:
                self.manifold_h.update_metric(H_t)
            if self.manifold_x is not None:
                self.manifold_x.update_metric(X_t)

            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
            
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            # MC-CADS: Multi-Channel Structure-Aware Condition Annealing (takes precedence over CADS)
            if use_mc_cads:
                # Compute decoder stability using CURRENT H_t and X_t
                with torch.no_grad():
                    _, X_next, atom_block_ids = mc_cads_decoder(H_t, X_t)
                    
                    # Structure stability: compare consecutive decoder outputs
                    # Small change between steps = stable prediction = high stability
                    if prev_X_next is not None:
                        atom_change = (X_next - prev_X_next).norm(dim=-1)  # [Natom]
                        
                        # Aggregate to block level (mean change per block)
                        from torch_scatter import scatter_mean
                        block_change = scatter_mean(atom_change, atom_block_ids, dim=0, dim_size=H_t.shape[0])
                        
                        # Normalize: small change = high stability
                        # Add tolerance: ignore changes smaller than 0.1A (noise floor) to allow S=1.0
                        block_change = (block_change - 0.1).clamp(min=0.0)
                        
                        struct_scale = 1.5  # Angstrom - relaxed scale to allow higher stability
                        stability = torch.exp(-block_change / struct_scale)  # [Nblock], 0 to 1
                    else:
                        # First step: default to 0 stability (full exploration)
                        stability = torch.zeros(H_t.shape[0], device=H_t.device)
                    
                    # Update prev_X_next for next iteration
                    prev_X_next = X_next.clone()
                
                # Structure masks
                scaf_mask = ~generate_mask  # Context region
                back_mask = generate_mask & is_aa.bool()  # Amino acids in generation
                sub_mask = generate_mask & ~is_aa.bool()  # Non-amino acids in generation
                
                # Compute gamma per channel: high stability -> high gamma (preserve condition)
                def compute_gamma(s, tau):
                    tau_low, tau_high = tau
                    return ((s - tau_low) / (tau_high - tau_low + 1e-8)).clamp(0, 1)
                
                gamma = torch.zeros_like(stability)
                gamma[scaf_mask] = compute_gamma(stability[scaf_mask], mc_cads_tau_scaf)
                gamma[back_mask] = compute_gamma(stability[back_mask], mc_cads_tau_back)
                gamma[sub_mask] = compute_gamma(stability[sub_mask], mc_cads_tau_sub)
                
                # Log stability and gamma statistics (every 10 steps to avoid spam)
                if t % 10 == 0 or t == self.num_steps:
                    log_parts = [f"[MC-CADS t={t}]"]
                    if scaf_mask.any():
                        s = stability[scaf_mask].mean()
                        log_parts.append(f"Scaf: S={s:.3f} γ={gamma[scaf_mask].mean():.3f}")
                    if back_mask.any():
                        s = stability[back_mask].mean()
                        log_parts.append(f"Back: S={s:.3f} γ={gamma[back_mask].mean():.3f}")
                    if sub_mask.any():
                        s = stability[sub_mask].mean()
                        log_parts.append(f"Sub: S={s:.3f} γ={gamma[sub_mask].mean():.3f}")
                    print(" | ".join(log_parts))
                
                # Per-node annealing (formula 1): ŷ = √γ·y + s·√(1-γ)·n
                gamma_exp = gamma.unsqueeze(-1)  # [N, 1]
                noise = torch.randn_like(cond_embedding)
                y_hat = (
                    torch.sqrt(gamma_exp) * cond_embedding +
                    mc_cads_noise_scale * torch.sqrt(1 - gamma_exp) * noise
                )
                
                # Rescaling (formula 3): normalize ŷ back to original mean/std
                if mc_cads_rescale:
                    mu_in = cond_embedding.mean(dim=-1, keepdim=True)
                    sigma_in = cond_embedding.std(dim=-1, keepdim=True) + 1e-8
                    y_hat_rescaled = (y_hat - y_hat.mean(dim=-1, keepdim=True)) / (y_hat.std(dim=-1, keepdim=True) + 1e-8)
                    y_hat_rescaled = y_hat_rescaled * sigma_in + mu_in
                    
                    # Mixing (formula 4): final = ψ·rescaled + (1-ψ)·corrupted
                    cond_embedding_t = mc_cads_mixing_factor * y_hat_rescaled + (1 - mc_cads_mixing_factor) * y_hat
                else:
                    cond_embedding_t = y_hat
            # CADS: Condition-Annealed Sampling (original implementation)
            elif use_cads:
                t_norm = t / self.num_steps  # 1.0 at start, 0.0 at end
                # Piecewise linear schedule: gamma=1 if t<=tau1, gamma=0 if t>=tau2, else linear
                gamma = max(0.0, min(1.0, (cads_tau2 - t_norm) / (cads_tau2 - cads_tau1))) if cads_tau1 < cads_tau2 else 0.0
                if gamma > 0:
                    # Core CADS formula: y' = sqrt(gamma)*y + noise_scale*sqrt(1-gamma)*noise
                    cond_embedding_t = math.sqrt(gamma) * cond_embedding + cads_noise_scale * math.sqrt(1 - gamma) * torch.randn_like(cond_embedding)
                else:
                    cond_embedding_t = cond_embedding
            else:
                cond_embedding_t = cond_embedding

            eps_H, eps_X = self.eps_net(
                H_t, X_t, cond_embedding_t, edges, edge_types, generate_mask, batch_ids, beta
            )

            H_next = self.trans_h.denoise(H_t, eps_H, generate_mask, batch_ids, t_tensor, log_var=Zh_log_var, training_free_anisotropic=training_free_anisotropic, noise_beta=noise_beta)
            # X_next = self.trans_x.denoise(X_t, eps_X, generate_mask, batch_ids, t_tensor, guidance=energy_eps_X, guidance_weight=energy_lambda)
            X_next = self.trans_x.denoise(X_t, eps_X, generate_mask, batch_ids, t_tensor, noise_beta=noise_beta)  # X also uses colored noise

            traj[t-1] = (X_next, H_next)
            traj[t] = (traj[t][0].cpu(), traj[t][1].cpu()) # Move previous states to cpu memory.
        
        return traj
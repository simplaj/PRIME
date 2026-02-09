"""
Riemannian Geometry for Latent Diffusion Models.
Minimal implementation: provides semantic-aware noise sampling.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional


class SemanticManifold(nn.Module):
    """
    Provides semantic-aware noise sampling based on the decoder's geometry.
    Uses the pullback metric G(z) = J^T J to sample anisotropic noise.
    
    Args:
        latent_dim: Dimension of the latent space
        momentum: EMA momentum for metric updates (0.9 = slow update)
        use_diagonal: If True, only use diagonal of G (faster)
        regularization: Small value for numerical stability
        update_freq: Update G every N calls (1=every step, 10=every 10 steps)
        warmup_steps: Number of steps to warmup before using learned G
    """
    
    def __init__(
        self,
        latent_dim: int,
        momentum: float = 0.9,
        use_diagonal: bool = True,
        regularization: float = 1e-4,
        update_freq: int = 10,      # Update G every 10 steps by default
        warmup_steps: int = 100,    # Warmup with isotropic noise
        name: str = "",             # Name for logging (e.g. "H" or "X")
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.momentum = momentum
        self.use_diagonal = use_diagonal
        self.regularization = regularization
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.name = name
        
        # Running average of metric diagonal
        # Initialize to ones (isotropic) - will be updated during warmup
        self.register_buffer('G_diag', torch.ones(latent_dim))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
        # Decoder function (set externally)
        self.decoder_fn: Optional[Callable] = None
        self.aggregator_fn: Optional[Callable] = None
    
    def set_decoder(self, decoder_fn: Callable):
        """Set decoder function for Jacobian computation."""
        self.decoder_fn = decoder_fn

    def set_aggregator(self, aggregator_fn: Callable):
        """Set aggregator function to map output squared grads to input batch size."""
        self.aggregator_fn = aggregator_fn
        
    def clear_context(self):
        """Clear decoder and aggregator to make module picklable."""
        self.decoder_fn = None
        self.aggregator_fn = None
    
    @property
    def is_warmed_up(self) -> bool:
        """Check if warmup is complete."""
        return self.step_count.item() >= self.warmup_steps
    
    @torch.no_grad()
    def update_metric(self, z: Tensor, mask: Optional[Tensor] = None):
        """
        Update metric tensor diagonal with EMA.
        
        Args:
            z: Latent points [N, d]
            mask: Optional boolean mask [N] to select samples for averaging
        """
        if self.decoder_fn is None or not self.training:
            if not self.training and not getattr(self, '_logged_inference_skip', False):
                 print(f"[Riemannian/{self.name}] Inference Mode: Skipping metric update. Using pre-trained G.")
                 self._logged_inference_skip = True
            return
        
        # Only update every update_freq steps
        if self.step_count.item() % self.update_freq != 0:
            return
        
        try:
            # from torch.func import jvp # Removed to avoid C++ scatter incompatibility
            
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            N, d = z.shape
            G_diag_current = torch.zeros(d, device=z.device, dtype=z.dtype)
            epsilon = 1e-3
            
            # Compute G_ii = ||∂f/∂z_i||^2 using Central Finite Differences
            # Robust to non-differentiable ops (like scatter_min)
            # NOTE: We use a serial loop here instead of batching (parallelizing) across dimensions.
            # Reason: The decoder closure captures fixed-size batch context (e.g., graph edges, 
            # block_ids, masks) corresponding to batch size N. Batching perturbations (size N*D) 
            # would require replicating and shifting this complex graph context, which is 
            # computationally expensive and memory-intensive (Graph Batching).
            # vmap() also fails due to C++ scatter operations.
            num_dims = min(d, 32)  # Limit for efficiency
            
            # Pre-compute baseline to ensure valid context
            # f0 = self.decoder_fn(z) 
            
            for i in range(num_dims):
                # Z + eps
                z_plus = z.clone()
                z_plus[:, i] += epsilon
                out_plus = self.decoder_fn(z_plus)
                
                # Z - eps
                z_minus = z.clone()
                z_minus[:, i] -= epsilon
                out_minus = self.decoder_fn(z_minus)
                
                # Jacobian column approximation
                J_col = (out_plus - out_minus) / (2 * epsilon)
                
                # J_col is [N_out, ...output_dims...]
                # Calculate squared norm per sample/element of output
                squared_grads = (J_col.reshape(J_col.shape[0], -1) ** 2).sum(dim=-1) # [N_out]
                
                # Aggregate if necessary (N_out -> N_in)
                if self.aggregator_fn is not None:
                    squared_grads = self.aggregator_fn(squared_grads)
                
                # Validation
                if squared_grads.shape[0] != N:
                     # Fallback: if aggregation failed or not provided but shapes mismatch
                     # We can't safely proceed with per-sample masking unless we take global mean
                     pass 
                
                if mask is not None:
                    # Flatten mask to match N if needed
                    mask_flat = mask.view(-1)
                    if mask_flat.shape[0] == squared_grads.shape[0]:
                        squared_grads = squared_grads[mask_flat]
                    
                G_diag_current[i] = squared_grads.mean()
            
            # Fill remaining with average
            if d > 32:
                avg_val = G_diag_current[:32].mean()
                G_diag_current[32:] = avg_val
            
            # Handle first update (no momentum)
            if self.step_count.item() == 0:
                self.G_diag = G_diag_current.clone()
            else:
                # EMA update
                self.G_diag = self.momentum * self.G_diag + (1 - self.momentum) * G_diag_current
                
            # print(f"[Riemannian/{self.name}] Updated G. Current mean: {self.G_diag.mean().item():.4f}")
                
        except Exception as e:
            print(f"[Riemannian/{self.name}] Metric update failed: {e}")
            import traceback
            traceback.print_exc()
    
    def sample_noise(self, z: Tensor, mask_generate: Optional[Tensor] = None) -> Tensor:
        """
        Sample semantic-aware noise.
        
        Args:
            z: Latent points [N, d] or [N, ...]
            mask_generate: Optional mask for generation
            
        Returns:
            noise: Anisotropic noise, same shape as z
        """
        # Sample standard Gaussian
        eps = torch.randn_like(z)
        
        if self.is_warmed_up:
            # Scale by inverse sqrt of metric diagonal
            # noise_i = eps_i / sqrt(G_ii + reg)
            G_inv_sqrt = 1.0 / torch.sqrt(self.G_diag + self.regularization)
            
            # Broadcast to match z shape
            if z.dim() == 2:
                scaled_eps = eps * G_inv_sqrt.unsqueeze(0)
            else:
                # Handle more complex shapes
                scaled_eps = eps * G_inv_sqrt.view(*([1] * (z.dim() - 1)), -1)
        else:
            scaled_eps = eps
            
        if self.training:
            self.step_count += 1
            if mask_generate is not None:
                # Ensure mask is boolean
                if mask_generate.dtype != torch.bool:
                    mask_valid = mask_generate > 0
                else:
                    mask_valid = mask_generate
                    
                if mask_valid.any():
                    self.update_metric(z, mask=mask_valid)
            
            # Log every 100 steps
            if self.step_count.item() % 100 == 0:
                prefix = f"Riemannian/{self.name}" if self.name else "Riemannian"
                
                # Calculate stats
                g_mean = self.G_diag.mean().item()
                g_max = self.G_diag.max().item()
                g_min = self.G_diag.min().item()
                
                std_iso = eps.std()
                std_semantic = scaled_eps.std()
                ratio = std_semantic / (std_iso + 1e-8)
                ratio_val = ratio.item()
                
                # 1. Console Logging
                print(f"[{prefix}] Step {self.step_count.item()}: G_mean={g_mean:.4f}, G_max={g_max:.4f}, Ratio={ratio_val:.4f}")

                # 2. WandB Visualization
                try:
                    import wandb
                    if wandb.run is not None:
                        # Calculate norms for comparison
                        norms_iso = eps.norm(dim=-1).view(-1)
                        norms_semantic = scaled_eps.norm(dim=-1).view(-1)
                        
                        wandb.log({
                            f"{prefix}_G_mean": g_mean,
                            f"{prefix}_G_max": g_max,
                            f"{prefix}_G_min": g_min,
                            f"{prefix}_noise_std_ratio": ratio_val,
                            f"{prefix}_G_hist": wandb.Histogram(self.G_diag.detach().cpu().numpy()),
                            f"{prefix}_noise_iso_norm_hist": wandb.Histogram(norms_iso.detach().cpu().float().numpy()),
                            f"{prefix}_noise_semantic_norm_hist": wandb.Histogram(norms_semantic.detach().cpu().float().numpy()),
                        }, commit=False)
                except ImportError:
                    pass
        
        return scaled_eps


def create_semantic_manifold(latent_dim: int, opt: dict = {}, name: str = "") -> SemanticManifold:
    """Factory function to create SemanticManifold from config."""
    return SemanticManifold(
        latent_dim=latent_dim,
        momentum=opt.get('momentum', 0.9),
        use_diagonal=opt.get('use_diagonal', True),
        regularization=opt.get('regularization', 1e-4),
        update_freq=opt.get('update_freq', 10),
        warmup_steps=opt.get('warmup_steps', 100),
        name=name,
    )


# ============ VAE-based Semantic Metric (Alternative to Jacobian) ============

class SemanticMetricFromVAE(nn.Module):
    """
    Uses VAE encoder's posterior variance as semantic metric.
    More stable than Jacobian-based approach.
    
    Key insight:
    - VAE encoder predicts q(z|x) = N(μ, σ²)
    - σ² reflects decoder's sensitivity to each latent dimension
    - Low variance → decoder sensitive → use smaller noise
    - High variance → decoder insensitive → can use larger noise
    """
    
    def __init__(
        self,
        latent_dim: int,
        min_scale: float = 0.1,
        max_scale: float = 3.0,
        diversity_factor: float = 1.0,
        momentum: float = 0.9,
        use_running_stats: bool = True,
        name: str = "",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.diversity_factor = diversity_factor
        self.momentum = momentum
        self.use_running_stats = use_running_stats
        self.name = name
        
        self.register_buffer('running_log_var', torch.zeros(latent_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_initialized', torch.tensor(False))
    
    def update_running_stats(self, log_var: Tensor, mask: Optional[Tensor] = None):
        """Update running statistics with new batch of log_var."""
        if not self.training:
            return
        with torch.no_grad():
            if mask is not None:
                log_var = log_var[mask]
            if log_var.numel() == 0:
                return
            batch_mean = log_var.mean(dim=0)
            if not self.is_initialized:
                self.running_log_var.copy_(batch_mean)
                self.is_initialized.fill_(True)
            else:
                self.running_log_var.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            self.num_batches_tracked += 1
    
    def get_noise_scale(self, log_var: Optional[Tensor] = None) -> Tensor:
        """Get noise scaling factor based on variance."""
        if log_var is None:
            log_var = self.running_log_var
        
        # Normalize log_var: subtract median so median → 0 (per-sample)
        median = torch.median(log_var, dim=-1, keepdim=True).values if log_var.dim() > 1 else torch.median(log_var)
        normalized = log_var - median
        
        # Riemannian noise scaling: scale = exp(normalized / 2)
        # < median (sensitive) → scale < 1 (shrink)
        # = median → scale = 1 (standard)
        # > median (insensitive) → scale > 1 (amplify)
        scale = torch.exp(normalized / 2)
        print(f"[SemanticMetric/{self.name}] normalized: [{normalized.min():.3f}, {normalized.max():.3f}] | scale: [{scale.min():.3f}, {scale.max():.3f}]")
        
        # Apply diversity boosting
        # This increases noise in all dimensions, but the effect is absolute:
        # - Sensitive dims (scale=0.1) -> 0.1 * 1.5 = 0.15 (Still protected)
        # - Insensitive dims (scale=1.0) -> 1.0 * 1.5 = 1.5 (Boosted exploration)
        if self.diversity_factor != 1.0:
            scale = scale * self.diversity_factor
            
        return scale.clamp(min=self.min_scale, max=self.max_scale)
    
    def sample_noise(self, z: Tensor, log_var: Optional[Tensor] = None, mask_generate: Optional[Tensor] = None) -> Tensor:
        """Sample semantic-aware noise scaled by VAE variance."""
        eps = torch.randn_like(z)
        scale = None
        
        if log_var is not None:
            scale = self.get_noise_scale(log_var)
        elif self.use_running_stats and self.is_initialized:
            scale = self.get_noise_scale(None).unsqueeze(0).expand_as(z)
            
        if scale is None:
            return eps
            
        scaled_eps = eps * scale
        
        # Logging to WandB (similar to SemanticManifold)
        if self.training and self.num_batches_tracked % 100 == 0:
             try:
                import wandb
                if wandb.run is not None:
                    prefix = f"Riemannian/{self.name}" if self.name else "Riemannian"
                    scale_flat = scale.detach().view(-1)
                    
                    # Calculate stats
                    s_mean = scale_flat.mean().item()
                    s_max = scale_flat.max().item()
                    s_min = scale_flat.min().item()
                    s_std = scale_flat.std().item()
                    
                    # Calculate noise magnitudes
                    mag_orig = eps.abs().mean().item()
                    mag_scaled = scaled_eps.abs().mean().item()
                    
                    # Visualization Data
                    log_dict = {
                        f"{prefix}_scale_mean": s_mean,
                        f"{prefix}_scale_max": s_max,
                        f"{prefix}_scale_min": s_min,
                        f"{prefix}_scale_std": s_std,
                        f"{prefix}_running_log_var_mean": self.running_log_var.mean().item(),
                        f"{prefix}_noise_mag_orig": mag_orig,
                        f"{prefix}_noise_mag_scaled": mag_scaled,
                    }
                    
                    # 1. Distributions (Histograms)
                    log_dict[f"{prefix}_dist_scale"] = wandb.Histogram(scale_flat.cpu().float().numpy())
                    log_dict[f"{prefix}_dist_noise_raw"] = wandb.Histogram(eps.detach().cpu().float().numpy())
                    log_dict[f"{prefix}_dist_noise_scaled"] = wandb.Histogram(scaled_eps.detach().cpu().float().numpy())
                    
                    if log_var is not None:
                         log_dict[f"{prefix}_dist_log_var_input"] = wandb.Histogram(log_var.detach().cpu().float().numpy())

                    # 2. Heatmaps (Vectors) - Visualize first 16 samples
                    # Flatten all dims except batch to visualize as 2D heatmap [Batch, Dim]
                    scale_tensor = scale.detach()
                    if scale_tensor.dim() == 1:
                         scale_tensor = scale_tensor.unsqueeze(0) # [1, D]
                    
                    # Take up to 16 samples
                    n_viz = min(16, scale_tensor.shape[0])
                    # Flatten: [B, D1, D2] -> [B, D1*D2]
                    scale_viz = scale_tensor[:n_viz].flatten(1).float()
                    
                    # Normalize for display (0.0 to 1.0 mapped from min_scale to max_scale)
                    # We simply divide by max_scale to keep relative intensity
                    img_tensor = scale_viz / self.max_scale
                    log_dict[f"{prefix}_viz_scale_heatmap"] = wandb.Image(
                        img_tensor.cpu().numpy(), 
                        caption=f"Noise Scale (First {n_viz} samples). Dark=Sensitive, Bright=Insensitive"
                    )

                    wandb.log(log_dict, commit=False)
                    
                    print(f"[{prefix}] Scale Mean: {s_mean:.4f} | Range: [{s_min:.4f}, {s_max:.4f}] | Std: {s_std:.4f}")
                    print(f"[{prefix}] Noise Mag : {mag_orig:.4f} -> {mag_scaled:.4f} (Reduction: {(1 - mag_scaled/mag_orig)*100:.1f}%)")
             except ImportError:
                pass
             except Exception as e:
                print(f"[Riemannian] Logging failed: {e}")
                
        return scaled_eps


def create_semantic_metric(latent_dim: int, opt: dict = {}, name: str = "") -> SemanticMetricFromVAE:
    """Factory function to create SemanticMetricFromVAE from config."""
    return SemanticMetricFromVAE(
        latent_dim=latent_dim,
        min_scale=opt.get('min_scale', 0.1),
        max_scale=opt.get('max_scale', 3.0),
        diversity_factor=opt.get('diversity_factor', 1.0),
        momentum=opt.get('momentum', 0.9),
        use_running_stats=opt.get('use_running_stats', True),
        name=name,
    )


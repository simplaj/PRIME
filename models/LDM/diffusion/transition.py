import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pink_noise_1d(shape, beta=1.0, device=None, dtype=torch.float32):
    """
    Generate 1/f^β noise along the first dimension (sequence).
    
    For proteins: correlation along residue sequence, independent across features.
    β=0: white, β=1: pink (recommended), β=2: red/brown
    
    Args:
        shape: (N, D) where N is sequence length, D is feature dim
        beta: noise color parameter (1.0 = pink noise)
    """
    if beta == 0:
        return torch.randn(shape, device=device, dtype=dtype)
    
    N = shape[0]
    rest_shape = shape[1:] if len(shape) > 1 else ()
    n_freqs = N // 2 + 1
    
    # Scaling: S(f) ∝ 1/f^β → amplitude ∝ 1/f^(β/2)
    freqs = torch.arange(1, n_freqs, device=device, dtype=dtype)
    scale = freqs ** (-beta / 2)
    
    # Generate shaped noise in frequency domain
    full_shape = (n_freqs,) + rest_shape
    real = torch.randn(full_shape, device=device, dtype=dtype)
    imag = torch.randn(full_shape, device=device, dtype=dtype)
    real[1:] *= scale.view(-1, *([1] * len(rest_shape)))
    imag[1:] *= scale.view(-1, *([1] * len(rest_shape)))
    real[0] = 0  # Zero DC component
    
    # IRFFT along first dimension, normalize per feature
    noise = torch.fft.irfft(torch.complex(real, imag), n=N, dim=0)
    noise = noise / (noise.std(dim=0, keepdim=True) + 1e-8)
    return noise


def graph_laplacian_noise(shape, batch_ids, beta=1.0, device=None, dtype=torch.float32):
    """
    Generate 1/λ^β noise using graph Laplacian on a chain graph (1D lattice).
    
    Uses sequential chain structure (node i connects to i±1 within same batch):
    - Low λ (low freq): global/coordinated changes across sequence
    - High λ (high freq): local/independent changes
    
    Same frequency scaling as pink_noise_1d (1/f^β) but using graph Laplacian framework.
    
    Args:
        shape: output noise shape (N, D)
        batch_ids: [N] batch indices (defines chain boundaries)
        beta: spectral exponent (0=white, 1=pink, 2=red)
    
    Returns:
        noise: [N, D] structure-aware spectral noise
    """
    if device is None:
        device = batch_ids.device
    if beta == 0:
        return torch.randn(shape, device=device, dtype=dtype)
    
    N = shape[0]
    D = shape[1] if len(shape) > 1 else 1
    
    unique_batches = batch_ids.unique()
    noise = torch.zeros(shape, device=device, dtype=dtype)
    
    for b in unique_batches:
        mask = batch_ids == b
        n = mask.sum().item()
        
        if n <= 1:
            noise[mask] = torch.randn(n, D, device=device, dtype=dtype)
            continue
        
        # Build chain adjacency matrix: A[i,i+1] = A[i+1,i] = 1
        A = torch.zeros(n, n, device=device, dtype=dtype)
        idx = torch.arange(n - 1, device=device)
        A[idx, idx + 1] = 1.0
        A[idx + 1, idx] = 1.0
        
        # Graph Laplacian: L = D - A (for chain: tridiagonal)
        D_diag = A.sum(dim=1)  # [2, 2, ..., 2, 1] at boundaries
        L = torch.diag(D_diag) - A
        
        # Spectral decomposition: L = U Λ U^T
        eig_vals, eig_vecs = torch.linalg.eigh(L)
        
        # Energy scaling: S(λ) ∝ 1/λ^β → amplitude ∝ 1/λ^(β/2)
        eig_vals_safe = eig_vals.clamp(min=1e-6)
        scale = eig_vals_safe ** (-beta / 2)
        scale[0] = 0  # Zero the DC mode
        
        # Generate white noise in spectral domain and scale
        xi = torch.randn(n, D, device=device, dtype=dtype)
        xi_spectral = eig_vecs.T @ xi  # Transform to spectral domain
        xi_scaled = scale.unsqueeze(-1) * xi_spectral  # Apply 1/λ^(β/2) scaling
        noise_b = eig_vecs @ xi_scaled  # Transform back
        
        # Normalize to unit variance
        noise_b = noise_b / (noise_b.std() + 1e-8)
        noise[mask] = noise_b
    
    return noise


def construct_transition(_type, num_steps, opt):
    if _type == 'Diffusion':
        return ContinuousTransition(num_steps, opt)
    elif _type == 'FlowMatching':
        return FlowMatchingTransition(num_steps, opt)
    else:
        raise NotImplementedError(f'transition type {_type} not implemented')


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class ContinuousTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
        # Optional: Riemannian manifold for semantic-aware noise
        self.manifold = None
        # Optional: VAE-based semantic metric (alternative to manifold)
        self.semantic_metric = None

    def set_manifold(self, manifold):
        """Optional: Set Riemannian manifold for semantic-aware noise."""
        self.manifold = manifold

    def set_semantic_metric(self, semantic_metric):
        """Optional: Set VAE-based semantic metric for noise scaling."""
        self.semantic_metric = semantic_metric

    def get_timestamp(self, t):
        # use beta as timestamp
        return self.var_sched.betas[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t, log_var=None):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
            log_var: [N, d] optional log variance from VAE encoder
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha_bar = self.var_sched.alpha_bars[t] # [batch_size]
        alpha_bar = alpha_bar[batch_ids]  # [N]

        c0 = torch.sqrt(alpha_bar).view(*expand_shape)
        c1 = torch.sqrt(1 - alpha_bar).view(*expand_shape)

        # Sample noise (VAE metric > Riemannian manifold > isotropic)
        if self.semantic_metric is not None and log_var is not None:
            e_rand = self.semantic_metric.sample_noise(p_0, log_var, mask_generate.squeeze())
        elif self.manifold is not None:
            e_rand = self.manifold.sample_noise(p_0, mask_generate.squeeze())
        else:
            e_rand = torch.randn_like(p_0)
        
        supervise_e_rand = e_rand.clone()
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, supervise_e_rand

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t, guidance=None, guidance_weight=1.0, log_var=None, training_free_anisotropic=False, noise_beta=0.0):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )[batch_ids]
        alpha_bar = self.var_sched.alpha_bars[t][batch_ids]
        sigma = self.var_sched.sigmas[t][batch_ids].view(*expand_shape)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(*expand_shape)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(*expand_shape)

        # Sample noise: choose base noise type (white vs pink/colored)
        if noise_beta > 0:
            base_noise = pink_noise_1d(p_t.shape, beta=noise_beta, device=p_t.device, dtype=p_t.dtype)
        else:
            base_noise = torch.randn_like(p_t)
        
        # Apply semantic scaling if enabled
        if training_free_anisotropic and log_var is not None:
            # Training-Free Anisotropic: normalize log_var by subtracting median, then scale
            median = torch.median(log_var, dim=-1, keepdim=True).values if log_var.dim() > 1 else torch.median(log_var)
            normalized = log_var - median
            std = torch.exp(0.5 * normalized).clamp(min=0.1, max=3.0)
            z_noise = base_noise * std
        elif self.semantic_metric is not None:
            z_noise = self.semantic_metric.sample_noise(p_t, log_var, mask_generate.squeeze())
        elif self.manifold is not None:
            z_noise = self.manifold.sample_noise(p_t, mask_generate.squeeze())
        else:
            z_noise = base_noise
        
        z = torch.where(
            (t > 1).view(*expand_shape).expand_as(p_t),
            z_noise,
            torch.zeros_like(p_t),
        )

        if guidance is not None:
            eps_p = eps_p - torch.sqrt(1 - alpha_bar).view(*expand_shape) * guidance * guidance_weight

        # if guidance is not None:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z + sigma * sigma * guidance_weight * guidance
        # else:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next


# TODO: flow matching (uniform or OT), not done yet
class FlowMatchingTransition(nn.Module):

    def __init__(self, num_steps, opt={}):
        super().__init__()
        self.num_steps = num_steps
        # TODO: number of steps T or T + 1
        c1 = torch.arange(0, num_steps + 1).float() / num_steps
        c0 = 1 - c1
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)

    def get_timestamp(self, t):
        # use c1 as timestamp
        return self.c1[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        c0 = self.c0[t][batch_ids].view(*expand_shape)
        c1 = self.c1[t][batch_ids].view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, (e_rand - p_0)

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        p_next = p_t - eps_p / self.num_steps
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next

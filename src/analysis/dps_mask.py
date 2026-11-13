"""DPS with an arbitrary point-observation mask (for the density sweep)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from Models.sde import get_score, mu_sigma


def _data_score(x_noisy, x_obs, t, model, option, mask, C, sigma_measure, clip=None):
    with torch.enable_grad():
        x_in = x_noisy.clone().detach().requires_grad_(True)
        mu, sigma = mu_sigma(t, option)
        eps = model(x_in, t)
        x0 = (1.0 / mu) * (x_in - sigma * eps)
        pred_obs = x0 * mask
        mse = ((pred_obs - x_obs) ** 2).mean(dim=[1, 2])
        grad = torch.autograd.grad(mse, x_in, grad_outputs=torch.ones_like(mse))[0]
        if clip is not None:
            q = torch.quantile(grad.abs().flatten(), clip)
            grad = torch.clamp(grad, -q, q)
        sigma_scaled = sigma_measure ** 2 + C * (sigma / mu) ** 2
        return (-(0.5 / sigma_scaled) * grad).detach()


def sample_dps_mask(x_init, x_obs, mask, seed, num_samples, diffusion_time_steps,
                    num_corrections, model, corr_scale, C=2e-5,
                    sigma_measure=1e-2, option=0, clip=None):
    torch.manual_seed(seed)
    T = diffusion_time_steps
    N = num_samples
    x_obs = x_obs.repeat([N, 1, 1])
    x = torch.randn_like(x_init.repeat([N, 1, 1]))
    t_i = torch.ones(1, device=x_init.device)
    mu_i, sigma_i = mu_sigma(t_i, option)
    traj = torch.zeros(N, *x_init.shape, T + 1, device=x_init.device)
    traj[..., 0] = x

    with torch.no_grad():
        for step_idx, i in enumerate(range(1, T + 1), start=1):
            t_d = torch.tensor(1 - i / T, device=x_init.device)
            mu_d, sigma_d = mu_sigma(t_d, option)
            r = mu_d / mu_i
            pred = model(x, t_i)
            score = get_score(pred, x, t_i, option) + _data_score(
                x, x_obs, t_i, model, option, mask, C, sigma_measure, clip)
            if i != T:
                x1 = r * x + (r - sigma_d / sigma_i) * (sigma_i ** 2) * score
            else:
                x1 = x + (sigma_i ** 2) * score
            for _ in range(num_corrections):
                z = torch.randn_like(x)
                pred = model(x1, t_d)
                eps = get_score(pred, x1, t_d, option) + _data_score(
                    x1, x_obs, t_d, model, option, mask, C, sigma_measure, clip)
                denom = max(F.mse_loss(eps, torch.zeros_like(eps)), 1e-8)
                delta = corr_scale / denom
                x1 = x1 + delta * eps + torch.sqrt(2 * delta) * z
            x = x1
            traj[..., step_idx] = x
            mu_i, sigma_i, t_i = mu_d, sigma_d, t_d
    return traj

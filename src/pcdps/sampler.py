"""Physics-constrained DPS (PC-DPS) sampler.

The reverse loop is a faithful re-implementation of
third_party/qg-beta-sr/Models/samplers.py:get_samples_corrected with
DPS.flag=True, plus an optional QG-residual gradient.  At lambda_phys=0 the
sampler must match the released DPS baseline (checked in tests/).

Pair mode reconstructs two consecutive snapshots (t, t+dt) at once:
  - each frame has its own observation/data term;
  - the QG residual couples the two frames.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from Models.samplers import A_les_interleaved
from Models.sde import get_score, mu_sigma


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x ** 2).mean() + 1e-12)


def _rms_per_sample(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x ** 2).mean(dim=(-2, -1), keepdim=True) + 1e-12)


def dps_terms(x_noisy, x_measured, diff_time, model_uncond, option, dps,
              grid, spectral_operator, ops=None, lambda_phys=0.0, dt=0.5,
              pair=False, triple=False, n_pair=None, time_average=True, residual_kmax=None,
              field_std=1.0, phys_mask=None, grad_clip=None, phys_t_min=0.0, phys_t_max=1.0, phys_lowpass=None,
              per_sample_scale=False, phys_late_boost=0.0):
    """Return (score_total, x0, diagnostics).

    score_total = unconditional score contribution from the DPS data term,
    optionally plus the negative QG-residual gradient.  The caller adds the
    model's own score (-pred/sigma).
    """
    with torch.enable_grad():
        x_in = x_noisy.clone().detach().requires_grad_(True)
        mu, sigma = mu_sigma(diff_time, option)
        eps_pred = model_uncond(x_in, diff_time)
        x0 = (1.0 / mu) * (x_in - sigma * eps_pred)

        pred_obs, sigma_scaled = A_les_interleaved(
            x0, dps, diff_time, grid, spectral_operator, option)
        mse = F.mse_loss(pred_obs, x_measured, reduction="none").mean(dim=[1, 2])
        g_data = torch.autograd.grad(mse, x_in, grad_outputs=torch.ones_like(mse), retain_graph=True)[0]
        data_score = -(0.5 / sigma_scaled) * g_data

        diag = {"x0": x0.detach(), "mse": mse.detach(),
                "data_score_rms": _rms(data_score).detach()}
        if (lambda_phys == 0.0 or ops is None or float(diff_time) < phys_t_min
                or float(diff_time) > phys_t_max):
            return data_score.detach(), x0.detach(), diag

        if not (pair or triple):
            raise ValueError("physics term requires pair=True or triple=True")
        if triple:
            assert n_pair is not None and x0.shape[0] == 3 * n_pair
            qm = x0[:n_pair] * field_std
            q0 = x0[n_pair:2 * n_pair] * field_std
            q1 = x0[2 * n_pair:] * field_std
            if phys_lowpass is not None:
                qm = ops.lowpass(qm, phys_lowpass)
                q0 = ops.lowpass(q0, phys_lowpass)
                q1 = ops.lowpass(q1, phys_lowpass)
            tm = ops.residual_terms(qm, q0, dt, time_average=False)
            t0 = ops.residual_terms(q0, q0, dt, time_average=False)
            t1 = ops.residual_terms(q1, q1, dt, time_average=False)
            w = (0.25, 0.5, 0.25)
            def comp(t):
                return t["J"] + t["drag"] + t["diffusion"] + ops.beta * t["beta_basis"]
            r = ((q1 - qm) / (2 * dt) + w[0] * comp(tm) + w[1] * comp(t0)
                 + w[2] * comp(t1) - tm["F"])
        else:
            assert n_pair is not None and x0.shape[0] == 2 * n_pair
            # Model space is divided by field_std; the QG coefficients/F are physical.
            q0 = x0[:n_pair] * field_std
            q1 = x0[n_pair:] * field_std
            if phys_lowpass is not None:
                q0 = ops.lowpass(q0, phys_lowpass)
                q1 = ops.lowpass(q1, phys_lowpass)
            r = ops.residual(q0, q1, dt=dt, time_average=time_average)
        if residual_kmax is not None:
            r = ops.lowpass(r, residual_kmax)
        if phys_mask is not None:
            m = phys_mask.to(r.device, r.dtype)
            phys_loss = (r ** 2 * m).sum() / (m.sum() + 1e-12)
        else:
            phys_loss = (r ** 2).mean()
        g_phys = torch.autograd.grad(phys_loss, x_in, retain_graph=False)[0]
        if grad_clip is not None:
            thr = torch.quantile(g_phys.abs().flatten(), grad_clip)
            g_phys = torch.clamp(g_phys, -thr, thr)
        if per_sample_scale:
            scale = _rms_per_sample(data_score) / (_rms_per_sample(g_phys) + 1e-12)
        else:
            scale = _rms(data_score) / (_rms(g_phys) + 1e-12)
        lam_eff = lambda_phys * (1.0 + phys_late_boost * (1.0 - float(diff_time)))
        total = data_score - lam_eff * scale * g_phys
        diag["phys_rms"] = _rms(r).detach()
        diag["g_phys_rms"] = _rms(g_phys).detach()
        diag["phys_scale"] = scale.detach()
        return total.detach(), x0.detach(), diag


def sample_pcdps(x_init, x_cond, seed, num_samples, diffusion_time_steps,
                 num_corrections, model_uncond, corr_scale, dps, grid,
                 spectral_operator, ops=None, lambda_phys=0.0, dt=0.5,
                 pair=False, triple=False, option=0, time_average=True, record_diag=False,
                 residual_kmax=None, field_std=1.0, phys_mask=None, grad_clip=None,
                 phys_t_min=0.0, phys_t_max=1.0, phys_lowpass=None, per_sample_scale=False,
                 phys_late_boost=0.0):
    """Reverse diffusion with DPS + optional QG physics.

    x_init: [H,W] reference field used only for shape/RNG (truth is not used
            to initialize the reverse process).
    x_cond: [H,W] or [2,H,W] observation(s), normalized by field_std.
    pair:   if True, x_cond must be [2,H,W] and num_samples refers to the
            number of *pairs*; the returned tensor is [2*num_samples,H,W].
    """
    torch.manual_seed(seed)
    T = diffusion_time_steps
    if triple:
        assert x_cond.ndim == 3 and x_cond.shape[0] == 3, "triple mode needs [3,H,W] obs"
        x_cond = torch.cat([
            x_cond[0:1].repeat([num_samples, 1, 1]),
            x_cond[1:2].repeat([num_samples, 1, 1]),
            x_cond[2:3].repeat([num_samples, 1, 1]),
        ], dim=0)  # [3N,H,W]
        n_pair = num_samples
        N = 3 * num_samples
    elif pair:
        assert x_cond.ndim == 3 and x_cond.shape[0] == 2, "pair mode needs [2,H,W] obs"
        # stack frame 0 samples first, then frame 1 samples
        x_cond = torch.cat([
            x_cond[0:1].repeat([num_samples, 1, 1]),
            x_cond[1:2].repeat([num_samples, 1, 1]),
        ], dim=0)  # [2N,H,W]
        n_pair = num_samples
        N = 2 * num_samples
    else:
        n_pair = None
        N = num_samples
        x_cond = x_cond.repeat([num_samples, 1, 1])  # [N,H,W]

    x = torch.randn_like(x_init.repeat([N, 1, 1]))
    t_i = torch.ones(1, device=x_init.device)
    mu_i, sigma_i = mu_sigma(t_i, option)

    traj = torch.zeros(N, x_init.shape[-2], x_init.shape[-1], T + 1,
                       device=x_init.device)
    traj[..., 0] = x
    diagnostics = []

    with torch.no_grad():
        for step_idx, i in enumerate(range(1, T + 1), start=1):
            t_i_dash = torch.tensor(1 - i / T, device=x_init.device)
            mu_d, sigma_d = mu_sigma(t_i_dash, option)
            r = mu_d / mu_i

            pred = model_uncond(x, t_i)
            total_score, x0, diag = dps_terms(
                x, x_cond, t_i, model_uncond, option, dps, grid,
                spectral_operator, ops=ops, lambda_phys=lambda_phys, dt=dt,
                pair=pair, triple=triple, n_pair=n_pair, time_average=time_average,
                residual_kmax=residual_kmax, field_std=field_std,
                phys_mask=phys_mask, grad_clip=grad_clip, phys_t_min=phys_t_min,
                phys_t_max=phys_t_max, phys_lowpass=phys_lowpass,
                per_sample_scale=per_sample_scale, phys_late_boost=phys_late_boost)
            score = get_score(pred, x, t_i, option) + total_score

            if i != T:
                x_i_1 = r * x + (r - sigma_d / sigma_i) * (sigma_i ** 2) * score
            else:
                x_i_1 = x + (sigma_i ** 2) * score

            for _ in range(num_corrections):
                z = torch.randn_like(x)
                pred = model_uncond(x_i_1, t_i_dash)
                total_score, x0, diag = dps_terms(
                    x_i_1, x_cond, t_i_dash, model_uncond, option, dps, grid,
                    spectral_operator, ops=ops, lambda_phys=lambda_phys, dt=dt,
                    pair=pair, triple=triple, n_pair=n_pair, time_average=time_average,
                    residual_kmax=residual_kmax, field_std=field_std,
                    phys_mask=phys_mask, grad_clip=grad_clip, phys_t_min=phys_t_min,
                    phys_t_max=phys_t_max, phys_lowpass=phys_lowpass,
                    per_sample_scale=per_sample_scale, phys_late_boost=phys_late_boost)
                eps = get_score(pred, x_i_1, t_i_dash, option) + total_score
                denom = max(F.mse_loss(eps, torch.zeros_like(eps)), 1e-8)
                delta = corr_scale / denom
                x_i_1 = x_i_1 + delta * eps + torch.sqrt(2 * delta) * z

            if record_diag and (i == 1 or i == T or i % 8 == 0):
                diagnostics.append({
                    "step": i,
                    "t": float(t_i_dash),
                    "mse": float(diag["mse"].mean()),
                    "data_score_rms": float(diag["data_score_rms"]),
                    "phys_rms": float(diag.get("phys_rms", torch.tensor(float("nan")))),
                })

            x = x_i_1
            traj[..., step_idx] = x
            mu_i, sigma_i = mu_d, sigma_d
            t_i = t_i_dash

    return traj, diagnostics

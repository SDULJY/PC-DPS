"""Differentiable 2D beta-plane QG operators on the 64x64 periodic grid.

Equation implemented (matching third_party/qg-2d/Operators/operators.py and
Config/Run02721.py):

    dq/dt + J(psi, q) + mu*q - nu*lap(q) + beta*dpsi/dx - F = 0
    q = lap(psi),  psi_hat = -q_hat / k^2

with F = A cos(B x) + D cos(E y), A=-0.1, B=2, D=0.1, E=2.
The nonlinear operator in qg-2d is -div(u q, v q) = -J; we implement the same
form and then use +J in the residual.

All operations use torch.fft with norm='forward', 2/3 de-aliasing, and are
autograd-compatible so the residual can be differentiated w.r.t. the field.
"""
from __future__ import annotations

import math

import torch


class QGOps:
    def __init__(self, n: int = 64, L: float = 2 * math.pi, device: str = "cuda",
                 dtype: torch.dtype = torch.float32,
                 mu: float = 2e-2, beta: float = 2.5,
                 nu: float | None = None, re: int | None = None,
                 forcing: bool = True, beta_sign: float = -1.0):
        self.n = n
        self.L = L
        self.device = device
        self.dtype = dtype
        self.mu = mu
        self.beta = beta
        # The released jet data is consistent with -beta*dpsi/dx; the code
        # config Run02721 writes +i*B*kx/k^2, which is the opposite sign.
        self.beta_sign = beta_sign
        self.nu = (1.0 / re) if nu is None and re is not None else (nu if nu is not None else 1e-3)
        self.use_forcing = forcing

        kx = torch.fft.rfftfreq(n, d=L / (n * 2 * math.pi)).to(device=device, dtype=dtype)
        ky = torch.fft.fftfreq(n, d=L / (n * 2 * math.pi)).to(device=device, dtype=dtype)
        self.kx = kx.view(1, -1)
        self.ky = ky.view(-1, 1)
        self.k2 = self.kx ** 2 + self.ky ** 2
        safe_k2 = torch.where(self.k2 > 0, self.k2, torch.ones_like(self.k2))
        self.irsq = torch.where(self.k2 > 0, 1.0 / safe_k2,
                                torch.zeros_like(self.k2))

        # qg-2d dealias(): kcut = sqrt(2) * (1 - 1/3) * min(max(ky), max(kr))
        kcut = math.sqrt(2.0) * (1.0 - 1.0 / 3.0) * min(float(ky.abs().max()), float(kx.max()))
        self.dealias_mask = (torch.sqrt(self.k2) <= kcut).to(dtype)

        if forcing:
            # Match third_party/qg-2d/Initial_forcing/forcing.py, which uses
            # torch.linspace(0, L, N) (endpoint included).
            x = torch.linspace(0, L, n, device=device, dtype=dtype)
            X = x.view(1, -1)
            Y = x.view(-1, 1)
            self.F = (-0.1 * torch.cos(2 * X) + 0.1 * torch.cos(2 * Y))
        else:
            self.F = torch.zeros((n, n), device=device, dtype=dtype)

    # ------------------------------------------------------------------ FFT
    def to_spectral(self, q: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(q, dim=(-2, -1), norm="forward")

    def to_physical(self, qh: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(qh, s=(self.n, self.n), dim=(-2, -1), norm="forward")

    def psi(self, q: torch.Tensor) -> torch.Tensor:
        return self.to_physical(-self.to_spectral(q) * self.irsq)

    def d_dx(self, q: torch.Tensor) -> torch.Tensor:
        return self.to_physical(1j * self.kx * self.to_spectral(q))

    def d_dy(self, q: torch.Tensor) -> torch.Tensor:
        return self.to_physical(1j * self.ky * self.to_spectral(q))

    def laplacian(self, q: torch.Tensor) -> torch.Tensor:
        return self.to_physical(-self.k2 * self.to_spectral(q))

    def lowpass(self, q: torch.Tensor, kmax: float) -> torch.Tensor:
        mask = (torch.sqrt(self.k2) <= kmax).to(q.dtype)
        return self.to_physical(self.to_spectral(q) * mask)

    # -------------------------------------------------------------- nonlinear
    def jacobian(self, psi: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """J(psi, q) = psi_x q_y - psi_y q_x, via qg-2d's -div(u q, v q) form."""
        psih = self.to_spectral(psi)
        qh = self.to_spectral(q)
        # u = -psi_y, v = psi_x
        u = self.to_physical(-1j * self.ky * psih)
        v = self.to_physical(1j * self.kx * psih)
        uq = u * q
        vq = v * q
        out_h = (-1j * self.kx * self.to_spectral(uq)
                 - 1j * self.ky * self.to_spectral(vq))
        out_h = out_h * self.dealias_mask          # 2/3 de-aliasing (out-of-place)
        # qg-2d's operator is -J; return +J for the residual.
        return -self.to_physical(out_h)

    # ---------------------------------------------------------------- residual
    def residual_terms(self, q0: torch.Tensor, q1: torch.Tensor, dt: float,
                       time_average: bool = True) -> dict:
        """Return individual terms of the QG residual for a consecutive pair.

        q0, q1: (..., n, n) fields at t and t+dt.
        The returned dict contains dqdt, J, drag, diffusion, beta_term, F.
        """
        psi0 = self.psi(q0)
        psi1 = self.psi(q1)
        J0 = self.jacobian(psi0, q0)
        J1 = self.jacobian(psi1, q1)
        lap0 = self.laplacian(q0)
        lap1 = self.laplacian(q1)
        psi_x0 = self.d_dx(psi0)
        psi_x1 = self.d_dx(psi1)
        if time_average:
            J = 0.5 * (J0 + J1)
            qbar = 0.5 * (q0 + q1)
            lap = 0.5 * (lap0 + lap1)
            psi_x = 0.5 * (psi_x0 + psi_x1)
        else:
            J, qbar, lap, psi_x = J0, q0, lap0, psi_x0
        return {
            "dqdt": (q1 - q0) / dt,
            "J": J,
            "drag": self.mu * qbar,
            "diffusion": -self.nu * lap,
            "beta_basis": self.beta_sign * psi_x,
            "F": self.F,
        }

    def residual(self, q0: torch.Tensor, q1: torch.Tensor, dt: float,
                 beta: float | None = None, time_average: bool = True) -> torch.Tensor:
        terms = self.residual_terms(q0, q1, dt, time_average=time_average)
        b = self.beta if beta is None else beta
        r = (terms["dqdt"] + terms["J"] + terms["drag"]
             + terms["diffusion"] + b * terms["beta_basis"] - terms["F"])
        return r

    def residual_no_beta(self, q0: torch.Tensor, q1: torch.Tensor, dt: float,
                         time_average: bool = True) -> torch.Tensor:
        """r0 for the beta regression: all terms except the beta term."""
        terms = self.residual_terms(q0, q1, dt, time_average=time_average)
        return terms["dqdt"] + terms["J"] + terms["drag"] + terms["diffusion"] - terms["F"]

    def beta_basis(self, q0: torch.Tensor, q1: torch.Tensor,
                   time_average: bool = True) -> torch.Tensor:
        """b such that residual = r0 + beta*b (data-consistent sign)."""
        psi0, psi1 = self.psi(q0), self.psi(q1)
        if time_average:
            b = 0.5 * (self.d_dx(psi0) + self.d_dx(psi1))
        else:
            b = self.d_dx(psi0)
        return self.beta_sign * b

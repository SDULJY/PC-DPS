"""Self-tests for the differentiable QG operators (P3 must pass these first)."""
import math

import torch

from src.pcdps.qg_ops import QGOps


def _grid(n=64, L=2 * math.pi, device="cpu", dtype=torch.float64):
    x = torch.linspace(0, L, n + 1, device=device, dtype=dtype)[:-1]
    return x.view(1, -1), x.view(-1, 1)


def test_spectral_derivatives():
    ops = QGOps(n=64, device="cpu", dtype=torch.float64, mu=0, beta=0, nu=0, forcing=False)
    X, Y = _grid(dtype=torch.float64)
    q = torch.sin(3 * X + 2 * Y)
    qx = ops.d_dx(q)
    qy = ops.d_dy(q)
    lap = ops.laplacian(q)
    ex = 3 * torch.cos(3 * X + 2 * Y)
    ey = 2 * torch.cos(3 * X + 2 * Y)
    el = -13 * torch.sin(3 * X + 2 * Y)
    for got, want, name in [(qx, ex, "dx"), (qy, ey, "dy"), (lap, el, "lap")]:
        err = (got - want).abs().max() / want.abs().max()
        print(name, "rel max err", float(err))
        assert err < 1e-10


def test_single_mode_jacobian_zero():
    ops = QGOps(n=64, device="cpu", dtype=torch.float64, mu=0, beta=0, nu=0, forcing=False)
    X, Y = _grid(dtype=torch.float64)
    q = torch.sin(3 * X + 2 * Y)
    psi = ops.psi(q)
    J = ops.jacobian(psi, q)
    err = J.abs().max() / (q.abs().max() + 1e-30)
    print("single-mode J rel max", float(err))
    assert err < 1e-10


def test_jacobian_matches_finite_difference():
    torch.manual_seed(0)
    ops = QGOps(n=32, device="cpu", dtype=torch.float64, mu=0, beta=0, nu=0, forcing=False)
    q = torch.randn(32, 32, dtype=torch.float64)
    # Keep only low modes so the product has no energy above the 2/3 cut-off;
    # otherwise the de-aliased operator and the direct product differ by design.
    low = (torch.sqrt(ops.k2) < 4).to(torch.float64)
    q = torch.fft.irfftn(
        torch.fft.rfftn(q, dim=(-2, -1)) * low,
        s=(32, 32), dim=(-2, -1),
    )
    psi = ops.psi(q)
    J = ops.jacobian(psi, q)
    # J = psi_x q_y - psi_y q_x; check one interior-like point via analytic derivatives
    psix, psiy = ops.d_dx(psi), ops.d_dy(psi)
    qx, qy = ops.d_dx(q), ops.d_dy(q)
    J_direct = psix * qy - psiy * qx
    err = (J - J_direct).abs().max() / (J_direct.abs().max() + 1e-30)
    print("J vs direct rel max", float(err))
    assert err < 1e-10


def test_residual_gradient():
    torch.manual_seed(0)
    ops = QGOps(n=16, device="cpu", dtype=torch.float64, mu=2e-2, beta=2.5, nu=1e-3,
                forcing=True)
    q0 = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    q1 = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    r = ops.residual(q0, q1, dt=0.5)
    loss = (r ** 2).mean()
    g0, g1 = torch.autograd.grad(loss, [q0, q1])
    # finite difference on a few random entries
    eps = 1e-6
    for _ in range(3):
        i = int(torch.randint(0, 16, (1,)))
        j = int(torch.randint(0, 16, (1,)))
        q0p = q0.detach().clone(); q0p[i, j] += eps
        q0m = q0.detach().clone(); q0m[i, j] -= eps
        fd = ((ops.residual(q0p, q1.detach(), 0.5) ** 2).mean()
              - (ops.residual(q0m, q1.detach(), 0.5) ** 2).mean()) / (2 * eps)
        rel = abs(fd - g0[i, j]) / (abs(fd) + 1e-12)
        print("grad check", float(rel))
        assert rel < 1e-5


def test_beta_oracle_linear_rossby_wave():
    # With mu=nu=0 and no forcing, a single Rossby wave is an exact solution:
    # q = cos(kx x + ky y - omega t), omega = -beta kx / k^2.
    n, L = 64, 2 * math.pi
    beta, kx, ky, dt = 2.5, 3, 2, 1e-3
    k2 = kx ** 2 + ky ** 2
    omega = -beta * kx / k2
    ops = QGOps(n=n, L=L, device="cpu", dtype=torch.float64, mu=0, beta=beta, nu=0,
                forcing=False, beta_sign=1.0)
    x = torch.linspace(0, L, n + 1, dtype=torch.float64)[:-1]
    X, Y = x.view(1, -1), x.view(-1, 1)
    theta = kx * X + ky * Y
    q0 = torch.cos(theta)
    # Use a first-order-exact pair so the forward-difference time derivative
    # equals the analytic dq/dt exactly; this isolates the spatial operators.
    q1 = q0 + dt * (omega * torch.sin(theta))
    r = ops.residual(q0, q1, dt=dt, time_average=False)
    print("wave residual rel", float(r.abs().max() / q0.abs().max()))
    assert r.abs().max() < 1e-10
    r0 = ops.residual_no_beta(q0, q1, dt=dt, time_average=False)
    b = ops.beta_basis(q0, q1, time_average=False)
    beta_hat = -(r0 * b).sum() / (b * b).sum()
    print("beta_hat", float(beta_hat))
    assert abs(beta_hat - beta) / beta < 1e-6

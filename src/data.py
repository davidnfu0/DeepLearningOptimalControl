import torch
import math


@torch.no_grad()
def make_donut(n_per_class=500, r_inner=1.0, r_outer=3.0, noise=0.1, device="cpu"):
    N0, N1 = n_per_class, n_per_class

    theta0 = 2 * math.pi * torch.rand(N0, device=device)
    theta1 = 2 * math.pi * torch.rand(N1, device=device)

    r0 = r_inner + noise * torch.randn(N0, device=device)
    r1 = r_outer + noise * torch.randn(N1, device=device)

    x0 = torch.stack([r0 * torch.cos(theta0), r0 * torch.sin(theta0)], dim=1)
    x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [torch.zeros(N0, device=device), torch.ones(N1, device=device)], dim=0
    ).long()

    idx = torch.randperm(X.size(0), device=device)
    return X[idx], y[idx]


@torch.no_grad()
def make_spiral(n_per_class=500, turns=1, noise=0.1, device="cpu"):
    N0, N1 = n_per_class, n_per_class
    t0 = torch.linspace(0.0, 1.0, N0, device=device)
    t1 = torch.linspace(0.0, 1.0, N1, device=device)

    th0 = turns * 2 * math.pi * t0
    th1 = turns * 2 * math.pi * t1 + math.pi

    r0 = t0
    r1 = t1

    x0 = torch.stack(
        [r0 * torch.cos(th0), r0 * torch.sin(th0)], dim=1
    ) + noise * torch.randn(N0, 2, device=device)
    x1 = torch.stack(
        [r1 * torch.cos(th1), r1 * torch.sin(th1)], dim=1
    ) + noise * torch.randn(N1, 2, device=device)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [torch.zeros(N0, device=device), torch.ones(N1, device=device)], dim=0
    ).long()

    idx = torch.randperm(X.size(0), device=device)
    return X[idx], y[idx]


@torch.no_grad()
def make_linear_dataset(n_per_class=500, device="cpu"):
    x0 = torch.randn(n_per_class, 2, device=device) * 0.4 - 1.0
    x1 = torch.randn(n_per_class, 2, device=device) * 0.4 + 1.0

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_per_class, device=device),
            torch.ones(n_per_class, device=device),
        ],
        dim=0,
    ).long()
    idx = torch.randperm(X.size(0), device=device)
    return X[idx], y[idx]


@torch.no_grad()
def make_3d_nested_spheres(n_per_class=1500, device="cpu"):
    r0 = torch.rand(n_per_class, device=device) ** (1 / 3)
    theta0 = 2 * torch.pi * torch.rand(n_per_class, device=device)
    phi0 = torch.acos(2 * torch.rand(n_per_class, device=device) - 1)

    dir0 = torch.stack(
        [
            torch.sin(phi0) * torch.cos(theta0),
            torch.sin(phi0) * torch.sin(theta0),
            torch.cos(phi0),
        ],
        dim=1,
    )

    x0 = r0.unsqueeze(1) * dir0

    r1 = 2 + torch.rand(n_per_class, device=device)
    theta1 = 2 * torch.pi * torch.rand(n_per_class, device=device)
    phi1 = torch.acos(2 * torch.rand(n_per_class, device=device) - 1)

    dir1 = torch.stack(
        [
            torch.sin(phi1) * torch.cos(theta1),
            torch.sin(phi1) * torch.sin(theta1),
            torch.cos(phi1),
        ],
        dim=1,
    )

    x1 = r1.unsqueeze(1) * dir1

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long().to(device)
    return X, y


@torch.no_grad()
def make_3d_double_helix(
    n_per_class: int = 1000,
    turns: float = 3.0,
    noise: float = 0.1,
    device: str = "cpu",
):
    t = torch.linspace(0, turns * 2 * torch.pi, n_per_class, device=device)

    x1 = torch.cos(t)
    y1 = torch.sin(t)
    z1 = t / (2 * torch.pi)
    helix1 = torch.stack([x1, y1, z1], dim=1)
    helix1 = helix1 + noise * torch.randn_like(helix1)

    x2 = torch.cos(t + torch.pi)
    y2 = torch.sin(t + torch.pi)
    z2 = t / (2 * torch.pi)
    helix2 = torch.stack([x2, y2, z2], dim=1)
    helix2 = helix2 + noise * torch.randn_like(helix2)

    X = torch.cat([helix1, helix2], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long().to(device)

    return X, y

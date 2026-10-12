"""The ground surface, as a differentiable bilinear lookup.

Shared: a walker rides this cloth, and a chunk model needs height-above-ground
to find the breast-height slice. A second implementation with a half-cell
offset would shift every detection it touched, so there is one.
"""
from __future__ import annotations

import numpy as np
import torch

class Dem:
    def __init__(self, npz_path, device="cuda"):
        d = np.load(npz_path, allow_pickle=True)
        self.z = torch.from_numpy(d["z"].astype(np.float32)).to(device)
        self.x0 = float(d["x0"]); self.y0 = float(d["y0"])
        self.cell = float(d["cell"])
        self.ny, self.nx = self.z.shape

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear ground height at xy (B,2); differentiable in xy."""
        u = ((xy[:, 0] - self.x0) / self.cell).clamp(0, self.nx - 1.001)
        v = ((xy[:, 1] - self.y0) / self.cell).clamp(0, self.ny - 1.001)
        u0 = u.detach().floor().long(); v0 = v.detach().floor().long()
        du = (u - u0.float()).clamp(0, 1); dv = (v - v0.float()).clamp(0, 1)
        z00 = self.z[v0, u0]; z01 = self.z[v0, u0 + 1]
        z10 = self.z[v0 + 1, u0]; z11 = self.z[v0 + 1, u0 + 1]
        return (z00 * (1 - du) * (1 - dv) + z01 * du * (1 - dv)
                + z10 * (1 - du) * dv + z11 * du * dv)

    def project(self, c: torch.Tensor) -> torch.Tensor:
        """Drop centers (B,3) onto the cloth: z := dem(x, y).

        The height VALUE is used (so 3D distances stay honest) but its gradient
        w.r.t. xy is cut: the cloth is a step function at building edges, where
        d(z)/d(xy) reaches ~6 m/m, and that derivative — multiplied along the
        8-step chain — dominated everything and blew training up. Walkers ride
        the ground either way, so z carries no direction signal worth its
        gradient.
        """
        return torch.cat([c[:, :2], self.height(c[:, :2]).detach()[:, None]], dim=1)

import math

import torch
from torch import nn

from asterion.contracts.errors import ContractError


def smooth_cutoff(distance: torch.Tensor, cutoff: float):
    x = (distance / cutoff).clamp(0, 1)
    # Quintic switch has zero first and second derivatives at both ends.
    return torch.where(
        distance < cutoff, 1 - 10 * x**3 + 15 * x**4 - 6 * x**5, torch.zeros_like(distance)
    )


class RadialBasis(nn.Module):
    frequencies: torch.Tensor

    def __init__(self, count: int, cutoff: float, smooth=True):
        super().__init__()
        if count < 2 or cutoff <= 0:
            raise ContractError("invalid radial basis")
        self.count = count
        self.cutoff = cutoff
        self.smooth = smooth
        self.register_buffer(
            "frequencies", torch.arange(1, count + 1, dtype=torch.float64) * math.pi
        )

    def envelope(self, distance):
        if self.smooth:
            return smooth_cutoff(distance, self.cutoff)
        return (distance < self.cutoff).to(distance.dtype)

    def forward(self, distance):
        x = distance[:, None] / self.cutoff
        # sinc supplies its finite r=0 limit without a branch containing division by zero.
        return torch.sinc(x * self.frequencies[None] / math.pi) * self.frequencies[None]

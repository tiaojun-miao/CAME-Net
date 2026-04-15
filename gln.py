"""
gln.py - Grade-wise normalization for PGA multivectors.

The implementation keeps grades separated and avoids injecting fixed
non-scalar offsets into non-scalar grades.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pga_algebra import GRADE_INDICES, Multivector, geometric_product


class GradewiseLayerNorm(nn.Module):
    """
    Grade-wise multivector normalization.

    For each grade r:
        X_r / sqrt(|<X_r * reverse(X_r)>_0| + eps)

    A learnable per-grade scale is kept by default. Bias is disabled by
    default because a shared additive offset inside a non-scalar grade
    generally breaks equivariance.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        learnable_scale: bool = True,
        learnable_bias: bool = False,
    ):
        super().__init__()
        self.eps = eps

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(5))
        else:
            self.register_buffer("scale", torch.ones(5))

        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer("bias", torch.zeros(5))

        self.learnable_bias = learnable_bias

    def forward(self, x: Multivector) -> Multivector:
        result = torch.zeros_like(x.data)

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue

            grade_mv = torch.zeros_like(x.data)
            grade_mv[..., indices] = x.data[..., indices]

            reversed_grade = Multivector(grade_mv).reverse().data
            scalar_part = geometric_product(grade_mv, reversed_grade)[..., 0]

            denom = torch.sqrt(torch.abs(scalar_part).unsqueeze(-1) + self.eps)
            normalized = grade_mv[..., indices] / denom
            normalized = normalized * self.scale[grade]

            if self.learnable_bias and grade in (0, 4):
                normalized = normalized + self.bias[grade]

            result[..., indices] = normalized

        return Multivector(result)

"""
gca.py - Geometric Clifford Attention and grade-preserving utility layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gln import GradewiseLayerNorm
from .pga_algebra import GRADE_INDICES, MULTIVECTOR_DIM, REVERSION_SIGNS, SCALAR_PART_TABLE, Multivector


def _grade_linear(in_dim: int, out_dim: int, grade: int) -> nn.Linear:
    return nn.Linear(in_dim, out_dim, bias=(grade in (0, 4)))


class GradeWiseLinear(nn.Module):
    """Apply an independent linear map inside each PGA grade."""

    def __init__(self, bias: bool = True):
        super().__init__()
        self.projections = nn.ModuleDict()
        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            self.projections[str(grade)] = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Multivector) -> Multivector:
        output = torch.zeros_like(x.data)
        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            output[..., indices] = self.projections[str(grade)](x.data[..., indices])
        return Multivector(output)


class GradeWiseMLP(nn.Module):
    """A small per-grade feed-forward network that preserves grade structure."""

    def __init__(self, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            hidden_dim = max(dim * expansion, dim)
            self.blocks[str(grade)] = nn.Sequential(
                _grade_linear(dim, hidden_dim, grade),
                nn.GELU(),
                nn.Dropout(dropout),
                _grade_linear(hidden_dim, dim, grade),
            )

    def forward(self, x: Multivector) -> Multivector:
        output = torch.zeros_like(x.data)
        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            output[..., indices] = self.blocks[str(grade)](x.data[..., indices])
        return Multivector(self.dropout(output))


class GeometricCliffordAttention(nn.Module):
    """
    Multi-head attention in multivector space.

    Q, K and V are projected independently inside each grade. Attention scores
    are computed from the scalar part of the geometric product:
        <Q * reverse(K)>_0 / sqrt(d_M)
    """

    def __init__(
        self,
        multivector_dim: int = MULTIVECTOR_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.multivector_dim = multivector_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(float(multivector_dim))
        self.dropout = nn.Dropout(dropout)

        self.grade_query_projs = nn.ModuleDict()
        self.grade_key_projs = nn.ModuleDict()
        self.grade_value_projs = nn.ModuleDict()
        self.grade_out_projs = nn.ModuleDict()

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            self.grade_query_projs[str(grade)] = _grade_linear(dim, dim * num_heads, grade)
            self.grade_key_projs[str(grade)] = _grade_linear(dim, dim * num_heads, grade)
            self.grade_value_projs[str(grade)] = _grade_linear(dim, dim * num_heads, grade)
            self.grade_out_projs[str(grade)] = _grade_linear(dim * num_heads, dim, grade)

    def _project(self, x: Multivector, projections: nn.ModuleDict) -> torch.Tensor:
        batch_size, num_tokens, _ = x.data.shape
        projected = torch.zeros(
            batch_size,
            self.num_heads,
            num_tokens,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            grade_data = x.data[..., indices]
            dim = len(indices)
            grade_proj = projections[str(grade)](grade_data)
            grade_proj = grade_proj.view(batch_size, num_tokens, self.num_heads, dim)
            projected[..., indices] = grade_proj.permute(0, 2, 1, 3)

        return projected

    def forward(
        self,
        x: Multivector,
        context: Optional[Multivector] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Multivector:
        context = x if context is None else context

        q_heads = self._project(x, self.grade_query_projs)
        k_heads = self._project(context, self.grade_key_projs)
        v_heads = self._project(context, self.grade_value_projs)

        reversion = REVERSION_SIGNS.to(device=x.data.device, dtype=x.data.dtype).view(1, 1, 1, -1)
        k_reversed = k_heads * reversion

        scalar_table = SCALAR_PART_TABLE.to(device=x.data.device, dtype=x.data.dtype)
        attn_scores = torch.einsum("bhqi,bhkj,ij->bhqk", q_heads, k_reversed, scalar_table)
        attn_scores = attn_scores * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        head_outputs = torch.einsum("bhqk,bhki->bhqi", attn_weights, v_heads)

        batch_size, num_queries = x.data.shape[:2]
        output = torch.zeros(
            batch_size,
            num_queries,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            grade_head = head_outputs[..., indices]
            grade_head = grade_head.permute(0, 2, 1, 3).reshape(batch_size, num_queries, dim * self.num_heads)
            output[..., indices] = self.grade_out_projs[str(grade)](grade_head)

        return Multivector(self.dropout(output))


class MotorValuedAttention(nn.Module):
    """
    Optional motor-valued attention block.

    It is kept as a lightweight auxiliary module, but the main model uses
    GeometricCliffordAttention.
    """

    def __init__(
        self,
        multivector_dim: int = MULTIVECTOR_DIM,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = GradeWiseLinear()
        self.key_proj = GradeWiseLinear()
        self.value_proj = GradeWiseLinear()
        self.out_proj = GradeWiseLinear()
        self.score_norm = GradewiseLayerNorm()

    def forward(self, x: Multivector) -> Multivector:
        q = self.query_proj(self.score_norm(x)).data
        k = self.key_proj(self.score_norm(x)).data
        v = self.value_proj(x).data

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        weights = F.softmax(logits, dim=-1)
        output = torch.matmul(weights, v)
        return self.out_proj(Multivector(output))


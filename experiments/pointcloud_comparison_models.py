"""
pointcloud_comparison_models.py - Lightweight point-cloud baseline models for comparison experiments.

These models are compact, paper-inspired baselines intended for matched small-budget
comparisons rather than exact reproductions of the original published architectures.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from method.came_net import CAMENet
from .controlled_geometry_experiments import CoefficientDotProductAttention


def _pairwise_knn(features: torch.Tensor, k: int) -> torch.Tensor:
    num_points = features.shape[1]
    if num_points <= 1:
        raise ValueError("At least two points are required for KNN-based baselines.")
    k = min(k, num_points - 1)
    pairwise_dist = torch.cdist(features, features)
    eye_mask = torch.eye(num_points, device=features.device, dtype=torch.bool).unsqueeze(0)
    pairwise_dist = pairwise_dist.masked_fill(eye_mask, float("inf"))
    return pairwise_dist.topk(k=k, dim=-1, largest=False).indices


def _gather_neighbors(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, channels = tensor.shape
    expanded = tensor.unsqueeze(1).expand(batch_size, num_points, num_points, channels)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
    return torch.gather(expanded, 2, gather_index)


def _sample_centroids(coords: torch.Tensor, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_points, _ = coords.shape
    sample_count = min(sample_count, num_points)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count == num_points:
        indices = torch.arange(num_points, device=coords.device).unsqueeze(0).expand(batch_size, -1)
    else:
        base = torch.linspace(0, num_points - 1, steps=sample_count, device=coords.device)
        indices = base.round().long().unsqueeze(0).expand(batch_size, -1)
    sampled = torch.gather(coords, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    return sampled, indices


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        features = self.point_mlp(point_coords)
        pooled = features.max(dim=1).values
        return self.classifier(pooled)


class SetAbstractionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, sample_ratio: float, k: int):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 3, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, coords: torch.Tensor, features: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = coords.shape
        sample_count = max(1, int(math.ceil(num_points * self.sample_ratio)))
        sampled_coords, _ = _sample_centroids(coords, sample_count)
        distance_matrix = torch.cdist(sampled_coords, coords)
        neighbor_count = min(self.k, num_points)
        knn_indices = distance_matrix.topk(k=neighbor_count, dim=-1, largest=False).indices

        expanded_coords = coords.unsqueeze(1).expand(batch_size, sample_count, num_points, 3)
        grouped_coords = torch.gather(expanded_coords, 2, knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        relative_coords = grouped_coords - sampled_coords.unsqueeze(2)

        if features is None:
            grouped_features = relative_coords
        else:
            expanded_features = features.unsqueeze(1).expand(batch_size, sample_count, num_points, features.shape[-1])
            grouped_input_features = torch.gather(
                expanded_features,
                2,
                knn_indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]),
            )
            grouped_features = torch.cat([relative_coords, grouped_input_features], dim=-1)

        updated = self.mlp(grouped_features)
        return sampled_coords, updated.max(dim=2).values


class PointNetPPStyleClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.sa1 = SetAbstractionLayer(input_dim=0, output_dim=hidden_dim * 2, sample_ratio=0.25, k=16)
        self.sa2 = SetAbstractionLayer(input_dim=hidden_dim * 2, output_dim=hidden_dim * 4, sample_ratio=0.25, k=16)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        coords1, features1 = self.sa1(point_coords, None)
        _, features2 = self.sa2(coords1, features1)
        pooled = features2.max(dim=1).values
        return self.classifier(pooled)


class EdgeConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        knn_indices = _pairwise_knn(features, self.k)
        neighbors = _gather_neighbors(features, knn_indices)
        central = features.unsqueeze(2).expand_as(neighbors)
        edge_features = torch.cat([central, neighbors - central], dim=-1)
        updated = self.mlp(edge_features)
        return updated.max(dim=2).values


class DGCNNStyleClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.edge1 = EdgeConvBlock(3, hidden_dim)
        self.edge2 = EdgeConvBlock(hidden_dim, hidden_dim * 2)
        self.edge3 = EdgeConvBlock(hidden_dim * 2, hidden_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        feat1 = self.edge1(point_coords)
        feat2 = self.edge2(feat1)
        feat3 = self.edge3(feat2)
        pooled = torch.cat(
            [
                feat1.max(dim=1).values,
                feat2.max(dim=1).values,
                feat3.max(dim=1).values,
            ],
            dim=-1,
        )
        return self.classifier(pooled)


class PointAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(hidden_dim / max(1, num_heads))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, hidden_dim = features.shape
        normalized = self.norm1(features)
        q = self.q_proj(normalized)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)
        rel_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
        pos_bias = self.pos_proj(rel_pos)

        q = q.view(batch_size, num_points, self.num_heads, hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, num_points, self.num_heads, hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, num_points, self.num_heads, hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        pos_bias = pos_bias.view(batch_size, num_points, num_points, self.num_heads, hidden_dim // self.num_heads).permute(0, 3, 1, 2, 4)

        attn_logits = (q.unsqueeze(3) * (k.unsqueeze(2) + pos_bias)).sum(dim=-1) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attended = torch.sum(attn_weights.unsqueeze(-1) * (v.unsqueeze(2) + pos_bias), dim=3)
        attended = attended.permute(0, 2, 1, 3).reshape(batch_size, num_points, hidden_dim)
        features = features + self.dropout(self.out_proj(attended))
        features = features + self.dropout(self.ffn(self.norm2(features)))
        return features


class PointTransformerV2StyleClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList(
            [PointAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        features = self.input_proj(point_coords)
        for layer in self.layers:
            features = layer(features, point_coords)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


class ScalarVectorBlock(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, dropout: float = 0.0, gated: bool = False):
        super().__init__()
        self.scalar_q = nn.Linear(scalar_dim, scalar_dim)
        self.scalar_k = nn.Linear(scalar_dim, scalar_dim)
        self.scalar_v = nn.Linear(scalar_dim, scalar_dim)
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, scalar_dim),
            nn.GELU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(scalar_dim, vector_dim),
            nn.Sigmoid(),
        )
        self.scalar_out = nn.Linear(scalar_dim, scalar_dim)
        self.vector_out = nn.Linear(vector_dim, vector_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_dim * 2, scalar_dim),
        )
        self.vector_norm = nn.LayerNorm(vector_dim)
        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.dropout = nn.Dropout(dropout)
        self.gated = gated

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True)
        direction = rel / dist.clamp_min(1e-6)

        scalar_normed = self.scalar_norm(scalar)
        q = self.scalar_q(scalar_normed)
        k = self.scalar_k(scalar_normed)
        v = self.scalar_v(scalar_normed)
        bias = self.dist_mlp(dist)
        attn_logits = (q.unsqueeze(2) * (k.unsqueeze(1) + bias)).sum(dim=-1) / math.sqrt(q.shape[-1])
        attn = torch.softmax(attn_logits, dim=-1)

        scalar_update = torch.sum(attn.unsqueeze(-1) * (v.unsqueeze(1) + bias), dim=2)
        scalar = scalar + self.dropout(self.scalar_out(scalar_update))
        scalar = scalar + self.dropout(self.scalar_ffn(self.scalar_norm(scalar)))

        vector_channels = vector.shape[-1]
        vector_neighbors = vector.unsqueeze(1).expand(-1, scalar.shape[1], -1, -1, -1)
        directional = direction.unsqueeze(-1) * self.vector_gate(scalar).unsqueeze(2).unsqueeze(2)
        vector_message = vector_neighbors + directional
        aggregated = torch.sum(attn.unsqueeze(-1).unsqueeze(-1) * vector_message, dim=2)
        if self.gated:
            aggregated = aggregated * self.vector_gate(scalar).unsqueeze(2)
        reshaped = aggregated.reshape(aggregated.shape[0], aggregated.shape[1] * aggregated.shape[2], vector_channels)
        projected = self.vector_out(self.vector_norm(reshaped)).reshape_as(aggregated)
        vector = vector + self.dropout(projected)
        return scalar, vector


class SE3TransformerStyleClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        vector_dim = max(4, hidden_dim // 4)
        self.scalar_input = nn.Linear(3, hidden_dim)
        self.vector_input = nn.Linear(3, vector_dim)
        self.layers = nn.ModuleList(
            [ScalarVectorBlock(hidden_dim, vector_dim, dropout=dropout, gated=False) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + vector_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        scalar = self.scalar_input(point_coords)
        vector_channels = self.vector_input(point_coords)
        vector = point_coords.unsqueeze(-1) * vector_channels.unsqueeze(2)
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, point_coords)
        pooled_scalar = scalar.mean(dim=1)
        pooled_vector = vector.norm(dim=2).mean(dim=1)
        return self.classifier(torch.cat([pooled_scalar, pooled_vector], dim=-1))


class EquiformerV2StyleClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        vector_dim = max(4, hidden_dim // 4)
        self.scalar_input = nn.Linear(3, hidden_dim)
        self.vector_input = nn.Linear(3, vector_dim)
        self.layers = nn.ModuleList(
            [ScalarVectorBlock(hidden_dim, vector_dim, dropout=dropout, gated=True) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + vector_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        scalar = self.scalar_input(point_coords)
        vector_channels = self.vector_input(point_coords)
        vector = point_coords.unsqueeze(-1) * vector_channels.unsqueeze(2)
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, point_coords)
        pooled_scalar = scalar.mean(dim=1)
        pooled_vector = vector.norm(dim=2).mean(dim=1)
        return self.classifier(torch.cat([pooled_scalar, pooled_vector], dim=-1))


class GATrStyleBaseline(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 32, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.backbone = CAMENet(
            num_classes=num_classes,
            point_feature_dim=0,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            multimodal=False,
        )
        for layer in self.backbone.came_layers:
            layer.attention = CoefficientDotProductAttention(layer.attention)

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone(point_coords=point_coords, point_features=kwargs.get("point_features"))

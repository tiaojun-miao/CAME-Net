"""
mpe.py - Multimodal Projective Embedding modules for CAME-Net.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from pga_algebra import GRADE_INDICES, Multivector, create_point_pga


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for _ in range(max(1, depth - 1)):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class PointCloudMPE(nn.Module):
    """
    Structured point-cloud embedding into PGA multivectors.

    The point-cloud branch uses closed-form geometric objects:
    - grade 1: local tangent plane through each point
    - grade 2: local normal direction encoded as a Euclidean bivector
    - grade 3: exact OPNS PGA point

    Only the scalar channel is learned from rigid-motion invariants and
    optional pose-independent point features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        knn_k: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.knn_k = knn_k

        self.invariant_encoder = _make_mlp(6, hidden_dim, hidden_dim, depth=num_layers)
        self.feature_encoder = (
            _make_mlp(feature_dim, hidden_dim, hidden_dim, depth=num_layers)
            if feature_dim > 0
            else None
        )

        shared_input_dim = hidden_dim * (2 if feature_dim > 0 else 1)
        self.shared_encoder = _make_mlp(shared_input_dim, hidden_dim, hidden_dim, depth=num_layers)

        self.grade0_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _gather_neighbors(coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = coords.shape
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        expanded = coords.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
        return torch.gather(expanded, dim=2, index=gather_index)

    def _estimate_local_geometry(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = coords.shape
        device = coords.device

        if num_points < 2:
            raise ValueError("PointCloudMPE requires at least two points to estimate local geometry.")

        neighbor_count = min(self.knn_k, num_points - 1)
        pairwise_dist = torch.cdist(coords, coords)
        eye_mask = torch.eye(num_points, device=device, dtype=torch.bool).unsqueeze(0)
        pairwise_dist = pairwise_dist.masked_fill(eye_mask, float("inf"))
        knn_indices = pairwise_dist.topk(k=neighbor_count, dim=-1, largest=False).indices

        neighbors = self._gather_neighbors(coords, knn_indices)
        relative = neighbors - coords.unsqueeze(2)

        covariance = torch.matmul(relative.transpose(-1, -2), relative)
        covariance = covariance / float(neighbor_count)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        normals = eigenvectors[..., 0]

        global_center = coords.mean(dim=1, keepdim=True)
        outward = coords - global_center
        orientation = torch.sign((normals * outward).sum(dim=-1, keepdim=True))
        orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
        normals = normals * orientation
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        radius = relative.norm(dim=-1).mean(dim=-1, keepdim=True)
        centered_radius = outward.norm(dim=-1, keepdim=True)
        eig_sum = eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        dominance = eigenvalues[..., 2:3] / eig_sum

        invariants = torch.cat(
            [
                eigenvalues,
                radius,
                centered_radius,
                dominance,
            ],
            dim=-1,
        )
        return normals, invariants

    @staticmethod
    def _plane_from_point_normal(coords: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        plane = torch.zeros(coords.shape[:-1] + (4,), device=coords.device, dtype=coords.dtype)
        plane[..., 0] = normals[..., 0]
        plane[..., 1] = normals[..., 1]
        plane[..., 2] = -normals[..., 2]
        plane[..., 3] = -(coords * normals).sum(dim=-1)
        return plane

    @staticmethod
    def _euclidean_bivector_from_direction(normals: torch.Tensor) -> torch.Tensor:
        bivector = torch.zeros(normals.shape[:-1] + (6,), device=normals.device, dtype=normals.dtype)
        bivector[..., 0] = normals[..., 0]
        bivector[..., 1] = normals[..., 1]
        bivector[..., 2] = normals[..., 2]
        return bivector

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Multivector:
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected coords of shape (B, N, 3), got {coords.shape}")

        batch_size, num_points, _ = coords.shape
        normals, invariants = self._estimate_local_geometry(coords)
        invariant_hidden = self.invariant_encoder(invariants)
        shared_inputs = [invariant_hidden]

        if self.feature_encoder is not None and features is not None:
            feature_hidden = self.feature_encoder(features)
            shared_inputs.append(feature_hidden)

        hidden = self.shared_encoder(torch.cat(shared_inputs, dim=-1))

        output = torch.zeros(batch_size, num_points, 16, device=coords.device, dtype=coords.dtype)
        output[..., GRADE_INDICES[0]] = self.grade0_proj(hidden)
        output[..., GRADE_INDICES[1]] = self._plane_from_point_normal(coords, normals)
        output[..., GRADE_INDICES[2]] = self._euclidean_bivector_from_direction(normals)
        output[..., GRADE_INDICES[3]] = create_point_pga(coords).data[..., GRADE_INDICES[3]]
        output[..., GRADE_INDICES[4]] = 0.0

        return Multivector(output)


class ImageMPE(nn.Module):
    """Map image patch features to scalar and pseudoscalar PGA components."""

    def __init__(
        self,
        patch_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        del num_heads, num_layers
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scalar_proj = nn.Linear(hidden_dim, 1)
        self.pseudoscalar_proj = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, patch_features: torch.Tensor) -> Multivector:
        batch_size, num_patches, _ = patch_features.shape
        hidden = self.layer_norm(self.input_proj(patch_features))

        output = torch.zeros(batch_size, num_patches, 16, device=patch_features.device, dtype=patch_features.dtype)
        output[..., GRADE_INDICES[0]] = self.scalar_proj(hidden)
        output[..., GRADE_INDICES[4]] = self.pseudoscalar_proj(hidden)
        return Multivector(output)


class TextMPE(nn.Module):
    """Map text token features to scalar and pseudoscalar PGA components."""

    def __init__(
        self,
        token_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 512,
    ):
        super().__init__()
        del num_heads, num_layers, max_length
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scalar_proj = nn.Linear(hidden_dim, 1)
        self.pseudoscalar_proj = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, token_embeddings: torch.Tensor) -> Multivector:
        batch_size, seq_len, _ = token_embeddings.shape
        hidden = self.layer_norm(self.input_proj(token_embeddings))

        output = torch.zeros(batch_size, seq_len, 16, device=token_embeddings.device, dtype=token_embeddings.dtype)
        output[..., GRADE_INDICES[0]] = self.scalar_proj(hidden)
        output[..., GRADE_INDICES[4]] = self.pseudoscalar_proj(hidden)
        return Multivector(output)


class MultimodalMPE(nn.Module):
    """
    Embed each modality independently and concatenate tokens in multivector space.

    Cross-modal interaction is handled later by GCA, not inside the embedding
    module.
    """

    def __init__(
        self,
        point_feature_dim: int = 0,
        image_patch_dim: int = 128,
        text_token_dim: int = 128,
        hidden_dim: int = 64,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.point_mpe = PointCloudMPE(feature_dim=point_feature_dim, hidden_dim=hidden_dim)
        self.image_mpe = ImageMPE(patch_dim=image_patch_dim, hidden_dim=hidden_dim)
        self.text_mpe = TextMPE(token_dim=text_token_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        return_splits: bool = False,
    ) -> Union[Multivector, Tuple[Multivector, Dict[str, Tuple[int, int]]]]:
        embeddings = []
        splits: Dict[str, Tuple[int, int]] = {}
        offset = 0

        if point_coords is not None:
            point_embedding = self.point_mpe(point_coords, point_features)
            embeddings.append(point_embedding.data)
            splits["point"] = (offset, offset + point_embedding.data.shape[1])
            offset += point_embedding.data.shape[1]

        if image_patches is not None:
            image_embedding = self.image_mpe(image_patches)
            embeddings.append(image_embedding.data)
            splits["image"] = (offset, offset + image_embedding.data.shape[1])
            offset += image_embedding.data.shape[1]

        if text_tokens is not None:
            text_embedding = self.text_mpe(text_tokens)
            embeddings.append(text_embedding.data)
            splits["text"] = (offset, offset + text_embedding.data.shape[1])
            offset += text_embedding.data.shape[1]

        if not embeddings:
            raise ValueError("At least one modality must be provided")

        if self.fusion_mode != "concat":
            raise ValueError(f"Unsupported fusion_mode '{self.fusion_mode}'. Use 'concat'.")

        combined = Multivector(torch.cat(embeddings, dim=1))
        if return_splits:
            return combined, splits
        return combined



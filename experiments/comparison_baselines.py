"""
comparison_baselines.py - Lightweight baseline and ablation models for paper-facing comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from method.came_net import CAMENet
from .controlled_geometry_experiments import (
    CoefficientDotProductAttention,
    IdentityMultivectorModule,
    MixedGeometricCoefficientAttention,
    NormalizedGeometricAttention,
    ScalarOnlyMPEWrapper,
    UnconstrainedBivectorPointMPEWrapper,
)
from .pointcloud_comparison_models import (
    DGCNNStyleClassifier,
    EquiformerV2StyleClassifier,
    GATrStyleBaseline,
    PointNetClassifier,
    PointNetPPStyleClassifier,
    PointTransformerV2StyleClassifier,
    SE3TransformerStyleClassifier,
)


DEFAULT_PROMPT_TEMPLATE = "a 3d shape of a {class_name}"
_PROMPT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 -_"
_PROMPT_PAD_INDEX = 0
_PROMPT_STOI = {character: index + 1 for index, character in enumerate(_PROMPT_ALPHABET)}


@dataclass(frozen=True)
class ComparisonMethodSpec:
    name: str
    family: str
    description: str
    uses_equivariance_regularizer: bool
    uses_auxiliary_loss: bool


def list_comparison_methods() -> List[str]:
    return list(get_comparison_method_specs().keys())


def get_comparison_method_specs() -> Dict[str, ComparisonMethodSpec]:
    return {
        "came": ComparisonMethodSpec(
            name="came",
            family="ours",
            description="Reference CAME-Net with geometric attention and optional soft equivariance regularization.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "pointnet": ComparisonMethodSpec(
            name="pointnet",
            family="baseline",
            description="Compact PointNet baseline with per-point MLPs and global max pooling.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "pointnetpp_style": ComparisonMethodSpec(
            name="pointnetpp_style",
            family="baseline",
            description="Lightweight PointNet++-style hierarchical set abstraction baseline.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "dgcnn_style": ComparisonMethodSpec(
            name="dgcnn_style",
            family="baseline",
            description="Lightweight DGCNN-style EdgeConv baseline.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "point_transformer_v2_style": ComparisonMethodSpec(
            name="point_transformer_v2_style",
            family="baseline",
            description="Lightweight Point Transformer V2-style coordinate-aware self-attention baseline.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "se3_transformer_style": ComparisonMethodSpec(
            name="se3_transformer_style",
            family="baseline",
            description="Lightweight SE(3)-Transformer-style scalar/vector message-passing baseline.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "gatr_style": ComparisonMethodSpec(
            name="gatr_style",
            family="baseline",
            description="Lightweight GATr-style PGA token transformer with coefficient-space attention.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "equiformer_v2_style": ComparisonMethodSpec(
            name="equiformer_v2_style",
            family="baseline",
            description="Lightweight EquiformerV2-style gated scalar/vector transformer baseline.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "came_no_gln": ComparisonMethodSpec(
            name="came_no_gln",
            family="ablation",
            description="Trainable ablation that removes grade-wise layer normalization.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "came_no_equiv_reg": ComparisonMethodSpec(
            name="came_no_equiv_reg",
            family="ablation",
            description="Trainable ablation that disables the soft equivariance regularizer.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "came_non_geometric_fusion_reg": ComparisonMethodSpec(
            name="came_non_geometric_fusion_reg",
            family="ablation",
            description="Trainable ablation that replaces scalar-part geometric scoring with coefficient-space dot-product attention while retaining the soft equivariance regularizer.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "came_unconstrained_bivector": ComparisonMethodSpec(
            name="came_unconstrained_bivector",
            family="ablation",
            description="Trainable ablation that copies Euclidean bivector activations into ideal bivector slots.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "came_scalar_only": ComparisonMethodSpec(
            name="came_scalar_only",
            family="ablation",
            description="Trainable ablation that keeps only scalar channels before the geometric core.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "came_non_geometric_fusion": ComparisonMethodSpec(
            name="came_non_geometric_fusion",
            family="ablation",
            description="Trainable ablation that replaces scalar-part geometric scoring with coefficient-space dot-product attention.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "came_normalized_geometric_attention": ComparisonMethodSpec(
            name="came_normalized_geometric_attention",
            family="ablation",
            description="Trainable ablation that normalizes geometric scalar scores by query/key coefficient norms.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "came_geom_coeff_mix": ComparisonMethodSpec(
            name="came_geom_coeff_mix",
            family="ablation",
            description="Trainable ablation that mixes geometric scalar scores with coefficient-space dot-product scores.",
            uses_equivariance_regularizer=True,
            uses_auxiliary_loss=False,
        ),
        "pointclip_style": ComparisonMethodSpec(
            name="pointclip_style",
            family="baseline",
            description="Lightweight PointCLIP-style baseline: point cloud to fixed orthographic views, then image-text similarity classification.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "label_prior": ComparisonMethodSpec(
            name="label_prior",
            family="baseline",
            description="Label-prior sanity baseline that predicts from train-split label frequencies only.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=False,
        ),
        "ulip_style": ComparisonMethodSpec(
            name="ulip_style",
            family="baseline",
            description="Lightweight ULIP-style baseline: jointly align point, view, and text embeddings with an auxiliary tri-modal loss.",
            uses_equivariance_regularizer=False,
            uses_auxiliary_loss=True,
        ),
    }


def apply_came_variant(model: CAMENet, method: str) -> CAMENet:
    return _apply_came_variant(model, method)


def build_comparison_model(
    *,
    method: str,
    class_names: Sequence[str],
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    image_size: int = 32,
    point_feature_dim: int = 0,
) -> nn.Module:
    specs = get_comparison_method_specs()
    if method not in specs:
        raise ValueError(f"Unknown comparison method: {method}")

    if method in {
        "came",
        "came_no_gln",
        "came_no_equiv_reg",
        "came_non_geometric_fusion_reg",
        "came_unconstrained_bivector",
        "came_scalar_only",
        "came_non_geometric_fusion",
        "came_normalized_geometric_attention",
        "came_geom_coeff_mix",
    }:
        model = CAMENet(
            num_classes=len(class_names),
            point_feature_dim=point_feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            multimodal=False,
        )
        return apply_came_variant(model, method)

    if method == "pointclip_style":
        return PointCLIPStyleBaseline(
            class_names=class_names,
            image_size=image_size,
            embed_dim=hidden_dim * 2,
        )

    if method == "pointnet":
        return PointNetClassifier(num_classes=len(class_names), hidden_dim=hidden_dim, dropout=dropout)

    if method == "pointnetpp_style":
        return PointNetPPStyleClassifier(num_classes=len(class_names), hidden_dim=hidden_dim, dropout=dropout)

    if method == "dgcnn_style":
        return DGCNNStyleClassifier(num_classes=len(class_names), hidden_dim=hidden_dim, dropout=dropout)

    if method == "point_transformer_v2_style":
        return PointTransformerV2StyleClassifier(
            num_classes=len(class_names),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    if method == "se3_transformer_style":
        return SE3TransformerStyleClassifier(
            num_classes=len(class_names),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    if method == "gatr_style":
        return GATrStyleBaseline(
            num_classes=len(class_names),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    if method == "equiformer_v2_style":
        return EquiformerV2StyleClassifier(
            num_classes=len(class_names),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    if method == "ulip_style":
        return ULIPStyleBaseline(
            class_names=class_names,
            image_size=image_size,
            embed_dim=hidden_dim * 2,
            point_hidden_dim=hidden_dim,
        )

    raise ValueError(f"Unhandled comparison method: {method}")


def describe_comparison_method(method: str) -> str:
    specs = get_comparison_method_specs()
    if method not in specs:
        raise ValueError(f"Unknown comparison method: {method}")
    spec = specs[method]
    return (
        f"# Comparison Method\n\n"
        f"- Name: {spec.name}\n"
        f"- Family: {spec.family}\n"
        f"- Description: {spec.description}\n"
        f"- Uses equivariance regularizer: {spec.uses_equivariance_regularizer}\n"
        f"- Uses auxiliary loss: {spec.uses_auxiliary_loss}\n"
    )


def _apply_came_variant(model: CAMENet, method: str) -> CAMENet:
    if method == "came":
        return model
    if method == "came_no_gln":
        for layer in model.came_layers:
            layer.norm1 = IdentityMultivectorModule()
            layer.norm2 = IdentityMultivectorModule()
        return model
    if method == "came_unconstrained_bivector":
        model.mpe.point_mpe = UnconstrainedBivectorPointMPEWrapper(model.mpe.point_mpe)
        return model
    if method == "came_scalar_only":
        model.mpe = ScalarOnlyMPEWrapper(model.mpe)
        return model
    if method == "came_non_geometric_fusion_reg":
        for layer in model.came_layers:
            layer.attention = CoefficientDotProductAttention(layer.attention)
        return model
    if method == "came_non_geometric_fusion":
        for layer in model.came_layers:
            layer.attention = CoefficientDotProductAttention(layer.attention)
        return model
    if method == "came_normalized_geometric_attention":
        for layer in model.came_layers:
            layer.attention = NormalizedGeometricAttention(layer.attention)
        return model
    if method == "came_geom_coeff_mix":
        for layer in model.came_layers:
            layer.attention = MixedGeometricCoefficientAttention(layer.attention)
        return model
    if method == "came_no_equiv_reg":
        return model
    raise ValueError(f"Unknown CAME variant: {method}")


def _tokenize_prompts(class_names: Sequence[str], template: str, max_length: int = 48) -> torch.Tensor:
    token_rows = []
    for class_name in class_names:
        prompt = template.format(class_name=class_name.lower())
        token_ids = [_PROMPT_STOI.get(character, _PROMPT_PAD_INDEX) for character in prompt[:max_length]]
        if len(token_ids) < max_length:
            token_ids.extend([_PROMPT_PAD_INDEX] * (max_length - len(token_ids)))
        token_rows.append(token_ids)
    return torch.tensor(token_rows, dtype=torch.long)


def _normalize_point_cloud(point_coords: torch.Tensor) -> torch.Tensor:
    centered = point_coords - point_coords.mean(dim=1, keepdim=True)
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return centered / scale


def render_point_cloud_views(point_coords: torch.Tensor, image_size: int = 32) -> torch.Tensor:
    if point_coords.ndim != 3 or point_coords.shape[-1] != 3:
        raise ValueError("point_coords must have shape (B, N, 3)")

    normalized = _normalize_point_cloud(point_coords)
    batch_size, num_points, _ = normalized.shape
    views = torch.zeros(batch_size, 3, 1, image_size, image_size, device=point_coords.device, dtype=point_coords.dtype)
    view_specs = [
        ((0, 1), 2),
        ((1, 2), 0),
        ((0, 2), 1),
    ]

    for batch_index in range(batch_size):
        sample = normalized[batch_index]
        for view_index, (axes, depth_axis) in enumerate(view_specs):
            image = torch.full(
                (image_size, image_size),
                -1.0,
                device=point_coords.device,
                dtype=point_coords.dtype,
            )
            projected = sample[:, axes]
            depth = sample[:, depth_axis]
            x_coords = (((projected[:, 0] + 1.0) * 0.5) * (image_size - 1)).round().clamp(0, image_size - 1).long()
            y_coords = (((1.0 - (projected[:, 1] + 1.0) * 0.5)) * (image_size - 1)).round().clamp(0, image_size - 1).long()
            depth_values = (depth + 1.0) * 0.5

            for point_index in range(num_points):
                y_coord = int(y_coords[point_index].item())
                x_coord = int(x_coords[point_index].item())
                value = depth_values[point_index]
                if value > image[y_coord, x_coord]:
                    image[y_coord, x_coord] = value

            image = torch.where(image < 0, torch.zeros_like(image), image)
            views[batch_index, view_index, 0] = image

    return views


def build_label_prior_logits(label_priors: torch.Tensor) -> torch.Tensor:
    priors = label_priors.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(priors / (1.0 - priors))


class CharPromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 48):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(len(_PROMPT_STOI) + 1, embed_dim, padding_idx=_PROMPT_PAD_INDEX)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = (token_ids != _PROMPT_PAD_INDEX).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (embedded * mask).sum(dim=1) / denom
        return self.proj(pooled)


class LabelPriorBaseline(nn.Module):
    def __init__(self, label_priors: torch.Tensor):
        super().__init__()
        prior_logits = build_label_prior_logits(label_priors.detach().float())
        self.register_buffer("prior_logits", prior_logits, persistent=True)

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        batch_size = point_coords.shape[0]
        return self.prior_logits.unsqueeze(0).expand(batch_size, -1)


class SmallViewEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images).flatten(1)
        return self.proj(features)


class SmallPointEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, point_coords: torch.Tensor) -> torch.Tensor:
        features = self.mlp(point_coords)
        return features.max(dim=1).values


class PointCLIPStyleBaseline(nn.Module):
    def __init__(
        self,
        *,
        class_names: Sequence[str],
        image_size: int = 32,
        embed_dim: int = 64,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.image_size = image_size
        self.view_encoder = SmallViewEncoder(embed_dim=embed_dim)
        self.prompt_encoder = CharPromptEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("prompt_token_ids", _tokenize_prompts(self.class_names, prompt_template), persistent=False)

    def encode_views(self, point_coords: torch.Tensor) -> torch.Tensor:
        views = render_point_cloud_views(point_coords, image_size=self.image_size)
        batch_size, num_views = views.shape[:2]
        encoded = self.view_encoder(views.view(batch_size * num_views, 1, self.image_size, self.image_size))
        return encoded.view(batch_size, num_views, -1).mean(dim=1)

    def encode_text(self) -> torch.Tensor:
        return self.prompt_encoder(self.prompt_token_ids.to(self.logit_scale.device))

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        view_features = F.normalize(self.encode_views(point_coords), dim=-1)
        text_features = F.normalize(self.encode_text(), dim=-1)
        scale = self.logit_scale.exp()
        return scale * view_features @ text_features.t()


class ULIPStyleBaseline(nn.Module):
    def __init__(
        self,
        *,
        class_names: Sequence[str],
        image_size: int = 32,
        embed_dim: int = 64,
        point_hidden_dim: int = 32,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.image_size = image_size
        self.point_encoder = SmallPointEncoder(embed_dim=embed_dim, hidden_dim=point_hidden_dim)
        self.view_encoder = SmallViewEncoder(embed_dim=embed_dim)
        self.prompt_encoder = CharPromptEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("prompt_token_ids", _tokenize_prompts(self.class_names, prompt_template), persistent=False)

    def encode_points(self, point_coords: torch.Tensor) -> torch.Tensor:
        return self.point_encoder(_normalize_point_cloud(point_coords))

    def encode_views(self, point_coords: torch.Tensor) -> torch.Tensor:
        views = render_point_cloud_views(point_coords, image_size=self.image_size)
        batch_size, num_views = views.shape[:2]
        encoded = self.view_encoder(views.view(batch_size * num_views, 1, self.image_size, self.image_size))
        return encoded.view(batch_size, num_views, -1).mean(dim=1)

    def encode_text(self) -> torch.Tensor:
        return self.prompt_encoder(self.prompt_token_ids.to(self.logit_scale.device))

    def forward(self, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        point_features = F.normalize(self.encode_points(point_coords), dim=-1)
        text_features = F.normalize(self.encode_text(), dim=-1)
        scale = self.logit_scale.exp()
        return scale * point_features @ text_features.t()

    def compute_auxiliary_loss(self, point_coords: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        point_features = F.normalize(self.encode_points(point_coords), dim=-1)
        view_features = F.normalize(self.encode_views(point_coords), dim=-1)
        text_features = F.normalize(self.encode_text(), dim=-1)
        matched_text_features = text_features[labels]

        temperature = self.logit_scale.exp().clamp_min(1e-6)
        logits_point_image = temperature * point_features @ view_features.t()
        targets = torch.arange(point_features.shape[0], device=point_features.device)
        point_image_loss = 0.5 * (
            F.cross_entropy(logits_point_image, targets) +
            F.cross_entropy(logits_point_image.t(), targets)
        )

        point_text_alignment = 1.0 - (point_features * matched_text_features).sum(dim=-1).mean()
        view_text_alignment = 1.0 - (view_features * matched_text_features).sum(dim=-1).mean()
        return point_image_loss + point_text_alignment + view_text_alignment

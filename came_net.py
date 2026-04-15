"""
came_net.py - CAME-Net main architecture.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from gca import GeometricCliffordAttention, GradeWiseMLP
from gln import GradewiseLayerNorm
from mpe import MultimodalMPE
from pga_algebra import MULTIVECTOR_DIM, Multivector


class CAMELayer(nn.Module):
    """One grade-preserving CAME block."""

    def __init__(
        self,
        multivector_dim: int = MULTIVECTOR_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.multivector_dim = multivector_dim
        self.use_ffn = use_ffn

        self.attention = GeometricCliffordAttention(
            multivector_dim=multivector_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = GradewiseLayerNorm()
        self.norm2 = GradewiseLayerNorm()
        self.ffn = GradeWiseMLP(dropout=dropout) if use_ffn else None

    def forward(self, x: Multivector) -> Multivector:
        attn_output = self.attention(self.norm1(x))
        x = Multivector(x.data + attn_output.data)

        if self.ffn is not None:
            ffn_output = self.ffn(self.norm2(x))
            x = Multivector(x.data + ffn_output.data)

        return x


class GlobalMeanPooling(nn.Module):
    """Permutation-invariant pooling that preserves multivector type."""

    def forward(self, x: Multivector, mask: Optional[torch.Tensor] = None) -> Multivector:
        if mask is None:
            return Multivector(x.data.mean(dim=1, keepdim=True))

        weights = (~mask).to(dtype=x.data.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x.data * weights).sum(dim=1, keepdim=True) / denom
        return Multivector(pooled)


class CAMENet(nn.Module):
    """
    CAME-Net for point-cloud or multimodal classification.

    The core geometric processing path stays inside multivector space. The final
    task head is standard because the downstream task output is not itself a
    multivector.
    """

    def __init__(
        self,
        num_classes: int = 40,
        point_feature_dim: int = 0,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        multimodal: bool = False,
        image_patch_dim: int = 128,
        text_token_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.multimodal = multimodal

        self.mpe = MultimodalMPE(
            point_feature_dim=point_feature_dim,
            image_patch_dim=image_patch_dim,
            text_token_dim=text_token_dim,
            hidden_dim=hidden_dim,
            fusion_mode="concat",
        )

        self.came_layers = nn.ModuleList(
            [
                CAMELayer(
                    multivector_dim=MULTIVECTOR_DIM,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_ffn=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.pooling = GlobalMeanPooling()
        self.classification_head = nn.Sequential(
            nn.Linear(MULTIVECTOR_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_inputs(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        return_splits: bool = False,
    ) -> Union[Multivector, Tuple[Multivector, Dict[str, Tuple[int, int]]]]:
        if (image_patches is not None or text_tokens is not None) and not self.multimodal:
            raise ValueError("This model was created with multimodal=False but image/text inputs were provided.")

        return self.mpe(
            point_coords=point_coords,
            point_features=point_features,
            image_patches=image_patches,
            text_tokens=text_tokens,
            return_splits=return_splits,
        )

    def get_latent_multivector(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        return_splits: bool = False,
    ) -> Union[Multivector, Tuple[Multivector, Dict[str, Tuple[int, int]]]]:
        encoded = self._encode_inputs(
            point_coords=point_coords,
            point_features=point_features,
            image_patches=image_patches,
            text_tokens=text_tokens,
            return_splits=return_splits,
        )

        if return_splits:
            latent, splits = encoded
        else:
            latent = encoded
            splits = None

        for layer in self.came_layers:
            latent = layer(latent)

        if return_splits:
            return latent, splits
        return latent

    def get_point_cloud_embedding(
        self,
        point_coords: torch.Tensor,
        point_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.get_latent_multivector(point_coords=point_coords, point_features=point_features)
        pooled = self.pooling(latent)
        return pooled.data.squeeze(1)

    def forward(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.get_latent_multivector(
            point_coords=point_coords,
            point_features=point_features,
            image_patches=image_patches,
            text_tokens=text_tokens,
        )
        pooled = self.pooling(latent)
        return self.classification_head(pooled.data.squeeze(1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

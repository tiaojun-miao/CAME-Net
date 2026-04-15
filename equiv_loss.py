"""
equiv_loss.py - Soft equivariance regularization for CAME-Net.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pga_algebra import GRADE_INDICES, Multivector, create_point_pga, extract_point_coordinates, geometric_product, random_motor, random_rotation, random_translation

DEFAULT_SIGMA_ROT = math.pi / 6
DEFAULT_SIGMA_TRANS = 1.0

def multivector_distance(
    a: Multivector,
    b: Multivector,
    eta: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    if eta is None:
        eta = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    diff = a.data - b.data
    total = diff.new_zeros(())

    for grade, indices in GRADE_INDICES.items():
        grade_diff = torch.zeros_like(diff)
        grade_diff[..., indices] = diff[..., indices]
        reversed_grade = Multivector(grade_diff).reverse().data
        scalar_part = geometric_product(grade_diff, reversed_grade)[..., 0].abs().mean()
        total = total + eta.get(grade, 0.0) * scalar_part

    return total


def compute_equivariance_error(
    original_output,
    transformed_output,
    transformation,
) -> torch.Tensor:
    """
    Compare Phi(M.X) and M.Phi(X).
    """
    if hasattr(original_output, "apply_motor") and hasattr(transformed_output, "data"):
        target = original_output.apply_motor(transformation).data
        return F.mse_loss(transformed_output.data, target)

    raise TypeError("compute_equivariance_error expects multivector outputs.")


def apply_random_transformation(
    points: torch.Tensor,
    sigma_rot: float = DEFAULT_SIGMA_ROT,
    sigma_trans: float = DEFAULT_SIGMA_TRANS,
    device: Optional[torch.device] = None,
):
    device = points.device if device is None else device
    motor = random_motor(points.shape[0], sigma_rot=sigma_rot, sigma_trans=sigma_trans, device=device, dtype=points.dtype)
    transformed_points = create_point_pga(points).apply_motor(motor)
    return extract_point_coordinates(transformed_points), motor


def equivariance_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    num_samples: int = 1,
    sigma_rot: float = DEFAULT_SIGMA_ROT,
    sigma_trans: float = DEFAULT_SIGMA_TRANS,
) -> torch.Tensor:
    return equivariance_loss_efficient(
        model=model,
        point_coords=batch["point_coords"],
        labels=batch.get("labels"),
        point_features=batch.get("point_features"),
        image_patches=batch.get("image_patches"),
        text_tokens=batch.get("text_tokens"),
        num_samples=num_samples,
        sigma_rot=sigma_rot,
        sigma_trans=sigma_trans,
    )


def equivariance_loss_efficient(
    model: nn.Module,
    point_coords: torch.Tensor,
    labels: Optional[torch.Tensor],
    point_features: Optional[torch.Tensor] = None,
    image_patches: Optional[torch.Tensor] = None,
    text_tokens: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    sigma_rot: float = DEFAULT_SIGMA_ROT,
    sigma_trans: float = DEFAULT_SIGMA_TRANS,
) -> torch.Tensor:
    del labels
    device = point_coords.device
    original_latent = model.get_latent_multivector(
        point_coords=point_coords,
        point_features=point_features,
        image_patches=image_patches,
        text_tokens=text_tokens,
    )

    total_loss = torch.zeros((), device=device, dtype=point_coords.dtype)

    for _ in range(num_samples):
        motor = random_motor(
            batch_size=point_coords.shape[0],
            sigma_rot=sigma_rot,
            sigma_trans=sigma_trans,
            device=device,
            dtype=point_coords.dtype,
        )
        transformed_coords = extract_point_coordinates(create_point_pga(point_coords).apply_motor(motor))
        transformed_latent = model.get_latent_multivector(
            point_coords=transformed_coords,
            point_features=point_features,
            image_patches=image_patches,
            text_tokens=text_tokens,
        )
        target_latent = original_latent.apply_motor(motor)
        total_loss = total_loss + F.mse_loss(transformed_latent.data, target_latent.data)

    return total_loss / max(1, num_samples)


def grade_wise_equivariance_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    num_samples: int = 1,
) -> Dict[str, torch.Tensor]:
    point_coords = batch["point_coords"]
    point_features = batch.get("point_features")
    image_patches = batch.get("image_patches")
    text_tokens = batch.get("text_tokens")
    device = point_coords.device

    original_latent = model.get_latent_multivector(
        point_coords=point_coords,
        point_features=point_features,
        image_patches=image_patches,
        text_tokens=text_tokens,
    )

    losses = {str(grade): torch.zeros((), device=device, dtype=point_coords.dtype) for grade in GRADE_INDICES}

    for _ in range(num_samples):
        transformed_coords, motor = apply_random_transformation(point_coords)
        transformed_latent = model.get_latent_multivector(
            point_coords=transformed_coords.to(device),
            point_features=point_features,
            image_patches=image_patches,
            text_tokens=text_tokens,
        )
        target_latent = original_latent.apply_motor(motor.to(device))

        for grade, indices in GRADE_INDICES.items():
            losses[str(grade)] = losses[str(grade)] + F.mse_loss(
                transformed_latent.data[..., indices],
                target_latent.data[..., indices],
            )

    for key in losses:
        losses[key] = losses[key] / max(1, num_samples)
    return losses


def rotational_equivariance_loss(
    model: nn.Module,
    point_coords: torch.Tensor,
    point_features: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    sigma: float = DEFAULT_SIGMA_ROT,
) -> torch.Tensor:
    original_latent = model.get_latent_multivector(point_coords=point_coords, point_features=point_features)
    total_loss = torch.zeros((), device=point_coords.device, dtype=point_coords.dtype)

    for _ in range(num_samples):
        rotation = random_rotation(point_coords.shape[0], sigma=sigma, device=point_coords.device, dtype=point_coords.dtype)
        rotated_coords = extract_point_coordinates(create_point_pga(point_coords).apply_motor(rotation))
        rotated_latent = model.get_latent_multivector(point_coords=rotated_coords, point_features=point_features)
        target_latent = original_latent.apply_motor(rotation)
        total_loss = total_loss + F.mse_loss(rotated_latent.data, target_latent.data)

    return total_loss / max(1, num_samples)


def translational_equivariance_loss(
    model: nn.Module,
    point_coords: torch.Tensor,
    point_features: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    sigma: float = DEFAULT_SIGMA_TRANS,
) -> torch.Tensor:
    original_latent = model.get_latent_multivector(point_coords=point_coords, point_features=point_features)
    total_loss = torch.zeros((), device=point_coords.device, dtype=point_coords.dtype)

    for _ in range(num_samples):
        translation = random_translation(point_coords.shape[0], sigma=sigma, device=point_coords.device, dtype=point_coords.dtype)
        translated_coords = extract_point_coordinates(create_point_pga(point_coords).apply_motor(translation))
        translated_latent = model.get_latent_multivector(point_coords=translated_coords, point_features=point_features)
        target_latent = original_latent.apply_motor(translation)
        total_loss = total_loss + F.mse_loss(translated_latent.data, target_latent.data)

    return total_loss / max(1, num_samples)


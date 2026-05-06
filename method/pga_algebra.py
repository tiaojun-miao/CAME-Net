"""
pga_algebra.py - Projective Geometric Algebra (PGA) utilities for CAME-Net.

This module keeps the blade multiplication table used by the Clifford attention
score, and provides a numerically stable motor construction for rigid motions.

Two representation choices are important here:
1. General multivectors are stored in a 16D coefficient vector.
2. Euclidean points are encoded as standard OPNS PGA points:
   P = e123 + x e032 + y e013 + z e021.

Using OPNS points keeps motor sandwich actions well-defined for translations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

MULTIVECTOR_DIM = 16

GRADE_INDICES = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8, 9, 10],
    3: [11, 12, 13, 14],
    4: [15],
}

# Basis ordering:
# [1, e1, e2, e3, e0, e23, e31, e12, e01, e02, e03, e123, e032, e013, e021, e0123]
_BLADE_ORDERS = [
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (1, 2),
    (2, 0),
    (0, 1),
    (3, 0),
    (3, 1),
    (3, 2),
    (0, 1, 2),
    (3, 2, 1),
    (3, 0, 2),
    (3, 0, 1),
    (0, 1, 2, 3),
]

_VECTOR_METRIC = (1.0, 1.0, 1.0, 0.0)

# Reverse sign per blade grade: (-1)^(r(r-1)/2)
REVERSION_SIGNS = torch.tensor(
    [
        1.0,
        1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0,
        1.0,
    ],
    dtype=torch.float32,
)


def _orientation_sign(order):
    inversions = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inversions += 1
    return -1.0 if inversions % 2 else 1.0


def _sorted_bits(order):
    bits = 0
    for idx in sorted(order):
        bits |= 1 << idx
    return bits


def _build_blade_metadata():
    metadata = []
    bits_to_index = {}
    for idx, order in enumerate(_BLADE_ORDERS):
        orientation = _orientation_sign(order)
        bits = _sorted_bits(order)
        metadata.append(
            {
                "order": order,
                "grade": len(order),
                "orientation": orientation,
                "bits": bits,
            }
        )
        bits_to_index[bits] = idx
    return metadata, bits_to_index


BLADE_METADATA, BITS_TO_INDEX = _build_blade_metadata()


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _canonical_swaps(bits_a: int, bits_b: int) -> int:
    swaps = 0
    for basis_idx in range(4):
        if bits_a & (1 << basis_idx):
            lower_mask = (1 << basis_idx) - 1
            swaps += _popcount(bits_b & lower_mask)
    return swaps


def _metric_factor(common_bits: int) -> float:
    factor = 1.0
    for basis_idx in range(4):
        if common_bits & (1 << basis_idx):
            metric = _VECTOR_METRIC[basis_idx]
            if metric == 0.0:
                return 0.0
            factor *= metric
    return factor


def _blade_mul(i: int, j: int) -> Tuple[float, Optional[int]]:
    info_i = BLADE_METADATA[i]
    info_j = BLADE_METADATA[j]

    bits_i = info_i["bits"]
    bits_j = info_j["bits"]

    coeff = info_i["orientation"] * info_j["orientation"]

    swaps = _canonical_swaps(bits_i, bits_j)
    if swaps % 2:
        coeff = -coeff

    common = bits_i & bits_j
    metric = _metric_factor(common)
    if metric == 0.0:
        return 0.0, None
    coeff *= metric

    result_bits = bits_i ^ bits_j
    res_idx = BITS_TO_INDEX[result_bits]

    coeff /= BLADE_METADATA[res_idx]["orientation"]
    return coeff, res_idx


def _build_multiplication_table() -> torch.Tensor:
    table = torch.zeros((MULTIVECTOR_DIM, MULTIVECTOR_DIM, MULTIVECTOR_DIM), dtype=torch.float32)
    for i in range(MULTIVECTOR_DIM):
        for j in range(MULTIVECTOR_DIM):
            coeff, res_idx = _blade_mul(i, j)
            if res_idx is not None and coeff != 0.0:
                table[i, j, res_idx] = coeff
    return table


MULTIPLICATION_TABLE = _build_multiplication_table()
SCALAR_PART_TABLE = MULTIPLICATION_TABLE[:, :, 0]


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the PGA geometric product of two equally-shaped tensors."""
    if a.shape != b.shape:
        raise ValueError("Inputs to geometric_product must share the same shape")
    table = MULTIPLICATION_TABLE.to(device=a.device, dtype=a.dtype)
    return torch.einsum("...i,...j,ijk->...k", a, b, table)


def _identity_multivector(batch_shape, device, dtype) -> torch.Tensor:
    data = torch.zeros(batch_shape + (MULTIVECTOR_DIM,), device=device, dtype=dtype)
    data[..., 0] = 1.0
    return data


def create_point_pga(coords: torch.Tensor) -> "Multivector":
    """
    Convert Euclidean coordinates to an OPNS PGA point.

    P = e123 + x e032 + y e013 + z e021
    """
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected 3D coordinates, got shape {coords.shape}")

    shape = coords.shape[:-1]
    data = torch.zeros(shape + (MULTIVECTOR_DIM,), device=coords.device, dtype=coords.dtype)
    data[..., 11] = 1.0
    data[..., 12] = coords[..., 0]
    data[..., 13] = coords[..., 1]
    data[..., 14] = coords[..., 2]
    return Multivector(data)


def extract_point_coordinates(points: "Multivector", eps: float = 1e-8) -> torch.Tensor:
    """Recover Euclidean coordinates from an OPNS PGA point multivector."""
    weight = points.data[..., 11:12]
    safe_weight = torch.where(weight.abs() > eps, weight, torch.ones_like(weight))
    coords = points.data[..., 12:15] / safe_weight
    return coords


def make_translation_motor(translation: torch.Tensor) -> "Multivector":
    """
    Construct a unit translator from Euclidean translation parameters.

    With the basis ordering used in this repository, the axis signs are:
    Tx = 1 - 0.5 dx e01, Ty = 1 - 0.5 dy e02, Tz = 1 + 0.5 dz e03.
    """
    if translation.shape[-1] != 3:
        raise ValueError(f"Expected translation of shape (..., 3), got {translation.shape}")

    data = _identity_multivector(translation.shape[:-1], translation.device, translation.dtype)
    data[..., 8] = -0.5 * translation[..., 0]
    data[..., 9] = -0.5 * translation[..., 1]
    data[..., 10] = 0.5 * translation[..., 2]
    return Multivector(data)


def _axis_rotor(angle: torch.Tensor, basis_index: int, sign: float) -> "Multivector":
    data = _identity_multivector(angle.shape, angle.device, angle.dtype)
    half = 0.5 * angle
    data[..., 0] = torch.cos(half)
    data[..., basis_index] = sign * torch.sin(half)
    return Multivector(data)


def make_rotation_motor(rotation: torch.Tensor) -> "Multivector":
    """
    Construct a rotor from XYZ Euler-style increments.

    The three parameters correspond to the e23, e31 and e12 generators.
    Sequential composition is used instead of a naive Euclidean normalization,
    which keeps the result in the unit motor manifold.
    """
    if rotation.shape[-1] != 3:
        raise ValueError(f"Expected rotation of shape (..., 3), got {rotation.shape}")

    rot_x = _axis_rotor(rotation[..., 0], basis_index=5, sign=1.0)
    rot_y = _axis_rotor(rotation[..., 1], basis_index=6, sign=1.0)
    rot_z = _axis_rotor(rotation[..., 2], basis_index=7, sign=-1.0)
    return compose_motors(rot_z, compose_motors(rot_y, rot_x))


def exp_bivector(omega: torch.Tensor) -> "Multivector":
    """
    Map six sampled motion parameters to a valid motor.

    The first three entries are rotation coefficients and the last three are
    translation coefficients. The implementation composes a rotor and a
    translator, which is stable and exact for the sampled rigid motions used
    in training.
    """
    if omega.shape[-1] != 6:
        raise ValueError(f"Expected bivector parameters of shape (..., 6), got {omega.shape}")

    rotor = make_rotation_motor(omega[..., :3])
    translator = make_translation_motor(omega[..., 3:])
    return compose_motors(translator, rotor)


class Multivector(nn.Module):
    """A lightweight wrapper around a PGA multivector tensor."""

    GRADE_INDICES = GRADE_INDICES

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data

    def __repr__(self) -> str:
        return f"Multivector(shape={self.data.shape})"

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    def clone(self) -> "Multivector":
        return Multivector(self.data.clone())

    def to(self, *args, **kwargs) -> "Multivector":
        self.data = self.data.to(*args, **kwargs)
        return self

    def grade_projection(self, r: int) -> "Multivector":
        indices = self.GRADE_INDICES.get(r, [])
        result = torch.zeros_like(self.data)
        if indices:
            result[..., indices] = self.data[..., indices]
        return Multivector(result)

    def reverse(self) -> "Multivector":
        signs = REVERSION_SIGNS.to(device=self.data.device, dtype=self.data.dtype)
        return Multivector(self.data * signs)

    def norm(self, r: Optional[int] = None, eps: float = 1e-8) -> torch.Tensor:
        mv = self.grade_projection(r) if r is not None else self
        product = geometric_product(mv.data, mv.reverse().data)
        scalar_part = product[..., 0]
        return torch.sqrt(torch.abs(scalar_part) + eps)

    def apply_motor(self, motor: "Multivector") -> "Multivector":
        if len(motor.shape) == 2 and len(self.shape) == 3:
            batch_size, num_tokens = self.shape[:2]
            motor_data = motor.data.unsqueeze(1).expand(batch_size, num_tokens, -1).contiguous()
            motor_mult = Multivector(motor_data)
            motor_rev = motor.reverse().data.unsqueeze(1).expand(batch_size, num_tokens, -1).contiguous()
            motor_rev_mult = Multivector(motor_rev)
            return motor_mult * self * motor_rev_mult
        return motor * self * motor.reverse()

    def scalar_part(self) -> torch.Tensor:
        return self.data[..., 0]

    def get_grade_component(self, r: int) -> torch.Tensor:
        indices = self.GRADE_INDICES.get(r, [])
        result = torch.zeros_like(self.data)
        if indices:
            result[..., indices] = self.data[..., indices]
        return result

    def extract_grade(self, r: int) -> "Multivector":
        return Multivector(self.get_grade_component(r))

    def normalized(self, eps: float = 1e-8) -> "Multivector":
        denom = torch.sqrt(torch.sum(self.data ** 2, dim=-1, keepdim=True) + eps)
        return Multivector(self.data / denom)

    def __mul__(self, other: "Multivector") -> "Multivector":
        return Multivector(geometric_product(self.data, other.data))

    def __add__(self, other: "Multivector") -> "Multivector":
        return Multivector(self.data + other.data)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return Multivector(self.data - other.data)

    def __neg__(self) -> "Multivector":
        return Multivector(-self.data)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector(self.data * scalar)


def compose_motors(motor1: Multivector, motor2: Multivector) -> Multivector:
    """Compose two motors. The resulting motor applies motor2 then motor1."""
    return motor1 * motor2


def random_motor(
    batch_size: int,
    sigma_rot: float = 0.5,
    sigma_trans: float = 0.5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Multivector:
    rotation = torch.randn(batch_size, 3, device=device, dtype=dtype) * sigma_rot
    translation = torch.randn(batch_size, 3, device=device, dtype=dtype) * sigma_trans
    return exp_bivector(torch.cat([rotation, translation], dim=-1))


def random_rotation(
    batch_size: int,
    sigma: float = 0.3,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Multivector:
    rotation = torch.randn(batch_size, 3, device=device, dtype=dtype) * sigma
    return make_rotation_motor(rotation)


def random_translation(
    batch_size: int,
    sigma: float = 0.5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Multivector:
    translation = torch.randn(batch_size, 3, device=device, dtype=dtype) * sigma
    return make_translation_motor(translation)


def apply_transformation(
    points: Multivector,
    rotation: Optional[Multivector] = None,
    translation: Optional[Multivector] = None,
) -> Multivector:
    result = points
    if rotation is not None:
        result = result.apply_motor(rotation)
    if translation is not None:
        result = result.apply_motor(translation)
    return result

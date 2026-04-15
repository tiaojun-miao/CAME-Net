"""
test_came_net.py - Focused regression tests for the refactored CAME-Net stack.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from came_net import CAMENet, count_parameters
from equiv_loss import equivariance_loss, equivariance_loss_efficient
from gca import GeometricCliffordAttention
from gln import GradewiseLayerNorm
from mpe import ImageMPE, MultimodalMPE, PointCloudMPE, TextMPE
from pga_algebra import (
    GRADE_INDICES,
    Multivector,
    create_point_pga,
    extract_point_coordinates,
    exp_bivector,
    geometric_product,
    random_motor,
)


def test_motor_geometry():
    print("=" * 60)
    print("Testing Motor Geometry")
    print("=" * 60)

    coords = torch.tensor([[[1.0, 2.0, 3.0]]])

    translation_motor = exp_bivector(torch.tensor([[0.0, 0.0, 0.0, 1.0, -2.0, 0.5]]))
    translated = extract_point_coordinates(create_point_pga(coords).apply_motor(translation_motor))
    expected = torch.tensor([[[2.0, 0.0, 3.5]]])
    print(f"Translated point: {translated}")
    assert torch.allclose(translated, expected, atol=1e-4), "Translation motor is incorrect"

    random_m = random_motor(batch_size=2, sigma_rot=0.3, sigma_trans=0.5)
    identity_check = geometric_product(random_m.data, random_m.reverse().data)
    print(f"M * reverse(M): {identity_check}")
    assert torch.allclose(identity_check[..., 0], torch.ones_like(identity_check[..., 0]), atol=1e-4)
    assert torch.allclose(identity_check[..., 1:], torch.zeros_like(identity_check[..., 1:]), atol=1e-4)

    print("[PASS] Motor geometry tests passed!\n")


def test_mpe_modules():
    print("=" * 60)
    print("Testing MPE Modules")
    print("=" * 60)

    batch_size, num_points = 2, 64
    point_coords = torch.randn(batch_size, num_points, 3)
    point_features = torch.randn(batch_size, num_points, 8)
    point_mpe = PointCloudMPE(feature_dim=8, hidden_dim=32)
    point_mv = point_mpe(point_coords, point_features)
    print(f"PointCloudMPE output shape: {point_mv.data.shape}")
    assert point_mv.data.shape == (batch_size, num_points, 16)
    assert torch.allclose(point_mv.data[..., [5, 6, 7]], torch.zeros_like(point_mv.data[..., [5, 6, 7]]), atol=1e-6)
    assert point_mv.data[..., [8, 9, 10]].abs().sum().item() > 0.0

    image_features = torch.randn(batch_size, 10, 128)
    image_mv = ImageMPE(patch_dim=128, hidden_dim=32)(image_features)
    print(f"ImageMPE output shape: {image_mv.data.shape}")
    assert image_mv.data.shape == (batch_size, 10, 16)
    assert torch.allclose(image_mv.data[..., GRADE_INDICES[1]], torch.zeros_like(image_mv.data[..., GRADE_INDICES[1]]))
    assert torch.allclose(image_mv.data[..., GRADE_INDICES[2]], torch.zeros_like(image_mv.data[..., GRADE_INDICES[2]]))
    assert torch.allclose(image_mv.data[..., GRADE_INDICES[3]], torch.zeros_like(image_mv.data[..., GRADE_INDICES[3]]))

    text_features = torch.randn(batch_size, 12, 128)
    text_mv = TextMPE(token_dim=128, hidden_dim=32)(text_features)
    print(f"TextMPE output shape: {text_mv.data.shape}")
    assert text_mv.data.shape == (batch_size, 12, 16)
    assert torch.allclose(text_mv.data[..., GRADE_INDICES[1]], torch.zeros_like(text_mv.data[..., GRADE_INDICES[1]]))

    multi_mpe = MultimodalMPE(point_feature_dim=8, hidden_dim=32)
    combined_mv, splits = multi_mpe(
        point_coords=point_coords,
        point_features=point_features,
        image_patches=image_features,
        text_tokens=text_features,
        return_splits=True,
    )
    print(f"Multimodal embedding shape: {combined_mv.data.shape}")
    print(f"Splits: {splits}")
    assert combined_mv.data.shape == (batch_size, 86, 16)
    assert splits == {"point": (0, 64), "image": (64, 74), "text": (74, 86)}

    print("[PASS] MPE module tests passed!\n")


def test_point_mpe_equivariance():
    print("=" * 60)
    print("Testing PointCloudMPE Equivariance")
    print("=" * 60)

    torch.manual_seed(0)
    point_mpe = PointCloudMPE(feature_dim=0, hidden_dim=32, knn_k=8)
    point_coords = torch.randn(1, 64, 3)
    original = point_mpe(point_coords)

    motions = {
        "translation": torch.tensor([[0.0, 0.0, 0.0, 1.0, -2.0, 0.5]]),
        "rotation": torch.tensor([[0.2, -0.3, 0.6, 0.0, 0.0, 0.0]]),
    }

    for name, omega in motions.items():
        motor = exp_bivector(omega)
        transformed_coords = extract_point_coordinates(create_point_pga(point_coords).apply_motor(motor))
        transformed = point_mpe(transformed_coords)
        pushed = original.apply_motor(motor)

        grade1_error = torch.mean((transformed.data[..., GRADE_INDICES[1]] - pushed.data[..., GRADE_INDICES[1]]) ** 2)
        grade2_error = torch.mean((transformed.data[..., GRADE_INDICES[2]] - pushed.data[..., GRADE_INDICES[2]]) ** 2)
        grade3_error = torch.mean((transformed.data[..., GRADE_INDICES[3]] - pushed.data[..., GRADE_INDICES[3]]) ** 2)

        print(f"{name} grade-1 MSE: {grade1_error.item():.6e}")
        print(f"{name} grade-2 MSE: {grade2_error.item():.6e}")
        print(f"{name} grade-3 MSE: {grade3_error.item():.6e}")

        assert grade1_error.item() < 1e-8
        assert grade2_error.item() < 1e-8
        assert grade3_error.item() < 1e-8

    print("[PASS] PointCloudMPE equivariance tests passed!\n")


def test_gca_and_gln():
    print("=" * 60)
    print("Testing GCA and GLN")
    print("=" * 60)

    x = Multivector(torch.randn(2, 32, 16))
    gca = GeometricCliffordAttention(multivector_dim=16, num_heads=4, dropout=0.0)
    out = gca(x)
    print(f"GCA output shape: {out.data.shape}")
    assert out.data.shape == x.data.shape

    pure_grade4 = Multivector(torch.zeros(1, 4, 16))
    pure_grade4.data[..., 15] = 2.0
    grade4_out = gca(pure_grade4)
    print(f"Grade-4 response sum: {grade4_out.data[..., 15].abs().sum().item():.4f}")
    assert grade4_out.data[..., 15].abs().sum().item() > 0.0

    gln = GradewiseLayerNorm()
    normalized = gln(x)
    print(f"GLN output shape: {normalized.data.shape}")
    assert normalized.data.shape == x.data.shape

    print("[PASS] GCA and GLN tests passed!\n")


def test_came_net_forward_pass():
    print("=" * 60)
    print("Testing CAME-Net Forward Pass")
    print("=" * 60)

    model = CAMENet(num_classes=7, point_feature_dim=8, num_layers=2, num_heads=4, hidden_dim=32, multimodal=True)
    print(f"Model parameters: {count_parameters(model):,}")

    point_coords = torch.randn(2, 64, 3)
    point_features = torch.randn(2, 64, 8)
    image_patches = torch.randn(2, 10, 128)
    text_tokens = torch.randn(2, 12, 128)

    logits = model(
        point_coords=point_coords,
        point_features=point_features,
        image_patches=image_patches,
        text_tokens=text_tokens,
    )
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (2, 7)

    embedding = model.get_point_cloud_embedding(point_coords, point_features)
    print(f"Point embedding shape: {embedding.shape}")
    assert embedding.shape == (2, 16)

    latent, splits = model.get_latent_multivector(
        point_coords=point_coords,
        point_features=point_features,
        image_patches=image_patches,
        text_tokens=text_tokens,
        return_splits=True,
    )
    print(f"Latent multivector shape: {latent.data.shape}")
    print(f"Latent splits: {splits}")
    assert latent.data.shape == (2, 86, 16)

    print("[PASS] CAME-Net forward pass tests passed!\n")


def test_equivariance_losses():
    print("=" * 60)
    print("Testing Equivariance Losses")
    print("=" * 60)

    model = CAMENet(num_classes=5, num_layers=2, num_heads=2, hidden_dim=32)
    model.eval()

    point_coords = torch.randn(2, 64, 3)
    batch = {
        "point_coords": point_coords,
        "labels": torch.randint(0, 5, (2,)),
    }

    loss = equivariance_loss(model, batch, num_samples=2)
    efficient_loss = equivariance_loss_efficient(model, point_coords, None, num_samples=2)
    print(f"Equivariance loss: {loss.item():.6f}")
    print(f"Efficient equivariance loss: {efficient_loss.item():.6f}")

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert not torch.isnan(efficient_loss)
    assert not torch.isinf(efficient_loss)

    print("[PASS] Equivariance loss tests passed!\n")


def test_training_step():
    print("=" * 60)
    print("Testing Training Step")
    print("=" * 60)

    model = CAMENet(num_classes=6, point_feature_dim=4, num_layers=2, num_heads=2, hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    point_coords = torch.randn(2, 64, 3)
    point_features = torch.randn(2, 64, 4)
    labels = torch.randint(0, 6, (2,))

    optimizer.zero_grad()
    logits = model(point_coords=point_coords, point_features=point_features)
    loss = criterion(logits, labels)
    print(f"Task loss: {loss.item():.6f}")
    loss.backward()

    gradients_exist = any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Gradients exist: {gradients_exist}")
    assert gradients_exist

    optimizer.step()
    print("[PASS] Training step test passed!\n")


def test_grade_indices():
    print("=" * 60)
    print("Testing GRADE_INDICES")
    print("=" * 60)

    all_indices = []
    for indices in GRADE_INDICES.values():
        all_indices.extend(indices)

    assert sorted(all_indices) == list(range(16))
    print(f"GRADE_INDICES: {GRADE_INDICES}")
    print("[PASS] GRADE_INDICES test passed!\n")


def run_all_tests():
    print("\n" + "=" * 60)
    print("CAME-Net Regression Test Suite")
    print("=" * 60 + "\n")

    test_motor_geometry()
    test_mpe_modules()
    test_point_mpe_equivariance()
    test_gca_and_gln()
    test_came_net_forward_pass()
    test_equivariance_losses()
    test_training_step()
    test_grade_indices()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

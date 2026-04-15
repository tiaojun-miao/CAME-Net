"""Focused method-alignment tests for CAME-Net."""

import torch

from gca import GeometricCliffordAttention, GradeWiseMLP
from gln import GradewiseLayerNorm
from mpe import PointCloudMPE
from pga_algebra import Multivector


def test_point_mpe_uses_euclidean_bivectors_only():
    torch.manual_seed(0)
    mpe = PointCloudMPE(feature_dim=8, hidden_dim=16, knn_k=8)
    coords = torch.randn(2, 32, 3)
    features = torch.randn(2, 32, 8)

    mv = mpe(coords, features).data

    assert torch.allclose(mv[..., [8, 9, 10]], torch.zeros_like(mv[..., [8, 9, 10]]), atol=1e-6)
    assert mv[..., [5, 6, 7]].abs().sum().item() > 0.0


def test_non_scalar_grade_maps_do_not_use_bias():
    gca = GeometricCliffordAttention(multivector_dim=16, num_heads=2, dropout=0.0)
    mlp = GradeWiseMLP(dropout=0.0)

    for grade in (1, 2, 3):
        assert gca.grade_query_projs[str(grade)].bias is None
        assert gca.grade_key_projs[str(grade)].bias is None
        assert gca.grade_value_projs[str(grade)].bias is None
        assert gca.grade_out_projs[str(grade)].bias is None
        assert mlp.blocks[str(grade)][0].bias is None
        assert mlp.blocks[str(grade)][3].bias is None

    for grade in (0, 4):
        assert gca.grade_query_projs[str(grade)].bias is not None
        assert gca.grade_out_projs[str(grade)].bias is not None


def test_gln_bias_is_applied_only_to_scalar_and_pseudoscalar():
    gln = GradewiseLayerNorm()
    with torch.no_grad():
        gln.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    x = Multivector(torch.zeros(1, 1, 16))
    out = gln(x).data

    assert torch.allclose(out[..., 0], torch.ones_like(out[..., 0]))
    assert torch.allclose(out[..., 15], torch.full_like(out[..., 15], 5.0))
    assert torch.allclose(out[..., 1:15], torch.zeros_like(out[..., 1:15]))

def test_multivector_distance_respects_grade_weights():
    from equiv_loss import multivector_distance

    a = Multivector(torch.zeros(1, 1, 16))
    b = Multivector(torch.zeros(1, 1, 16))
    b.data[..., 5] = 2.0

    distance_disabled = multivector_distance(a, b, eta={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    distance_enabled = multivector_distance(a, b, eta={0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0})

    assert distance_disabled.item() == 0.0
    assert distance_enabled.item() > 0.0


def test_training_equivariance_helper_matches_eval_mode_and_restores_train_state():
    from equiv_loss import equivariance_loss_efficient
    from train import _equivariance_loss_from_batch

    torch.manual_seed(0)
    model = GeometricCliffordAttention  # keep linters quiet about local imports
    del model
    net = __import__("came_net").CAMENet(num_classes=5, num_layers=2, num_heads=2, hidden_dim=32, dropout=0.1)
    net.train()
    batch = {
        "point_coords": torch.randn(2, 64, 3),
        "labels": torch.randint(0, 5, (2,)),
    }

    torch.manual_seed(123)
    helper_loss = _equivariance_loss_from_batch(equivariance_loss_efficient, net, batch, torch.device("cpu"))

    torch.manual_seed(123)
    net.eval()
    direct_eval_loss = equivariance_loss_efficient(net, batch["point_coords"], None)
    net.train()

    assert torch.allclose(helper_loss, direct_eval_loss, atol=1e-6)
    assert net.training


if __name__ == "__main__":
    test_point_mpe_uses_euclidean_bivectors_only()
    test_non_scalar_grade_maps_do_not_use_bias()
    test_gln_bias_is_applied_only_to_scalar_and_pseudoscalar()
    test_multivector_distance_respects_grade_weights()
    test_training_equivariance_helper_matches_eval_mode_and_restores_train_state()
    print("method alignment tests passed")

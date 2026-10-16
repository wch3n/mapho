import numpy as np
import torch
from e3nn import o3
from mapho.vasp.paw_conditioning import (
    scalar_projection,
    overlap_coordinates,
    ScaledPAWHead,
    ConditionedPAWModel,
)
from mapho.vasp.model import native_projector_transform, PAWModel, atomic_inputs
from mapho.field_model import FieldModelConfig, mace_graph


def test_overlap_scaling_handles_signs_and_preserves_native_rotations():
    channels = [0, 0, 1, 1, 2, 2]
    n = sum(2 * l + 1 for l in channels)
    rng = np.random.default_rng(19)
    q = scalar_projection(rng.normal(size=(n, n)), channels)
    b, w, info = overlap_coordinates(q, channels)
    np.testing.assert_allclose(w @ b, np.eye(n), atol=2e-14)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        r = -o3.rand_matrix()
        t = native_projector_transform(channels)
        u = (
            t
            @ o3.Irreps([(1, (l, (-1) ** l)) for l in channels]).D_from_matrix(r)
            @ t.T
        ).numpy()
        np.testing.assert_allclose(u @ b, b @ u, atol=1e-13)
        x = rng.normal(size=(n, n))
        x = (x + x.T) / 2
        np.testing.assert_allclose(
            b @ (u @ x @ u.T) @ b.T, u @ (b @ x @ b.T) @ u.T, atol=1e-12
        )
    finally:
        torch.set_default_dtype(old)
    assert info["floored_eigenvalues"] == 0


def test_small_radial_directions_survive_double_precision_reconstruction():
    q = np.array([[1.0, 1.0 - 2e-8], [1.0 - 2e-8, 1.0]])
    b, w, _ = overlap_coordinates(q, [0, 0], relative_floor=1e-10)
    head = ScaledPAWHead(o3.Irreps("2x0e"), [0, 0], b, np.eye(2), 1.0)
    encoded = torch.tensor([[[0.1, 0.03], [0.03, -0.2]]], requires_grad=True)
    physical = head.physical(encoded)
    assert physical.dtype == torch.float64
    recovered = w @ physical.detach().numpy() @ w - np.eye(2)
    np.testing.assert_allclose(recovered, encoded.detach().numpy(), atol=1e-8)
    physical.square().sum().backward()
    assert torch.isfinite(encoded.grad).all()


def test_frozen_field_is_unchanged_and_corrected_matrix_is_symmetric():
    config = FieldModelConfig(channels=2, radial_count=4, grid_size=6, correlation=2)
    base = PAWModel(config, mode="full").eval()
    for p in base.local.coefficients.parameters():
        torch.nn.init.normal_(p, std=0.02)
    inputs = atomic_inputs(
        np.array([14, 14]),
        np.array([[0.0, 0.0, 0.0], [1.3, 1.3, 1.3]]),
        np.eye(3) * 5.4,
        np.zeros((6, 6, 6)),
    )
    graph = mace_graph(inputs, config)
    q = np.eye(8)
    b, _, _ = overlap_coordinates(q, [0, 0, 1, 1])
    spec = {
        14: dict(
            channels=[0, 0, 1, 1],
            decode=b.tolist(),
            origin=np.eye(8).tolist(),
            scale=0.1,
        )
    }
    model = ConditionedPAWModel(base, spec)
    before = base.coefficients(graph)
    after = model.coefficients(graph)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
    torch.testing.assert_close(
        after[2], torch.eye(8, dtype=torch.float64).expand(2, -1, -1), rtol=0, atol=0
    )
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())


def test_joint_losses_reach_backbone_field_fourier_and_conditioned_heads():
    from mapho.vasp.model import point_basis

    torch.manual_seed(12)
    config = FieldModelConfig(
        channels=2, radial_count=4, grid_size=4, correlation=2, fno_width=2
    )
    base = PAWModel(config, mode="full").eval()
    for module in (base.local.coefficients, base.local.offset, base.correction.readout):
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.04)
    inputs = atomic_inputs(
        np.array([14, 14]),
        np.array([[0.0, 0.0, 0.0], [1.3, 1.3, 1.3]]),
        np.eye(3) * 5.4,
        np.zeros((4, 4, 4)),
    )
    graph = mace_graph(inputs, config)
    fractions = np.stack(np.unravel_index(np.arange(64), (4, 4, 4)), -1) / 4
    b, _, _ = overlap_coordinates(np.eye(8), [0, 0, 1, 1])
    spec = {
        14: dict(
            channels=[0, 0, 1, 1],
            decode=b.tolist(),
            origin=np.eye(8).tolist(),
            scale=0.1,
        )
    }
    frozen = ConditionedPAWModel(base, spec).eval()
    joint = ConditionedPAWModel(base, spec, train_base=True).eval()
    joint.heads.load_state_dict(frozen.heads.state_dict())
    item = dict(
        graph=graph,
        shape=(4, 4, 4),
        probes=point_basis(inputs, fractions, config),
        baseline=torch.zeros(4, 4, 4),
        envelope=torch.tensor(inputs.reference_density, dtype=torch.float32),
        cell=torch.tensor(inputs.cell, dtype=torch.float32),
    )
    before = frozen(item)
    after = joint(item)
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
    for p in joint.heads.parameters():
        torch.nn.init.normal_(p, std=0.03)
    field, d, encoded = joint(item, return_encoded=True)
    field_loss = (field - 0.3).square().mean()
    paw_loss = sum((v - 0.1).square().mean() for v in encoded.values())
    backbone = [p for p in joint.base.local.backbone.parameters() if p.requires_grad]
    for component in (field_loss, paw_loss):
        gradients = torch.autograd.grad(
            component, backbone, retain_graph=True, allow_unused=True
        )
        assert any(g is not None and bool(torch.count_nonzero(g)) for g in gradients)
    loss = field_loss + paw_loss
    loss.backward()
    for prefix in (
        "base.local.backbone",
        "base.local.coefficients",
        "base.local.offset",
        "base.correction",
        "heads",
    ):
        grads = [
            p.grad
            for n, p in joint.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        ]
        assert any(g is not None and bool(torch.count_nonzero(g)) for g in grads), (
            prefix
        )
        assert all(g is None or torch.isfinite(g).all() for g in grads), prefix
    assert all(
        p.grad is None and not p.requires_grad
        for n, p in joint.base.named_parameters()
        if n.startswith("onsite_")
    )
    assert d.dtype == torch.float64

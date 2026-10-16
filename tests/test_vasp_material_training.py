import json
import h5py
import numpy as np
import torch

from mapho.vasp.train import prepare, load_items, onsite_mse, load_model, predict
from mapho.vasp.model import PAWModel
from mapho.field_model import FieldModelConfig
from mapho.vasp.paw import sha


def test_binary_rectangular_training_cache_and_native_prediction(tmp_path):
    dataset = tmp_path / "data"
    atomic = tmp_path / "atomic"
    dataset.mkdir()
    atomic.mkdir()
    split = {}
    channels = [[0, 1], [0, 1, 2]]
    for i, key in enumerate(["train", "validation", "test"]):
        name = f"sample_{i:03d}"
        source = dataset / "samples" / name
        source.mkdir(parents=True)
        baseline = atomic / name
        baseline.mkdir()
        (source / "POSCAR").write_text(f"geometry {i}")
        split[key] = [f"samples/{name}/operator.h5"]
        x = np.arange(6)[:, None, None]
        field = np.broadcast_to(np.cos(x), (6, 6, 8)).copy()
        for folder, is_target in [(baseline, False), (source, True)]:
            with h5py.File(folder / "operator.h5", "w") as f:
                f.attrs["schema"] = "mlham.vasp_paw_operator.v1"
                f.attrs["metadata_json"] = json.dumps(
                    {"reference_manifest": {"settings": {"ICHARG": 12}}}
                )
                f["cell_A"] = np.diag([6.0, 6.0, 8.0])
                f["positions_A"] = [[0.7 + i * 0.01, 0.3, 0.5], [2.4, 1.2, 1.7]]
                f["atomic_numbers"] = [16, 42]
                f["paw/types"] = [0, 1]
                d = np.zeros((2, 9, 9))
                d[0, :4, :4] = np.eye(4) * (0.1 if is_target else 0.0)
                d[1] = np.eye(9) * (0.2 if is_target else 0.0)
                f["paw/d"] = d
                f["paw/q"] = np.zeros_like(d)
                f["paw"].attrs["channels_json"] = json.dumps(channels)
                f["potential_eV"] = field + (0.1 * np.sin(x) if is_target else 0.0)
    (dataset / "manifest.json").write_text(json.dumps({"splits": split}))
    (dataset / "data_audit.json").write_text('{"passed":true}')
    (atomic / "completion.json").write_text('{"complete":true}')
    (atomic / "manifest.json").write_text(
        json.dumps({"dataset_manifest_sha256": sha(dataset / "manifest.json")})
    )
    root = tmp_path / "training"
    m = prepare(
        dataset, atomic, root, grid_size=6, grid_shape=(6, 6, 8), point_count=16
    )
    assert m["species_channels"] == {16: [0, 1], 42: [0, 1, 2]}
    assert m["config"]["atomic_numbers"] == (16, 42)
    config = FieldModelConfig(**m["config"])
    model = PAWModel(config, mode="none", species_channels=m["species_channels"])
    data = load_items(root, "train", config, "cpu")[0]
    field, d = model(data)
    assert data["shape"] == (6, 6, 8) and d.shape == (2, 9, 9)
    loss = (field - data["target"]).square().mean() + onsite_mse(d, data)
    loss.backward()
    assert torch.isfinite(loss)
    v, dd = predict(model, m["entries"]["test"][0], "cpu")
    with h5py.File(m["entries"]["test"][0]["baseline"]) as f:
        np.testing.assert_array_equal(v, f["potential_eV"][:])
        np.testing.assert_array_equal(dd, f["paw/d"][:])

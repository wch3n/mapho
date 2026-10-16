"""Exercise tutorial orchestration with synthetic operators and no native VASP."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from ase.io import write
import h5py
import numpy as np
import pytest

from mapho.vasp.paw import save_json, sha


@pytest.fixture
def tutorial(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/vasp/si_bands.py"
    spec = importlib.util.spec_from_file_location("tutorial_bands", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    training, dataset, source = (
        tmp_path / n for n in ("training", "dataset", "source")
    )
    for folder in (training / "full", dataset, source):
        folder.mkdir(parents=True)
    executable = tmp_path / "synthetic_executable"
    executable.write_text("Never executed; native calls are replaced in this test.")
    (source / "POTCAR").write_text("Synthetic identity; never supplied to VASP.")
    (training / "full/model.pt").write_text("Synthetic checkpoint identity.")
    baseline = source / "operator.h5"
    with h5py.File(baseline, "w") as f:
        f.attrs["metadata_json"] = json.dumps(
            dict(
                reference=str(source),
                reference_manifest=dict(
                    settings=dict(ENCUT=600),
                    inputs={"POTCAR": sha(source / "POTCAR")},
                ),
            )
        )
    save_json(
        dataset / "manifest.json",
        dict(
            executable=str(executable),
            executable_sha256=sha(executable),
        ),
    )
    save_json(
        training / "manifest.json",
        dict(
            target_kind="pbe",
            config=dict(atomic_numbers=[14]),
            dataset=str(dataset),
            entries=dict(
                train=[dict(baseline=str(baseline), baseline_sha256=sha(baseline))]
            ),
        ),
    )
    events = []
    levels = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])

    def prepare(folder, atoms, potcars, *, encut, mesh, nbands):
        folder.mkdir()
        write(folder / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
        (folder / "POTCAR").write_text("Synthetic PAW identity.")
        (folder / "KPOINTS").write_text("Synthetic mesh.")
        return dict(settings=dict(ENCUT=encut, ICHARG=2, NBANDS=nbands))

    def export(folder, output):
        native = example.read_json(folder / "manifest.json")
        with h5py.File(output, "w") as f:
            f.attrs["schema"] = "mapho.vasp_paw_operator.v1"
            f.attrs["metadata_json"] = json.dumps(
                dict(reference=str(folder), reference_manifest=native)
            )
            f["potential_eV"] = np.ones((4, 4, 4))
            f["cell_A"] = np.eye(3) * 5
            f["positions_A"] = [[0, 0, 0], [1, 1, 1]]
            f["atomic_numbers"] = [14, 14]
            f["paw/d"] = np.tile(np.eye(8), (2, 1, 1))
            f["paw/q"] = np.zeros((2, 8, 8))
            f["paw/types"] = [0, 0]
            f["paw"].attrs["channels_json"] = "[[0,0,1,1]]"
            f["bands/eigenvalues_eV"] = levels[None]
            f["bands/kpoints"] = [[0.0, 0.0, 0.0]]
            f["native_pot_header"] = (folder / "POSCAR").read_text()
            f["native_pot_suffix"] = ""

    def report(folder):
        save_json(
            folder / "report.json",
            dict(complete=True, nocc=4, energy_eV=-10.0, sampled_gap_eV=2.0),
        )

    def predict(model, entry, device):
        assert set(entry) == {"baseline", "baseline_sha256"}
        assert Path(entry["baseline"]).parent.name == "atomic"
        events.append("predict")
        return np.ones((4, 4, 4)) * 2, np.tile(np.eye(8) * 2, (2, 1, 1))

    def bands(folder):
        points = example.silicon_path(np.eye(3))["kpoints"]
        shift = 0.025 if folder.name == "mapho_bands" else 0
        return dict(
            kpoints=np.asarray(points), eigenvalues_eV=np.tile(levels + shift, (81, 1))
        )

    monkeypatch.setattr(example, "load_model", lambda *a: object())
    monkeypatch.setattr(example, "prepare_reference", prepare)
    monkeypatch.setattr(
        example, "run", lambda folder, *a, **kw: events.append(folder.name)
    )
    monkeypatch.setattr(example, "export_operator", export)
    monkeypatch.setattr(example, "case_report", report)
    monkeypatch.setattr(example, "predict", predict)
    monkeypatch.setattr(example, "compare", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(example, "read_bands", bands)
    args = argparse.Namespace(
        training=training,
        output=tmp_path / "bands",
        mode="full",
        device="cpu",
        meshes=[14, 16],
        launcher="",
    )
    return example, args, events


def test_prediction_precedes_references_and_errors_preserve_energy_offset(tutorial):
    pytest.importorskip("matplotlib")
    example, args, events = tutorial
    metrics = example.execute(args)
    assert events == [
        "atomic",
        "predict",
        "reference_14",
        "reference_16",
        "replay_control",
        "pbe_bands",
        "mapho_bands",
    ]
    assert metrics["frontier_mae_meV"] == pytest.approx(25)
    assert metrics["pbe_path_gap_eV"] == pytest.approx(metrics["predicted_path_gap_eV"])
    with (args.output / "band_source_data.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 81 * 8
    assert all(float(row["error_meV"]) == pytest.approx(25) for row in rows)
    assert (args.output / "bands.pdf").stat().st_size > 0
    assert example.read_json(args.output / "completion.json")["complete"]
    imported = example.read_json(args.output / "mapho_bands/manifest.json")
    assert imported["settings"]["ICHARG"] == 14
    assert Path(imported["operator"]).parent.name == "atomic"


@pytest.mark.parametrize("gate", ["sampling", "replay"])
def test_failed_native_controls_prevent_band_comparison(tutorial, monkeypatch, gate):
    example, args, events = tutorial
    function = "compare_operators" if gate == "sampling" else "compare"
    monkeypatch.setattr(example, function, lambda *a, **kw: dict(passed=False))
    with pytest.raises(RuntimeError, match="sampling|replay"):
        example.execute(args)
    assert "mapho_bands" not in events
    assert not (args.output / "completion.json").exists()
    assert not (args.output / "bands.pdf").exists()


def test_changed_executable_rejected_before_output_creation(tutorial):
    example, args, events = tutorial
    dataset = example.read_json(args.training / "manifest.json")["dataset"]
    executable = Path(example.read_json(Path(dataset) / "manifest.json")["executable"])
    executable.write_text("Changed synthetic identity.")
    with pytest.raises(ValueError, match="executable changed"):
        example.execute(args)
    assert not args.output.exists()
    assert events == []

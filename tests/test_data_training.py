import json

import h5py
import numpy as np
import pytest
import torch

from mapho.data import save_sample, load_sample
from mapho.io import read_blocks_h5, write_blocks_h5, read_wannier90_hr
from mapho.model import ModelConfig
from mapho.synthetic import synthetic_sample
from mapho.train import fit, load_checkpoint


def test_sample_round_trip_and_invalid_graph(tmp_path):
    sample = synthetic_sample(1, "spd")
    path = tmp_path / "sample.npz"
    save_sample(path, sample)
    loaded = load_sample(path)
    torch.testing.assert_close(loaded.hamiltonian, sample.hamiltonian)
    torch.testing.assert_close(loaded.graph.pairs, sample.graph.pairs)
    with pytest.raises(FileExistsError):
        save_sample(path, sample)
    with np.load(path) as source:
        content = {key: source[key] for key in source.files}
    content["pairs"] = content["pairs"][:-1]
    np.savez(tmp_path / "bad.npz", **content)
    with pytest.raises(ValueError, match="complete canonical"):
        load_sample(tmp_path / "bad.npz")


def test_hdf5_round_trip_and_malformed_bounds(tmp_path):
    blocks = {
        (0, 0, 0, 0, 0): np.eye(2, dtype=complex),
        (1, 0, 0, 0, 1): np.array([[1 + 2j], [3]]),
    }
    path = tmp_path / "blocks.h5"
    write_blocks_h5(path, blocks)
    restored = read_blocks_h5(path)
    for key in blocks:
        np.testing.assert_array_equal(restored[key], blocks[key])
    with h5py.File(path, "r+") as data:
        data["chunk_boundaries"][1] = 3
    with pytest.raises(ValueError, match="Block sizes"):
        read_blocks_h5(path)


def test_wannier_degeneracy_and_complex_values(tmp_path):
    path = tmp_path / "toy_hr.dat"
    path.write_text(
        "generated fixture\n1\n3\n2 1 2\n"
        "-1 0 0 1 1 -2.0 -0.4\n0 0 0 1 1 0.5 0\n1 0 0 1 1 -2.0 0.4\n"
    )
    raw = read_wannier90_hr(path)
    k = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    expected = 0.5 - 2 * np.cos(2 * np.pi * k[:, 0]) - 0.4 * np.sin(2 * np.pi * k[:, 0])
    np.testing.assert_allclose(raw.bloch(k)[:, 0, 0], expected, atol=1e-12)
    assert np.iscomplexobj(raw.matrices)
    path.write_text("truncated\n1\n2\n")
    with pytest.raises(ValueError, match="Truncated"):
        read_wannier90_hr(path)


def test_training_reduces_error_and_reloads_checkpoint(tmp_path):
    samples = [synthetic_sample(seed, "s") for seed in range(12)]
    model, report, partitions = fit(
        samples,
        tmp_path / "fit",
        ModelConfig(basis="s", channels=4, layers=1),
        epochs=16,
        batch_size=8,
        learning_rate=0.01,
    )
    assert report["validation"]["mae_eV"] < report["initial_validation"]["mae_eV"] * 0.7
    restored, _ = load_checkpoint(tmp_path / "fit" / "model.pt")
    graph = partitions["test"][0].graph
    torch.testing.assert_close(restored(graph), model(graph))
    split = json.loads((tmp_path / "fit" / "split.json").read_text())
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
    assert sum(map(len, split.values())) == len(samples)


def test_complex_labels_rejected_before_training(tmp_path):
    samples = [synthetic_sample(seed, "s") for seed in range(8)]
    samples[0].hamiltonian = samples[0].hamiltonian.to(torch.complex64)
    with pytest.raises(ValueError, match="real spinless"):
        fit(samples, tmp_path / "fit", ModelConfig(basis="s"))


def test_inconsistent_energy_reference_rejected(tmp_path):
    samples = [synthetic_sample(seed, "s") for seed in range(8)]
    samples[0].metadata["energy_reference"] = "a different reference"
    with pytest.raises(ValueError, match="energy_reference"):
        fit(samples, tmp_path / "fit", ModelConfig(basis="s"))

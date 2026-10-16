"""Reproducible pilot comparing constrained MACE, pointwise and FNO exchange.

Evaluation isolates exchange with the exact hybrid local Hamiltonian. Reported
Ritz eigenvalues are eight-state subspace diagnostics, not a full ML band solve.
"""

from .schema import canonical_schema

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .exchange import (
    ExchangeGate,
    FockExchange,
    gygi_baldereschi_zero,
    pw_to_grid,
    pw_mask,
)
from .field_model import FieldInputs, FieldModelConfig, mace_graph
from .probes import PeriodicProbeGrid


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


class PreparedExchange:
    def __init__(self, directory, config, *, double=False, target_functional="HSE"):
        self.directory = Path(directory)
        p = dict(np.load(self.directory / "PBE/exchange.npz", allow_pickle=False))
        h = dict(
            np.load(
                self.directory / target_functional / "exchange.npz", allow_pickle=False
            )
        )
        self.report = json.loads(
            (self.directory / target_functional / "report.json").read_text()
        )
        if (
            target_functional not in ("HSE", "PBE0")
            or self.report["functional"] != target_functional
        ):
            raise ValueError("Hybrid target mismatch")
        p_report = json.loads((self.directory / "PBE/report.json").read_text())
        self.nocc = int(self.report["nocc_per_spin"])
        if self.nocc != p_report["nocc_per_spin"] or not 1 <= self.nocc <= 6:
            raise ValueError(
                "PBE/hybrid occupied-state counts differ or are unsupported"
            )
        for key in (
            "cell_bohr",
            "mill",
            "ngk",
            "shape",
            "kpoints",
            "weights",
            "numbers",
            "positions",
        ):
            if not np.allclose(p[key], h[key], rtol=0, atol=1e-12):
                raise ValueError(f"PBE/hybrid grids differ: {key}")
        self.data = h
        self.shape = tuple(map(int, h["shape"]))
        dtype = torch.float64 if double else torch.float32
        cdtype = torch.complex128 if double else torch.complex64

        def grids(coeff):
            return np.stack(
                [
                    pw_to_grid(c[:, :n], m[:n], self.shape)
                    for c, m, n in zip(coeff, h["mill"], h["ngk"])
                ]
            )

        masks = np.stack(
            [pw_mask(m[:n], self.shape) for m, n in zip(h["mill"], h["ngk"])]
        )
        self.psi = torch.tensor(grids(h["orbitals"]), dtype=cdtype)
        self.target = torch.tensor(grids(h["actions"]), dtype=cdtype)
        self.operator_options = dict(
            mu=self.report["screening_bohr_inverse"],
            alpha=self.report["exchange_fraction"],
            rydberg_eV=self.report["rydberg_eV"],
        )
        if target_functional == "PBE0":
            if (
                self.report["exxdiv_treatment"] != "gygi-baldereschi"
                or self.report["x_gamma_extrapolation"]
                or self.operator_options["mu"] != 0
            ):
                raise ValueError("Unsupported unscreened singularity treatment")
            # Derive from the PBE input geometry, not a fitted hybrid scalar.
            zero = gygi_baldereschi_zero(
                p["cell_bohr"],
                p_report["ecutwfc_ry"],
                p_report["ecutrho_ry"],
                self.report["kmesh"],
            )
            if not np.isclose(
                zero, self.report["coulomb_zero_mode_bohr2"], rtol=1e-10, atol=1e-10
            ):
                raise ValueError(
                    "Independent Coulomb singular correction disagrees with QE"
                )
            self.operator_options["zero_mode"] = zero
        self.op = FockExchange(
            torch.tensor(grids(p["orbitals"][:, : self.nocc]), dtype=cdtype),
            p["kpoints"],
            p["weights"],
            p["cell_bohr"],
            masks,
            **self.operator_options,
            recompute=False,
        )
        self.cell = torch.tensor(
            h["cell_bohr"] * self.report["bohr_angstrom"], dtype=dtype
        )
        self.density = torch.tensor(p["density"], dtype=dtype)
        self.hxc = torch.tensor(p["hxc"], dtype=dtype)
        inputs = FieldInputs(
            p["numbers"], p["positions"], self.cell.numpy(), p["density"], p["hxc"]
        )
        self.graph = mace_graph(inputs, config)
        self.probes = PeriodicProbeGrid(inputs, self.shape, config)

    def gate(self, model):
        return model(self.graph, self.probes, self.cell, self.density, self.hxc)


def action_loss(sample, gate, k, bands=6, start=0):
    error = (
        sample.op.apply(sample.psi[k, start : start + bands], k, gate)
        - sample.target[k, start : start + bands]
    )
    return error.abs().square().sum() / bands


def validation(model, samples):
    model.eval()
    values = []
    with torch.no_grad():
        for s in samples:
            gate = s.gate(model)
            for k in (0, 3):
                values.append(float(action_loss(s, gate, k)))
    return float(np.mean(values))


def train(root, output, variant, seed, epochs, train_samples, val_samples, config):
    out = output / f"{variant}_seed{seed}"
    out.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    model = ExchangeGate(variant, config)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.0001)
    best_loss = validation(model, val_samples)
    best = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [{"epoch": 0, "validation_action_mse_eV2": best_loss}]
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for i in rng.permutation(len(train_samples)):
            s = train_samples[i]
            # Cycle the six seen k points; each gets equal exposure over six epochs.
            cycle = epoch - 1 + int(i)
            k = cycle % 6
            band_start = 2 * ((cycle // 6) % 3)
            opt.zero_grad(set_to_none=True)
            gate = s.gate(model)
            loss = action_loss(s, gate, k, bands=2, start=band_start)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {"epoch": epoch, "train_action_mse_eV2": float(np.mean(losses))}
        if epoch % 6 == 0 or epoch == epochs:
            score = validation(model, val_samples)
            row["validation_action_mse_eV2"] = score
            if score < best_loss:
                best_loss = score
                best = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            print(
                f"{variant} seed={seed} epoch={epoch} validation={score**0.5 * 1000:.4f} meV",
                flush=True,
            )
        history.append(row)
    seconds = time.perf_counter() - started
    model.load_state_dict(best)
    changed = {}
    for prefix in (
        "local.backbone",
        "local.coefficients",
        "local.offset",
        "correction",
    ):
        changed[prefix] = sum(
            not torch.equal(v, initial[k])
            for k, v in best.items()
            if k.startswith(prefix)
        )
    report = {
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "updates": epochs * len(train_samples),
        "best_epoch": best_epoch,
        "best_validation_action_mse_eV2": best_loss,
        "training_wall_seconds": seconds,
        "changed_state_tensors": changed,
        "process_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "reference": "converged PBE occupied orbitals; same-spin Fock exchange",
        "target_functional": train_samples[0].report["functional"],
        "model": "P_k M_g K[PBE] M_g P_k; g=exp(0.1*tanh(MACE+correction))",
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    torch.save(
        {
            "schema": "mapho.exchange_gate.v1",
            "config": asdict(config),
            "variant": variant,
            "state_dict": best,
            "metadata": report,
        },
        out / "model.pt",
    )
    dump(out / "history.json", history)
    dump(out / "training.json", report)
    return model


def validate_native(root, output, config):
    manifest = json.loads((root / "manifest.json").read_text())
    name = manifest["splits"]["train"][0]
    s = PreparedExchange(
        root / name,
        config,
        double=True,
        target_functional=manifest.get("target_functional", "HSE"),
    )
    h = s.data
    op = FockExchange(
        s.psi[:, : s.nocc],
        h["kpoints"],
        h["weights"],
        h["cell_bohr"],
        s.op.masks.numpy(),
        **s.operator_options,
    )
    rows = []
    with torch.no_grad():
        for k in range(len(h["kpoints"])):
            y = op.apply(s.psi[k], k)
            error = y - s.target[k]
            rows.append(
                {
                    "k_index": k,
                    "max_action_error_eV": float(error.flatten(1).norm(dim=1).max()),
                    "relative_error": float(error.norm() / s.target[k].norm()),
                }
            )
    result = {
        "independent_fft_vs_native_hybrid": rows,
        "target_functional": s.report["functional"],
        "operator_options": s.operator_options,
        "max_error_eV": max(r["max_action_error_eV"] for r in rows),
        "native_ace_band_max_error_eV": s.report["ace_band_max_error_eV"],
        "passed": max(r["max_action_error_eV"] for r in rows) < 1e-8,
    }
    dump(output / "native_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Independent exchange does not reproduce QE")


def evaluate(root, output, models, config):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    timings = []
    probe_rows = []
    with torch.no_grad():
        for name in manifest["splits"]["test"]:
            s = PreparedExchange(
                root / name,
                config,
                target_functional=manifest.get("target_functional", "HSE"),
            )
            gates = {"pbe_reference": None}
            for label, model in models.items():
                model.eval()
                start = time.perf_counter()
                gates[label] = s.gate(model)
                timings.append(
                    {
                        "sample": name,
                        "model": label,
                        "gate_seconds": time.perf_counter() - start,
                        "gate_min": float(gates[label].min()),
                        "gate_max": float(gates[label].max()),
                        "gate_rms_deviation": float(
                            (gates[label] - 1).square().mean().sqrt()
                        ),
                    }
                )
            for k in range(8):
                psi = s.psi[k]
                target = s.target[k]
                n = int(s.data["ngk"][k])
                mill = s.data["mill"][k, :n]
                probe = torch.tensor(
                    pw_to_grid(s.data["probes"][k, :, :n], mill, s.shape),
                    dtype=psi.dtype,
                )
                probe_target = torch.tensor(
                    pw_to_grid(s.data["probe_actions"][k, :, :n], mill, s.shape),
                    dtype=psi.dtype,
                )
                target_matrix = psi.flatten(1).conj() @ target.flatten(1).T
                for label, gate in gates.items():
                    t = time.perf_counter()
                    pred = s.op.apply(psi, k, gate)
                    seconds = time.perf_counter() - t
                    error = pred - target
                    norms = error.flatten(1).norm(dim=1).numpy()
                    reference_norms = target.flatten(1).norm(dim=1).numpy()
                    matrix = psi.flatten(1).conj() @ pred.flatten(1).T
                    herm = float((matrix - matrix.conj().T).abs().max())
                    if herm > 5e-5 or not torch.isfinite(pred).all():
                        raise RuntimeError(
                            "Predicted exchange violates numerical operator checks"
                        )
                    delta = (matrix - target_matrix).numpy().astype(np.complex128)
                    ritz = np.linalg.eigvalsh(
                        np.diag(s.data["eigenvalues"][k])
                        + 0.5 * (delta + delta.conj().T)
                    )
                    for band in range(8):
                        rows.append(
                            {
                                "sample": name,
                                "model": label,
                                "k_index": k,
                                "band_index": band,
                                "unseen_k": k >= 6,
                                "unseen_band": band >= 6,
                                "occupied": band < s.nocc,
                                "action_error_eV": float(norms[band]),
                                "target_action_norm_eV": float(reference_norms[band]),
                                "reference_eigenvalue_eV": float(
                                    s.data["eigenvalues"][k, band]
                                ),
                                "ritz_eigenvalue_eV": float(ritz[band]),
                                "ritz_error_eV": float(
                                    ritz[band] - s.data["eigenvalues"][k, band]
                                ),
                                "matrix_hermiticity_error_eV": herm,
                            }
                        )
                    timings.append(
                        {
                            "sample": name,
                            "model": label,
                            "k_index": k,
                            "eight_state_action_seconds": seconds,
                        }
                    )
                    probe_error = s.op.apply(probe, k, gate) - probe_target
                    probe_rows.append(
                        {
                            "sample": name,
                            "model": label,
                            "k_index": k,
                            "action_mse_eV2": float(
                                probe_error.abs().square().sum() / 2
                            ),
                            "target_norm2_eV2": float(
                                probe_target.abs().square().sum() / 2
                            ),
                        }
                    )
                ace = s.data["ace_actions"][k] - s.data["actions"][k]
                for band in range(8):
                    rows.append(
                        {
                            "sample": name,
                            "model": "native_ace",
                            "k_index": k,
                            "band_index": band,
                            "unseen_k": k >= 6,
                            "unseen_band": band >= 6,
                            "occupied": band < s.nocc,
                            "action_error_eV": float(np.linalg.norm(ace[band])),
                            "target_action_norm_eV": float(
                                np.linalg.norm(s.data["actions"][k, band])
                            ),
                            "reference_eigenvalue_eV": float(
                                s.data["eigenvalues"][k, band]
                            ),
                            "ritz_eigenvalue_eV": None,
                            "ritz_error_eV": None,
                            "matrix_hermiticity_error_eV": None,
                        }
                    )
                timings.append(
                    {
                        "sample": name,
                        "model": "native_ace",
                        "k_index": k,
                        "eight_state_action_seconds": s.report["native_ace_seconds"][k],
                    }
                )
                timings.append(
                    {
                        "sample": name,
                        "model": "native_exact",
                        "k_index": k,
                        "eight_state_action_seconds": s.report["native_exact_seconds"][
                            k
                        ],
                    }
                )
                probe_rows.append(
                    {
                        "sample": name,
                        "model": "native_ace",
                        "k_index": k,
                        "action_mse_eV2": float(
                            np.sum(
                                abs(
                                    s.data["ace_probe_actions"][k]
                                    - s.data["probe_actions"][k]
                                )
                                ** 2
                            )
                            / 2
                        ),
                        "target_norm2_eV2": float(
                            np.sum(abs(s.data["probe_actions"][k]) ** 2) / 2
                        ),
                    }
                )
            print(f"evaluated {name}", flush=True)
    dump(output / "evaluation_rows.json", rows)
    dump(output / "timings.json", timings)
    dump(output / "orthogonal_probe_rows.json", probe_rows)
    summary = {}
    for label in ["pbe_reference", *models, "native_ace"]:
        records = [r for r in rows if r["model"] == label]
        groups = {}
        for group, predicate in [
            ("all", lambda r: True),
            ("seen_k_seen_band", lambda r: not r["unseen_k"] and not r["unseen_band"]),
            ("unseen_k", lambda r: r["unseen_k"]),
            ("unseen_band", lambda r: r["unseen_band"]),
            ("occupied", lambda r: r["occupied"]),
            ("unoccupied", lambda r: not r["occupied"]),
        ]:
            rr = [r for r in records if predicate(r)]
            mse = np.mean([r["action_error_eV"] ** 2 for r in rr])
            norm = np.mean([r["target_action_norm_eV"] ** 2 for r in rr])
            groups[group] = {
                "action_rmse_meV": float(np.sqrt(mse) * 1000),
                "relative_action_error": float(np.sqrt(mse / norm)),
                "ritz_mae_meV": None
                if label == "native_ace"
                else float(np.mean([abs(r["ritz_error_eV"]) for r in rr]) * 1000),
            }
        pp = [r for r in probe_rows if r["model"] == label]
        groups["orthogonal_probes"] = {
            "action_rmse_meV": float(
                np.sqrt(np.mean([r["action_mse_eV2"] for r in pp])) * 1000
            ),
            "relative_action_error": float(
                np.sqrt(
                    np.mean([r["action_mse_eV2"] for r in pp])
                    / np.mean([r["target_norm2_eV2"] for r in pp])
                )
            ),
        }
        summary[label] = groups
    dump(output / "comparison.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--seeds", type=int, nargs="+", default=[31, 47])
    p.add_argument(
        "--variants",
        choices=["mace", "pointwise", "fno"],
        nargs="+",
        default=["mace", "pointwise", "fno"],
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Requires complete training/validation samples; does not read test samples",
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Evaluate existing checkpoints instead of fitting",
    )
    a = p.parse_args()
    root = a.dataset.resolve()
    output = a.output.resolve()
    if a.epochs < 1 or len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Invalid training schedule")
    if len(set(a.variants)) != len(a.variants):
        raise ValueError("Duplicate model variant")
    if not a.train_only and not (root / "complete.json").exists():
        raise ValueError("Dataset incomplete")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text())
    config = FieldModelConfig(
        atomic_numbers=tuple(manifest.get("atomic_numbers", [14])),
        fno_width=4,
        fno_layers=2,
    )
    if a.checkpoints:
        models = {}
        expected_hash = hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest()
        for path in a.checkpoints:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if (
                canonical_schema(state["schema"]) != "mapho.exchange_gate.v1"
                or state["metadata"]["manifest_sha256"] != expected_hash
            ):
                raise ValueError("Checkpoint/dataset mismatch")
            if state["config"] != asdict(config):
                raise ValueError("Checkpoint configuration differs")
            label = f"{state['variant']}_seed{state['metadata']['seed']}"
            if label in models:
                raise ValueError("Duplicate model")
            model = ExchangeGate(state["variant"], config)
            model.load_state_dict(state["state_dict"])
            models[label] = model
        dump(
            output / "checkpoints.json",
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in a.checkpoints
            },
        )
        evaluate(root, output, models, config)
        return
    protocol = {
        "seeds": a.seeds,
        "epochs": a.epochs,
        "config": asdict(config),
        "variants": a.variants,
        "validation_k_indices": [0, 3],
        "training_batch": "two bands at one seen k per structure per epoch; all 18 pairs cycle every 18 epochs",
        "validation_band_indices": list(range(6)),
        "validation": "lowest held-out-structure action MSE, including identity epoch zero",
        "test": "six unseen structures; k indices 6,7 and band indices 6,7 excluded from all fitting and selection",
        "target_functional": manifest.get("target_functional", "HSE"),
        "ritz": "eight exact hybrid eigenstates; exact hybrid local part; exchange-only subspace diagnostic",
        "native_ace": "exact hybrid reference, constructed using all eight bands at each k; not a learned surrogate",
    }
    dump(output / "protocol.json", protocol)
    validate_native(root, output, config)
    train_samples = [
        PreparedExchange(
            root / name, config, target_functional=protocol["target_functional"]
        )
        for name in manifest["splits"]["train"]
    ]
    val_samples = [
        PreparedExchange(
            root / name, config, target_functional=protocol["target_functional"]
        )
        for name in manifest["splits"]["validation"]
    ]
    models = {}
    for seed in a.seeds:
        for variant in protocol["variants"]:
            models[f"{variant}_seed{seed}"] = train(
                root,
                output,
                variant,
                seed,
                a.epochs,
                train_samples,
                val_samples,
                config,
            )
    del train_samples, val_samples
    if not a.train_only:
        evaluate(root, output, models, config)


if __name__ == "__main__":
    main()

"""Validate standalone ACE against saved QE scientific exchange outputs.

No VASP code, files, or executable are used. Native QE ACE actions are an
independent comparison, never inputs to construction of our ACE factors.
"""

from ..source_snapshot import source_path

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from ..ace_operator import ACEOperator
from ..file_utils import sha


CASES = (
    ("Si-HSE", "hse_exchange_pilot_v1", "HSE"),
    ("LiF-HSE", "lif_hse_exchange_v1", "HSE"),
    ("LiF-PBE0", "lif_pbe0_exchange_v1", "PBE0"),
)
ACTION_TOL_EV = 1e-8


def maximum_action_error(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected, dim=0).max())


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def check_kpoint(data, report, k, device):
    n = int(data["ngk"][k])
    tensor = lambda key: torch.as_tensor(
        data[key][k, :, :n].T.copy(), dtype=torch.complex128, device=device
    )
    phi, actions = tensor("orbitals"), tensor("actions")
    probes, probe_actions = tensor("probes"), tensor("probe_actions")
    nband, nocc = phi.shape[1], int(report["nocc_per_spin"])
    overlap = phi.mH @ phi
    identity = torch.eye(nband, dtype=phi.dtype, device=device)
    normalization_error = float((overlap - identity).abs().max())
    if normalization_error > 1e-8:
        raise ValueError("Stored construction orbitals are not orthonormal")

    operator = ACEOperator.from_samples(phi, actions)
    predicted = operator.apply(phi)
    predicted_probes = operator.apply(probes)
    errors = torch.linalg.vector_norm(predicted_probes - probe_actions, dim=0)
    native_probe_error = torch.linalg.vector_norm(
        tensor("ace_probe_actions") - probe_actions, dim=0
    )

    # Generalized Ritz problem accounts for small normalization errors in the
    # saved eigenvectors. H0 is kept exact inside this eight-band subspace.
    eigenvalues = torch.tensor(
        data["eigenvalues"][k], dtype=torch.float64, device=device
    )
    reference_h = 0.5 * (
        overlap * eigenvalues[None, :] + eigenvalues[:, None] * overlap
    )
    lower = torch.linalg.cholesky(overlap)

    def ritz(matrix):
        left = torch.linalg.solve_triangular(lower, matrix, upper=False)
        transformed = torch.linalg.solve_triangular(lower, left.mH, upper=False).mH
        return torch.linalg.eigvalsh(0.5 * (transformed + transformed.mH))

    corrected_h = reference_h + phi.mH @ (predicted - actions)
    ritz_error = float((ritz(corrected_h) - ritz(reference_h)).abs().max())

    # A unitary change of construction orbitals must not change the operator.
    rng = torch.Generator().manual_seed(71 + k)
    unitary = torch.linalg.qr(
        torch.randn(nband, nband, dtype=torch.complex128, generator=rng)
    )[0].to(device)
    rotated = ACEOperator.from_samples(phi @ unitary, actions @ unitary)
    rotation_error = maximum_action_error(rotated.apply(probes), predicted_probes)

    # The occupied-only approximation need not reproduce empty bands.
    occupied = ACEOperator.from_samples(phi[:, :nocc], actions[:, :nocc])
    empty_error = maximum_action_error(occupied.apply(phi[:, nocc:]), actions[:, nocc:])
    # Supplying additional full actions must recover those additional states.
    enriched = ACEOperator.from_samples(
        torch.cat([phi, probes], dim=1), torch.cat([actions, probe_actions], dim=1)
    )
    enriched_error = maximum_action_error(enriched.apply(probes), probe_actions)

    block = torch.cat([phi, probes], dim=1)
    block_actions = operator.apply(block)
    matrix = block.mH @ block_actions
    hermitian_error = float((matrix - matrix.mH).abs().max())
    largest_exchange_ritz = float(
        torch.linalg.eigvalsh(0.5 * (matrix + matrix.mH)).max()
    )
    a, b = 0.4 + 0.7j, -0.2 + 0.3j
    mixed = a * phi[:, 0] + b * probes[:, 0]
    linearity_error = float(
        torch.linalg.vector_norm(
            operator.apply(mixed) - a * predicted[:, 0] - b * predicted_probes[:, 0]
        )
    )

    cpu = ACEOperator.from_samples(phi.cpu(), actions.cpu())
    cpu_difference = maximum_action_error(
        operator.apply(block).cpu(), cpu.apply(block.cpu())
    )
    row = {
        "k_index": k,
        "n_plane_waves": n,
        "construction_rank": nband,
        "nocc": nocc,
        "normalization_max_error": normalization_error,
        "construction_action_max_error_eV": maximum_action_error(predicted, actions),
        "qe_ace_bands_max_difference_eV": maximum_action_error(
            predicted, tensor("ace_actions")
        ),
        "qe_ace_probes_max_difference_eV": maximum_action_error(
            predicted_probes, tensor("ace_probe_actions")
        ),
        "eight_state_ritz_max_error_eV": ritz_error,
        "rotation_action_max_difference_eV": rotation_error,
        "linearity_max_error_eV": linearity_error,
        "hermiticity_max_error_eV": hermitian_error,
        "largest_exchange_ritz_eV": largest_exchange_ritz,
        "cpu_device_action_max_difference_eV": cpu_difference,
        "occupied_only_empty_action_max_error_eV": empty_error,
        "outside_subspace_squared_errors_eV2": errors.square().cpu().tolist(),
        "qe_ace_outside_subspace_squared_errors_eV2": native_probe_error.square()
        .cpu()
        .tolist(),
        "enriched_probe_action_max_error_eV": enriched_error,
        "factor_storage_bytes": operator.storage_bytes,
        "dense_exchange_storage_bytes": n * n * phi.element_size(),
    }
    required = [
        key
        for key in row
        if key.endswith("_eV")
        and key
        not in {"largest_exchange_ritz_eV", "occupied_only_empty_action_max_error_eV"}
    ]
    row["passed"] = all(
        np.isfinite(row[key]) and row[key] < ACTION_TOL_EV for key in required
    )
    row["passed"] &= largest_exchange_ritz < ACTION_TOL_EV
    return row


@torch.no_grad()
def check_fft(data, report, device):
    from ..exchange import FockExchange, gygi_baldereschi_zero, pw_mask, pw_to_grid

    shape = tuple(int(x) for x in data["shape"])
    nocc = int(report["nocc_per_spin"])
    occupied = np.stack(
        [
            pw_to_grid(c[:nocc, : int(n)], mill[: int(n)], shape)
            for c, mill, n in zip(data["orbitals"], data["mill"], data["ngk"])
        ]
    )
    masks = np.stack(
        [pw_mask(mill[: int(n)], shape) for mill, n in zip(data["mill"], data["ngk"])]
    )
    options = dict(
        alpha=report["exchange_fraction"],
        mu=report["screening_bohr_inverse"],
        rydberg_eV=report["rydberg_eV"],
        recompute=False,
    )
    zero_difference = None
    if report["functional"] == "PBE0":
        if (
            report["exxdiv_treatment"] != "gygi-baldereschi"
            or report["x_gamma_extrapolation"]
        ):
            raise ValueError("Unsupported saved unscreened-exchange convention")
        zero = gygi_baldereschi_zero(
            data["cell_bohr"],
            report["ecutwfc_ry"],
            report["ecutrho_ry"],
            report["kmesh"],
        )
        zero_difference = abs(zero - report["coulomb_zero_mode_bohr2"])
        if not np.isclose(
            zero, report["coulomb_zero_mode_bohr2"], rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                "Independent Coulomb singularity correction differs from QE"
            )
        options["zero_mode"] = zero
    op = FockExchange(
        torch.tensor(occupied, dtype=torch.complex128, device=device),
        data["kpoints"],
        data["weights"],
        data["cell_bohr"],
        masks,
        **options,
    )
    rows = []
    for k, n in enumerate(data["ngk"]):
        n = int(n)
        mill = data["mill"][k, :n]
        coefficients = np.concatenate(
            [data["orbitals"][k, :, :n], data["probes"][k, :, :n]]
        )
        grid = torch.tensor(
            pw_to_grid(coefficients, mill, shape), dtype=torch.complex128, device=device
        )
        actions_grid = op.apply(grid, k)
        reciprocal = torch.fft.fftn(actions_grid, dim=(-3, -2, -1), norm="ortho")
        indices = [
            torch.tensor(mill[:, axis] % shape[axis], dtype=torch.long, device=device)
            for axis in range(3)
        ]
        actions = reciprocal[:, indices[0], indices[1], indices[2]].T
        nband = data["orbitals"].shape[1]
        target = torch.tensor(
            np.concatenate(
                [data["actions"][k, :, :n], data["probe_actions"][k, :, :n]]
            ).T,
            dtype=torch.complex128,
            device=device,
        )
        phi = torch.tensor(
            coefficients[:nband].T, dtype=torch.complex128, device=device
        )
        ace = ACEOperator.from_samples(phi, actions[:, :nband])
        error = maximum_action_error(actions, target)
        ace_error = maximum_action_error(ace.apply(phi), target[:, :nband])
        rows.append(
            {
                "k_index": k,
                "fft_full_vs_qe_max_error_eV": error,
                "fft_built_ace_vs_qe_max_error_eV": ace_error,
                "passed": error < ACTION_TOL_EV and ace_error < ACTION_TOL_EV,
            }
        )
    return {
        "rows": rows,
        "coulomb_zero_mode_difference_bohr2": zero_difference,
        "passed": all(row["passed"] for row in rows),
    }


def summarize(rows):
    maxima = {
        key: max(row[key] for row in rows)
        for key in rows[0]
        if key.endswith("_eV") or key.endswith("_bytes")
    }
    errors = [
        value for row in rows for value in row["outside_subspace_squared_errors_eV2"]
    ]
    native_errors = [
        value
        for row in rows
        for value in row["qe_ace_outside_subspace_squared_errors_eV2"]
    ]
    return {
        **maxima,
        "outside_subspace_action_rmse_eV": float(np.sqrt(np.mean(errors))),
        "qe_ace_outside_subspace_action_rmse_eV": float(
            np.sqrt(np.mean(native_errors))
        ),
        "k_blocks": len(rows),
        "construction_states": sum(row["construction_rank"] for row in rows),
        "passed": all(row["passed"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--require-gpu-name")
    parser.add_argument(
        "--skip-fft",
        action="store_true",
        help="Check saved exchange actions only; do not recompute FFT exchange",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_name = (
        torch.cuda.get_device_name() if device.type == "cuda" else platform.processor()
    )
    if args.require_gpu_name and args.require_gpu_name not in device_name:
        raise RuntimeError(f"Required {args.require_gpu_name}; got {device_name}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "mapho.ace_validation.v1",
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "dtype": "complex128",
        "action_tolerance_eV": ACTION_TOL_EV,
        "scope": "Fixed-operator ACE; QE norm-conserving data; eight-state Ritz checks, not a full material band solve or hybrid SCF",
        "source_sha256": {
            name: sha(source_path(name))
            for name in [
                "ace_operator.py",
                "qe/ace_validation.py",
                "exchange.py",
                "file_utils.py",
            ]
        },
        "cases": {},
    }
    started = time.perf_counter()
    for label, directory, functional in CASES:
        base = args.runs / directory / "dataset"
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        samples = list(
            dict.fromkeys(
                name for split in manifest["splits"].values() for name in split
            )
        )
        rows, provenance = [], []
        fft_result = {
            "performed": False,
            "reason": "Requested saved-actions-only check",
        }
        for index, sample in enumerate(samples):
            folder = base / sample / functional
            data_path, report_path = folder / "exchange.npz", folder / "report.json"
            with np.load(data_path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            report = json.loads(report_path.read_text())
            if report["functional"] != functional:
                raise ValueError("Reference functional mismatch")
            provenance.append(
                {
                    "sample": sample,
                    "npz_sha256": sha(data_path),
                    "report_sha256": sha(report_path),
                }
            )
            for k in range(len(data["ngk"])):
                row = check_kpoint(data, report, k, device)
                row["sample"] = sample
                rows.append(row)
            if index == 0 and not args.skip_fft:
                print(
                    f"{label}: independent FFT/full-exchange cross-check on {sample}",
                    flush=True,
                )
                fft_result = check_fft(data, report, device)
                fft_result["performed"] = True
            print(f"{label}: {index + 1}/{len(samples)} structures checked", flush=True)
        case = {
            "summary": summarize(rows),
            "rows": rows,
            "fft_cross_check": fft_result,
            "manifest_sha256": sha(manifest_path),
            "inputs": provenance,
            "structures": len(samples),
        }
        case["passed"] = case["summary"]["passed"] and fft_result.get(
            "passed", args.skip_fft
        )
        result["cases"][label] = case
        dump(args.output / "results.json", result)
    result["wall_seconds"] = time.perf_counter() - started
    result["fft_checks_performed"] = not args.skip_fft
    result["passed"] = all(case["passed"] for case in result["cases"].values())
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    dump(args.output / "results.json", result)
    print(
        json.dumps(
            {label: case["summary"] for label, case in result["cases"].items()},
            indent=2,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit("ACE validation failed; inspect results.json")


if __name__ == "__main__":
    main()

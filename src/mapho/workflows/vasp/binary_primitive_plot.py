"""Plot certified primitive LiF and bilayer MoS2 spectra, retaining a common energy zero."""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visible_polyline(x, y, limits):
    """Geometrically clip the displayed straight segments; keep source data intact.

    Explicit clipping also keeps invisible off-axis PDF paths out of geometric
    text-collision audits, which do not interpret the PDF clipping stack.
    """
    lower, upper = limits
    points = []
    for i in range(len(x) - 1):
        dy = y[i + 1] - y[i]
        if abs(dy) < 1e-15:
            if not lower <= y[i] <= upper:
                if points and np.isfinite(points[-1]).all():
                    points.append([np.nan, np.nan])
                continue
            start, end = 0.0, 1.0
        else:
            crossings = sorted([(lower - y[i]) / dy, (upper - y[i]) / dy])
            start, end = max(0.0, crossings[0]), min(1.0, crossings[1])
            if start > end:
                if points and np.isfinite(points[-1]).all():
                    points.append([np.nan, np.nan])
                continue
        a = np.array(
            [x[i] + start * (x[i + 1] - x[i]), np.clip(y[i] + start * dy, lower, upper)]
        )
        b = np.array(
            [x[i] + end * (x[i + 1] - x[i]), np.clip(y[i] + end * dy, lower, upper)]
        )
        if not points or not np.allclose(points[-1], a, atol=1e-12, rtol=0):
            if points and np.isfinite(points[-1]).all():
                points.append([np.nan, np.nan])
            points.append(a)
        points.append(b)
    return np.asarray(points).reshape(-1, 2).T


def plot(root):
    root = Path(root)
    data = json.loads((root / "results.json").read_text())
    if not data["complete"]:
        raise ValueError(
            "Sampling and same-operator replay must be certified before delivery"
        )
    out = root / "figures"
    out.mkdir(exist_ok=True)
    joint = bool(data.get("joint_finetuning", False))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2047244, 6.1023622), gridspec_kw={"height_ratios": [1.4, 1]}
    )
    fig.subplots_adjust(
        left=0.10, right=0.975, bottom=0.15, top=0.82, wspace=0.32, hspace=0.43
    )
    colors = {"none": "#0072B2", "full": "#D55E00"}
    display = {"lif": "LiF", "mos2": "MoS₂ bilayer"}
    limits = {"lif": (-5, 15), "mos2": (-4, 4)}
    with (out / "band_source_data.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "material",
                "method",
                "k_index",
                "distance_inv_A",
                "kx",
                "ky",
                "kz",
                "band",
                "native_energy_eV",
                "energy_minus_reference_VBM_eV",
                "error_vs_native_PBE_eV",
            ]
        )
        for column, material in enumerate(("lif", "mos2")):
            result = data["materials"][material]
            spec = result["specification"]
            path = spec["path"]
            spectra = {
                k: np.asarray(v, dtype=float)
                for k, v in result["eigenvalues_eV"].items()
            }
            x = np.asarray(path["distance_inv_A"])
            nocc = spec["nocc"]
            ref = spectra["native_high"]
            assert np.all(np.diff(x) > 0)
            assert all(
                a.shape == (len(x), spec["nbands"]) and np.isfinite(a).all()
                for a in spectra.values()
            )
            zero = float(ref[:, nocc - 1].max())
            for method, e in spectra.items():
                for k in range(len(x)):
                    for band in range(e.shape[1]):
                        writer.writerow(
                            [
                                material,
                                method,
                                k,
                                x[k],
                                *path["kpoints"][k],
                                band + 1,
                                e[k, band],
                                e[k, band] - zero,
                                e[k, band] - ref[k, band],
                            ]
                        )
            upper, lower = axes[:, column]
            for band in range(spec["nbands"]):
                upper.plot(
                    *visible_polyline(x, ref[:, band] - zero, limits[material]),
                    color="#252525",
                    lw=0.9,
                )
                upper.plot(
                    *visible_polyline(
                        x, spectra["full"][:, band] - zero, limits[material]
                    ),
                    color=colors["full"],
                    ls="--",
                    lw=0.95,
                )
            for mode in ("none", "full"):
                delta = 1000 * (spectra[mode] - ref)
                for band, style in ((nocc - 1, "-"), (nocc, "--")):
                    lower.plot(x, delta[:, band], color=colors[mode], ls=style, lw=1.05)
            for ax in (upper, lower):
                ax.set_xlim(x[0], x[-1])
                indices = path["vertex_indices"]
                ax.set_xticks(
                    x[indices], [s.replace("Gamma", "Γ") for s in path["labels"]]
                )
                for i in indices[1:-1]:
                    ax.axvline(x[i], color="#C7C7C7", lw=0.55, zorder=0)
                ax.axhline(0, color="#C7C7C7", lw=0.55, zorder=0)
            upper.set_ylim(*limits[material])
            upper.set_ylabel("Energy − PBE VBM (eV)")
            lower.set_ylabel("Predicted − PBE energy (meV)")
            lower.set_xlabel("Wave vector")
            upper.set_title(
                f"{'ab'[column]}  {display[material]}",
                loc="left",
                fontsize=9,
                fontweight="bold",
                pad=9,
            )
            lower.set_title(
                f"{'cd'[column]}  {display[material]}: frontier errors",
                loc="left",
                fontsize=9,
                fontweight="bold",
                pad=9,
            )
    fig.text(
        0.10,
        0.965,
        "Primitive-cell bands after joint fine-tuning"
        if joint
        else "Primitive-cell bands from frozen supercell models",
        fontsize=11,
    )
    fig.text(
        0.10,
        0.925,
        "Nonmagnetic PBE / PAW  |  600 eV  |  Ideal parent geometries  |  "
        + ("No primitive-cell fitting" if joint else "No retraining"),
        fontsize=8,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#252525", lw=1, label="VASP PBE"),
            Line2D(
                [0], [0], color=colors["full"], lw=1, ls="--", label="MACE + FNO (SiLU)"
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.09, 0.894),
        ncol=2,
        handlelength=2.7,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=colors["none"], lw=1.2, label="MACE + pointwise"),
            Line2D([0], [0], color=colors["full"], lw=1.2, label="MACE + FNO"),
            Line2D([0], [0], color="#333333", lw=1.2, label="Top valence"),
            Line2D(
                [0], [0], color="#333333", lw=1.2, ls="--", label="Bottom conduction"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.09, 0.055),
        ncol=4,
        handlelength=2.4,
        columnspacing=1.3,
    )
    fig.text(
        0.10,
        0.025,
        "Common reference VBM zero; "
        + (
            "MoS₂: all active components jointly fine-tuned. LiF: original joint fit."
            if joint
            else "MoS₂ uses repaired PAW heads."
        ),
        fontsize=7,
    )
    fig.savefig(out / "lif_mos2_primitive_bands.pdf")
    # PNG is required for the user's requested in-chat display and visual inspection.
    fig.savefig(out / "lif_mos2_primitive_bands.png", dpi=300)
    plt.close(fig)
    caption = (
        "Primitive-cell transfer of frozen SiLU MACE+FNO models trained on 16-atom LiF and 24-atom "
        "MoS2-bilayer supercells. Top: native frozen-density PBE bands (black) and predictions of both "
        "the local potential and PAW on-site matrices (orange dashed). Bottom: signed errors in the "
        "highest occupied and lowest unoccupied level at each k point, including the matched pointwise "
        "control. Solid/dashed error curves denote valence/conduction levels. The same PBE VBM is "
        "subtracted from reference and prediction; no method-specific energy shifts are fitted. "
        "LiF uses a=4.03 A (conventional cubic), a two-atom primitive cell, and 81 X-Gamma-L points. "
        "The six-atom 2H MoS2 bilayer uses a=3.18 A, c=29.34 A, 20 A vacuum, and 73 Gamma-M-K-Gamma points. "
        "These are the ideal unrelaxed training-parent geometries; calculations are nonmagnetic PBE "
        "without SOC or empirical dispersion. Primitive references use independently checked denser "
        "SCF meshes and a 600 eV cutoff. MoS2 uses the repaired PAW heads with the original field model "
        "frozen. There is one fitted seed and one parent crystal per material; no statistical uncertainty "
        "is implied. Display windows are -5 to 15 eV (LiF) and -4 to 4 eV (MoS2); all calculated "
        "bands, all path points, and reference controls are retained in the source CSV. "
        "Reported gaps are minima/maxima along these paths, not a full Brillouin-zone search.\n"
    )
    if joint:
        caption = caption.replace(
            "MoS2 uses the repaired PAW heads with the original field model frozen.",
            "MoS2 uses joint fine-tuning of the repaired PAW heads, MACE backbone, field decoder, and correction network on the original training structures; validation selects the checkpoint. LiF retains the original joint checkpoint. No primitive labels enter fitting or selection.",
        )
    (out / "figure_caption.txt").write_text(caption)
    return out

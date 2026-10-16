"""Plot the completed primitive-cell transfer check with a shared energy zero."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(root):
    root = Path(root)
    data = json.loads((root / "results.json").read_text())
    if not data["complete"]:
        raise ValueError("Native sampling and replay checks must pass before plotting")
    spectra = {k: np.asarray(v) for k, v in data["eigenvalues_eV"].items()}
    x = np.asarray(data["path"]["distance_inv_A"])
    assert len(x) == 81 and np.all(np.diff(x) > 0)
    assert all(a.shape == (81, 8) and np.isfinite(a).all() for a in spectra.values())
    ref = spectra["native_16"]
    zero = float(ref[:, 3].max())
    with (root / "band_source_data.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "k_index",
                "distance_inv_A",
                "kx",
                "ky",
                "kz",
                "band",
                "native_energy_eV",
                "energy_minus_reference_VBM_eV",
                "error_vs_native_eV",
            ]
        )
        for method, e in spectra.items():
            for k in range(len(x)):
                for band in range(8):
                    writer.writerow(
                        [
                            method,
                            k,
                            x[k],
                            *data["path"]["kpoints"][k],
                            band + 1,
                            e[k, band],
                            e[k, band] - zero,
                            e[k, band] - ref[k, band],
                        ]
                    )
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
        1, 2, figsize=(7.0866, 4.0551), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.18, top=0.75, wspace=0.34)
    colors = {"local": "#777777", "none": "#3576A6", "full": "#C66A24"}
    names = {"local": "MACE", "none": "MACE + pointwise", "full": "MACE + FNO"}
    left, right = axes
    for band in range(8):
        left.plot(
            x,
            ref[:, band] - zero,
            color="#252525",
            lw=1.0,
            label="VASP PBE" if band == 0 else None,
        )
        left.plot(
            x,
            spectra["full"][:, band] - zero,
            color=colors["full"],
            lw=1.0,
            ls="--",
            label=names["full"] if band == 0 else None,
        )
    for mode in ("local", "none", "full"):
        delta = 1000 * (spectra[mode] - ref)
        for band, style in ((3, "-"), (4, "--")):
            right.plot(x, delta[:, band], color=colors[mode], ls=style, lw=1.15)
    tick_indices = data["path"]["vertex_indices"]
    for ax in axes:
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(x[tick_indices], ["X", "Γ", "L"])
        ax.set_xlabel("Wave vector")
        ax.axvline(x[tick_indices[1]], color="#BDBDBD", lw=0.6, zorder=0)
        ax.axhline(0, color="#BDBDBD", lw=0.6, zorder=0)
    left.set_ylabel("Energy − PBE VBM (eV)")
    right.set_ylabel("Predicted − PBE energy (meV)")
    left.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        borderaxespad=0,
        ncol=2,
        handlelength=2.2,
        columnspacing=1.1,
    )
    right.legend(
        handles=[
            Line2D([0], [0], color=colors[k], lw=1.3, label=names[k])
            for k in ("local", "none", "full")
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        borderaxespad=0,
        handlelength=2.0,
    )
    fig.text(0.085, 0.865, "a  Band dispersion", weight="bold", fontsize=9)
    fig.text(0.615, 0.865, "b  Frontier-level errors", weight="bold", fontsize=9)
    fig.text(
        0.085,
        0.965,
        "Primitive Si: prediction from the frozen Si16 models",
        fontsize=11,
    )
    fig.text(
        0.085,
        0.915,
        "2 atoms  |  a = 5.43 Å  |  Nonmagnetic PBE / PAW  |  No retraining",
        fontsize=8,
    )
    fig.text(0.615, 0.06, "Solid: top valence   Dashed: bottom conduction", fontsize=7)
    fig.text(0.085, 0.06, "Common PBE VBM zero; no fitted energy shifts", fontsize=7)
    fig.savefig(root / "primitive_si_bands.pdf")
    # Preview is needed for visual inspection and inline delivery.
    fig.savefig(root / "primitive_si_bands.png", dpi=300)
    plt.close(fig)
    caption = (
        "Primitive silicon at a = 5.43 Å, evaluated with the unchanged models trained on 16-atom cells. "
        "Left: all eight bands on an 81-point X–Γ–L path, comparing native PBE with MACE+FNO. "
        "A single energy zero, the native PBE valence-band maximum, is subtracted from both curves. "
        "Right: signed errors in the highest valence and lowest conduction level at each k point, "
        "including the MACE-only and matched pointwise controls. Solid and dashed curves denote "
        "valence and conduction levels, respectively. The reference uses a 16×16×16 SCF mesh, "
        "600 eV cutoff, and native frozen-density band calculations. The ML band calculations use "
        "both predicted potential and PAW on-site matrices from geometry and a fixed atomic-density "
        "reference. This is one pristine structure and one trained seed; no uncertainty bands or "
        "statistical tests are implied. Every calculated level is retained in the accompanying CSV.\n"
    )
    (root / "figure_caption.txt").write_text(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    plot(parser.parse_args().root)

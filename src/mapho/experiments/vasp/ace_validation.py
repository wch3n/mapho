"""Native reference, frozen-ACE replay, gauge and missing-exchange controls."""

from ...source_snapshot import snapshot_package, source_hashes, source_path
import json
from pathlib import Path
import shutil

from ...vasp.ace import export_hybrid, prepare_hybrid_replay
from ...vasp.exchange import compare_native_factors
from .hybrid_pilot import LAUNCHER, band_metrics
from ...vasp.paw import compare, run, save_json, sha, write_incar


def prepare(root, reference, executable, native_executable):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    source = json.loads((reference / "manifest.json").read_text())
    if source["functional"] != "PBE0" or source["settings"]["ISPIN"] != 1:
        raise ValueError("Expected scalar PBE0 seed")
    rows = []
    for name in ("native", "reference"):
        folder = root / name
        folder.mkdir()
        for item in ("POSCAR", "POTCAR", "KPOINTS", "WAVECAR"):
            shutil.copyfile(reference / item, folder / item)
        settings = dict(source["settings"])
        settings.pop("TIME", None)
        settings.update(
            ALGO="Normal",
            LFOCKACE=True,
            NELMIN=24,
            NELM=200,
            EDIFF=1e-10,
            IMIX=1,
            AMIX=0.2,
            ISTART=1,
            ICHARG=0,
            NELMDL=0,
        )
        write_incar(folder / "INCAR", settings)
        m = dict(
            source,
            settings=settings,
            purpose="Native PBE0 ACE export/replay validation",
            unpatched_control=name == "native",
            restart_source=str(reference),
        )
        m["inputs"] = {
            n: sha(folder / n)
            for n in ("POSCAR", "POTCAR", "KPOINTS", "INCAR", "WAVECAR")
        }
        if name == "reference":
            (folder / "MAPHO_EXPORT").touch()
            (folder / "MAPHO_ACE_EXPORT").touch()
            (folder / "MAPHO_FOCK_EXPORT").touch()
        save_json(folder / "manifest.json", m)
    snap = root / "source_snapshot/mapho"
    snap.mkdir(parents=True)
    snapshot_package(snap)
    study = dict(
        schema="mapho.vasp_ace_validation.v2",
        source_reference=str(reference),
        executable=str(Path(executable).resolve()),
        native_executable=str(Path(native_executable).resolve()),
        native_executable_sha256=sha(native_executable),
        source_sha256=source_hashes(snap),
        tolerance_eV=1e-4,
    )
    save_json(root / "study.json", study)
    return study


def execute(root):
    root = Path(root).resolve()
    m = json.loads((root / "study.json").read_text())
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError("Worker source changed")
    if sha(m["native_executable"]) != m["native_executable_sha256"]:
        raise ValueError("Native executable changed")
    report = dict(
        complete=False,
        passed=False,
        executable_sha256=sha(m["executable"]),
        scope="Same mesh, independently constructed and applied MAPHO ACE from full VASP actions; finite rank",
    )
    save_json(root / "validation.json", report)

    def calculate(name, native=False):
        folder = root / name
        imported = (folder / "MAPHO_ACE_IMPORT").exists()
        if imported and any((folder / item).exists() for item in ("WAVECAR", "CHGCAR")):
            raise RuntimeError(
                "Replay must start without reference wavefunctions or density"
            )
        result = run(
            folder,
            m["native_executable" if native else "executable"],
            launcher=LAUNCHER,
            timeout=1800,
        )
        text = (folder / "stdout.log").read_text()
        if "Offloading initialized" not in text:
            raise RuntimeError("CUDA offload not reported")
        if (
            (folder / "MAPHO_ACE_IMPORT").exists()
            and "frozen ACE imported; native reciprocal basis verified" not in text
        ):
            raise RuntimeError("ACE import marker missing")
        if (
            imported
            and "external ACE applied by the independent MAPHO contraction" not in text
        ):
            raise RuntimeError("Independent MAPHO contraction marker missing")
        return result

    try:
        calculate("native", native=True)
        calculate("reference")
        report["bridge_vs_native"] = compare(
            root / "native", root / "reference", tolerance_eV=m["tolerance_eV"]
        )
        if not report["bridge_vs_native"]["passed"]:
            raise RuntimeError("Bridge changed native PBE0 reference")
        ref = root / "reference"
        manifest = json.loads((ref / "manifest.json").read_text())
        manifest["initial_wavecar_sha256"] = manifest["inputs"].pop("WAVECAR")
        manifest["final_wavecar_sha256"] = sha(ref / "WAVECAR")
        save_json(ref / "manifest.json", manifest)
        export_hybrid(ref, root / "operator.h5")
        report["independent_factorization"] = compare_native_factors(
            ref / "MAPHO_FOCK.bin", root / "operator.ace.bin", ref / "MAPHO_ACE.bin"
        )
        save_json(root / "validation.json", report)
        if not report["independent_factorization"]["passed"]:
            raise RuntimeError(
                "Independent factorization disagrees with full actions or native ACE"
            )
        for name, shift, scale in [
            ("replay", 0.0, 1.0),
            ("shift", 0.25, 1.0),
            ("half_exchange", 0.0, 0.5),
        ]:
            prepare_hybrid_replay(
                root / "operator.h5", root / name, shift_eV=shift, ace_scale=scale
            )
            calculate(name)
            report[name] = compare(
                ref, root / name, shift_eV=shift, tolerance_eV=m["tolerance_eV"]
            )
            save_json(root / "validation.json", report)
        report["negative_control_detected"] = not report["half_exchange"]["passed"]
        report["bands"] = band_metrics(ref, root / "replay")
        report["replay_reads_reference_wavefunctions"] = (
            root / "replay/CHGCAR"
        ).exists() or (
            (root / "replay/WAVECAR").exists()
            and (root / "replay/WAVECAR").stat().st_size > 0
        )
        report["passed"] = (
            all(
                report[name]["passed"]
                for name in ("bridge_vs_native", "replay", "shift")
            )
            and report["negative_control_detected"]
            and report["independent_factorization"]["passed"]
            and not report["replay_reads_reference_wavefunctions"]
        )
        report.update(
            complete=True,
            outputs_sha256={
                str(p.relative_to(root)): sha(p)
                for p in root.rglob("*")
                if p.is_file()
                and p.name
                in (
                    "EIGENVAL",
                    "MAPHO_ACE.bin",
                    "MAPHO_FOCK.bin",
                    "POT",
                    "MAPHO_PAW.dat",
                    "operator.h5",
                    "operator.ace.bin",
                )
            },
        )
        if not report["passed"]:
            raise RuntimeError("Independent MAPHO ACE acceptance failed")
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save_json(root / "validation.json", report)
    return report

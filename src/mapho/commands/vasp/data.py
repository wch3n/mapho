"""VASP data commands with lazy imports of APIs and study implementations."""

import argparse
import json

from . import dispatch


def _atomic_reference(argv=None):
    from ...experiments.vasp import atomic_reference as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp data atomic-reference", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("dataset")
    q.add_argument("root")
    q.add_argument(
        "--export-only",
        action="store_true",
        help="Export the atomic operator before diagonalization (requires the private atomic hook)",
    )
    q.add_argument(
        "--executable", help="MAPHO-enabled VASP executable for the atomic references"
    )
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(
            a.dataset, a.root, export_only=a.export_only, executable=a.executable
        )
    elif a.command == "worker":
        implementation.worker(a.root, a.index)
    else:
        implementation.execute(a.root)


def _silicon_dataset(argv=None):
    from ...experiments.vasp import silicon_dataset as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp data silicon", description=implementation.__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root")
    p.add_argument("--potcar", required=True)
    p.add_argument("--executable", required=True)
    p.add_argument("--count", type=int, default=24)
    for name in ("execute", "convergence"):
        p = sub.add_parser(name)
        p.add_argument("root")
    p = sub.add_parser("worker")
    p.add_argument("root")
    p.add_argument("phase", choices=["calibration", "samples"])
    p.add_argument("index", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = implementation.prepare(
            args.root, args.potcar, args.executable, count=args.count
        )
        result = {
            k: result[k]
            for k in ("schema", "count", "natoms", "production_kppvol", "splits")
        }
    elif args.command == "worker":
        result = implementation.run_case(args.root, args.phase, args.index)
    else:
        result = getattr(implementation, args.command)(args.root)
    print(json.dumps(result, indent=2))


def _material_dataset(argv=None):
    from ...experiments.vasp import material_dataset as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp data material", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("material", choices=["lif", "mos2"])
    q.add_argument("--potcar-root", required=True)
    q.add_argument("--executable", required=True)
    q = sub.add_parser("execute")
    q.add_argument("root")
    q.add_argument("phase", choices=["calibration", "samples"])
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("phase")
    q.add_argument("index", type=int)
    q = sub.add_parser("audit")
    q.add_argument("root")
    a = p.parse_args(argv)
    if a.command == "prepare":
        m = implementation.prepare(a.root, a.material, a.potcar_root, a.executable)
        print(
            json.dumps(
                {
                    k: m[k]
                    for k in (
                        "material",
                        "count",
                        "natoms",
                        "nbands",
                        "production_kppvol",
                    )
                }
            )
        )
    elif a.command == "worker":
        implementation.worker(a.root, a.phase, a.index)
    elif a.command == "execute":
        implementation.execute(a.root, a.phase)
    else:
        implementation.audit(a.root)


def _reference_chunk(argv=None):
    from ...experiments.vasp import reference_chunk as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp data reference-chunk", description=implementation.__doc__
    )
    p.add_argument("root")
    p.add_argument("start", type=int)
    p.add_argument("stop", type=int)
    a = p.parse_args(argv)
    implementation.execute(a.root, a.start, a.stop)


COMMANDS = {
    "atomic-reference": _atomic_reference,
    "silicon": _silicon_dataset,
    "material": _material_dataset,
    "reference-chunk": _reference_chunk,
}


def main(argv=None):
    return dispatch(COMMANDS, argv, prog="mapho-vasp data", description=__doc__)


if __name__ == "__main__":
    main()

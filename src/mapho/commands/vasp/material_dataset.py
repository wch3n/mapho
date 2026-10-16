"""Command-line arguments for material dataset; implementation lives in mapho.workflows.vasp.material_dataset."""

import argparse
import json


def main(argv=None):
    from ...workflows.vasp import material_dataset as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
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


if __name__ == "__main__":
    main()

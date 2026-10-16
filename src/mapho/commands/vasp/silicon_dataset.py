"""Command-line arguments for silicon dataset; implementation lives in mapho.workflows.vasp.silicon_dataset."""

import argparse
import json


def main(argv=None):
    from ...workflows.vasp import silicon_dataset as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
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


if __name__ == "__main__":
    main()

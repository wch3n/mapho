"""Command-line arguments for paw; implementation lives in mapho.vasp.paw."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from ...vasp import paw as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("reference", type=Path)
    ex.add_argument("output", type=Path)
    rp = sub.add_parser("replay")
    rp.add_argument("operator", type=Path)
    rp.add_argument("folder", type=Path)
    rp.add_argument("--shift-eV", type=float, default=0.0)
    co = sub.add_parser("compare")
    co.add_argument("reference")
    co.add_argument("replay")
    co.add_argument("--shift-eV", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.command == "export":
        result = implementation.export_operator(args.reference, args.output)
    elif args.command == "replay":
        result = implementation.prepare_replay(
            args.operator, args.folder, shift_eV=args.shift_eV
        )
    else:
        result = implementation.compare(args.reference, args.replay, args.shift_eV)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

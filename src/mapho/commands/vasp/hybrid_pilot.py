"""Command-line arguments for hybrid pilot; implementation lives in mapho.workflows.vasp.hybrid_pilot."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from ...workflows.vasp import hybrid_pilot as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--potcar", type=Path, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--native-executable", type=Path, required=True)
    p.add_argument("--meshes", nargs="+", type=int, default=[2, 3])
    p = sub.add_parser("execute")
    p.add_argument("root", type=Path)
    p.add_argument("index", type=int)
    p = sub.add_parser("collect")
    p.add_argument("root", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = {
        "prepare": implementation.prepare,
        "execute": implementation.execute,
        "collect": implementation.collect,
    }[command](**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

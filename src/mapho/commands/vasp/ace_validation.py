"""Command-line arguments for ace validation; implementation lives in mapho.workflows.vasp.ace_validation."""

import argparse
import json


def main(argv=None):
    from ...workflows.vasp import ace_validation as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("root")
    s.add_argument("reference")
    s.add_argument("executable")
    s.add_argument("native_executable")
    s = sub.add_parser("execute")
    s.add_argument("root")
    a = vars(p.parse_args(argv))
    command = a.pop("command")
    print(
        json.dumps(
            (
                implementation.prepare
                if command == "prepare"
                else implementation.execute
            )(**a),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

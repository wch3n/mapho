"""Command-line arguments for material training; implementation lives in mapho.workflows.vasp.material_training."""

import argparse


def main(argv=None):
    from ...workflows.vasp import material_training as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("dataset")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("native-checks")
    q.add_argument("training")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.dataset)
    elif a.command == "execute":
        implementation.execute(a.root)
    elif a.command == "native-checks":
        implementation.native_checks(a.training)
    else:
        implementation.bands_case(a.root, a.index)


if __name__ == "__main__":
    main()

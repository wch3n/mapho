"""Command-line arguments for primitive bands; implementation lives in mapho.workflows.vasp.primitive_bands."""

import argparse


def main(argv=None):
    from ...workflows.vasp import primitive_bands as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    for stage in ("reference", "bands"):
        q = sub.add_parser(stage)
        q.add_argument("root")
        q.add_argument("name")
    for stage in ("execute", "analyze"):
        q = sub.add_parser(stage)
        q.add_argument("root")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training)
    elif a.command == "reference":
        implementation.reference(a.root, a.name)
    elif a.command == "bands":
        implementation.bands(a.root, a.name)
    elif a.command == "analyze":
        implementation.analyze(a.root)
    else:
        implementation.execute(a.root)


if __name__ == "__main__":
    main()

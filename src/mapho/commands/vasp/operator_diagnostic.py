"""Command-line arguments for operator diagnostic; implementation lives in mapho.workflows.vasp.operator_diagnostic."""

import argparse


def main(argv=None):
    from ...workflows.vasp import operator_diagnostic as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training)
    elif a.command == "worker":
        implementation.worker(a.root, a.index)
    else:
        implementation.execute(a.root)


if __name__ == "__main__":
    main()

"""Command-line arguments for atomic reference; implementation lives in mapho.workflows.vasp.atomic_reference."""

import argparse


def main(argv=None):
    from ...workflows.vasp import atomic_reference as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("dataset")
    q.add_argument("root")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.dataset, a.root)
    elif a.command == "worker":
        implementation.worker(a.root, a.index)
    else:
        implementation.execute(a.root)


if __name__ == "__main__":
    main()

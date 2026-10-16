"""Command-line arguments for train; implementation lives in mapho.vasp.train."""

import argparse


def main(argv=None):
    from ...vasp import train as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("dataset")
    q.add_argument("atomic")
    q.add_argument("output")
    q = sub.add_parser("train")
    q.add_argument("root")
    q.add_argument("mode", choices=["local", "none", "full"])
    q.add_argument("--epochs", type=int, default=300)
    q = sub.add_parser("evaluate")
    q.add_argument("root")
    q.add_argument("mode", choices=["local", "none", "full"])
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.dataset, a.atomic, a.output)
    elif a.command == "train":
        implementation.train(a.root, a.mode, a.epochs)
    else:
        implementation.evaluate(a.root, a.mode)


if __name__ == "__main__":
    main()

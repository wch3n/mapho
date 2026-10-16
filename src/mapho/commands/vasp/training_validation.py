"""Command-line arguments for training validation; implementation lives in mapho.workflows.vasp.training_validation."""

import argparse


def main(argv=None):
    from ...workflows.vasp import training_validation as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("native-checks", "pipeline"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("--epochs", type=int, default=300)
    q = sub.add_parser("fit")
    q.add_argument("root")
    q.add_argument("mode")
    q.add_argument("--epochs", type=int, default=300)
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "native-checks":
        implementation.native_checks(a.root)
    elif a.command == "bands":
        implementation.bands_case(a.root, a.index)
    elif a.command == "pipeline":
        implementation.pipeline(a.root, a.epochs)
    else:
        from ...vasp.train import train, evaluate

        train(a.root, a.mode, a.epochs)
        evaluate(a.root, a.mode)


if __name__ == "__main__":
    main()

"""Command-line arguments for paw joint; implementation lives in mapho.workflows.vasp.paw_joint."""

import argparse


def main(argv=None):
    from ...workflows.vasp import paw_joint as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("repaired")
    q.add_argument("primitive")
    q.add_argument("--epochs", type=int, default=300)
    for name in ("execute", "primitive-results"):
        q = sub.add_parser(name)
        q.add_argument("root")
    for name in ("fit", "predict"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("label", choices=implementation.LABELS)
        if name == "predict":
            q.add_argument("split", choices=["validation", "test", "primitive"])
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("split")
    q.add_argument("index", type=int)
    q.add_argument("label", choices=implementation.LABELS)
    q = sub.add_parser("summarize")
    q.add_argument("root")
    q.add_argument("split")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.repaired, a.primitive, a.epochs)
    elif a.command == "fit":
        implementation.fit(a.root, a.label)
    elif a.command == "predict":
        implementation.predictions(a.root, a.label, a.split)
    elif a.command == "bands":
        implementation.bands(a.root, a.split, a.index, a.label)
    elif a.command == "summarize":
        implementation.summarize(a.root, a.split)
    elif a.command == "primitive-results":
        implementation.primitive_results(a.root)
    else:
        implementation.execute(a.root)


if __name__ == "__main__":
    main()

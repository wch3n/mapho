"""Command-line arguments for paw repair; implementation lives in mapho.workflows.vasp.paw_repair."""

import argparse


def main(argv=None):
    from ...workflows.vasp import paw_repair as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    q.add_argument("--refine-from")
    q = sub.add_parser("prepare-evaluation")
    q.add_argument("root")
    q.add_argument("trained")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q.add_argument("phase", choices=["validation", "test"])
    for name in ("fit", "predict-test"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("label", choices=implementation.LABELS)
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("split")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training, a.refine_from)
    elif a.command == "prepare-evaluation":
        implementation.prepare_evaluation(a.root, a.trained)
    elif a.command == "fit":
        implementation.fit(a.root, a.label)
    elif a.command == "predict-test":
        implementation.predictions(a.root, a.label, "test")
    elif a.command == "bands":
        implementation.band_case(a.root, a.split, a.index)
    else:
        implementation.execute(a.root, a.phase)


if __name__ == "__main__":
    main()

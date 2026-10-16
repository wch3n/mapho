"""Command-line arguments for activation ablation; implementation lives in mapho.workflows.vasp.activation_ablation."""

import argparse


def main(argv=None):
    from ...workflows.vasp import activation_ablation as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("baseline")
    q.add_argument("primitive")
    for name in ("execute", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("root")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("mode", choices=["none", "full"])
    q.add_argument("index")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.baseline, a.primitive)
    elif a.command == "bands":
        implementation.band_case(a.root, a.mode, a.index)
    elif a.command == "summarize":
        implementation.summarize(a.root)
    else:
        implementation.execute(a.root)


if __name__ == "__main__":
    main()

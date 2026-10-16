"""Command-line arguments for reference chunk; implementation lives in mapho.workflows.vasp.reference_chunk."""

import argparse


def main(argv=None):
    from ...workflows.vasp import reference_chunk as implementation

    p = argparse.ArgumentParser(description=implementation.__doc__)
    p.add_argument("root")
    p.add_argument("start", type=int)
    p.add_argument("stop", type=int)
    a = p.parse_args(argv)
    implementation.execute(a.root, a.start, a.stop)


if __name__ == "__main__":
    main()

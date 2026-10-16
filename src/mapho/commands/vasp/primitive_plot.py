"""Command-line arguments for primitive plot; implementation lives in mapho.workflows.vasp.primitive_plot."""

import argparse


def main(argv=None):
    from ...workflows.vasp import primitive_plot as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
    parser.add_argument("root")
    implementation.plot(parser.parse_args(argv).root)


if __name__ == "__main__":
    main()

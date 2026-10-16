"""Command-line arguments for binary primitive plot; implementation lives in mapho.workflows.vasp.binary_primitive_plot."""

import argparse


def main(argv=None):
    from ...workflows.vasp import binary_primitive_plot as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
    parser.add_argument("root")
    implementation.plot(parser.parse_args(argv).root)


if __name__ == "__main__":
    main()

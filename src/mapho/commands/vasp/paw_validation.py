"""Command-line arguments for paw validation; implementation lives in mapho.workflows.vasp.paw_validation."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from ...workflows.vasp import paw_validation as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(implementation.validate(parser.parse_args(argv).root), indent=2))


if __name__ == "__main__":
    main()

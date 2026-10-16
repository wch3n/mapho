"""Command-line arguments for exchange; implementation lives in mapho.vasp.exchange."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from ...vasp import exchange as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            implementation.build_ace_from_samples(args.samples, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()

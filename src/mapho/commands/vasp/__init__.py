"""VASP command dispatcher; APIs and workflows are imported only when selected."""

import argparse
from importlib import import_module

COMMANDS = (
    "paw",
    "exchange",
    "train",
    "atomic-reference",
    "silicon-dataset",
    "material-dataset",
    "reference-chunk",
    "material-training",
    "paw-repair",
    "paw-joint",
    "primitive-bands",
    "binary-primitive-bands",
    "primitive-plot",
    "binary-primitive-plot",
    "paw-validation",
    "spin-validation",
    "training-validation",
    "ace-validation",
    "operator-diagnostic",
    "activation-ablation",
    "hybrid-pilot",
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mapho-vasp", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = import_module("." + args.command.replace("-", "_"), __name__)
    return command.main(args.arguments)

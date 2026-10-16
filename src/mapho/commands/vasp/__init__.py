"""VASP commands grouped by data preparation, training, prediction, and validation."""

import argparse
from importlib import import_module

GROUPS = ("data", "train", "predict", "validate")


def dispatch(commands, argv, *, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("command", choices=commands)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return commands[args.command](args.arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mapho-vasp", description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return import_module("." + args.group, __name__).main(args.arguments)

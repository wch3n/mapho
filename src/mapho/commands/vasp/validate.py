"""VASP validate commands with lazy imports of APIs and study implementations."""

from pathlib import Path
import argparse
import json

from . import dispatch


def _paw_validation(argv=None):
    from ...experiments.vasp import paw_validation as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp validate paw", description=implementation.__doc__
    )
    parser.add_argument("root", type=Path)
    print(json.dumps(implementation.validate(parser.parse_args(argv).root), indent=2))


def _spin_validation(argv=None):
    from ...experiments.vasp import spin_validation as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp validate spin", description=implementation.__doc__
    )
    parser.add_argument("root", type=Path)
    print(json.dumps(implementation.validate(parser.parse_args(argv).root), indent=2))


def _training_validation(argv=None):
    from ...experiments.vasp import training_validation as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp validate training", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("native-checks", "pipeline"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("--epochs", type=int, default=300)
    q = sub.add_parser("fit")
    q.add_argument("root")
    q.add_argument("mode")
    q.add_argument("--epochs", type=int, default=300)
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "native-checks":
        implementation.native_checks(a.root)
    elif a.command == "bands":
        implementation.bands_case(a.root, a.index)
    elif a.command == "pipeline":
        implementation.pipeline(a.root, a.epochs)
    else:
        from ...vasp.train import train, evaluate

        train(a.root, a.mode, a.epochs)
        evaluate(a.root, a.mode)


def _ace_validation(argv=None):
    from ...experiments.vasp import ace_validation as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp validate ace", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("root")
    s.add_argument("reference")
    s.add_argument("executable")
    s.add_argument("native_executable")
    s = sub.add_parser("execute")
    s.add_argument("root")
    a = vars(p.parse_args(argv))
    command = a.pop("command")
    print(
        json.dumps(
            (
                implementation.prepare
                if command == "prepare"
                else implementation.execute
            )(**a),
            indent=2,
        )
    )


def _exchange(argv=None):
    from ...vasp import exchange as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp validate exchange", description=implementation.__doc__
    )
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            implementation.build_ace_from_samples(args.samples, args.output), indent=2
        )
    )


def _operator_diagnostic(argv=None):
    from ...experiments.vasp import operator_diagnostic as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp validate operator", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training)
    elif a.command == "worker":
        implementation.worker(a.root, a.index)
    else:
        implementation.execute(a.root)


def _hybrid_pilot(argv=None):
    from ...experiments.vasp import hybrid_pilot as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp validate hybrid", description=implementation.__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--potcar", type=Path, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--native-executable", type=Path, required=True)
    p.add_argument("--meshes", nargs="+", type=int, default=[2, 3])
    p = sub.add_parser("execute")
    p.add_argument("root", type=Path)
    p.add_argument("index", type=int)
    p = sub.add_parser("collect")
    p.add_argument("root", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = {
        "prepare": implementation.prepare,
        "execute": implementation.execute,
        "collect": implementation.collect,
    }[command](**args)
    print(json.dumps(result, indent=2))


COMMANDS = {
    "paw": _paw_validation,
    "spin": _spin_validation,
    "training": _training_validation,
    "ace": _ace_validation,
    "exchange": _exchange,
    "operator": _operator_diagnostic,
    "hybrid": _hybrid_pilot,
}


def main(argv=None):
    return dispatch(COMMANDS, argv, prog="mapho-vasp validate", description=__doc__)


if __name__ == "__main__":
    main()

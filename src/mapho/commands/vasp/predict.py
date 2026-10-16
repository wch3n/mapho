"""VASP predict commands with lazy imports of APIs and study implementations."""

from pathlib import Path
import argparse
import json

from . import dispatch


def _paw(argv=None):
    from ...vasp import paw as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp predict paw", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("reference", type=Path)
    ex.add_argument("output", type=Path)
    rp = sub.add_parser("replay")
    rp.add_argument("operator", type=Path)
    rp.add_argument("folder", type=Path)
    rp.add_argument("--shift-eV", type=float, default=0.0)
    co = sub.add_parser("compare")
    co.add_argument("reference")
    co.add_argument("replay")
    co.add_argument("--shift-eV", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.command == "export":
        result = implementation.export_operator(args.reference, args.output)
    elif args.command == "replay":
        result = implementation.prepare_replay(
            args.operator, args.folder, shift_eV=args.shift_eV
        )
    else:
        result = implementation.compare(args.reference, args.replay, args.shift_eV)
    print(json.dumps(result, indent=2))


def _primitive_bands(argv=None):
    from ...experiments.vasp import primitive_bands as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp predict silicon-bands", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    for stage in ("reference", "bands"):
        q = sub.add_parser(stage)
        q.add_argument("root")
        q.add_argument("name")
    for stage in ("execute", "analyze"):
        q = sub.add_parser(stage)
        q.add_argument("root")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training)
    elif a.command == "reference":
        implementation.reference(a.root, a.name)
    elif a.command == "bands":
        implementation.bands(a.root, a.name)
    elif a.command == "analyze":
        implementation.analyze(a.root)
    else:
        implementation.execute(a.root)


def _binary_primitive_bands(argv=None):
    from ...experiments.vasp import binary_primitive_bands as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp predict binary-bands", description=implementation.__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "execute", "analyze"):
        p = sub.add_parser(command)
        p.add_argument("root")
    for command in ("reference", "predict", "bands"):
        p = sub.add_parser(command)
        p.add_argument("root")
        p.add_argument("material", choices=["lif", "mos2"])
        if command != "predict":
            p.add_argument("name")
    a = parser.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root)
    elif a.command == "reference":
        implementation.reference(a.root, a.material, a.name)
    elif a.command == "predict":
        implementation.predict_material(a.root, a.material)
    elif a.command == "bands":
        implementation.bands(a.root, a.material, a.name)
    elif a.command == "analyze":
        implementation.analyze(a.root)
    else:
        implementation.execute(a.root)


def _primitive_plot(argv=None):
    from ...experiments.vasp import primitive_plot as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp predict silicon-plot", description=implementation.__doc__
    )
    parser.add_argument("root")
    implementation.plot(parser.parse_args(argv).root)


def _binary_primitive_plot(argv=None):
    from ...experiments.vasp import binary_primitive_plot as implementation

    parser = argparse.ArgumentParser(
        prog="mapho-vasp predict binary-plot", description=implementation.__doc__
    )
    parser.add_argument("root")
    implementation.plot(parser.parse_args(argv).root)


COMMANDS = {
    "paw": _paw,
    "silicon-bands": _primitive_bands,
    "binary-bands": _binary_primitive_bands,
    "silicon-plot": _primitive_plot,
    "binary-plot": _binary_primitive_plot,
}


def main(argv=None):
    return dispatch(COMMANDS, argv, prog="mapho-vasp predict", description=__doc__)


if __name__ == "__main__":
    main()

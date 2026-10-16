"""Command-line arguments for binary primitive bands; implementation lives in mapho.workflows.vasp.binary_primitive_bands."""

import argparse


def main(argv=None):
    from ...workflows.vasp import binary_primitive_bands as implementation

    parser = argparse.ArgumentParser(description=implementation.__doc__)
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


if __name__ == "__main__":
    main()

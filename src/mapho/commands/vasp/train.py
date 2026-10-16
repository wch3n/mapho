"""VASP train commands with lazy imports of APIs and study implementations."""

import argparse
import json

from . import dispatch


def _train(argv=None):
    from ...vasp import train as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train model", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("dataset")
    q.add_argument("atomic")
    q.add_argument("output")
    q = sub.add_parser("train")
    q.add_argument("root")
    q.add_argument("mode", choices=["local", "none", "full"])
    q.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override manifest setting (default: 300)",
    )
    q.add_argument("--device", default=None, help="Override manifest device")
    q = sub.add_parser("evaluate")
    q.add_argument("root")
    q.add_argument("mode", choices=["local", "none", "full"])
    q.add_argument("--device", default=None, help="Override manifest device")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.dataset, a.atomic, a.output)
    elif a.command == "train":
        implementation.train(a.root, a.mode, a.epochs, device=a.device)
    else:
        implementation.evaluate(a.root, a.mode, device=a.device)


def _configured_run(argv=None):
    parser = argparse.ArgumentParser(
        prog="mapho-vasp train run",
        description="Prepare and fit a PAW model from YAML; CLI overrides take precedence.",
    )
    parser.add_argument("--config", required=True, help="Training YAML file")
    parser.add_argument("--output", help="New output directory, relative to cwd")
    parser.add_argument("--mode", choices=["local", "none", "full"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", help="cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print resolved settings without writing or training",
    )
    actions.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare caches and configuration without fitting",
    )
    args = vars(parser.parse_args(argv))
    config = args.pop("config")
    dry_run, prepare_only = args.pop("dry_run"), args.pop("prepare_only")
    from ...experiments.vasp.configured_training import run

    result = run(
        config,
        overrides={k: v for k, v in args.items() if v is not None},
        dry_run=dry_run,
        prepare_only=prepare_only,
    )
    print(json.dumps(result, indent=2))


def _material_training(argv=None):
    from ...experiments.vasp import material_training as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train material", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("dataset")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("native-checks")
    q.add_argument("training")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.dataset)
    elif a.command == "execute":
        implementation.execute(a.root)
    elif a.command == "native-checks":
        implementation.native_checks(a.training)
    else:
        implementation.bands_case(a.root, a.index)


def _paw_repair(argv=None):
    from ...experiments.vasp import paw_repair as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train paw-repair", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("training")
    q.add_argument("--refine-from")
    q = sub.add_parser("prepare-evaluation")
    q.add_argument("root")
    q.add_argument("trained")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q.add_argument("phase", choices=["validation", "test"])
    for name in ("fit", "predict-test"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("label", choices=implementation.LABELS)
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("split")
    q.add_argument("index", type=int)
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.training, a.refine_from)
    elif a.command == "prepare-evaluation":
        implementation.prepare_evaluation(a.root, a.trained)
    elif a.command == "fit":
        implementation.fit(a.root, a.label)
    elif a.command == "predict-test":
        implementation.predictions(a.root, a.label, "test")
    elif a.command == "bands":
        implementation.band_case(a.root, a.split, a.index)
    else:
        implementation.execute(a.root, a.phase)


def _paw_joint(argv=None):
    from ...experiments.vasp import paw_joint as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train paw-joint", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("repaired")
    q.add_argument("primitive")
    q.add_argument("--epochs", type=int, default=300)
    for name in ("execute", "primitive-results"):
        q = sub.add_parser(name)
        q.add_argument("root")
    for name in ("fit", "predict"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("label", choices=implementation.LABELS)
        if name == "predict":
            q.add_argument("split", choices=["validation", "test", "primitive"])
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("split")
    q.add_argument("index", type=int)
    q.add_argument("label", choices=implementation.LABELS)
    q = sub.add_parser("summarize")
    q.add_argument("root")
    q.add_argument("split")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.repaired, a.primitive, a.epochs)
    elif a.command == "fit":
        implementation.fit(a.root, a.label)
    elif a.command == "predict":
        implementation.predictions(a.root, a.label, a.split)
    elif a.command == "bands":
        implementation.bands(a.root, a.split, a.index, a.label)
    elif a.command == "summarize":
        implementation.summarize(a.root, a.split)
    elif a.command == "primitive-results":
        implementation.primitive_results(a.root)
    else:
        implementation.execute(a.root)


def _activation_ablation(argv=None):
    from ...experiments.vasp import activation_ablation as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train activation", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("baseline")
    q.add_argument("primitive")
    for name in ("execute", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("root")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("mode", choices=["none", "full"])
    q.add_argument("index")
    a = p.parse_args(argv)
    if a.command == "prepare":
        implementation.prepare(a.root, a.baseline, a.primitive)
    elif a.command == "bands":
        implementation.band_case(a.root, a.mode, a.index)
    elif a.command == "summarize":
        implementation.summarize(a.root)
    else:
        implementation.execute(a.root)


def _pbe0_heads(argv=None):
    from ...experiments.vasp import pbe0_learning as implementation

    p = argparse.ArgumentParser(
        prog="mapho-vasp train pbe0-heads", description=implementation.__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare-dataset")
    s.add_argument("root")
    s.add_argument("calibration")
    s.add_argument("--count", type=int, default=24)
    s = sub.add_parser("worker")
    s.add_argument("root")
    s.add_argument("index", type=int)
    s = sub.add_parser("audit")
    s.add_argument("root")
    s = sub.add_parser("prepare-training")
    s.add_argument("root")
    s.add_argument("dataset")
    for name in ("fit", "evaluate-bands"):
        s = sub.add_parser(name)
        s.add_argument("root")
        s.add_argument("mode", choices=["local", "full"])
        if name == "fit":
            s.add_argument("--epochs", type=int, default=None)
            s.add_argument("--device", default=None)
    args = vars(p.parse_args(argv))
    command = args.pop("command").replace("-", "_")
    return getattr(implementation, command)(**args)


COMMANDS = {
    "run": _configured_run,
    "pbe0-heads": _pbe0_heads,
    "model": _train,
    "material": _material_training,
    "paw-repair": _paw_repair,
    "paw-joint": _paw_joint,
    "activation": _activation_ablation,
}


def main(argv=None):
    return dispatch(COMMANDS, argv, prog="mapho-vasp train", description=__doc__)


if __name__ == "__main__":
    main()

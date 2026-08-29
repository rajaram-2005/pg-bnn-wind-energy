"""WindFusion command line interface.

Commands that need a dataset are explicit: they refuse to invent results and
print ``NOT MEASURED`` for anything that has not been measured on this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import WindFusionConfig, default_config
from .models import MODEL_REGISTRY, create_model, registry_table
from .provenance import PROVENANCE, provenance_markdown, provenance_table, validate_provenance

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _load_config(path: str | None) -> WindFusionConfig:
    return WindFusionConfig.from_yaml(path) if path else default_config()


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


# ── command implementations ────────────────────────────────────────────────


def _cmd_train(args) -> int:
    from .data.loaders import build_dataloaders
    from .training.loop import Trainer, save_checkpoint, write_history

    config = _load_config(args.config)
    if args.epochs:
        config = config.with_overrides(**{"training.epochs": args.epochs})
    if args.mode:
        config = config.with_overrides(**{"model.mode": args.mode})
    if args.turbines:
        config = config.with_overrides(**{"data.n_turbines": args.turbines})

    spec = MODEL_REGISTRY[config.model.mode]
    model = create_model(
        config.model.mode,
        config.model.input_features,
        config.model.outputs,
        config.model.physics_features,
        config.turbine,
    )
    bundle = build_dataloaders(
        config, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors
    )
    print(f"[windfusion] {config.model.mode}: {bundle.summary()}", file=sys.stderr)
    trainer = Trainer(model, config, device=args.device, verbose=not args.quiet)
    history = trainer.fit(bundle.train, bundle.val)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_checkpoint(
        model,
        out_dir / f"{config.model.mode}.pt",
        config,
        meta={"data": bundle.meta, "history_best": history.best},
    )
    write_history(history, out_dir / f"{config.model.mode}-history.json")
    _print(
        {
            "checkpoint": str(checkpoint),
            "model": config.model.mode,
            "best": history.best,
            "parameters": model.parameter_count,
            "data": bundle.meta,
            "note": "metrics below are training-loss values, NOT field accuracy",
        }
    )
    return 0


def _cmd_evaluate(args) -> int:
    from .data.loaders import build_dataloaders
    from .evaluation.evaluate import evaluate_model, evaluate_verification
    from .training.calibration import calibrate
    from .training.loop import load_checkpoint

    if args.checkpoint:
        model, config, meta = load_checkpoint(args.checkpoint, args.device)
    else:
        config = _load_config(args.config)
        model = create_model(
            args.mode or config.model.mode,
            config.model.input_features,
            config.model.outputs,
            config.model.physics_features,
            config.turbine,
        )
        meta = {"note": "untrained model: metrics are a pipeline check only"}
    spec = getattr(model, "spec", MODEL_REGISTRY[config.model.mode])
    bundle = build_dataloaders(
        config, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors
    )
    calibration = None
    if not args.no_calibration:
        calibration = calibrate(model, bundle.val, device=args.device)
    result = evaluate_model(
        model,
        bundle.test,
        config,
        device=args.device,
        mc_samples=args.mc_samples,
        calibration=calibration,
        keep_arrays=False,
    )
    verdicts = evaluate_verification(
        model, bundle.test, config, device=args.device, mc_samples=max(args.mc_samples // 2, 2)
    )
    _print(
        {
            "model": spec.model_id,
            "checkpoint_meta": meta,
            "metrics": result.metrics,
            "timing": result.timing,
            "calibration": calibration.as_dict() if calibration else None,
            "verification": verdicts,
            "dataset": bundle.meta,
        }
    )
    return 0


def _cmd_benchmark(args) -> int:
    from .evaluation.benchmark import benchmark_family

    config = _load_config(args.config)
    modes = [args.mode] if args.mode else (args.modes.split(",") if args.modes else None)
    report = benchmark_family(config, modes=modes, repeats=args.repeats)
    report["note"] = (
        "Measured on this machine with random inputs. Predictive accuracy is NOT MEASURED: "
        "run `windfusion train` then `windfusion evaluate` on a versioned dataset."
    )
    _print(report)
    return 0


def _cmd_distill(args) -> int:
    from .data.loaders import build_dataloaders
    from .training.distillation import DistillTrainer
    from .training.loop import save_checkpoint

    config = _load_config(args.config)
    teacher_id = args.teacher or config.distillation.teacher
    student_id = args.student or config.distillation.student
    teacher_spec = MODEL_REGISTRY[teacher_id]
    student_spec = MODEL_REGISTRY[student_id]
    teacher = create_model(teacher_id, config.model.input_features, config.model.outputs, config.model.physics_features, config.turbine)
    student = create_model(student_id, config.model.input_features, config.model.outputs, config.model.physics_features, config.turbine)
    if args.teacher_checkpoint:
        from .training.loop import load_checkpoint

        teacher, config, _ = load_checkpoint(args.teacher_checkpoint, args.device)
    bundle = build_dataloaders(
        config,
        forecast_horizon=max(teacher_spec.forecast_horizon, student_spec.forecast_horizon),
        neighbors=max(teacher_spec.neighbors, student_spec.neighbors),
    )
    if args.epochs:
        config = config.with_overrides(**{"training.epochs": args.epochs})
    trainer = DistillTrainer(student, teacher, config, device=args.device, verbose=not args.quiet)
    history = trainer.fit(bundle.train, bundle.val)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = save_checkpoint(
        student, out / f"{student_id}-distilled.pt", config, meta={"teacher": teacher_id}
    )
    _print({"student": student_id, "teacher": teacher_id, "checkpoint": str(path), "best": history.best})
    return 0


def _cmd_export(args) -> int:
    from .deployment.export import export_onnx, export_torchscript

    config = _load_config(args.config)
    model = create_model(
        args.mode or config.model.mode,
        config.model.input_features,
        config.model.outputs,
        config.model.physics_features,
        config.turbine,
    )
    if args.checkpoint:
        from .training.loop import load_checkpoint

        model, config, _ = load_checkpoint(args.checkpoint, args.device)
    suffix = Path(args.out).suffix
    if suffix == ".onnx":
        report = export_onnx(model, args.out, config.model.sequence_length, check_parity=not args.no_parity)
    else:
        report = export_torchscript(model, args.out, config.model.sequence_length)
    _print(report.as_dict())
    return 0


def _cmd_simulate(args) -> int:
    from .digital_twin.simulator import DigitalTwin
    from .digital_twin.state import TurbineState

    state = TurbineState(
        bearing_temperature=args.bearing_temperature,
        gearbox_temperature=args.gearbox_temperature,
        generator_temperature=args.generator_temperature,
        vibration=args.vibration,
        health=args.health,
        rul_days=args.rul_days,
    )
    twin = DigitalTwin(state)
    scenarios = {
        "coolant_loss": {"gearbox_temperature": args.gearbox_temperature + 30},
        "bearing_degradation": {"vibration": args.vibration + 3.0},
        "derate_20pct": {"torque": 0.8 * state.torque},
        "icing": {"icing_risk": 0.8, "wind_speed": 3.0},
    }
    _print({"baseline": twin.state.to_dict(), "scenarios": twin.scenario_table(scenarios)})
    return 0


def _cmd_explain(args) -> int:
    from .data.loaders import build_dataloaders
    from .explainability.report import explain, render_report
    from .physics.residuals import compute_residuals

    config = _load_config(args.config)
    model = create_model(
        args.mode or config.model.mode,
        config.model.input_features,
        config.model.outputs,
        config.model.physics_features,
        config.turbine,
    )
    if args.checkpoint:
        from .training.loop import load_checkpoint

        model, config, _ = load_checkpoint(args.checkpoint, args.device)
    spec = model.spec
    bundle = build_dataloaders(config, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors)
    batch = next(iter(bundle.test))
    prediction = model.predict(batch["sequence"], batch["physics"], batch.get("neighbors"), samples=args.mc_samples)
    residuals = compute_residuals(batch["raw"], config.turbine)
    report = explain(
        prediction,
        {k: float(v[0].abs().mean()) for k, v in residuals.items()},
        {"health": float(prediction["mean"][0, 1]), "rul_days": float(prediction["mean"][0, 2]) * 365},
        model=model,
        sequence=batch["sequence"],
        physics=batch["physics"],
    )
    print(render_report(report) if args.text else json.dumps(report, indent=2, default=str))
    return 0


def _cmd_fleet(args) -> int:
    from .data.loaders import build_dataloaders, per_turbine_loaders
    from .training.fleet import federated_train

    config = _load_config(args.config)
    model = create_model(
        args.mode or config.model.mode,
        config.model.input_features,
        config.model.outputs,
        config.model.physics_features,
        config.turbine,
    )
    spec = model.spec
    bundle = build_dataloaders(config, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors)
    clients = list(per_turbine_loaders(bundle.datasets.train, batch_size=32).values())[: args.clients]
    reports = federated_train(model, clients, config, device=args.device, rounds=args.rounds)
    _print(
        {
            "model": spec.model_id,
            "clients": len(clients),
            "rounds": [r.as_dict() for r in reports],
            "note": "federated averaging over per-turbine clients (synthetic data unless configured otherwise)",
        }
    )
    return 0


def _cmd_models(args) -> int:
    if args.markdown:
        print(registry_table())
        return 0
    _print({mode: spec.as_dict() for mode, spec in MODEL_REGISTRY.items()})
    return 0


def _cmd_provenance(args) -> int:
    if args.markdown:
        print(provenance_markdown())
        return 0
    if args.validate:
        problems = validate_provenance()
        _print({"problems": problems, "ok": not problems})
        return 1 if problems else 0
    _print(
        {
            "version": __version__,
            "concepts": [
                {
                    "concept_id": c.concept_id,
                    "source_repo": c.source_repo,
                    "upstream_path": c.upstream_path,
                    "license": c.upstream_license,
                    "reuse": c.reuse,
                    "local_path": c.local_path,
                    "validated_by": list(c.validated_by),
                }
                for c in PROVENANCE
            ],
            "table": provenance_table(),
        }
    )
    return 0


def _cmd_telemetry(args) -> int:
    from .edge.telemetry import AdaptiveTelemetryPolicy, evaluate_policy

    config = _load_config(args.config)
    policy = AdaptiveTelemetryPolicy(
        bypass_threshold=config.telemetry.bypass_threshold,
        detail_threshold=config.telemetry.detail_threshold,
        weights=dict(config.telemetry.weights),
        cooldown_steps=config.telemetry.cooldown_steps,
    )
    import numpy as np

    rng = np.random.default_rng(config.seed)
    windows = [np.cumsum(rng.normal(0, 0.2, args.window)) + 20 for _ in range(args.windows)]
    signals = [
        {
            "anomaly": float(rng.uniform(0, 0.2)),
            "epistemic": float(rng.uniform(0, 0.6)),
            "rate": float(rng.uniform(0, 0.3)),
            "physics_residual": float(rng.uniform(0, 0.4)),
        }
        for _ in range(args.windows)
    ]
    signals[args.windows // 2] = {"anomaly": 0.9, "epistemic": 0.8, "rate": 0.5, "physics_residual": 0.7}
    _print(evaluate_policy(policy, windows, signals))
    return 0


# ── parser ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="windfusion",
        description=f"WindFusion v{__version__} - physics-confidence fusion for wind-turbine PdM (advisory only)",
    )
    parser.add_argument("--version", action="version", version=f"windfusion {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, *aliases, **kwargs):
        p = sub.add_parser(name, aliases=list(aliases), **kwargs)
        return p

    p = add("train", help="train a model on the configured data source")
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--config")
    p.add_argument("--epochs", type=int)
    p.add_argument("--turbines", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/checkpoints")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = add("evaluate", help="evaluate a checkpoint or an untrained model")
    p.add_argument("--checkpoint")
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--mc-samples", dest="mc_samples", type=int, default=16)
    p.add_argument("--no-calibration", action="store_true")
    p.set_defaults(func=_cmd_evaluate)

    p = add("benchmark", help="measure parameters, size and CPU latency")
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--modes", help="comma separated model ids")
    p.add_argument("--config")
    p.add_argument("--repeats", type=int, default=50)
    p.set_defaults(func=_cmd_benchmark)

    p = add("distill", help="distil a teacher into a student")
    p.add_argument("--teacher", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--student", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--config")
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/checkpoints")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_distill)

    p = add("export", help="export ONNX or TorchScript with a parity check")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--checkpoint")
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-parity", action="store_true")
    p.set_defaults(func=_cmd_export)

    p = add("simulate", help="digital-twin counterfactual scenarios")
    p.add_argument("--bearing-temperature", type=float, default=60)
    p.add_argument("--gearbox-temperature", type=float, default=65)
    p.add_argument("--generator-temperature", type=float, default=80)
    p.add_argument("--vibration", type=float, default=2.5)
    p.add_argument("--health", type=float, default=0.9)
    p.add_argument("--rul-days", dest="rul_days", type=float, default=120)
    p.set_defaults(func=_cmd_simulate)

    p = add("explain", help="explain one prediction (routing, residuals, saliency)")
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--checkpoint")
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--mc-samples", dest="mc_samples", type=int, default=8)
    p.add_argument("--text", action="store_true")
    p.set_defaults(func=_cmd_explain)

    p = add("fleet", help="federated averaging over turbine clients")
    p.add_argument("--mode", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--clients", type=int, default=6)
    p.set_defaults(func=_cmd_fleet)

    p = add("models", help="list the model family")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=_cmd_models)

    p = add("provenance", help="print the provenance registry")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.set_defaults(func=_cmd_provenance)

    p = add("telemetry", help="evaluate the adaptive telemetry policy")
    p.add_argument("--config")
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--window", type=int, default=64)
    p.set_defaults(func=_cmd_telemetry)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

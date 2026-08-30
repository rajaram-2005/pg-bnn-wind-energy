"""End-to-end reproducible benchmark on the **synthetic** fleet.

Everything measured here is measured on simulated data with a known latent
damage state. It is a regression/ablation harness, **not** evidence of field
performance: every artefact this script writes is stamped ``SYNTHETIC``.

Stages
------
1. fleet          - deterministic synthetic fleet + checksum
2. footprint      - parameters, size, CPU latency for every model
3. supervised     - baselines and WindFusion models trained and scored
4. calibration    - temperature scaling / conformal coverage + verdict behaviour
5. distillation   - teacher -> student vs student trained from scratch
6. fleet          - federated averaging + few-shot adaptation to a held-out site
7. telemetry      - adaptive policy: bandwidth saved vs fidelity kept

Usage
-----
    python benchmarks/synthetic_benchmark.py                 # full run
    python benchmarks/synthetic_benchmark.py --quick          # smoke-sized run
    python benchmarks/synthetic_benchmark.py --epochs 25 --turbines 48
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from windfusion import __version__
from windfusion.config import WindFusionConfig
from windfusion.data.loaders import build_dataloaders, per_turbine_loaders
from windfusion.data.schema import RUL_SCALE_DAYS
from windfusion.evaluation.baselines import build_baselines, dataset_arrays
from windfusion.evaluation.benchmark import benchmark_family
from windfusion.evaluation.evaluate import evaluate_model, evaluate_verification
from windfusion.evaluation.metrics import summarise_all
from windfusion.models import MODEL_REGISTRY, create_model
from windfusion.training.calibration import calibrate
from windfusion.training.distillation import DistillTrainer
from windfusion.training.fleet import fewshot_adapt, federated_train, support_loader
from windfusion.training.loop import Trainer, set_determinism

SYNTHETIC_STAMP = "SYNTHETIC (simulated fleet, not field data)"


@dataclass
class BenchmarkReport:
    """Everything the run produced, ready to serialise."""

    stage: dict = field(default_factory=dict)

    def set(self, key: str, value) -> None:
        self.stage[key] = value
        print(f"[benchmark] {key}: done", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────


def _model_bundle(mode: str, config: WindFusionConfig):
    spec = MODEL_REGISTRY[mode]
    return (
        create_model(
            mode,
            config.model.input_features,
            config.model.outputs,
            config.model.physics_features,
            config.turbine,
        ),
        spec,
    )


def _prepare(mode: str, config: WindFusionConfig, runs=None):
    model, spec = _model_bundle(mode, config)
    bundle = build_dataloaders(
        config, runs=runs, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors
    )
    return model, bundle


def train_and_score(mode: str, config: WindFusionConfig, runs=None, epochs: int | None = None):
    """Train one model and return (metrics, calibration, verification, timing)."""
    cfg = config.with_overrides(**{"model.mode": mode})
    if epochs:
        cfg = cfg.with_overrides(**{"training.epochs": epochs})
    set_determinism(cfg.seed)
    model, bundle = _prepare(mode, cfg, runs)
    trainer = Trainer(model, cfg, verbose=False)
    started = time.perf_counter()
    history = trainer.fit(bundle.train, bundle.val)
    train_seconds = round(time.perf_counter() - started, 2)
    calibration = calibrate(model, bundle.val, mc_samples=4)
    result = evaluate_model(
        model, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration
    )
    verdicts = evaluate_verification(model, bundle.test, cfg, mc_samples=4)
    return {
        "model": mode,
        "parameters": model.parameter_count,
        "train_seconds": train_seconds,
        "epochs_run": len(history.epochs),
        "best_val_total": None if history.best is None else round(history.best.get("val_total", float("nan")), 6),
        "metrics": result.metrics,
        "timing": result.timing,
        "calibration": calibration.as_dict(),
        "verification": verdicts,
    }


def score_baselines(config: WindFusionConfig, bundle, epochs: int) -> dict:
    """Fit the non-neural baselines on the same windows as the models."""
    train = dataset_arrays(bundle.datasets.train)
    test = dataset_arrays(bundle.datasets.test)
    out = {}
    for name, baseline in build_baselines(epochs=epochs).items():
        started = time.perf_counter()
        baseline.fit(train)
        mean, std = baseline.predict(test)
        out[name] = {
            "model": name,
            "parameters": int(getattr(baseline, "coef_", np.zeros((1, 1))).size)
            if name == "ridge"
            else _count_parameters(baseline),
            "train_seconds": round(time.perf_counter() - started, 2),
            "metrics": summarise_all(
                test.y, mean, std, warning_horizon_days=config.evaluation.warning_horizon_days
            ),
        }
    return out


def _count_parameters(baseline) -> int:
    model = getattr(baseline, "model", None)
    if model is None:
        return 0
    return int(sum(p.numel() for p in model.parameters()))


# ── stages ─────────────────────────────────────────────────────────────────


def stage_fleet(config: WindFusionConfig, runs) -> dict:
    from windfusion.data import fleet_summary

    return {"summary": fleet_summary(runs), "config": config.data.to_dict()}


def stage_footprint(config: WindFusionConfig) -> dict:
    return benchmark_family(config, repeats=30)


def stage_supervised(config: WindFusionConfig, runs, modes: list[str], epochs: int, include_baselines: bool) -> dict:
    from windfusion.data.loaders import build_dataloaders

    bundle = build_dataloaders(config, runs=runs)
    results: dict[str, dict] = {}
    if include_baselines:
        results.update(score_baselines(config, bundle, epochs))
    for mode in modes:
        results[mode] = train_and_score(mode, config, runs, epochs=epochs)
    return results


def stage_self_learning(config: WindFusionConfig, runs, epochs: int | None = None) -> dict:
    """Fit the self-learning model on its own, then score it like every other model.

    ``autonomous_fit`` consumes no ground-truth labels: round 1 is pure
    self-supervision, round 2 adds the model's own gated pseudo-labels. The
    resulting model is then evaluated with the same supervised metric suite,
    which is exactly the point of the stage: how far does a model get when it
    trains on its own?
    """
    cfg = config.with_overrides(**{"model.mode": "windfusion-auto"})
    sc = cfg.self_training
    fit_epochs = max(1, min(sc.epochs, epochs or sc.epochs))
    set_determinism(cfg.seed)
    model, bundle = _prepare("windfusion-auto", cfg, runs)
    started = time.perf_counter()
    report = model.autonomous_fit(
        bundle.train, bundle.val, config=cfg, epochs=fit_epochs, verbose=False
    )
    seconds = round(time.perf_counter() - started, 2)
    calibration = calibrate(model, bundle.val, mc_samples=4)
    result = evaluate_model(
        model, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration
    )
    verdicts = evaluate_verification(model, bundle.test, cfg, mc_samples=4)
    return {
        "model": "windfusion-auto",
        "mode": report["mode"],
        "parameters": report["parameters"],
        "train_seconds": seconds,
        "rounds_run": report["rounds"],
        "epochs_per_round": fit_epochs,
        "best_objective": (
            None if report["best"] is None else round(report["best"]["total"], 6)
        ),
        "labels_consumed": False,
        "pseudo_labels": report["pseudo_labels"],
        "pseudo_stats": report["pseudo_stats"],
        "metrics": result.metrics,
        "timing": result.timing,
        "calibration": calibration.as_dict(),
        "verification": verdicts,
    }


def stage_distillation(config: WindFusionConfig, runs, epochs: int) -> dict:
    teacher_id = config.distillation.teacher
    student_id = config.distillation.student
    teacher_result = train_and_score(teacher_id, config, runs, epochs=epochs)

    # student from scratch (control)
    scratch = train_and_score(student_id, config, runs, epochs=epochs)

    # student with distillation from the freshly trained teacher
    set_determinism(config.seed)
    cfg = config.with_overrides(**{"training.epochs": epochs, "model.mode": student_id})
    teacher, _ = _model_bundle(teacher_id, cfg)
    teacher_trainer = Trainer(teacher, cfg, verbose=False)
    teacher_bundle = build_dataloaders(cfg, runs=runs, forecast_horizon=MODEL_REGISTRY[teacher_id].forecast_horizon)
    teacher_trainer.fit(teacher_bundle.train, teacher_bundle.val)

    student, spec = _model_bundle(student_id, cfg)
    bundle = build_dataloaders(
        cfg, runs=runs, forecast_horizon=spec.forecast_horizon, neighbors=spec.neighbors
    )
    trainer = DistillTrainer(student, teacher, cfg, verbose=False)
    history = trainer.fit(bundle.train, bundle.val)
    calibration = calibrate(student, bundle.val, mc_samples=4)
    result = evaluate_model(student, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration)
    distilled = {
        "model": f"{student_id}+distilled",
        "parameters": student.parameter_count,
        "epochs_run": len(history.epochs),
        "best_val_total": None if history.best is None else round(history.best.get("val_total", float("nan")), 6),
        "metrics": result.metrics,
        "calibration": calibration.as_dict(),
    }
    return {
        "teacher": teacher_result,
        "student_from_scratch": scratch,
        "student_distilled": distilled,
        "kd_terms": {k: round(v, 6) for k, v in history.epochs[-1].items() if k.startswith("kd_")},
    }


def stage_fleet_learning(config: WindFusionConfig, runs, epochs: int) -> dict:
    """Federated averaging plus few-shot adaptation to a held-out site."""
    mode = "windfusion-edge"
    cfg = config.with_overrides(**{"model.mode": mode, "training.epochs": epochs})
    set_determinism(cfg.seed)
    model, bundle = _prepare(mode, cfg, runs)

    # zero-shot: train on the training sites only, score the held-out test site
    trainer = Trainer(model, cfg, verbose=False)
    trainer.fit(bundle.train, bundle.val)
    calibration = calibrate(model, bundle.val, mc_samples=4)
    zero_shot = evaluate_model(
        model, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration
    )

    # federated averaging over per-turbine clients from the training split
    clients = list(per_turbine_loaders(bundle.datasets.train, batch_size=32).values())
    fed_model, _ = _model_bundle(mode, cfg)
    fed_model.load_state_dict(model.state_dict())
    rounds = federated_train(fed_model, clients[: cfg.fleet.clients_per_round * 2], cfg, rounds=2)
    fed_result = evaluate_model(
        fed_model, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration
    )

    # few-shot adaptation: a handful of labelled windows from the held-out site
    support = support_loader(
        bundle.datasets.test, n_windows=cfg.fleet.adaptation_windows, batch_size=16, seed=cfg.seed
    )
    adapted = fewshot_adapt(model, support, cfg, steps=cfg.fleet.fewshot_steps)
    adapted_result = evaluate_model(
        adapted, bundle.test, cfg, mc_samples=cfg.evaluation.mc_samples, calibration=calibration
    )

    def _rmse(entry) -> float:
        return float(entry.metrics["regression"]["rul_days"]["mae_days"])

    return {
        "protocol": "held-out site: turbines never seen during training",
        "zero_shot_rul_mae_days": round(_rmse(zero_shot), 3),
        "federated_rul_mae_days": round(_rmse(fed_result), 3),
        "fewshot_rul_mae_days": round(_rmse(adapted_result), 3),
        "fewshot_windows": cfg.fleet.adaptation_windows,
        "federated_rounds": [r.as_dict() for r in rounds],
        "zero_shot_metrics": zero_shot.metrics,
        "fewshot_metrics": adapted_result.metrics,
    }


def stage_telemetry(config: WindFusionConfig) -> dict:
    from windfusion.edge.telemetry import AdaptiveTelemetryPolicy, evaluate_policy

    policy = AdaptiveTelemetryPolicy(
        bypass_threshold=config.telemetry.bypass_threshold,
        detail_threshold=config.telemetry.detail_threshold,
        weights=dict(config.telemetry.weights),
        cooldown_steps=config.telemetry.cooldown_steps,
    )
    rng = np.random.default_rng(config.seed)
    windows = [np.cumsum(rng.normal(0, 0.2, 128)).astype(np.float32) + 40 for _ in range(256)]
    signals = []
    for i in range(256):
        # 8% of windows are "events": high uncertainty / rate / physics residual
        if i % 12 == 0:
            signals.append({"anomaly": 0.8, "epistemic": 0.7, "rate": 0.6, "physics_residual": 0.7})
        else:
            signals.append(
                {
                    "anomaly": float(rng.uniform(0, 0.15)),
                    "epistemic": float(rng.uniform(0, 0.3)),
                    "rate": float(rng.uniform(0, 0.1)),
                    "physics_residual": float(rng.uniform(0, 0.2)),
                }
            )
    return evaluate_policy(policy, windows, signals)


# ── reporting ──────────────────────────────────────────────────────────────


def _regression_row(name: str, entry: dict) -> str:
    regression = entry["metrics"]["regression"]
    calibration = entry["metrics"]["calibration"]
    classification = entry["metrics"]["classification"]
    return (
        f"| `{name}` | {entry.get('parameters', 0):,} | "
        f"{regression['health_index']['mae']:.4f} | {regression['rul_days']['mae_days']:.2f} | "
        f"{classification['f1']:.3f} | {classification['auroc']:.3f} | "
        f"{calibration['ece']:.4f} | {calibration['nll']:.3f} | "
        f"{calibration['empirical_coverage']:.3f} | {entry.get('train_seconds', 0):.1f} |"
    )


def as_markdown(report: dict) -> str:
    lines = [
        f"# WindFusion v{__version__} - synthetic benchmark results",
        "",
        f"> **{SYNTHETIC_STAMP}.** Every number below was measured on a "
        "simulated fleet with a known latent damage state. It is a regression and "
        "ablation harness, not evidence of field performance.",
        "",
        f"- Generated: {report['environment']['generated_at']}",
        f"- Fleet checksum: `{report['fleet']['summary']['checksum']}`",
        f"- Data: {report['fleet']['summary']['n_turbines']} turbines x "
        f"{report['fleet']['summary']['seq_len']} samples, "
        f"{report['fleet']['summary']['n_sites']} sites",
        f"- Seed: {report['config_snapshot']['seed']}, epochs: "
        f"{report['config_snapshot']['training']['epochs']}",
        f"- Machine: {report['environment']['platform']} (torch "
        f"{report['environment']['torch']}, {report['environment']['threads']} threads)",
        "",
        "## 1. Model footprint",
        "",
        "| Model | Parameters | FP32 MB | INT8 MB | CPU latency ms (median) | windows/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, entry in report["footprint"]["results"].items():
        lines.append(
            f"| `{mode}` | {entry['parameters']:,} | {entry['size_fp32_mb']:.3f} | "
            f"{entry['size_int8_mb']} | {entry['latency_ms_median']:.3f} | "
            f"{entry['throughput_windows_per_s']:.1f} |"
        )
    lines += [
        "",
        "## 2. Predictive quality (synthetic test split)",
        "",
        "| Model | Parameters | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage | train s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in report["supervised"].items():
        lines.append(_regression_row(name, entry))
    lines += [
        "",
        "## 3. Distillation (teacher -> student)",
        "",
        "| Variant | Parameters | health MAE | RUL MAE (days) | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("teacher", "student_from_scratch", "student_distilled"):
        entry = report["distillation"][key]
        lines.append(
            f"| `{entry['model']}` | {entry['parameters']:,} | "
            f"{entry['metrics']['regression']['health_index']['mae']:.4f} | "
            f"{entry['metrics']['regression']['rul_days']['mae_days']:.2f} | "
            f"{entry['metrics']['calibration']['ece']:.4f} |"
        )
    lines += [
        "",
        f"Knowledge-distillation terms on the final epoch: {report['distillation']['kd_terms']}",
        "",
        "## 4. Self-learning (`windfusion-auto`, autonomous fit — no labels consumed)",
        "",
        "| Model | Parameters | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage | train s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _regression_row("windfusion-auto", report["self_learning"]),
        "",
        f"Rounds: {report['self_learning']['rounds_run']} "
        f"(self-supervised + gated pseudo-labels), pseudo-labels accepted: "
        f"{report['self_learning']['pseudo_labels']} of "
        f"{report['self_learning']['pseudo_stats'].get('observed', 0)} observed, "
        f"best autonomous objective: {report['self_learning']['best_objective']}. "
        f"Labels consumed: no — the supervised metric suite is applied *after* training "
        f"only to score the result.",
        "",
        "## 5. Fleet learning (held-out site)",
        "",
        "| Protocol | RUL MAE (days) |",
        "|---|---:|",
        f"| zero-shot (no adaptation) | {report['fleet_learning']['zero_shot_rul_mae_days']:.2f} |",
        f"| + federated averaging | {report['fleet_learning']['federated_rul_mae_days']:.2f} |",
        f"| + few-shot ({report['fleet_learning']['fewshot_windows']} windows) | "
        f"{report['fleet_learning']['fewshot_rul_mae_days']:.2f} |",
        "",
        "## 6. Verification behaviour (fail-closed)",
        "",
        "| Model | abstention rate | coverage | accuracy on decided | recall NORMAL / WARNING / CRITICAL |"
        " false critical | missed critical |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in report["supervised"].items():
        verdicts = entry.get("verification")
        if not verdicts:
            continue
        recall = verdicts.get("recall_by_class", {})
        lines.append(
            f"| `{name}` | {verdicts['abstention_rate']:.3f} | {verdicts['coverage']:.3f} | "
            f"{verdicts['accuracy_on_decided']:.3f} | "
            f"{recall.get('NORMAL', 0.0):.2f} / {recall.get('WARNING', 0.0):.2f} / "
            f"{recall.get('CRITICAL', 0.0):.2f} | "
            f"{verdicts['false_critical_rate']:.3f} | "
            f"{verdicts['missed_critical_rate']:.3f} |"
        )
    first = next(
        (e["verification"] for e in report["supervised"].values() if e.get("verification")), None
    )
    if first:
        lines += [
            "",
            f"Class balance of the synthetic test split: {first.get('truth_histogram', {})} "
            f"(majority-class rate {first.get('majority_class_rate', 0.0):.3f} — a trivial "
            f"always-NORMAL advisor scores this, so compare against it, not against zero).",
            "",
            f"Verdicts actually emitted by `windfusion-edge`: "
            f"{report['supervised']['windfusion-edge']['verification'].get('verdict_histogram', {})}. "
            f"Abstention is triggered by epistemic uncertainty, physics inconsistency or low data "
            f"completeness; this split is in-distribution with ~99% complete windows, so the "
            f"abstention path is exercised by the unit tests rather than by this table.",
        ]
    lines += [
        "",
        "## 7. Adaptive telemetry",
        "",
        f"- windows: {report['telemetry']['windows']}, modes: {report['telemetry']['modes']}",
        f"- bandwidth reduction: {report['telemetry']['bandwidth_reduction']:.3f}",
        f"- mean |error| on reconstruction: {report['telemetry']['mean_mae']:.5f}",
        "",
        "## 8. What is NOT measured here",
        "",
        "- Real SCADA/CM data of any kind (no licensed dataset is bundled).",
        "- Energy per inference and on-device RAM: needs platform tooling.",
        "- Cross-OEM and cross-farm generalisation.",
        "- Any comparison against the upstream repositories' published numbers: "
        "those numbers come from different data and are not comparable.",
        "",
        "## 9. Fleet composition",
        "",
        f"```json\n{json.dumps(report['fleet']['summary'], indent=2)}\n```",
        "",
    ]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--turbines", type=int, default=36)
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--modes", default=None, help="comma separated model ids")
    parser.add_argument("--quick", action="store_true", help="tiny run for CI smoke tests")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--out-dir", default="benchmarks/results")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.epochs, args.turbines, args.seq_len = 3, 6, 400

    config = WindFusionConfig()
    config = config.with_overrides(
        **{
            "seed": args.seed,
            "data.n_turbines": args.turbines,
            "data.seq_len": args.seq_len,
            "data.seed": args.seed,
            "training.epochs": args.epochs,
            "evaluation.mc_samples": 8,
        }
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    from windfusion.data import load_fleet

    started = time.perf_counter()
    runs = load_fleet(config.data)
    report = BenchmarkReport()
    report.set("fleet", stage_fleet(config, runs))
    report.set("footprint", stage_footprint(config))

    modes = (
        args.modes.split(",")
        if args.modes
        else [
            "windfusion-edge",
            "windfusion-lite",
            "windfusion-research",
            "windfusion-auto",
            "ra-wind",
            "qinglong-wind",
            "vayu-wind",
            "odin-wind",
            "aeolus-wind",
        ]
    )
    report.set(
        "supervised",
        stage_supervised(config, runs, modes, args.epochs, not args.no_baselines),
    )
    report.set("self_learning", stage_self_learning(config, runs, args.epochs))
    report.set("distillation", stage_distillation(config, runs, args.epochs))
    report.set("fleet_learning", stage_fleet_learning(config, runs, args.epochs))
    report.set("telemetry", stage_telemetry(config))

    payload = {
        "schema": "windfusion-benchmark/2",
        "windfusion_version": __version__,
        "synthetic": True,
        "stamp": SYNTHETIC_STAMP,
        "fleet": report.stage["fleet"],
        "footprint": report.stage["footprint"],
        "supervised": report.stage["supervised"],
        "self_learning": report.stage["self_learning"],
        "distillation": report.stage["distillation"],
        "fleet_learning": report.stage["fleet_learning"],
        "telemetry": report.stage["telemetry"],
        "config_snapshot": {
            "seed": args.seed,
            "epochs": args.epochs,
            "training": config.training.to_dict(),
            "self_training": config.self_training.to_dict(),
            "data": config.data.to_dict(),
            "evaluation": config.evaluation.to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_seconds": round(time.perf_counter() - started, 2),
            "rul_scale_days": RUL_SCALE_DAYS,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = "quick" if args.quick else f"synthetic-v{__version__}"
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = out_dir / f"{stamp}.md"
    md_path.write_text(as_markdown(payload), encoding="utf-8")
    print(f"[benchmark] wrote {json_path} and {md_path} in {payload['environment']['total_seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

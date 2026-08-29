import argparse
import json

from .models import MODEL_REGISTRY, create_model


def main():
    p = argparse.ArgumentParser(prog="windfusion")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "distill", "simulate", "explain"):
        sub.add_parser(name)
    b = sub.add_parser("benchmark")
    b.add_argument("--mode", default="windfusion-lite", choices=MODEL_REGISTRY)
    e = sub.add_parser("export")
    e.add_argument("path")
    e.add_argument("--mode", default="windfusion-edge", choices=MODEL_REGISTRY)
    args = p.parse_args()
    if args.command == "benchmark":
        from benchmarks.run import run

        print(json.dumps(run(args.mode), indent=2))
    elif args.command == "export":
        from .deployment.export import export_onnx

        print(export_onnx(create_model(args.mode), args.path))
    else:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "pipeline scaffold ready",
                    "note": "provide a versioned dataset/config; synthetic results are never represented as field validation",
                }
            )
        )


if __name__ == "__main__":
    main()

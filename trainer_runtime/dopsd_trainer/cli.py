from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import validate_dataset
from .models import MODEL_REGISTRY, download_model, inspect_model_cache
from .profiles import list_production_profiles
from .probes import probe_environment
from .recipes import TrainingConfig, build_accelerate_command
from .settings import build_settings_summary


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="dopsd-trainer", description="D-OPSD Trainer runtime utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Probe local runtime environment")
    probe_parser.add_argument("--project-root", default=".")

    dataset_parser = subparsers.add_parser("validate-dataset", help="Validate a D-OPSD JSONL dataset")
    dataset_parser.add_argument("jsonl")
    dataset_parser.add_argument("--recipe-id", default="z-image-turbo-vlm")
    dataset_parser.add_argument("--project-root", default=".")

    command_parser = subparsers.add_parser("build-command", help="Build an accelerate command")
    command_parser.add_argument("--recipe-id", required=True)
    command_parser.add_argument("--exp-name", required=True)
    command_parser.add_argument("--project-root", default=".")
    command_parser.add_argument("--train-jsonl")
    command_parser.add_argument("--model")
    command_parser.add_argument("--max-train-steps", type=int)
    command_parser.add_argument("--launcher", choices=["accelerate", "python"], default="accelerate")
    command_parser.add_argument("--no-deepspeed", action="store_true")
    command_parser.add_argument("--low-vram", action="store_true")
    command_parser.add_argument("--block-offload", action="store_true")
    command_parser.add_argument("--block-offload-num-blocks", type=int, default=2)
    command_parser.add_argument("--resolution-scale")
    command_parser.add_argument("--no-save-samples", action="store_true")
    command_parser.add_argument("--no-save-checkpoints", action="store_true")
    command_parser.add_argument("--use-8bit-adam", action="store_true")
    command_parser.add_argument("--model-cache-root")
    command_parser.add_argument("--use-model-ids", action="store_true", help="Do not resolve cached local model snapshots")

    models_parser = subparsers.add_parser("models", help="Inspect registered model cache state")
    models_parser.add_argument("--cache-root")

    settings_parser = subparsers.add_parser("settings", help="Summarize trainer runtime settings")
    settings_parser.add_argument("--project-root", default=".")

    profiles_parser = subparsers.add_parser("profiles", help="List verified production starter profiles")
    profiles_parser.add_argument("--project-root", default=".")

    download_parser = subparsers.add_parser("download-model", help="Download a registered Hugging Face model")
    download_parser.add_argument("model_id", choices=sorted(MODEL_REGISTRY))
    download_parser.add_argument("--cache-root")

    args = parser.parse_args()

    if args.command == "probe":
        _print_json(probe_environment(args.project_root))
        return

    if args.command == "validate-dataset":
        _print_json(validate_dataset(args.jsonl, args.recipe_id, args.project_root).to_dict())
        return

    if args.command == "build-command":
        config = TrainingConfig(
            recipe_id=args.recipe_id,
            exp_name=args.exp_name,
            train_jsonl=args.train_jsonl,
            model=args.model,
            max_train_steps=args.max_train_steps,
            launcher=args.launcher,
            use_deepspeed=not args.no_deepspeed,
            low_vram=args.low_vram,
            block_offload=args.block_offload,
            block_offload_num_blocks=args.block_offload_num_blocks,
            resolution_scale=args.resolution_scale,
            save_samples=not args.no_save_samples,
            save_checkpoints=not args.no_save_checkpoints,
            use_8bit_adam=args.use_8bit_adam,
            model_cache_root=args.model_cache_root,
            prefer_local_models=not args.use_model_ids,
        )
        _print_json(build_accelerate_command(config, Path(args.project_root)))
        return

    if args.command == "models":
        _print_json([inspect_model_cache(model_id, args.cache_root) for model_id in MODEL_REGISTRY])
        return

    if args.command == "settings":
        _print_json(build_settings_summary(args.project_root))
        return

    if args.command == "profiles":
        _print_json(list_production_profiles())
        return

    if args.command == "download-model":
        _print_json(download_model(args.model_id, args.cache_root))
        return


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer_runtime"))

from dopsd_trainer.models import MODEL_REGISTRY, download_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download D-OPSD Trainer models")
    parser.add_argument("--all-default", action="store_true", help="Download all default models")
    parser.add_argument("--model-id", choices=sorted(MODEL_REGISTRY), help="Download one model")
    parser.add_argument("--cache-root")
    args = parser.parse_args()

    if not args.all_default and not args.model_id:
        parser.error("provide --all-default or --model-id")

    model_ids = [args.model_id] if args.model_id else [
        model_id for model_id, spec in MODEL_REGISTRY.items() if spec.default
    ]
    results = [download_model(model_id, args.cache_root) for model_id in model_ids]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import model_path_for_command


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    label: str
    work_dir: str
    train_script: str
    default_model: str
    default_dataset: str
    default_steps: int
    default_sample_steps: int
    default_checkpoint_steps: int
    default_learning_rate: str
    default_num_processes: int
    edit_sys_prompt: str | None = None


RECIPE_REGISTRY: dict[str, RecipeSpec] = {
    "z-image-turbo-vlm": RecipeSpec(
        recipe_id="z-image-turbo-vlm",
        label="Z-Image Turbo Style LoRA",
        work_dir="z-image-turbo_self-distill-vlm",
        train_script="train_dopsd.py",
        default_model="Tongyi-MAI/Z-Image-Turbo",
        default_dataset="dataset/style_Millennium/data.jsonl",
        default_steps=2000,
        default_sample_steps=100,
        default_checkpoint_steps=500,
        default_learning_rate="1e-4",
        default_num_processes=1,
    ),
    "flux2-klein-identity": RecipeSpec(
        recipe_id="flux2-klein-identity",
        label="FLUX2 Klein Identity LoRA",
        work_dir="flux2-klein_self-distill-edit",
        train_script="train_dopsd.py",
        default_model="black-forest-labs/FLUX.2-klein-4B",
        default_dataset="dataset/corgi/data.jsonl",
        default_steps=3001,
        default_sample_steps=100,
        default_checkpoint_steps=500,
        default_learning_rate="2e-5",
        default_num_processes=1,
        edit_sys_prompt="Make the main subject (facial/ip details), background, and overall composition consistent with the reference image.",
    ),
    "flux2-klein-editing": RecipeSpec(
        recipe_id="flux2-klein-editing",
        label="FLUX2 Klein Editing LoRA",
        work_dir="flux2-klein-edit-self-distill-gt-ref",
        train_script="train_dopsd.py",
        default_model="black-forest-labs/FLUX.2-klein-4B",
        default_dataset="dataset/interaction/data.jsonl",
        default_steps=3001,
        default_sample_steps=100,
        default_checkpoint_steps=500,
        default_learning_rate="2e-5",
        default_num_processes=1,
        edit_sys_prompt=(
            "Note that the last image is the output image I expect to be. Your output must be as similar as the "
            "last reference image, based on the first reference image and the editing instructions."
        ),
    ),
}


@dataclass
class TrainingConfig:
    recipe_id: str
    exp_name: str
    output_dir: str = "exp_results"
    model: str | None = None
    train_jsonl: str | None = None
    test_jsonl: str | None = None
    max_train_steps: int | None = None
    epochs: int | None = None
    sample_steps: int | None = None
    checkpoint_steps: int | None = None
    num_processes: int | None = None
    main_process_port: int = 60212
    launcher: str = "accelerate"
    use_deepspeed: bool = True
    mixed_precision: str = "bf16"
    batch_size: int = 1
    batch_size_test: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: str | None = None
    lora_rank: int = 64
    lora_alpha: int = 128
    num_training_steps: int = 4
    seed: int = 30
    ema_decay: str = "0.9999"
    enable_gc: bool = True
    vae_dtype: str = "bf16"
    adam_weight_decay: str = "0.0"
    edit_sys_prompt: str | None = None
    low_vram: bool = False
    block_offload: bool = False
    block_offload_num_blocks: int = 1
    layer_offload: bool = False
    layer_offload_transformer_percent: float = 1.0
    layer_offload_text_encoder_percent: float = 1.0
    resolution_scale: str | None = None
    save_samples: bool = True
    save_checkpoints: bool = True
    use_8bit_adam: bool = False
    prefer_local_models: bool = True
    model_cache_root: str | None = None
    qwen_vl_model: str | None = None


def recipe_as_dict(recipe_id: str) -> dict[str, Any]:
    return asdict(RECIPE_REGISTRY[recipe_id])


def _percent_arg(value: float) -> str:
    return str(max(0.0, min(1.0, float(value))))


def build_accelerate_command(config: TrainingConfig, project_root: str | Path | None = None) -> dict[str, Any]:
    if config.recipe_id not in RECIPE_REGISTRY:
        raise KeyError(f"unknown recipe: {config.recipe_id}")

    root = Path(project_root or Path.cwd()).resolve()
    spec = RECIPE_REGISTRY[config.recipe_id]
    work_dir = root / spec.work_dir

    train_jsonl = config.train_jsonl or spec.default_dataset
    test_jsonl = config.test_jsonl or train_jsonl
    model = config.model or spec.default_model
    if config.prefer_local_models and config.model is None:
        model = model_path_for_command(model, config.model_cache_root)
    max_train_steps = config.max_train_steps or spec.default_steps
    epochs = config.epochs or max_train_steps + 1
    sample_steps = config.sample_steps or spec.default_sample_steps
    checkpoint_steps = config.checkpoint_steps or spec.default_checkpoint_steps
    num_processes = config.num_processes or spec.default_num_processes
    learning_rate = config.learning_rate or spec.default_learning_rate
    edit_sys_prompt = config.edit_sys_prompt if config.edit_sys_prompt is not None else spec.edit_sys_prompt
    env: dict[str, str] = {}

    if config.recipe_id == "z-image-turbo-vlm":
        qwen_model = config.qwen_vl_model
        if qwen_model is None and config.prefer_local_models:
            qwen_model = model_path_for_command("Qwen/Qwen3-VL-4B-Instruct", config.model_cache_root)
        if qwen_model:
            env["DOPSD_QWEN_VL_MODEL"] = qwen_model

    if config.launcher == "python":
        args = ["python", spec.train_script]
    elif config.launcher == "accelerate":
        args = [
            "accelerate",
            "launch",
            "--config_file",
            "configs/default.yaml",
            "--main_process_port",
            str(config.main_process_port),
            "--num_processes",
            str(num_processes),
            spec.train_script,
        ]
    else:
        raise ValueError(f"unknown launcher: {config.launcher}")

    if config.use_deepspeed:
        args.extend([
            "--deepspeed-config",
            "configs/z2.json",
        ])

    args.extend([
        "--output-dir",
        config.output_dir,
        "--exp-name",
        config.exp_name,
        "--sample-steps",
        str(sample_steps),
        "--checkpoint-steps",
        str(checkpoint_steps),
        "--epochs",
        str(epochs),
        "--max-train-steps",
        str(max_train_steps),
        "--pretrained_model",
        model,
        "--num-training-steps",
        str(config.num_training_steps),
        "--use-lora",
        "2",
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--data-path-train-jsonl",
        train_jsonl,
        "--data-path-test-jsonl",
        test_jsonl,
        "--seed",
        str(config.seed),
        "--mixed-precision",
        config.mixed_precision,
        "--batch-size",
        str(config.batch_size),
        "--batch-size-test",
        str(config.batch_size_test),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--learning-rate-gen",
        str(learning_rate),
        "--adam-weight-decay",
        str(config.adam_weight_decay),
        "--vae-dtype",
        config.vae_dtype,
        "--ema-decay",
        str(config.ema_decay),
    ])

    if config.enable_gc:
        args.append("--enable-gc")
    if config.low_vram:
        args.append("--low-vram")
    if config.block_offload:
        args.extend(["--block-offload", "--block-offload-num-blocks", str(max(1, int(config.block_offload_num_blocks)))])
    if config.layer_offload:
        args.extend([
            "--layer-offload",
            "--layer-offload-transformer-percent",
            _percent_arg(config.layer_offload_transformer_percent),
            "--layer-offload-text-encoder-percent",
            _percent_arg(config.layer_offload_text_encoder_percent),
        ])
    if config.resolution_scale:
        args.extend(["--resolution-scale", config.resolution_scale])
    if not config.save_samples:
        args.append("--no-save-samples")
    if not config.save_checkpoints:
        args.append("--no-save-checkpoints")
    if config.use_8bit_adam:
        args.append("--use-8bit-adam")
    if edit_sys_prompt:
        args.extend(["--edit-sys-prompt", edit_sys_prompt])

    return {
        "recipe": recipe_as_dict(config.recipe_id),
        "cwd": str(work_dir),
        "env": env,
        "args": args,
        "display": " ".join(f'"{arg}"' if " " in arg else arg for arg in args),
    }

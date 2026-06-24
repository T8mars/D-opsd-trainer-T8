import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from peft import PeftModel
from PIL import Image
from torchvision.utils import make_grid

from utils import create_generator


def array2grid(x):
    n_images = x.size(0)
    height = x.size(2)
    width = x.size(3)
    aspect_ratio = width / height
    nrow = max(1, round(math.sqrt(n_images / aspect_ratio)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def dtype_from_name(name):
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def free_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def append_log(request, message):
    log_path = request.get("log_path")
    if not log_path:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = Path(log_path).expanduser().resolve()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[\033[34m{timestamp}\033[0m] [final-sampler] {message}\n")


def enable_transformer_block_offload(transformer, device, num_blocks_per_group):
    if getattr(transformer, "_dopsd_block_offload_enabled", False):
        return
    from diffusers.hooks import apply_group_offloading

    blocks_per_group = max(1, int(num_blocks_per_group))
    apply_group_offloading(
        transformer,
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=blocks_per_group,
        non_blocking=torch.cuda.is_available(),
        use_stream=False,
    )
    transformer._dopsd_block_offload_enabled = True
    print(f"Transformer block offload enabled: {blocks_per_group} block(s) per group")


def configure_tiled_vae(vae, request):
    try:
        vae.enable_slicing()
    except NotImplementedError:
        append_log(request, f"VAE slicing unavailable for {vae.__class__.__name__}")

    if not request.get("tiled_vae", True):
        append_log(request, "Tiled VAE disabled by request")
        return

    tile_size = max(16, int(request.get("vae_tile_size", 64)))
    overlap = min(0.75, max(0.0, float(request.get("vae_tile_overlap", 0.25))))
    block_channels = getattr(getattr(vae, "config", None), "block_out_channels", (1, 1, 1, 1))
    latent_scale = 2 ** max(0, len(block_channels) - 1)
    latent_tile_size = max(1, tile_size // latent_scale)

    if hasattr(vae, "tile_sample_min_size"):
        vae.tile_sample_min_size = tile_size
    if hasattr(vae, "tile_latent_min_size"):
        vae.tile_latent_min_size = latent_tile_size
    if hasattr(vae, "tile_overlap_factor"):
        vae.tile_overlap_factor = overlap
    vae.enable_tiling()

    message = (
        "Tiled VAE enabled: "
        f"tile_sample_min_size={tile_size}, "
        f"tile_latent_min_size={latent_tile_size}, "
        f"tile_overlap_factor={overlap}"
    )
    print(message)
    append_log(request, message)


def configure_pipeline_cpu_offload(pipeline, request, device, inference_dtype, vae_dtype):
    if request.get("final_sampler_cpu_offload", True) and torch.cuda.is_available():
        pipeline.vae.to(dtype=vae_dtype)
        pipeline.text_encoder.to(dtype=inference_dtype)
        pipeline.enable_model_cpu_offload(device=device)
        message = "Final sampler model CPU offload enabled"
        print(message)
        append_log(request, message)
        return True

    pipeline.to(device)
    pipeline.vae.to(device, dtype=vae_dtype)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    return False


def sample_flux2_with_optional_latent_decode(
    pipeline,
    vae_dtype,
    decode_latents_after_transformer_hooks=False,
    **pipeline_kwargs,
):
    if not decode_latents_after_transformer_hooks:
        return pipeline(output_type="pt", **pipeline_kwargs)[0]

    latents = pipeline(output_type="latent", **pipeline_kwargs)[0]
    free_cuda_memory()
    pipeline.vae.to(latents.device, dtype=vae_dtype)
    images = pipeline.vae.decode(latents.to(dtype=vae_dtype), return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type="pt")
    del latents
    free_cuda_memory()
    return images


def load_request(path):
    request_path = Path(path).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["_request_path"] = str(request_path)
    return request


def load_pipeline(request, device, inference_dtype, vae_dtype):
    pipeline = Flux2KleinPipeline.from_pretrained(
        request["model"],
        torch_dtype=inference_dtype,
        low_cpu_mem_usage=False,
    )
    adapter_dir = Path(request["adapter_dir"]).expanduser().resolve()
    student_dir = adapter_dir / "student"
    teacher_dir = adapter_dir / "teacher"
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        str(student_dir),
        adapter_name="student",
        is_trainable=False,
    )
    pipeline.transformer.load_adapter(str(teacher_dir), adapter_name="teacher", is_trainable=False)
    configure_tiled_vae(pipeline.vae, request)
    cpu_offload_enabled = configure_pipeline_cpu_offload(
        pipeline,
        request,
        device,
        inference_dtype,
        vae_dtype,
    )
    if request.get("block_offload") and not cpu_offload_enabled:
        enable_transformer_block_offload(
            pipeline.transformer,
            device,
            request.get("block_offload_num_blocks", 2),
        )
    return pipeline


@torch.inference_mode()
def run_final_samples(request):
    append_log(request, "Starting deferred final sample generation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dtype = dtype_from_name(request.get("inference_dtype", "bf16"))
    vae_dtype = dtype_from_name(request.get("vae_dtype", "bf16"))
    pipeline = load_pipeline(request, device, inference_dtype, vae_dtype)

    prompts = list(request["prompts"])
    image_paths = [Path(path).expanduser().resolve() for path in request["image_paths"]]
    height = int(request["height"])
    width = int(request["width"])
    display_height = max(64, int(request.get("source_height", height)) // 2)
    display_width = max(64, int(request.get("source_width", width)) // 2)
    global_step = int(request["global_step"])
    sample_dir = Path(request["sample_dir"]).expanduser().resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    generator_test = create_generator(prompts, int(request.get("seed", 2026)))
    decode_after_hooks = bool(request.get("block_offload"))

    pipeline.transformer.set_adapter("teacher")
    teacher_test_images = []
    teacher_edit_prompts = [f"{prompt} {request.get('edit_sys_prompt', '')}".strip() for prompt in prompts]
    for image_path, teacher_prompt, test_generator in zip(image_paths, teacher_edit_prompts, generator_test):
        with Image.open(image_path) as image:
            test_image_gt = image.convert("RGB")
        image_t = sample_flux2_with_optional_latent_decode(
            pipeline,
            vae_dtype,
            decode_latents_after_transformer_hooks=decode_after_hooks,
            image=test_image_gt,
            prompt=teacher_prompt,
            height=height,
            width=width,
            num_inference_steps=int(request["num_inference_steps"]),
            guidance_scale=float(request["guidance_scale"]),
            generator=test_generator,
        )
        teacher_test_images.append(image_t)
    images_t = torch.cat(teacher_test_images, dim=0)
    images_t = F.interpolate(images_t, size=(display_height, display_width), mode="bicubic", align_corners=False)
    if decode_after_hooks:
        images_t = images_t.to("cpu")
        free_cuda_memory()

    pipeline.transformer.set_adapter("student")
    images_s = sample_flux2_with_optional_latent_decode(
        pipeline,
        vae_dtype,
        decode_latents_after_transformer_hooks=decode_after_hooks,
        prompt=prompts,
        height=height,
        width=width,
        num_inference_steps=int(request["num_inference_steps"]),
        guidance_scale=float(request["guidance_scale"]),
        generator=generator_test,
    )
    images_s = F.interpolate(images_s, size=(display_height, display_width), mode="bicubic", align_corners=False)

    student_path = sample_dir / f"samples_step_{global_step}_student.png"
    teacher_path = sample_dir / f"samples_step_{global_step}_teacher.png"
    Image.fromarray(array2grid(images_s.to(torch.float32))).save(student_path)
    Image.fromarray(array2grid(images_t.to(torch.float32))).save(teacher_path)
    print(f"Saved final student sample to {student_path}")
    print(f"Saved final teacher sample to {teacher_path}")
    append_log(request, f"Saved final student sample to {student_path}")
    append_log(request, f"Saved final teacher sample to {teacher_path}")
    append_log(request, "Training completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run deferred FLUX2 final sample generation.")
    parser.add_argument("--request", required=True, help="Path to final sampler request JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    request = load_request(args.request)
    run_final_samples(request)


if __name__ == "__main__":
    main()

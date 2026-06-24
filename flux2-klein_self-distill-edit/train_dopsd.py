import os
import time
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.deepspeed import get_active_deepspeed_plugin
from accelerate.logging import get_logger
from diffusers import Flux2KleinPipeline
from diffusers.utils.torch_utils import is_compiled_module
import tqdm
import logging
from pathlib import Path
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import math
from torchvision.utils import make_grid
from dataset import TextImageDataset, AspectBatchSampler, CustomDataLoader, parse_ratios
from dataset_validate import TextPromptDataset
from dopsd_layer_offload import attach_layer_offload
from local_paths import resolve_existing_path
from PIL import Image
from arguments import parse_args
from utils import create_generator
from ema_utils import *

logger = get_logger(__name__)


def array2grid(x):
    n_images = x.size(0)
    height = x.size(2)
    width = x.size(3)
    aspect_ratio = width / height
    nrow = max(1, round(math.sqrt(n_images / aspect_ratio)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return grid


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def create_tensorboard_writer(save_dir, args, logger):
    if not getattr(args, "tensorboard", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as error:
        logger.warning(f"TensorBoard disabled: {error}")
        return None
    tensorboard_dir = getattr(args, "tensorboard_dir", "tensorboard")
    if not os.path.isabs(tensorboard_dir):
        tensorboard_dir = os.path.join(save_dir, tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    return SummaryWriter(tensorboard_dir)


def write_tensorboard_scalars(tb_writer, logs, global_step):
    if tb_writer is None:
        return
    tb_writer.add_scalar("loss/total", float(logs.get("loss_total", 0.0)), global_step)
    tb_writer.add_scalar("loss/dopsd", float(logs.get("loss_dopsd", 0.0)), global_step)
    tb_writer.add_scalar("train/grad_norm", float(logs.get("grad_n", 0.0)), global_step)
    tb_writer.add_scalar("train/epoch", float(logs.get("epoch", 0.0)), global_step)
    tb_writer.flush()


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def scale_resolutions(resolutions, scale, target_resolution=1024):
    if scale <= 0:
        raise ValueError("--resolution-scale must be greater than 0")
    if target_resolution <= 0:
        raise ValueError("--target-resolution must be greater than 0")

    effective_scale = scale * (target_resolution / 1024.0)
    if effective_scale == 1.0:
        return resolutions

    scaled = []
    for width, height in resolutions:
        scaled_width = max(64, int(round((width * effective_scale) / 16)) * 16)
        scaled_height = max(64, int(round((height * effective_scale) / 16)) * 16)
        scaled.append((scaled_width, scaled_height))
    return scaled

def free_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def configure_tiled_vae(vae, args, logger):
    try:
        vae.enable_slicing()
    except NotImplementedError:
        logger.info(f"VAE slicing unavailable for {vae.__class__.__name__}")

    if not args.tiled_vae:
        logger.info("Tiled VAE disabled by request")
        return

    tile_size = max(16, int(args.vae_tile_size))
    overlap = min(0.75, max(0.0, float(args.vae_tile_overlap)))
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

    logger.info(
        "Tiled VAE enabled: "
        f"tile_sample_min_size={tile_size}, "
        f"tile_latent_min_size={latent_tile_size}, "
        f"tile_overlap_factor={overlap}"
    )

def move_frozen_conditioners_to_device(pipeline, device, text_dtype, vae_dtype):
    pipeline.text_encoder.to(device, dtype=text_dtype)
    pipeline.vae.to(device, dtype=vae_dtype)

def prepare_pipeline_for_sampling(pipeline, device, text_dtype, vae_dtype):
    pipeline.to(device)
    move_frozen_conditioners_to_device(pipeline, device, text_dtype, vae_dtype)

def offload_frozen_conditioners(pipeline):
    pipeline.text_encoder.to("cpu")
    pipeline.vae.to("cpu")
    free_cuda_memory()

def enable_transformer_block_offload(transformer, device, num_blocks_per_group, logger):
    if getattr(transformer, "_dopsd_block_offload_enabled", False):
        return
    try:
        from diffusers.hooks import apply_group_offloading
    except Exception as exc:
        raise RuntimeError(
            "--block-offload requires a Diffusers build with apply_group_offloading support."
        ) from exc

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
    logger.info(f"Transformer block offload enabled: {blocks_per_group} block(s) per group")

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

def should_defer_final_sampling(args, global_step):
    if global_step != args.max_train_steps:
        return False
    if args.final_sample_mode == "subprocess":
        return True
    if args.final_sample_mode == "inline":
        return False
    return bool(args.block_offload)

def _resolved_run_dir(args):
    run_dir = Path(args.output_dir).expanduser() / args.exp_name
    return run_dir.resolve()

def scale_final_sample_size(height, width, scale):
    if scale <= 0:
        raise ValueError("--final-sample-resolution-scale must be greater than 0")
    scaled_h = max(64, int(round((height * scale) / 16)) * 16)
    scaled_w = max(64, int(round((width * scale) / 16)) * 16)
    return scaled_h, scaled_w

def scale_sample_size(height, width, scale):
    if scale <= 0:
        raise ValueError("--sample-resolution-scale must be greater than 0")
    scaled_h = max(64, int(round((height * scale) / 16)) * 16)
    scaled_w = max(64, int(round((width * scale) / 16)) * 16)
    return scaled_h, scaled_w

def write_final_sampler_request(
    args,
    global_step,
    gen_model,
    accelerator,
    test_prompts,
    gt_image_paths,
    test_h,
    test_w,
    logger,
):
    if args.use_lora <= 1:
        raise RuntimeError("--final-sample-mode subprocess requires LoRA adapters.")

    run_dir = _resolved_run_dir(args)
    final_sampler_dir = run_dir / "final_sampler"
    adapter_dir = final_sampler_dir / f"lora_gen_step_{global_step}"
    sample_dir = run_dir / "samples"
    final_sampler_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    unwrap_model(gen_model, accelerator).save_pretrained(str(adapter_dir))
    final_h, final_w = scale_final_sample_size(
        int(test_h),
        int(test_w),
        args.final_sample_resolution_scale,
    )

    request = {
        "model": args.pretrained_model,
        "adapter_dir": str(adapter_dir),
        "sample_dir": str(sample_dir),
        "log_path": str(run_dir / "log.txt"),
        "global_step": global_step,
        "prompts": list(test_prompts),
        "image_paths": [str(path) for path in gt_image_paths],
        "height": final_h,
        "width": final_w,
        "source_height": int(test_h),
        "source_width": int(test_w),
        "resolution_scale": float(args.final_sample_resolution_scale),
        "num_inference_steps": int(args.num_training_steps),
        "guidance_scale": 0.0 if args.num_training_steps < 10 else 4.0,
        "seed": 2026,
        "edit_sys_prompt": args.edit_sys_prompt,
        "vae_dtype": args.vae_dtype,
        "inference_dtype": "bf16" if args.mixed_precision == "bf16" else "fp16",
        "final_sampler_cpu_offload": bool(args.final_sampler_cpu_offload) and not bool(args.block_offload),
        "tiled_vae": bool(args.tiled_vae),
        "vae_tile_size": int(args.vae_tile_size),
        "vae_tile_overlap": float(args.vae_tile_overlap),
        "requested_block_offload": bool(args.block_offload),
        "block_offload": bool(args.block_offload),
        "block_offload_num_blocks": int(args.block_offload_num_blocks),
        "sampler_script": str(Path(__file__).resolve().with_name("sample_flux2_final.py")),
    }
    request_path = final_sampler_dir / f"request_step_{global_step}.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    logger.info(f"Deferred final samples to subprocess request {request_path}")
    return {"request_path": str(request_path)}

def compute_empirical_mu(image_seq_len, num_steps):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _prepare_text_ids(x, t_coord=None):
    batch_size, seq_len, _ = x.shape
    out_ids = []

    for i in range(batch_size):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(seq_len)
        out_ids.append(torch.cartesian_prod(t, h, w, l))

    return torch.stack(out_ids)


def _encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    max_sequence_length=512,
    hidden_states_layers=(9, 18, 27),
    dtype=None,
):
    dtype = text_encoder.dtype if dtype is None else dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids = []
    all_attention_masks = []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
    )

    hidden_states = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    hidden_states = hidden_states.to(dtype=dtype, device=device)
    del output, input_ids, attention_mask

    batch_size, num_layers, seq_len, hidden_dim = hidden_states.shape
    prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_layers * hidden_dim
    )

    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // 4, height * 2, width * 2)
    return latents


def _unpack_latents_with_ids(x, x_ids):
    x_list = []
    for data, pos in zip(x, x_ids):
        _, channels = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        height = int(torch.max(h_ids).item()) + 1
        width = int(torch.max(w_ids).item()) + 1
        flat_ids = h_ids * width + w_ids

        out = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        x_list.append(out.view(height, width, channels).permute(2, 0, 1))

    return torch.stack(x_list, dim=0)


@torch.no_grad()
def prepare_batch_image_latents(
    pipeline,
    images,
    device,
    vae_dtype,
    latent_dtype,
    generator=None,
):
    reference_images = images.to(device=device, dtype=vae_dtype)
    encoded_image_latents = pipeline._encode_vae_image(image=reference_images, generator=generator)
    image_latent_ids = pipeline._prepare_image_ids([encoded_image_latents[:1]])
    image_latent_ids = image_latent_ids.repeat(encoded_image_latents.shape[0], 1, 1).to(device)
    image_latents = pipeline._pack_latents(encoded_image_latents).to(device=device, dtype=latent_dtype)
    return image_latents, image_latent_ids


@torch.no_grad()
def decode_flux_packed_x0_to_images(
    x_0,
    img_ids,
    pipeline,
    vae_dtype,
    latents_bn_mean,
    latents_bn_std,
):
    x_0_pix = _unpack_latents_with_ids(x_0, img_ids).to(device=x_0.device, dtype=vae_dtype)
    x_0_pix = x_0_pix * latents_bn_std + latents_bn_mean
    x_0_pix = _unpatchify_latents(x_0_pix)
    x_0_pix = pipeline.vae.decode(x_0_pix, return_dict=False)[0]
    return (x_0_pix / 2 + 0.5).clamp(0, 1)


def save_student_teacher_trajectory(
    pipeline,
    student_x0_traj,
    teacher_x0_traj,
    latent_ids,
    save_dir,
    global_step,
    accelerator,
    vae_dtype,
    latents_bn_mean,
    latents_bn_std,
    max_size=None,
):
    import os, math, numpy as np
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(save_dir, exist_ok=True)

    def to_uint8(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1))
        return x if x.dtype == np.uint8 else (np.clip(x, 0, 1) * 255).round().astype(np.uint8)

    def grid(x, nrow=4, pad=2, bg=255):
        assert x.ndim == 4, f"grid expects 4D input, got {x.shape}"
        n, h, w, c = x.shape
        nrow = max(1, min(nrow, n))
        ncol = math.ceil(n / nrow)
        g = np.full((ncol * h + pad * (ncol - 1), nrow * w + pad * (nrow - 1), c), bg, np.uint8)
        for k in range(n):
            r, col = divmod(k, nrow)
            y, z = r * (h + pad), col * (w + pad)
            g[y:y + h, z:z + w] = x[k]
        return g

    def add_titles_and_concat(a, b, pad=16, title_h=36, bg=255):
        h, w1, c = a.shape
        _, w2, _ = b.shape
        canvas = np.full((title_h + max(a.shape[0], b.shape[0]), w1 + pad + w2, c), bg, np.uint8)
        canvas[title_h:title_h + a.shape[0], :w1] = a
        canvas[title_h:title_h + b.shape[0], w1 + pad:w1 + pad + w2] = b
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), "Student", fill=(0, 0, 0), font=font)
        draw.text((w1 + pad + 10, 10), "Teacher", fill=(0, 0, 0), font=font)
        return img

    for i, (sx0, tx0) in enumerate(zip(student_x0_traj, teacher_x0_traj)):
        s = decode_flux_packed_x0_to_images(
            sx0[:4],
            latent_ids[:4],
            pipeline,
            vae_dtype,
            latents_bn_mean,
            latents_bn_std,
        ).float()
        t = decode_flux_packed_x0_to_images(
            tx0[:4],
            latent_ids[:4],
            pipeline,
            vae_dtype,
            latents_bn_mean,
            latents_bn_std,
        ).float()

        if accelerator.is_main_process:

            t_dir = os.path.join(save_dir, f"t{i}")
            os.makedirs(t_dir, exist_ok=True)

            s_np = to_uint8(s)
            t_np = to_uint8(t)

            single_img_dir = f"{t_dir}/one_img"
            os.makedirs(single_img_dir, exist_ok=True)
            Image.fromarray(s_np[0]).save(os.path.join(single_img_dir, f"step_{global_step}_student_single.png"))
            Image.fromarray(t_np[0]).save(os.path.join(single_img_dir, f"step_{global_step}_teacher_single.png"))

            nrow = 4 if s_np.shape[1] <= 1024 else 2
            img = add_titles_and_concat(grid(s_np, nrow=nrow), grid(t_np, nrow=nrow))

            if max_size is not None:
                w, h = img.size
                scale = min(max_size[0] / w, max_size[1] / h, 1.0)
                if scale < 1:
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

            img.save(os.path.join(t_dir, f"step_{global_step}_student_teacher_x0.png"))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ds_config = args.deepspeed_config

    zero2_plugin_a = None
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "project_config": accelerator_project_config,
    }
    if ds_config:
        zero2_plugin_a = DeepSpeedPlugin(hf_ds_config=ds_config)
        accelerator_kwargs["deepspeed_plugins"] = {"z2_a": zero2_plugin_a}

    accelerator = Accelerator(**accelerator_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger(__name__)
    tb_writer = None

    if accelerator.is_main_process:
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)

        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        tb_writer = create_tensorboard_writer(save_dir, args, logger)

    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Create pipe :
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=inference_dtype,
        low_cpu_mem_usage=False,
    )

    num_channels_latents = pipeline.transformer.config.in_channels // 4

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(args.use_lora <= 1)
    tokenizer = pipeline.tokenizer


    # disable progress bar for cold start
    pipeline.set_progress_bar_config(disable=True)

    if args.layer_offload and args.layer_offload_transformer_percent > 0:
        attach_layer_offload(
            pipeline.transformer,
            accelerator.device,
            args.layer_offload_transformer_percent,
            logger,
            label="transformer",
        )

    # init lora
    if args.use_lora > 1:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        pipeline.transformer = init_dual_lora_transformer(
            transformer=pipeline.transformer,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            current_adapter_name="student",
            old_adapter_name="teacher",
            old_init_from_current=True,
        )

    # we use ema in full-finetune
    else:
        raise NotImplementedError("Full finetuning is not implemented here, please set --use-lora to > 1 for now.")

    # Move vae and text_encoder to device and cast to inference_dtype
    if args.vae_dtype == "fp32":
        vae_dtype = torch.float32
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    else:
        vae_dtype = inference_dtype
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    # avoid OOM in both inline sample generation and training VAE paths
    configure_tiled_vae(pipeline.vae, args, logger)
    if args.layer_offload and args.layer_offload_text_encoder_percent > 0:
        attach_layer_offload(
            pipeline.text_encoder,
            accelerator.device,
            args.layer_offload_text_encoder_percent,
            logger,
            label="text encoder",
        )
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    gen_model = pipeline.transformer
    gen_model_trainable_parameters = list(filter(lambda p: p.requires_grad, gen_model.parameters()))

    # enable gradient checkpointing
    if args.enable_gc:
        gen_model.enable_gradient_checkpointing()


    # Setup optimizer and learning rate scheduler:
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer_gen = optimizer_cls(
        gen_model_trainable_parameters,
        lr=args.learning_rate_gen,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup dataset:
    all_ratios = [
        '1024x1024 ( 1:1 index_0 )',
        '1152x896 ( 9:7 index_1 )',
        '896x1152 ( 7:9 index_2 )',
        '1152x864 ( 4:3 index_3 )',
        '864x1152 ( 3:4 index_4 )',
        '1248x832 ( 3:2 index_5 )',
        '832x1248 ( 2:3 index_6 )',
        '1280x720 ( 16:9 index_7 )',
        '720x1280 ( 9:16 index_8 )',
        '1344x576 ( 21:9 index_9 )',
        '576x1344 ( 9:21 index_10 )'
    ]

    prompt_keys = ['short_en', 'detailed_en', 'short_zh', 'detailed_zh', 'medium_zh', 'medium_en', "user_prompt_en",
                   "user_prompt_zh"]
    test_prompt_keys = ['short_en', 'short_zh', 'medium_zh', 'medium_en', "user_prompt_en", "user_prompt_zh"]
    select_ratio_index = [j for j in range(len(all_ratios))]
    select_ratio = [all_ratios[i] for i in select_ratio_index]
    target_resolutions = scale_resolutions(parse_ratios(select_ratio), args.resolution_scale, args.target_resolution)

    dataset_root = Path(__file__).resolve().parent
    train_jsonl_path = resolve_existing_path(args.data_path_train_jsonl, dataset_root)
    test_jsonl_path = resolve_existing_path(args.data_path_test_jsonl, dataset_root)

    if '1024x1024 ( 1:1 index_0 )' in select_ratio:
        test_w, test_h = target_resolutions[select_ratio.index('1024x1024 ( 1:1 index_0 )')]
    else:
        test_w, test_h = target_resolutions[0]

    train_dataset = TextImageDataset(
        str(train_jsonl_path),
        target_resolutions=target_resolutions,
        data_root=dataset_root,
    )

    train_sampler = AspectBatchSampler(
        buckets=train_dataset.buckets,
        batch_size=args.batch_size,
        target_resolutions=target_resolutions,
        prompt_keys=prompt_keys,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )

    num_samples = len(train_dataset)
    local_batch_size = int(args.batch_size)

    # Create data loaders:
    train_dataloader = CustomDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.save_samples:
        # validation dataset
        num_test_samples = args.batch_size_test * accelerator.num_processes
        test_dataset = TextPromptDataset(
            str(test_jsonl_path),
            prompt_keys=test_prompt_keys,
            num_prompts=num_test_samples,
            have_gt=True,
            data_root=dataset_root,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False
        )
    else:
        test_dataloader = None

    # printing
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_samples} samples")
        logger.info(f"Training bucket resolutions: {target_resolutions}")
        if not ds_config:
            logger.info("DeepSpeed disabled; using standard Accelerate prepare")
        if args.low_vram:
            logger.info("Low VRAM mode enabled: frozen VAE/text encoder offload between batches")
        if args.block_offload:
            logger.info("Transformer block offload requested for final sampling only")
        if args.layer_offload:
            logger.info("AI Toolkit-style layer offloading enabled for training-time transformer/text encoder")
        if not args.save_samples:
            logger.info("Sample generation disabled")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(
            f"Total trainable parameters in gen_model: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad)}")

        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)

    # prepare file to log loss
    if accelerator.is_main_process:
        # clean the log files if they exist
        if os.path.exists(log_gen):
            os.remove(log_gen)
        # add a header to the log files
        with open(log_gen, 'w') as f:
            f.write("loss for few step generator\n")

    if zero2_plugin_a is not None:
        assert get_active_deepspeed_plugin(accelerator.state) is zero2_plugin_a
        if test_dataloader is None:
            gen_model, optimizer_gen, train_dataloader = accelerator.prepare(
                gen_model, optimizer_gen, train_dataloader
            )
        else:
            gen_model, optimizer_gen, train_dataloader, test_dataloader = accelerator.prepare(
                gen_model, optimizer_gen, train_dataloader, test_dataloader
            )
    else:
        gen_model, optimizer_gen = accelerator.prepare(gen_model, optimizer_gen)

    # Keep pipeline-based sampling on the same prepared transformer used for training.
    pipeline.transformer = gen_model

    if args.low_vram:
        offload_frozen_conditioners(pipeline)

    global_step = 0
    final_sampler_payload = None
    epoch_start = -1
    # resume (we now leave it blank, users can add their own checkpoints)

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    ############################################### Train Loop ######################################################

    if args.save_samples:
        # get sample prompts, free to change
        test_prompts, gt_image_paths = next(iter(test_dataloader))
        sample_h, sample_w = scale_sample_size(test_h, test_w, args.sample_resolution_scale)
        if args.sample_resolution_scale != 1.0:
            logger.info(f"Inline sample resolution scaled to {sample_w}x{sample_h}")
        test_images_gt = []
        for image_path in gt_image_paths:
            with Image.open(image_path) as img:
                test_images_gt.append(img.convert("RGB"))

        generator_test = create_generator(test_prompts, 2026)
        if not args.skip_initial_sample:
            with torch.no_grad():
                # sample multistep images for comparison
                prepare_pipeline_for_sampling(pipeline, accelerator.device, inference_dtype, vae_dtype)
                pipeline.vae.to(accelerator.device, dtype=inference_dtype)
                with accelerator.autocast():
                    with pipeline.transformer.disable_adapter() if args.use_lora > 1 else torch.no_grad():
                        images = pipeline(
                            prompt=test_prompts,
                            height=sample_h,
                            width=sample_w,
                            num_inference_steps=args.num_training_steps,
                            guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                            generator=generator_test,
                            output_type="pt",
                        )[0]

                # resize to 1/2 resolution according to its original size
                images = torch.nn.functional.interpolate(images, size=(sample_h // 2, sample_w // 2), mode='bicubic',
                                                         align_corners=False)

                # Save images locally
                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(images.to(torch.float32))

                pipeline.vae.to(accelerator.device, dtype=vae_dtype)

                # Save as grid images
                out_samples = Image.fromarray(array2grid(out_samples))
                if accelerator.is_main_process:
                    base_dir = os.path.join(args.output_dir, args.exp_name)
                    sample_dir = os.path.join(base_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    out_samples.save(f"{sample_dir}/samples_original.png")
                    logger.info(f"Saved original sample images to {sample_dir}/samples_original.png")
                del images, out_samples
                free_cuda_memory()
        else:
            logger.info("Skipped initial sample generation by request")

    grad_norm = 0
    for epoch in range(epoch_start + 1, args.epochs):
        for batch in train_dataloader:

            with accelerator.accumulate(gen_model):


                images = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dtype)
                train_dtype = inference_dtype
                prompts = batch["prompts"]

                bsz = images.shape[0]
                h, w = images.shape[2], images.shape[3]

                seqlen = (h * w) // (16 * 16)
                mu = compute_empirical_mu(seqlen, args.num_training_steps)
                pipeline.scheduler.set_timesteps(
                    args.num_training_steps,
                    device=accelerator.device,
                    mu=mu,
                )
                timesteps = pipeline.scheduler.timesteps

                with torch.no_grad():
                    if args.low_vram:
                        move_frozen_conditioners_to_device(
                            pipeline,
                            accelerator.device,
                            inference_dtype,
                            vae_dtype,
                        )
                    with accelerator.autocast():
                        prompt_embeds, txt_ids = _encode_prompt(
                            pipeline.text_encoder,
                            tokenizer,
                            prompts,
                            max_sequence_length=512,
                            device=accelerator.device,
                        )
                        teacher_edit_prompts = [f"{p} {args.edit_sys_prompt}".strip() for p in prompts]
                        teacher_prompt_embeds, teacher_txt_ids = _encode_prompt(
                            pipeline.text_encoder,
                            tokenizer,
                            teacher_edit_prompts,
                            max_sequence_length=512,
                            device=accelerator.device,
                        )
                        teacher_image_latents, teacher_image_latent_ids = prepare_batch_image_latents(
                            pipeline=pipeline,
                            images=images,
                            device=accelerator.device,
                            vae_dtype=vae_dtype,
                            latent_dtype=train_dtype,
                            generator=None,
                        )

                        latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
                            device=accelerator.device,
                            dtype=vae_dtype,
                        )
                        latents_bn_std = torch.sqrt(
                            pipeline.vae.bn.running_var.view(1, -1, 1, 1)
                            + pipeline.vae.config.batch_norm_eps
                        ).to(
                            device=accelerator.device,
                            dtype=vae_dtype,
                        )
                    if args.low_vram:
                        offload_frozen_conditioners(pipeline)

                latents_begin = pipeline.prepare_latents(
                    batch_size=bsz,
                    num_latents_channels=num_channels_latents,
                    height=h,
                    width=w,
                    dtype=train_dtype,
                    device=accelerator.device,
                    generator=None,
                    latents=None,
                )
                latents_begin, latent_ids = latents_begin

                latents_student = latents_begin
                latents_teacher = latents_begin

                total_loss = 0.0
                loss_dopsd_whole = []
                student_x0_traj = []
                teacher_x0_traj = []

                for back_step in range(len(timesteps)):
                    t = timesteps[back_step].expand(bsz) / 1000
                    t = t.to(device=accelerator.device, dtype=train_dtype)

                    if back_step < len(timesteps) - 1:
                        next_t = timesteps[back_step + 1].expand(bsz) / 1000
                    else:
                        next_t = torch.zeros_like(t)
                    next_t = next_t.to(device=accelerator.device, dtype=train_dtype)

                    dt = next_t - t

                    # detach current state to avoid cross-timestep BPTT
                    latents_student = latents_student.detach().requires_grad_(True)
                    latents_teacher = latents_teacher.detach()

                    # teacher
                    with torch.no_grad():
                        with accelerator.autocast():
                            gen_model.set_adapter("teacher")
                            teacher_hidden_states = torch.cat([latents_student, teacher_image_latents], dim=1)
                            teacher_img_ids = torch.cat([latent_ids, teacher_image_latent_ids], dim=1)
                            v_pred_teacher = gen_model(
                                hidden_states=teacher_hidden_states,
                                timestep=t,
                                guidance=None,
                                encoder_hidden_states=teacher_prompt_embeds,
                                txt_ids=teacher_txt_ids,
                                img_ids=teacher_img_ids,
                                return_dict=False,
                            )[0]
                            v_pred_teacher = v_pred_teacher[:, :latents_student.size(1)]

                        latents_teacher_cur = latents_student
                        x_0_teacher = latents_teacher_cur + (0 - t).reshape(bsz, 1, 1) * v_pred_teacher
                        latents_teacher = latents_teacher_cur + v_pred_teacher * dt.reshape(bsz, 1, 1)

                    # student
                    with accelerator.autocast():
                        gen_model.set_adapter("student")
                        v_pred_student = gen_model(
                            hidden_states=latents_student,
                            timestep=t,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=txt_ids,
                            img_ids=latent_ids,
                            return_dict=False,
                        )[0]
                        v_pred_student = v_pred_student[:, :latents_student.size(1)]

                    latents_student_cur = latents_student
                    x_0_student = latents_student_cur + (0 - t).reshape(bsz, 1, 1) * v_pred_student
                    latents_student = latents_student_cur + v_pred_student * dt.reshape(bsz, 1, 1)
                    
                    loss_dopsd =  F.mse_loss(
                        x_0_student, x_0_teacher.detach(), reduction="mean"
                    )
                    total_loss = total_loss + loss_dopsd
                    loss_dopsd_whole.append(loss_dopsd.detach())

                    if args.save_samples and args.sample_steps > 0 and accelerator.sync_gradients and ((global_step + 1) % args.sample_steps == 0):
                        student_x0_traj.append(x_0_student.detach())
                        teacher_x0_traj.append(x_0_teacher.detach())

                total_loss = total_loss / len(loss_dopsd_whole)
                accelerator.backward(total_loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(gen_model.parameters(), args.max_grad_norm)

                optimizer_gen.step()
                optimizer_gen.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    if accelerator.num_processes == 1:
                        loss_dopsd_value = torch.stack(loss_dopsd_whole).detach().mean().item()
                        loss_total_value = total_loss.detach().mean().item()
                    else:
                        loss_dopsd_value = accelerator.gather(torch.stack(loss_dopsd_whole).detach()).mean().item()
                        loss_total_value = accelerator.gather(total_loss.detach()).mean().item()

                    logs = {
                        "loss_dopsd": loss_dopsd_value,
                        "loss_total": loss_total_value,
                        "glo_s": global_step,
                        "epoch": epoch,
                        "grad_n": float(grad_norm) if grad_norm is not None else 0.0,
                    }

                    if accelerator.trackers:
                        accelerator.log(logs, step=global_step)
                    ema_update_lora_adapter(
                        gen_model,
                        src_adapter="student",
                        dst_adapter="teacher",
                        ema_decay=args.ema_decay,
                    )

                    if accelerator.is_main_process:
                        with open(log_gen, "a") as f_log_gen:
                            f_log_gen.write(f"{json.dumps(logs)}\n")
                        write_tensorboard_scalars(tb_writer, logs, global_step)

                    # save model
                    if args.save_checkpoints and (global_step % args.checkpoint_steps == 0 or global_step == args.max_train_steps):
                        # save checkpoint
                        if accelerator.is_main_process:
                            if args.use_lora > 1:
                                lora_dict_gen = os.path.join(checkpoint_dir, f"lora_gen_step_{global_step}")
                                os.makedirs(lora_dict_gen, exist_ok=True)
                                unwrap_model(gen_model, accelerator).save_pretrained(lora_dict_gen)
                            else:
                                ckpt_dict_gen = os.path.join(checkpoint_dir, f"gen_model_step_{global_step}.pt")
                                accelerator.save(unwrap_model(gen_model, accelerator).state_dict(), ckpt_dict_gen)
                                logger.info(f"Saved model checkpoint to {checkpoint_dir} at step {global_step}")

                    # visualize samples
                    if args.save_samples and args.sample_steps > 0 and (global_step % args.sample_steps == 0 or global_step == args.max_train_steps):
                        with torch.no_grad():
                            prepare_pipeline_for_sampling(pipeline, accelerator.device, inference_dtype, vae_dtype)
                            pipeline.vae.to(accelerator.device, dtype=vae_dtype)

                            traj_dir = os.path.join(args.output_dir, args.exp_name)
                            traj_dir = os.path.join(traj_dir, "samples_trajectory")
                            save_student_teacher_trajectory(
                                pipeline,
                                student_x0_traj,
                                teacher_x0_traj,
                                latent_ids,
                                traj_dir,
                                global_step,
                                accelerator,
                                vae_dtype,
                                latents_bn_mean,
                                latents_bn_std,
                                max_size=(2048, 2048),
                            )
                            del student_x0_traj, teacher_x0_traj
                            del images, prompts
                            del prompt_embeds, txt_ids, teacher_prompt_embeds, teacher_txt_ids
                            del teacher_image_latents, teacher_image_latent_ids
                            del latents_begin, latent_ids, latents_student, latents_teacher
                            del latents_student_cur, latents_teacher_cur
                            del teacher_hidden_states, teacher_img_ids
                            del v_pred_student, v_pred_teacher, x_0_student, x_0_teacher
                            del loss_dopsd, loss_dopsd_whole, total_loss
                            del latents_bn_mean, latents_bn_std
                            free_cuda_memory()

                            defer_final_sample = should_defer_final_sampling(args, global_step)
                            if defer_final_sample:
                                if accelerator.is_main_process:
                                    final_sampler_payload = write_final_sampler_request(
                                        args,
                                        global_step,
                                        gen_model,
                                        accelerator,
                                        test_prompts,
                                        gt_image_paths,
                                        test_h,
                                        test_w,
                                        logger,
                                    )
                                logger.info("Final sample generation deferred to a clean subprocess")
                            else:
                                if args.block_offload and global_step == args.max_train_steps:
                                    enable_transformer_block_offload(
                                        unwrap_model(gen_model, accelerator),
                                        accelerator.device,
                                        args.block_offload_num_blocks,
                                        logger,
                                    )

                                # sample multistep images for comparison
                                gen_model.set_adapter("teacher")
                                with accelerator.autocast():
                                    teacher_test_images = []
                                    teacher_edit_prompts = [
                                        f"{p} {args.edit_sys_prompt}".strip() for p in test_prompts
                                    ]
                                    for test_image_gt, teacher_prompt, test_generator in zip(
                                        test_images_gt,
                                        teacher_edit_prompts,
                                        generator_test,
                                    ):
                                        image_t = sample_flux2_with_optional_latent_decode(
                                            pipeline,
                                            vae_dtype,
                                            decode_latents_after_transformer_hooks=args.block_offload,
                                            image=test_image_gt,
                                            prompt=teacher_prompt,
                                            height=sample_h,
                                            width=sample_w,
                                            num_inference_steps=args.num_training_steps,
                                            guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                            generator=test_generator,
                                        )
                                        teacher_test_images.append(image_t)
                                    images_t = torch.cat(teacher_test_images, dim=0)
                                image_t = torch.nn.functional.interpolate(images_t, size=(sample_h // 2, sample_w // 2), mode='bicubic', align_corners=False)
                                if args.block_offload:
                                    image_t = image_t.to("cpu")
                                    del images_t, teacher_test_images
                                    free_cuda_memory()

                                gen_model.set_adapter("student")
                                with accelerator.autocast():
                                    images_s = sample_flux2_with_optional_latent_decode(
                                        pipeline,
                                        vae_dtype,
                                        decode_latents_after_transformer_hooks=args.block_offload,
                                        prompt=test_prompts,
                                        height=sample_h,
                                        width=sample_w,
                                        num_inference_steps=args.num_training_steps,
                                        guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                        generator=generator_test,
                                    )

                                images_s = torch.nn.functional.interpolate(images_s, size=(sample_h // 2, sample_w // 2),
                                                                         mode='bicubic', align_corners=False)

                                # Save images locally
                                accelerator.wait_for_everyone()
                                out_samples = accelerator.gather(images_s.to(torch.float32))
                                out_samples_t = accelerator.gather(image_t.to(torch.float32))

                                # Save as grid images
                                out_samples = Image.fromarray(array2grid(out_samples))
                                out_samples_t = Image.fromarray(array2grid(out_samples_t))
                                if accelerator.is_main_process:

                                    base_dir = os.path.join(args.output_dir, args.exp_name)
                                    sample_dir = os.path.join(base_dir, "samples")
                                    os.makedirs(sample_dir, exist_ok=True)
                                    out_samples.save(f"{sample_dir}/samples_step_{global_step}_student.png")
                                    out_samples_t.save(f"{sample_dir}/samples_step_{global_step}_teacher.png")
                                    logger.info(f"Saved sample images to {sample_dir}/samples_step_{global_step}.png")

                                pipeline.vae.to(accelerator.device, dtype=vae_dtype)
                                del images_s, image_t, out_samples, out_samples_t
                                free_cuda_memory()
                                if args.low_vram:
                                    offload_frozen_conditioners(pipeline)
            progress_bar.set_postfix(**logs)

            ############################################### End Train Loop ######################################################

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    is_main_process = accelerator.is_main_process
    if final_sampler_payload:
        del pipeline, gen_model
        del optimizer_gen, train_dataloader
        if test_dataloader is not None:
            del test_dataloader
        free_cuda_memory()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()

    if tb_writer is not None:
        tb_writer.close()

    accelerator.end_training()

    if is_main_process and final_sampler_payload:
        logger.info("Final sample generation request ready for runner")
    elif is_main_process:
        logger.info("Training completed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)




































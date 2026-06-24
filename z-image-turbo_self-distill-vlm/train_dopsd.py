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
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps
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
from utils import _encode_prompt, create_generator
from ema_utils import *
from vlm_utils import load_matching_state_dict,get_qwen3vl_zimage_prompt_embeds

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

def offload_zimage_conditioners(pipeline, vl_model):
    pipeline.text_encoder.to("cpu")
    pipeline.vae.to("cpu")
    vl_model.to("cpu")
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

def move_prompt_embeds_to_device(prompt_embeds, device, dtype):
    return [embed.to(device=device, dtype=dtype) for embed in prompt_embeds]

def encode_zimage_prompts_for_sampling(pipeline, tokenizer, prompts, sample_device, dtype, low_vram):
    prompt_input = prompts if isinstance(prompts, str) else list(prompts)
    prompt_embeds = _encode_prompt(
        pipeline.text_encoder,
        tokenizer,
        prompt_input,
        max_sequence_length=512,
        device="cpu" if low_vram else sample_device,
    )
    return move_prompt_embeds_to_device(prompt_embeds, sample_device, dtype)

@torch.no_grad()
def sample_zimage_from_prompt_embeds(
    pipeline,
    prompt_embeds,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    generator,
    device,
    output_type="pt",
):
    if guidance_scale != 0.0:
        raise NotImplementedError("Explicit-device Z-Image sampling currently supports guidance_scale=0.0 only.")

    pipeline._guidance_scale = guidance_scale
    pipeline._joint_attention_kwargs = None
    pipeline._interrupt = False
    pipeline._cfg_normalization = False
    pipeline._cfg_truncation = 1.0

    batch_size = len(prompt_embeds)
    num_channels_latents = pipeline.transformer.in_channels
    latents = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        torch.float32,
        device,
        generator,
        None,
    )

    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    pipeline.scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)
    pipeline.scheduler.set_begin_index(0)

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, timestep in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            timestep_model_input = timestep.expand(latents.shape[0])
            timestep_model_input = (1000 - timestep_model_input) / 1000
            latent_model_input = latents.to(pipeline.transformer.dtype)
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = pipeline.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = torch.stack([item.float() for item in model_out_list], dim=0)
            noise_pred = -noise_pred.squeeze(2)

            latents = pipeline.scheduler.step(noise_pred.to(torch.float32), timestep, latents, return_dict=False)[0]
            assert latents.dtype == torch.float32

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    if output_type == "latent":
        return latents

    latents = latents.to(pipeline.vae.dtype)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(images, output_type=output_type)



@torch.no_grad()
def decode_latents_to_images(latents, pipeline):
    latents = latents.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def save_student_teacher_trajectory(pipeline, student_x0_traj, teacher_x0_traj, save_dir, global_step, accelerator, max_size=None):
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
        s = decode_latents_to_images(sx0[:4], pipeline).float()
        t = decode_latents_to_images(tx0[:4], pipeline).float()

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

    if accelerator.is_main_process:
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)

        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

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
    pipeline = ZImagePipeline.from_pretrained(
        args.pretrained_model,
        low_cpu_mem_usage=False,
        torch_dtype=inference_dtype,
    )

    num_channels_latents = pipeline.transformer.in_channels

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(args.use_lora <= 1)
    tokenizer = pipeline.tokenizer


    # get vlm encoder
    vl_model_name = os.environ.get("DOPSD_QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    min_pixels = 512 * 512
    max_pixels = 768 * 768
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(vl_model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    vl_model = AutoModelForImageTextToText.from_pretrained(
        vl_model_name,
        torch_dtype=inference_dtype,
    )
    missing_keys, unexpected_keys = load_matching_state_dict(
        target_module=vl_model.model.language_model,
        source_state_dict=pipeline.text_encoder.state_dict(),
        verbose=False,
    )
    vl_model.requires_grad_(False)
    if args.low_vram:
        vl_model.to("cpu", dtype=inference_dtype)
    else:
        vl_model.to(accelerator.device, dtype=inference_dtype)

    if accelerator.is_main_process:
        logger.info(f"VLM loaded, dtype: {vl_model.parameters().__next__().dtype}")




    # disable progress bar for cold start
    pipeline.set_progress_bar_config(disable=True)

    if args.layer_offload and args.layer_offload_transformer_percent > 0:
        zimage_ignore_modules = [
            module
            for module in (
                getattr(pipeline.transformer, "x_pad_token", None),
                getattr(pipeline.transformer, "cap_pad_token", None),
            )
            if module is not None
        ]
        attach_layer_offload(
            pipeline.transformer,
            accelerator.device,
            args.layer_offload_transformer_percent,
            logger,
            label="transformer",
            ignore_modules=zimage_ignore_modules,
        )

    # init lora
    if args.use_lora > 1:
        # Set correct lora layers
        target_modules = [
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "attention.to_k",
            "attention.to_q",
            "attention.to_v",
            "attention.to_out.0",
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
    else:
        vae_dtype = inference_dtype
    # avoid OOM
    pipeline.vae.enable_slicing()
    if args.layer_offload and args.layer_offload_text_encoder_percent > 0:
        attach_layer_offload(
            pipeline.text_encoder,
            accelerator.device,
            args.layer_offload_text_encoder_percent,
            logger,
            label="text encoder",
        )
    if args.low_vram:
        pipeline.vae.to("cpu", dtype=vae_dtype)
        pipeline.text_encoder.to("cpu", dtype=inference_dtype)
        free_cuda_memory()
        if accelerator.is_main_process:
            logger.info("Low VRAM mode enabled: frozen VAE/text encoder/VLM offload to CPU.")
    if args.layer_offload and accelerator.is_main_process:
        logger.info("AI Toolkit-style layer offloading enabled for training-time transformer/text encoder")
    else:
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
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

    # printing
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_samples} samples")
        if not args.save_checkpoints:
            logger.info("Checkpoint saving disabled")
        if not args.save_samples:
            logger.info("Sample generation disabled")
        if args.block_offload:
            logger.info("Transformer block offload requested for final sampling only")
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
        gen_model, optimizer_gen, test_dataloader = accelerator.prepare(
            gen_model, optimizer_gen, test_dataloader
        )
    else:
        gen_model, optimizer_gen = accelerator.prepare(gen_model, optimizer_gen)

    # Keep pipeline-based sampling on the same prepared transformer used for training.
    pipeline.transformer = gen_model

    global_step = 0
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
        test_images_gt = []
        for image_path in gt_image_paths:
            with Image.open(image_path) as img:
                test_images_gt.append(img.convert("RGB"))

        generator_test = create_generator(test_prompts, 2026)
        if not args.skip_initial_sample:
            with torch.no_grad():
                # sample multistep images for comparison
                prompt_embeds_test = encode_zimage_prompts_for_sampling(
                    pipeline,
                    tokenizer,
                    test_prompts,
                    accelerator.device,
                    inference_dtype,
                    args.low_vram,
                )
                pipeline.vae.to(accelerator.device, dtype=inference_dtype)
                with accelerator.autocast():
                    with pipeline.transformer.disable_adapter() if args.use_lora > 1 else torch.no_grad():
                        images = sample_zimage_from_prompt_embeds(
                            pipeline,
                            prompt_embeds_test,
                            height=test_h,
                            width=test_w,
                            num_inference_steps=9 if args.num_training_steps < 10 else 50,
                            guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                            generator=generator_test,
                            device=accelerator.device,
                            output_type="pt",
                        )

                # resize to 1/2 resolution according to its original size
                images = torch.nn.functional.interpolate(images, size=(test_h // 2, test_w // 2), mode='bicubic',
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
                del images, out_samples, prompt_embeds_test
                free_cuda_memory()
                if args.low_vram:
                    offload_zimage_conditioners(pipeline, vl_model)
        else:
            logger.info("Skipped initial sample generation by request")

    grad_norm = 0
    for epoch in range(epoch_start + 1, args.epochs):
        for batch in train_dataloader:

            with accelerator.accumulate(gen_model):


                images = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dtype)
                train_dtype = inference_dtype
                prompts = batch["prompts"]


                images_vl = (images + 1) / 2
                if args.low_vram:
                    images_vl = list(images_vl.to("cpu").unbind(dim=0))
                else:
                    images_vl = list(images_vl.unbind(dim=0))

                bsz = images.shape[0]
                h, w = images.shape[2], images.shape[3]


                if args.num_training_steps == 4:
                    timesteps = [0, 100.0000014901161, 250, 500]
                elif args.num_training_steps == 8:
                    timesteps = [1000.0000,  976.8991,  947.7647,  909.8782,  858.5987,  785.2998,
                         671.9212,  473.2203]
                    timesteps = [1000 - t for t in timesteps]
                else:
                    raise NotImplementedError

                #change to list of tensor for timesteps range from (0~1) equal /1000
                timesteps = [torch.tensor(t, device=accelerator.device, dtype=train_dtype) for t in timesteps]


                with torch.no_grad():
                    if args.low_vram:
                        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
                    with accelerator.autocast():
                        prompt_embeds_list = _encode_prompt(
                            pipeline.text_encoder,
                            tokenizer,
                            prompts,
                            max_sequence_length=512,
                            device="cpu" if args.low_vram else accelerator.device,
                        )
                    if args.low_vram:
                        prompt_embeds_list = move_prompt_embeds_to_device(
                            prompt_embeds_list,
                            accelerator.device,
                            inference_dtype,
                        )
                        free_cuda_memory()

                    with torch.no_grad():
                        if not args.low_vram:
                            vl_model.to(accelerator.device, dtype=inference_dtype)
                        prompt_embeds_list_vl = get_qwen3vl_zimage_prompt_embeds(
                            vl_model=vl_model,
                            processor=processor,
                            prompts=prompts,
                            images=images_vl,
                            device="cpu" if args.low_vram else accelerator.device,
                            dtype=inference_dtype,
                            max_sequence_length=1024,
                            num_images_per_prompt=1,
                             hidden_state_layer=-2,
                            use_system_prompt=False,
                        )
                        if args.low_vram:
                            prompt_embeds_list_vl = move_prompt_embeds_to_device(
                                prompt_embeds_list_vl,
                                accelerator.device,
                                inference_dtype,
                            )
                        else:
                            prompt_embeds_list_vl = move_prompt_embeds_to_device(
                                prompt_embeds_list_vl,
                                accelerator.device,
                                inference_dtype,
                            )

                    with torch.no_grad():
                        if args.low_vram:
                            pipeline.vae.to(accelerator.device, dtype=vae_dtype)
                        images = pipeline.vae.encode(images).latent_dist.mode()
                        images = (images - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                    if args.low_vram:
                        offload_zimage_conditioners(pipeline, vl_model)


                latents_begin = pipeline.prepare_latents(
                    batch_size=bsz,
                    num_channels_latents=num_channels_latents,
                    height=h,
                    width=w,
                    dtype=train_dtype,
                    device=accelerator.device,
                    generator=None,
                    latents=None,
                )

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
                        next_t = torch.ones_like(t)
                    next_t = next_t.to(device=accelerator.device, dtype=train_dtype)

                    dt = next_t - t

                    # detach current state to avoid cross-timestep BPTT
                    latents_student = latents_student.detach().requires_grad_(True)
                    latents_teacher = latents_teacher.detach()

                    latents_student_in = latents_student.unsqueeze(2)
                    latents_student_list = list(latents_student_in.unbind(dim=0))

                    latents_teacher_in = latents_teacher.unsqueeze(2)
                    latents_teacher_list = list(latents_teacher_in.unbind(dim=0))

                    # teacher
                    with torch.no_grad():
                        with accelerator.autocast():
                            gen_model.set_adapter("teacher")
                            v_pred_teacher = gen_model(
                                latents_student_list,
                                t,
                                prompt_embeds_list_vl,
                                return_dict=False,
                            )[0]
                            v_pred_teacher = torch.stack(v_pred_teacher, dim=0).squeeze(2)

                        latents_teacher_cur = latents_student
                        x_0_teacher = latents_teacher_cur + (1 - t.reshape(bsz, 1, 1, 1)) * v_pred_teacher
                        latents_teacher = latents_teacher_cur + v_pred_teacher * dt.reshape(bsz, 1, 1, 1)

                    # student
                    with accelerator.autocast():
                        gen_model.set_adapter("student")
                        v_pred_student = gen_model(
                            latents_student_list,
                            t,
                            prompt_embeds_list,
                            return_dict=False,
                        )[0]
                        v_pred_student = torch.stack(v_pred_student, dim=0).squeeze(2)

                    latents_student_cur = latents_student
                    x_0_student = latents_student_cur + (1 - t.reshape(bsz, 1, 1, 1)) * v_pred_student
                    latents_student = latents_student_cur + v_pred_student * dt.reshape(bsz, 1, 1, 1)
                    
                    #  we use x_0 loss  here, which is different as shown in our paper, It can be regarded as a weighted sum of v loss and t (with a greater weight in the early steps). We found that this leads to faster convergence.
                    # legacy:
                    # loss_dopsd = F.mse_loss(
                    #     v_pred_student,v_pred_teacher.detach(), reduction="mean"
                    # )
                    
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

                    logs = {
                        "loss_dopsd": accelerator.gather(torch.stack(loss_dopsd_whole).detach()).mean().item(),
                        "loss_total": accelerator.gather(total_loss.detach()).mean().item(),
                        "glo_s": global_step,
                        "epoch": epoch,
                        "grad_n": float(grad_norm) if grad_norm is not None else 0.0,
                    }

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
                        if args.block_offload and global_step == args.max_train_steps:
                            enable_transformer_block_offload(
                                unwrap_model(gen_model, accelerator),
                                accelerator.device,
                                args.block_offload_num_blocks,
                                logger,
                            )
                        with torch.no_grad():
                            prompt_embeds_test = encode_zimage_prompts_for_sampling(
                                pipeline,
                                tokenizer,
                                test_prompts,
                                accelerator.device,
                                inference_dtype,
                                args.low_vram,
                            )
                            pipeline.vae.to(accelerator.device, dtype=inference_dtype)

                            traj_dir = os.path.join(args.output_dir, args.exp_name)
                            traj_dir = os.path.join(traj_dir, "samples_trajectory")
                            save_student_teacher_trajectory(
                                pipeline,
                                student_x0_traj,
                                teacher_x0_traj,
                                traj_dir,
                                global_step,
                                accelerator,
                                max_size=(2048, 2048),
                            )

                            # sample multistep images for comparison
                            gen_model.set_adapter("student")
                            with accelerator.autocast():
                                images_s = sample_zimage_from_prompt_embeds(
                                    pipeline,
                                    prompt_embeds_test,
                                    height=test_h,
                                    width=test_w,
                                    num_inference_steps=9 if args.num_training_steps < 10 else 50,
                                    guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                    generator=generator_test,
                                    device=accelerator.device,
                                    output_type="pt",
                                )

                            images_s = torch.nn.functional.interpolate(images_s, size=(test_h // 2, test_w // 2),
                                                                     mode='bicubic', align_corners=False)

                            gen_model.set_adapter("teacher")
                            with accelerator.autocast():
                                prompt_embeds_vl_test = get_qwen3vl_zimage_prompt_embeds(
                                    vl_model=vl_model,
                                    processor=processor,
                                    prompts=test_prompts,
                                    images=test_images_gt,
                                    device="cpu" if args.low_vram else accelerator.device,
                                    dtype=inference_dtype,
                                    max_sequence_length=1024,
                                    num_images_per_prompt=1,
                                     hidden_state_layer=-2,
                                    use_system_prompt=False,
                        )
                                prompt_embeds_vl_test = move_prompt_embeds_to_device(
                                    prompt_embeds_vl_test,
                                    accelerator.device,
                                    inference_dtype,
                                )
                                images_t = sample_zimage_from_prompt_embeds(
                                    pipeline,
                                    prompt_embeds_vl_test,
                                    height=test_h,
                                    width=test_w,
                                    num_inference_steps=9 if args.num_training_steps < 10 else 50,
                                    guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                    generator=generator_test,
                                    device=accelerator.device,
                                    output_type="pt",
                                )
                            image_t = torch.nn.functional.interpolate(images_t, size=(test_h // 2, test_w // 2), mode='bicubic', align_corners=False)

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
                            del images_s, images_t, image_t, out_samples, out_samples_t
                            free_cuda_memory()
                            if args.low_vram:
                                offload_zimage_conditioners(pipeline, vl_model)
            progress_bar.set_postfix(**logs)

            ############################################### End Train Loop ######################################################

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)




































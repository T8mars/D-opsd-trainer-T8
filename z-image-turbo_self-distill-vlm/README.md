# Z-Image-Turbo Tuning with D-OPSD (vlm condition as context)



##  🌀  Training


#### Single Node 4 GPUs (The specific path settings inside need to be changed.)
```bash
cd z-image-turbo_self-distill-vlm
bash scripts/train_lora.sh
```
Note that we conduct the training in 4 steps, which will accelerate the training speed. During inference, it can maintain the same 8-step inference process as the default setting of the Z-Image-Turbo.

The output directory structure will be like:
```output_dir/
├── checkpoints/
│   │   └── lora_gen_step_i/
├──  samples_trajectory/
│   │   └──t0/
│   │   └──ti/
├── loss_logs/
│   │   └── loss_gen_log.jsonl
├── samples/
│   │   └── samples_original.png
│   │   └── samples_step_i_student.png
│   │   └── samples_step_i_teacher.png
├── args.json
└── log.txt
```

##  🌠 Inference

After training, loading the trained LoRA weights to perform inference. The inference pipeline is the same as the original Z-Image Turbo.

demo code
```python
import torch
from diffusers import ZImagePipeline
from peft import PeftModel

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# 2. Load the LoRA weights
lora_weights_path = "exp_results/dopsd_vlmcon_ema0.9999_onpolicy_4steptrain_styleqianxi_bsz4_lora_lr1e-4/lora_gen_step_i/student"
pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        lora_weights_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

prompt = """A deer stands in the magical forest, surrounded by plants and mist. The ground is
covered with fallen leaves, and the entire scene has an [V] style"""

# 3. Generate Image

# without lora
with pipe.transformer.disable_adapter():
    image_original = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
# with lora
image_lora = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

#save imgs
image_original.save("samples/samples_original.png")
image_lora.save("samples/samples_step_i_student.png")
```


## 🤝🏻 Acknowledgement

This code is mainly built upon [DMDR](https://github.com/vvvvvjdy/dmdr/), [Z-Image](https://github.com/Tongyi-MAI/Z-Image) repositories. 
Thanks for  their contributions to the community.



#


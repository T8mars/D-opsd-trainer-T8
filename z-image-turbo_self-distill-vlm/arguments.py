import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Training")

    #deepspeed
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--enable-gc", action=argparse.BooleanOptionalAction, default=False, help="Enable model gradient checkpointing.")

    # logging:
    parser.add_argument("--output-dir", type=str, default="dopsd-exps")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                        help="Write TensorBoard scalar event files next to the JSONL loss log.")
    parser.add_argument("--tensorboard-dir", type=str, default="tensorboard",
                        help="TensorBoard event directory, relative to the experiment output folder unless absolute.")

    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoint-steps", type=int, default=200000)
    parser.add_argument("--max-train-steps", type=int, default=200000)
    parser.add_argument("--save-samples", action=argparse.BooleanOptionalAction, default=True,
                        help="Save original/final sample images during training.")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                        help="Save LoRA checkpoints during training.")
    parser.add_argument("--skip-initial-sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Skip the pre-training original sample pass to reduce startup VRAM pressure.")


    # Gen model
    parser.add_argument("--pretrained_model", type=str, default="z-turbo")
    parser.add_argument("--use-lora",type=float, default=1, help="use if > 1")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--num-training-steps", type=int, default=8, help="number of diffusion steps for training.")
    parser.add_argument("--ema-decay", type=float, default=0.9, help="EMA decay for teacher model.")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False,
                        help="Offload frozen VAE/text encoder/VLM to CPU between conditioning and transformer training.")
    parser.add_argument("--block-offload", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Diffusers group block offload for the transformer when low VRAM is not enough.")
    parser.add_argument("--block-offload-num-blocks", type=int, default=1,
                        help="Transformer blocks per CPU/GPU transfer group for --block-offload.")
    parser.add_argument("--layer-offload", action=argparse.BooleanOptionalAction, default=False,
                        help="Use ai-toolkit-style training-time layer offloading for eligible Linear/Conv layers.")
    parser.add_argument("--layer-offload-transformer-percent", type=float, default=1.0,
                        help="Fraction of transformer Linear/Conv layers to offload when --layer-offload is enabled.")
    parser.add_argument("--layer-offload-text-encoder-percent", type=float, default=1.0,
                        help="Fraction of text encoder Linear/Conv layers to offload when --layer-offload is enabled.")

    #vae
    parser.add_argument("--vae-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="VAE precision.")

    # dataset
    parser.add_argument("--data-path-train-jsonl", type=str, default="../data/x.jsonl", help="Path to the training data jsonl file.")
    parser.add_argument("--data-path-test-jsonl", type=str, default="../data/x.jsonl", help="Path to the testing data jsonl file.")
    parser.add_argument("--batch-size", type=int, default=4, help="local batch size.")
    parser.add_argument("--batch-size-test", type=int, default=1, help="local batch size test.")
    parser.add_argument("--target-resolution", type=int, default=1024,
                        help="Base bucket resolution before --resolution-scale is applied, ai-toolkit-style.")
    parser.add_argument("--resolution-scale", type=float, default=1.0,
                        help="Scale training bucket resolutions. Use values below 1.0 for local smoke tests.")

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use-8bit-adam", action=argparse.BooleanOptionalAction, default=False,)

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate-gen", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=30)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)



    args = parser.parse_args()

    return args

# D-OPSD Trainer Skill

## Purpose

Use this file as the local project operating guide for building and maintaining the D-OPSD Trainer.

The project goal is to turn upstream `D-OPSD` into a local model training workbench with a clear Liquid Glass UI, reliable environment checks, model management, dataset validation, job queueing, live logs, loss charts, samples, and checkpoint handling.

## Required Context Preflight

Before any work involving code, architecture, debugging, frontend UI, testing, deployment, GitHub, code review, or technical documentation:

1. Read this `SKILL.md`.
2. Read `meta.json` if it exists.
3. Read `features.json` if it exists.
4. Read `roadmap.md` if it exists.
5. Read relevant configuration files.
6. Read relevant source files and tests before editing them.

This rule is especially important after context compaction or a long pause.

## Project Sources

Primary training core:

- `z-image-turbo_self-distill-vlm`
- `flux2-klein_self-distill-edit`
- `flux2-klein-edit-self-distill-gt-ref`

Reference project:

- `ostris/ai-toolkit`, used for workflow inspiration only: dashboard, jobs, queues, dataset tools, logs, samples, settings, and GPU monitor.

Do not copy unrelated ai-toolkit model families into this trainer unless they directly support D-OPSD.

## Implementation Principles

- Keep D-OPSD training logic intact at first. Wrap it with a runtime bridge before rewriting internals.
- Prefer WSL2/Linux for actual training when DeepSpeed, Triton, or bitsandbytes are involved.
- Make single-GPU 16GB operation stable before enabling larger multi-GPU or 9B recipes.
- Treat model downloads as managed assets with status, path, size, auth/licensing, and verification.
- Treat datasets as first-class objects, not plain paths.
- Keep generated commands inspectable before launch.
- Every long-running operation should write logs and status that the UI can recover after restart.
- For OOM-sensitive profiles, prefer conservative defaults first: direct Python launcher, bf16, batch size 1, gradient checkpointing, 8-bit Adam, scaled resolutions, and optional sample/checkpoint skipping for smoke runs.
- Use `ostris/ai-toolkit` as a reference for low-VRAM concepts such as `low_vram`, quantization, 8-bit optimizer, and layer/block offload. Do not copy unrelated model families or broad abstractions unless D-OPSD needs them.
- Optional transformer block offload is wired through Diffusers group offloading with `--block-offload --block-offload-num-blocks 2`. It is scoped to final sampling only; do not attach it to the training transformer path because a real FLUX2 backward pass failed with a cuda/cpu device mismatch. For FLUX2 Identity `save_samples=True` on the 16GB profile, the verified fallback is now deferred final sampling with tiled VAE encode/decode and Diffusers model CPU offload; the group block-offload hook itself remains experimental and is not used in the verified sampler request.

## UI Principles

- The first screen is the usable dashboard.
- Use Liquid Glass as a functional layer for navigation, top bars, floating controls, modals, and status chips.
- Keep logs, forms, charts, and tables high-contrast and readable.
- Do not use decorative glass that makes training information harder to scan.
- Use icons for tool buttons and tooltips for unfamiliar actions.
- Keep page sections unframed where possible; use cards for repeated items, modals, and specific tool panels.
- Avoid nested cards.
- Ensure responsive behavior on desktop and mobile.
- Default the user-facing UI to Chinese, keep a Chinese/English language toggle reachable on desktop and mobile, and show the visible product brand as `T8 D-OPSD Tranier`. Keep `/api/project` metadata as `D-OPSD Trainer` unless launcher health checks are intentionally updated.

## Feature Registry

`features.json` is the authoritative local feature ledger. Update it when a feature is planned, started, completed, blocked, or verified.

Feature statuses:

- `planned`
- `in_progress`
- `implemented`
- `verified`
- `blocked`

## Verification Rules

Before marking a feature verified:

- Identify the command or visual/runtime evidence that proves it.
- Run the relevant command or inspect the relevant artifact.
- Record the evidence in `features.json`.
- Do not treat a build passing as proof that training works.
- Do not treat a model cache path as proof that the model can load.
- Do not treat an empty artifact directory as proof that samples or checkpoints exist. Check for actual entries/files.
- For frontend QA, confirm Tailwind CSS is loaded in the browser. If `layout.css` has zero rules or `.grid` computes to `display: block`, restart the Next dev server before trusting layout results.
- If `npm run build` runs while `next dev` is already serving the UI, restart the dev server before browser QA. The production build can rewrite `.next` chunks and leave the running dev server with stale module references.
- When Browser automation is unavailable, run `scripts/check_ui_smoke.ps1` against the running launcher URL. It verifies project metadata, key API data, Chinese-default primary page HTML, the `T8 D-OPSD Tranier` visible brand, New Job preflight and verified 16GB profile text, Next CSS assets, and Liquid Glass CSS; it does not replace desktop/mobile visual inspection for overflow or clipped controls.

## Runtime Targets

Primary local target:

- Windows host UI.
- Windows UI launcher entrypoint is `D-OPSD-Trainer.cmd`, backed by `scripts/start_trainer.ps1`.
- Electron desktop package target is `trainer-ui/release/win-unpacked/T8 D-OPSD Tranier.exe`, built with `npm run pack:win --prefix trainer-ui`; installer/portable artifacts are built with `npm run dist:win --prefix trainer-ui`.
- The packaged Electron app runs the Next standalone server with `ELECTRON_RUN_AS_NODE=1`, sets `DOPSD_PROJECT_ROOT`, and copies bundled project files into the writable workspace `%APPDATA%\d-opsd-trainer-ui\workspace`.
- Electron main-process and embedded Next logs are written to `%APPDATA%\d-opsd-trainer-ui\logs\electron-main.log`; Next child stdout/stderr must use safe stream writes plus file logging so Windows GUI launches do not crash on `EPIPE`.
- Packaged smoke tests can set `DOPSD_ELECTRON_PORT` and `DOPSD_ELECTRON_SMOKE_FILE`, then launch `T8 D-OPSD Tranier.exe --smoke-test`; the verified run wrote `ok: true`, URL `http://127.0.0.1:19075`, and the AppData workspace path.
- WSL2 Ubuntu training backend when needed.
- RTX 4060 Ti 16GB as the current default hardware profile.
- WSL trainer environment should source `scripts/dopsd_wsl_env.sh` before dependency, model, or training commands.
- Fresh Windows/WSL machines should run `scripts/setup_wsl_trainer.ps1` to create/update `trainer-data/venvs/dopsd`, install `requirements-trainer.txt`, run `pip check`, and run runtime probe/settings checks. `-SkipPipInstall -SkipProbe` is only for a fast script-path smoke.
- The WSL trainer venv is `trainer-data/venvs/dopsd`.
- Hugging Face assets should live under `trainer-data/hf-home` unless explicitly overridden.
- The launcher defaults to `http://127.0.0.1:8675`, writes logs/PID under `trainer-data/launcher`, supports `-Port`, `-NoBrowser`, `-Wait`, and `-SmokeTest`, and warns if WSL is unavailable. Root double-click uses `-Wait` by default so the UI server stays alive with the launcher session; root `.cmd` launches with `-Wait` for any non-smoke invocation, even when extra arguments such as `-NoBrowser` or `-Port` are supplied. The root `.cmd` normalizes `PATH`, prefers `pwsh.exe` with Windows PowerShell fallback, checks WSL with `wsl.exe --status` to avoid false `wsl -l -v` host-encoding warnings, and verifies `/api/project` returns D-OPSD Trainer metadata before treating an existing port as ready.
- Launcher detach checker: `scripts/check_launcher_detach.ps1 -Port 18861 -TimeoutSeconds 120 -HoldSeconds 5`. Run it only when no project UI is already running; it rejects concurrent project Next dev servers because they share `.next` and can invalidate routes. The verified run started `scripts/start_trainer.ps1` without `-Wait` on a temporary port, observed the parent launcher exit, confirmed `/api/project` stayed healthy afterward, and cleaned only project-owned temporary UI processes while restoring the previous `launcher.pid`.
- UI smoke checker: `scripts/check_ui_smoke.ps1 -BaseUrl http://127.0.0.1:8675 -TimeoutSeconds 30`. The verified run checked 6 Chinese-default pages, the `T8 D-OPSD Tranier` visible brand, model custom path/open-folder copy, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, 3 dataset-ready summaries, and the New Job verified 16GB profile controls.
- Release readiness checker: run with PowerShell 7, `scripts/check_release_readiness.ps1 -BaseUrl http://127.0.0.1:8675 -UiTimeoutSeconds 30`. It runs Python runtime tests, TypeScript typecheck, JSON validation, `git diff --check`, PowerShell parser checks, WSL smoke-script syntax checks, production profile contract checks, UI smoke, training observability smoke, live queue smoke, runner recovery smoke, WSL residual training-process checks, and `nvidia-smi`. Use `-SkipTrainingObservability`, `-SkipLiveQueue`, or `-SkipRunnerRecovery` only for a faster inner loop, not final release evidence. The latest verified full run returned Ok true with 19 checks, 58 Python tests, production profile contract included, training observability/queue/recovery enabled, and GPU memory about `1895 MiB / 16380 MiB`.

Fallback target:

- Native Windows for UI and non-training utilities.

## Model Policy

Default managed models:

- `Tongyi-MAI/Z-Image-Turbo`
- `Qwen/Qwen3-VL-4B-Instruct`
- `black-forest-labs/FLUX.2-klein-4B`

Experimental models:

- `black-forest-labs/FLUX.2-klein-9B`

If a model is gated, record the blocker clearly and keep the app functional with the available models.

Current model notes:

- The three default models are cached locally under `trainer-data/hf-home/hub`.
- `Qwen/Qwen3-VL-4B-Instruct` should be passed to Z-Image through `DOPSD_QWEN_VL_MODEL` using the local snapshot path, because local ID loading can still attempt remote metadata checks.
- The 9B FLUX2 model is experimental and should not be downloaded or selected by default on the 16GB hardware profile.
- `/api/models` reads the real Hugging Face cache status through the runtime model registry.
- The Models page shows default readiness, cache size, snapshot count, primary snapshot, cache directory, auth/gated hints, and experimental status.
- The Models page also supports per-model custom local path overrides stored in `trainer-data/models/custom-model-paths.json` and can open either the custom path or the Hugging Face cache directory from the UI.

## Low-VRAM Operating Notes

The verified local FLUX2 smoke path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_identity_smoke_1step_lowvram_python timeout 1200 bash scripts/run_flux2_smoke.sh"
```

This path uses:

- direct `python train_dopsd.py` launch,
- local FLUX2 4B snapshot,
- `--low-vram`,
- `--resolution-scale 0.25`,
- `--use-8bit-adam`,
- bf16,
- batch size 1,
- LoRA rank 4,
- `--no-save-samples`,
- `--no-save-checkpoints`.

The smoke proves a one-step 256px low-VRAM training pass on RTX 4060 Ti 16GB. It does not prove production-resolution training, sample generation, or checkpoint writing.

## Recommended 16GB Production Starter Profiles

The current UI and runner default to verified starter profiles, not the old one-step 0.25 smoke profile:

- FLUX2 Identity: `RESOLUTION_SCALE=0.625`, `SAMPLE_RESOLUTION_SCALE=0.5`, `FINAL_SAMPLE_RESOLUTION_SCALE=0.5`, `MAX_TRAIN_STEPS=5`, `SAMPLE_STEPS=5`, `CHECKPOINT_STEPS=5`, samples and checkpoints enabled, block offload disabled. Evidence: `flux2_identity_res0625_artifacts_scale05_5step_20260623045623`.
- FLUX2 Editing: `RESOLUTION_SCALE=0.5625`, `SAMPLE_RESOLUTION_SCALE=0.5`, `MAX_TRAIN_STEPS=5`, `SAMPLE_STEPS=5`, `CHECKPOINT_STEPS=5`, samples and checkpoints enabled, block offload disabled. Evidence: `flux2_editing_res05625_artifacts_scale05_5step_20260623044037`.
- Z-Image Turbo: `RESOLUTION_SCALE=0.5`, `MAX_TRAIN_STEPS=2`, `SAMPLE_STEPS=2`, `CHECKPOINT_STEPS=2`, samples and checkpoints enabled, block offload disabled. Evidence: `zimage_style_res05_artifacts_2step_202606221528`.

These profiles are defined in `trainer_runtime/dopsd_trainer/profiles.py`, exposed through `python scripts/check_runtime.py profiles --project-root .`, surfaced in `/api/settings` as `production_profiles`, displayed in Settings and New Job, and checked by `scripts/check_production_profiles.ps1`.

Keep the following boundaries explicit: native full-size FLUX2 Identity 0.625 inline sampling OOMs, native full-size FLUX2 Editing 0.375 sampling OOMs, Z-Image 1.0 OOMs, FLUX2 artifact writing beyond the listed five-step starter profiles is unverified, and Diffusers group block offload remains experimental.

Current FLUX2 low-VRAM implementation moves frozen VAE/text encoder components to CPU between conditioning and transformer training. The FLUX2 Identity and FLUX2 Editing main training scripts configure tiled VAE before any inline sample decode, and the Identity deferred final sampler also uses tiled VAE plus model CPU offload. ai-toolkit-style deeper block/layer offload remains a future option for larger recipes or higher resolutions.

The same smoke path has also been verified through the UI/API runner:

- job id: `e7ddf6dc-520a-48b7-979f-de30081e78a3`
- exp name: `flux2_identity_20260621160921727`
- output: `trainer-data/smoke-runs/flux2_identity_20260621160921727`
- result: completed, step 1, loss `1.9178619384765625`, trainable params `983040`, runner exit code `0`
- observed peak VRAM was about 16054 MiB used on the RTX 4060 Ti 16GB, so do not raise the default smoke resolution without fresh OOM testing.

The verified FLUX2 Identity 0.3125-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res03125_1step_noartifact_202606221710 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3000 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res03125_1step_noartifact_202606221710`
- output: `trainer-data/smoke-runs/flux2_identity_res03125_1step_noartifact_202606221710`
- result: completed, step 1, latest loss `1.7702149152755737`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory back near idle

This verifies a small FLUX2 Identity no-artifact step above the 0.25 smoke baseline. It does not prove sample generation, checkpoint writing, longer training, or production-resolution training.

The verified FLUX2 Identity 0.3125-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res03125_ckpt_1step_202606221755 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 3000 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res03125_ckpt_1step_202606221755`
- output: `trainer-data/smoke-runs/flux2_identity_res03125_ckpt_1step_202606221755`
- result: completed, step 1, latest loss `1.7702149152755737`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Identity checkpoint writing at `0.3125` resolution on RTX 4060 Ti 16GB. It does not prove sample generation, longer training, or production-resolution training.

The verified FLUX2 Identity 0.3125-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res03125_sample_tiledinline_1step_202606221900 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 4200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res03125_sample_tiledinline_1step_202606221900`
- output: `trainer-data/smoke-runs/flux2_identity_res03125_sample_tiledinline_1step_202606221900`
- result: completed, step 1, latest loss `1.7823187112808228`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training logged `Tiled VAE enabled: tile_sample_min_size=64, tile_latent_min_size=8, tile_overlap_factor=0.25` before `Saved original sample images`, and the deferred final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully

This verifies FLUX2 Identity sample writing at `0.3125` resolution on RTX 4060 Ti 16GB after moving tiled VAE configuration into the main inline sample path. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.375-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0375_1step_noartifact_202606222050 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3600 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0375_1step_noartifact_202606222050`
- output: `trainer-data/smoke-runs/flux2_identity_res0375_1step_noartifact_202606222050`
- result: completed, step 1, latest loss `1.61357581615448`, trainable params `983040`
- args: `resolution_scale=0.375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `384x384` through `512x224`, and main training logged tiled VAE before training
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Identity no-artifact training at `0.375` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.375-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0375_ckpt_1step_202606222100 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 3600 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0375_ckpt_1step_202606222100`
- output: `trainer-data/smoke-runs/flux2_identity_res0375_ckpt_1step_202606222100`
- result: completed, step 1, latest loss `1.61357581615448`, trainable params `983040`
- args: `resolution_scale=0.375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Identity checkpoint writing at `0.375` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.375-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0375_sample_1step_tiledinline_202606222110 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 4200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0375_sample_1step_tiledinline_202606222110`
- output: `trainer-data/smoke-runs/flux2_identity_res0375_sample_1step_tiledinline_202606222110`
- result: completed, step 1, latest loss `1.61357581615448`, trainable params `983040`
- args: `resolution_scale=0.375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, deferred final sampling to a clean subprocess, then logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully

This verifies FLUX2 Identity sample writing at `0.375` resolution on RTX 4060 Ti 16GB. It does not prove longer training, higher-than-0.375 FLUX2 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.4375-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res04375_1step_noartifact_202606222250 RESOLUTION_SCALE=0.4375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 4200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res04375_1step_noartifact_202606222250`
- output: `trainer-data/smoke-runs/flux2_identity_res04375_1step_noartifact_202606222250`
- result: completed, step 1, latest loss `1.462178111076355`, trainable params `983040`
- args: `resolution_scale=0.4375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `448x448` through `592x256`, and main training logged tiled VAE before training
- post-run checks found no training process and GPU memory back near idle at about `494 MiB`

This verifies FLUX2 Identity no-artifact training at `0.4375` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.4375-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res04375_ckpt_1step_202606222300 RESOLUTION_SCALE=0.4375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 4200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res04375_ckpt_1step_202606222300`
- output: `trainer-data/smoke-runs/flux2_identity_res04375_ckpt_1step_202606222300`
- result: completed, step 1, latest loss `1.462178111076355`, trainable params `983040`
- args: `resolution_scale=0.4375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory back near idle at about `566 MiB`

This verifies FLUX2 Identity checkpoint writing at `0.4375` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.4375-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res04375_sample_1step_tiledinline_202606222310 RESOLUTION_SCALE=0.4375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 4800 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res04375_sample_1step_tiledinline_202606222310`
- output: `trainer-data/smoke-runs/flux2_identity_res04375_sample_1step_tiledinline_202606222310`
- result: completed, step 1, latest loss `1.462178111076355`, trainable params `983040`
- args: `resolution_scale=0.4375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, deferred final sampling to a clean subprocess, then logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory back near idle at about `832 MiB`

This verifies FLUX2 Identity sample writing at `0.4375` resolution as an intermediate boundary on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.5-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05_1step_noartifact_202606222335 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 4800 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05_1step_noartifact_202606222335`
- output: `trainer-data/smoke-runs/flux2_identity_res05_1step_noartifact_202606222335`
- result: completed, step 1, latest loss `1.3801677227020264`, trainable params `983040`
- args: `resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `512x512` through `672x288`, and main training logged tiled VAE before training
- post-run checks found no training process and GPU memory back near idle at about `487 MiB`

This verifies FLUX2 Identity no-artifact training at `0.5` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.5-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05_ckpt_1step_202606222345 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 4800 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05_ckpt_1step_202606222345`
- output: `trainer-data/smoke-runs/flux2_identity_res05_ckpt_1step_202606222345`
- result: completed, step 1, latest loss `1.3801677227020264`, trainable params `983040`
- args: `resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory near idle at about `599 MiB`

This verifies FLUX2 Identity checkpoint writing at `0.5` resolution on RTX 4060 Ti 16GB. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.5-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05_sample_1step_tiledinline_202606222355 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 5400 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05_sample_1step_tiledinline_202606222355`
- output: `trainer-data/smoke-runs/flux2_identity_res05_sample_1step_tiledinline_202606222355`
- result: completed, step 1, latest loss `1.3801677227020264`, trainable params `983040`
- args: `resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, deferred final sampling to a clean subprocess, then logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory back near idle at about `530 MiB`

This verifies FLUX2 Identity sample writing at `0.5` resolution on RTX 4060 Ti 16GB. It does not prove multi-step artifact writing, artifact writing above `0.5`, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.5-resolution two-step artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05_artifacts_2step_202606221900 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05_artifacts_2step_202606221900`
- output: `trainer-data/smoke-runs/flux2_identity_res05_artifacts_2step_202606221900`
- result: completed, step 2, latest loss `1.62534499168396`, trainable params `983040`
- args: `resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_2_student.png`, and `samples/samples_step_2_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, checkpoint files at step 2, deferred final sampling to a clean subprocess, then logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_2_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `613 MiB`

This verifies FLUX2 Identity two-step sample and checkpoint artifact writing at `0.5` resolution on RTX 4060 Ti 16GB. It does not prove `0.625` artifact writing, higher-than-0.625 FLUX2 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Identity 0.5625-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05625_1step_noartifact_202606221915 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 6000 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05625_1step_noartifact_202606221915`
- output: `trainer-data/smoke-runs/flux2_identity_res05625_1step_noartifact_202606221915`
- result: completed, step 1, latest loss `1.3002043962478638`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `576x576` through `752x320`, main training logged tiled VAE before training, low VRAM mode was enabled, and sample generation was disabled
- post-run checks found no training process and GPU memory near idle

This verifies FLUX2 Identity no-artifact training at `0.5625` resolution on RTX 4060 Ti 16GB. This no-artifact path itself does not prove sample generation, checkpoint writing, longer training, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.5625-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05625_ckpt_1step_202606221937 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 6000 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05625_ckpt_1step_202606221937`
- output: `trainer-data/smoke-runs/flux2_identity_res05625_ckpt_1step_202606221937`
- result: completed, step 1, latest loss `1.3002043962478638`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `576x576` through `752x320`, main training logged tiled VAE before training, sample generation was disabled, and training completed
- post-run checks found no training process and GPU memory near idle at about `617 MiB`

This verifies FLUX2 Identity checkpoint writing at `0.5625` resolution on RTX 4060 Ti 16GB. This checkpoint-only path does not prove sample writing, combined sample/checkpoint artifact writing, longer training, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.5625-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05625_sample_1step_tiledinline_202606221951 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05625_sample_1step_tiledinline_202606221951`
- output: `trainer-data/smoke-runs/flux2_identity_res05625_sample_1step_tiledinline_202606221951`
- result: completed, step 1, latest loss `1.3002043962478638`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `621 MiB`

This verifies FLUX2 Identity sample writing at `0.5625` resolution on RTX 4060 Ti 16GB. It does not prove combined sample/checkpoint artifact writing, longer training, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.5625-resolution combined artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res05625_artifacts_1step_202606222007 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res05625_artifacts_1step_202606222007`
- output: `trainer-data/smoke-runs/flux2_identity_res05625_artifacts_1step_202606222007`
- result: completed, step 1, latest loss `1.3002043962478638`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: main training wrote `samples_original.png`, checkpoint files at step 1, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `739 MiB`

This verifies FLUX2 Identity combined sample and checkpoint artifact writing at `0.5625` resolution on RTX 4060 Ti 16GB. It does not prove longer training, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Identity 0.625-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_1step_noartifact_202606222023 RESOLUTION_SCALE=0.625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_1step_noartifact_202606222023`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_1step_noartifact_202606222023`
- result: completed, step 1, latest loss `1.2559603452682495`, trainable params `983040`
- args: `resolution_scale=0.625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `640x640` through `832x352`, main training logged tiled VAE before training, low VRAM mode was enabled, sample generation was disabled, and training completed
- post-run checks found no training process and GPU memory near idle at about `631 MiB`

This verifies FLUX2 Identity no-artifact training at `0.625` resolution on RTX 4060 Ti 16GB. It does not prove sample generation, checkpoint writing, longer training, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.625-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_ckpt_1step_202606222045 RESOLUTION_SCALE=0.625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_ckpt_1step_202606222045`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_ckpt_1step_202606222045`
- result: completed, step 1, latest loss `1.2559603452682495`, trainable params `983040`
- args: `resolution_scale=0.625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `640x640` through `832x352`, main training logged tiled VAE before training, low VRAM mode was enabled, sample generation was disabled, trainable params were logged, and training completed
- post-run checks found no training process and GPU memory near idle at about `636 MiB`

This verifies FLUX2 Identity checkpoint writing at `0.625` resolution on RTX 4060 Ti 16GB. It does not prove sample writing, combined sample/checkpoint artifact writing, longer training, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples at this resolution.

The checked FLUX2 Identity 0.625-resolution sample path without inline sample scaling is not stable on this 16GB profile:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_sample_1step_tiledinline_202606222120 RESOLUTION_SCALE=0.625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_sample_1step_tiledinline_202606222120`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_sample_1step_tiledinline_202606222120`
- result: failed before step 1, latest step `0`, no loss points, trainable params `983040`
- failure reason: `RuntimeError: CUDA driver error: out of memory` while the initial inline original sample still used the full 0.625 sample size
- stack tail: OOM occurred in the FLUX2 transformer attention path at `transformer_flux2.py`, `torch.cat([hidden_states, mlp_hidden_states], dim=-1)`

Do not treat native 0.625 inline sample generation as stable on RTX 4060 Ti 16GB until a separate run proves it.

The verified FLUX2 Identity 0.625-resolution sample-writing path uses scaled inline preview samples:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_sample_scale05_1step_202606230105 RESOLUTION_SCALE=0.625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_sample_scale05_1step_202606230105`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_sample_scale05_1step_202606230105`
- result: completed, step 1, latest loss `1.2559603452682495`, trainable params `983040`
- args: `resolution_scale=0.625`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training logged `Inline sample resolution scaled to 320x320`, wrote `samples_original.png`, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `623 MiB`
- verification checks: targeted red/green sample-scale test passed, Python unittest discovery ran 45 tests OK, `py_compile` passed for FLUX2 Identity training/sampler files and tests, WSL `bash -n` passed for all three smoke scripts, `npm run typecheck --prefix trainer-ui` passed, `npm run build --prefix trainer-ui` passed, `git diff --check` passed, and the restarted UI on port 8675 returned healthy `/api/project` and `/api/models` responses

This verifies FLUX2 Identity sample writing at `0.625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. This sample-only run did not include checkpoint writing; the combined sample/checkpoint path is separately verified below. It does not prove native full-size inline sample generation, longer training, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.625-resolution combined artifact-writing path uses the same scaled inline preview setting:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_artifacts_scale05_1step_202606230140 RESOLUTION_SCALE=0.625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_artifacts_scale05_1step_202606230140`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_artifacts_scale05_1step_202606230140`
- result: completed, step 1, latest loss `1.2559603452682495`, trainable params `983040`
- args: `resolution_scale=0.625`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png` (`41438` bytes), `samples/samples_step_1_student.png` (`143147` bytes), and `samples/samples_step_1_teacher.png` (`157922` bytes)
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: main training logged `Inline sample resolution scaled to 320x320`, wrote `samples_original.png`, saved step 1 checkpoints, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `881 MiB`
- verification checks: summary assertion matched `flux2_identity_res0625_artifacts_scale05_1step_202606230140` exactly, visual inspection opened `samples/samples_step_1_student.png`, Python unittest discovery ran 45 tests OK, `py_compile` passed for FLUX2 Identity training/sampler files and tests, WSL `bash -n` passed for all three smoke scripts, JSON validation passed for `features.json` and `meta.json`, `npm run typecheck --prefix trainer-ui` passed, `npm run build --prefix trainer-ui` passed, `git diff --check` passed before this verification note was added, and the launcher restarted port 8675 with healthy `/api/project` plus `/api/models` responses

This verifies FLUX2 Identity one-step combined sample and checkpoint artifact writing at `0.625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Two-step, three-step, and five-step artifact writing are separately verified below. It does not prove native full-size inline sample generation, artifact writing beyond five steps, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.625-resolution two-step combined artifact-writing path uses the same scaled inline preview setting:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_artifacts_scale05_2step_20260622221738 RESOLUTION_SCALE=0.625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_artifacts_scale05_2step_20260622221738`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_artifacts_scale05_2step_20260622221738`
- result: completed, step 2, latest loss `1.4668035507202148`, trainable params `983040`
- args: `resolution_scale=0.625`, `sample_resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png`, `samples/samples_step_2_student.png`, and `samples/samples_step_2_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: main training logged `Inline sample resolution scaled to 320x320`, wrote `samples_original.png`, saved step 2 checkpoints, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_2_student.png` opened successfully
- post-run checks found no training process and GPU memory near idle at about `612 MiB`
- verification checks: summary assertion matched `flux2_identity_res0625_artifacts_scale05_2step_20260622221738` exactly, visual inspection opened `samples/samples_step_2_student.png`, Python unittest discovery ran 47 tests OK, `py_compile` passed for both FLUX2 training scripts and tests, WSL `bash -n` passed for all three smoke scripts, JSON validation passed for `features.json` and `meta.json`, `npm run typecheck --prefix trainer-ui` passed, `npm run build --prefix trainer-ui` passed, the launcher restarted port 8675 in `-Wait` mode with healthy `/api/project`, `scripts/check_ui_smoke.ps1` returned `Ok True`, `git diff --check` passed, and final WSL/GPU checks showed no training process with GPU memory near idle at about `672 MiB`

This verifies FLUX2 Identity two-step combined sample and checkpoint artifact writing at `0.625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Three-step and five-step artifact writing are separately verified below. It does not prove native full-size inline sample generation, artifact writing beyond five steps, higher-than-0.625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.625-resolution three-step combined artifact-writing path uses the same scaled inline preview setting:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_artifacts_scale05_3step_20260623004403 RESOLUTION_SCALE=0.625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=3 SAVE_SAMPLES=1 SAMPLE_STEPS=3 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=3 BLOCK_OFFLOAD=1 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_artifacts_scale05_3step_20260623004403`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_artifacts_scale05_3step_20260623004403`
- result: completed, step 3, latest loss `1.170734167098999`, trainable params `983040`
- loss points: step 1 `1.2559603452682495`, step 2 `1.4681332111358643`, step 3 `1.170734167098999`
- args: `resolution_scale=0.625`, `sample_resolution_scale=0.5`, `max_train_steps=3`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=3`, `save_checkpoints=true`, `checkpoint_steps=3`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- sample files: `samples/samples_original.png` (`41438` bytes), `samples/samples_step_3_student.png` (`142790` bytes), and `samples/samples_step_3_teacher.png` (`158002` bytes)
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_3/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_3/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, 3 loss points, and no failure reason
- log evidence: main training logged `Inline sample resolution scaled to 320x320`, wrote `samples_original.png`, saved step 3 checkpoints, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples
- visual check: `samples/samples_step_3_student.png` opened successfully
- post-run checks found no D-OPSD training or final-sampler process; `nvidia-smi` showed about `2600 MiB / 16380 MiB` used after exit, with no D-OPSD process still attached
- verification checks: summary assertion matched `flux2_identity_res0625_artifacts_scale05_3step_20260623004403` exactly, visual inspection opened `samples/samples_step_3_student.png`, output listing found the expected sample, trajectory, and checkpoint files, Python unittest discovery ran 51 tests OK, `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests, WSL `bash -n` passed for all three smoke scripts, JSON validation passed for `features.json` and `meta.json`, `npm run typecheck --prefix trainer-ui` passed, `scripts/check_ui_smoke.ps1` returned `Ok True`, `git diff --check` passed, trailing whitespace checks passed, and final WSL/GPU checks showed no D-OPSD training process with GPU memory at about `2607 MiB / 16380 MiB`

This verifies FLUX2 Identity three-step combined sample and checkpoint artifact writing at `0.625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The five-step artifact path is separately verified below. It does not prove native full-size inline sample generation, artifact writing beyond five steps, higher-than-0.625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples at this resolution.

The verified FLUX2 Identity 0.625-resolution five-step combined artifact-writing path uses the same scaled inline preview setting:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_identity_res0625_artifacts_scale05_5step_20260623045623 RESOLUTION_SCALE=0.625 SAMPLE_RESOLUTION_SCALE=0.5 FINAL_SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=5 SAVE_SAMPLES=1 SAMPLE_STEPS=5 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=5 BLOCK_OFFLOAD=1 timeout 9000 bash scripts/run_flux2_smoke.sh"
```

- exp name: `flux2_identity_res0625_artifacts_scale05_5step_20260623045623`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_artifacts_scale05_5step_20260623045623`
- result: completed, step 5, latest loss `1.2803046703338623`, trainable params `983040`
- loss points: five points, ending at epoch `1`, grad norm `2.467148542404175`
- args: `resolution_scale=0.625`, `sample_resolution_scale=0.5`, `max_train_steps=5`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=5`, `save_checkpoints=true`, `checkpoint_steps=5`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `final_sampler_cpu_offload=true`, `final_sample_resolution_scale=0.5`, `block_offload=true`
- final sampler request: `requested_block_offload=true` but `block_offload=false`, so the verified path stayed on tiled VAE plus model CPU offload rather than the experimental Diffusers group hook
- sample files: `samples/samples_original.png` (`41438` bytes), `samples/samples_step_5_student.png` (`143027` bytes), and `samples/samples_step_5_teacher.png` (`158004` bytes)
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_5/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_5/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 5 checkpoint files, 5 loss points, and no failure reason
- log evidence: main training logged `Inline sample resolution scaled to 320x320`, wrote `samples_original.png`, saved step 5 checkpoints, deferred final sampling to a clean subprocess, then the final sampler logged tiled VAE plus model CPU offload before writing final student/teacher samples and `Training completed.`
- visual check: `samples/samples_step_5_student.png` opened successfully
- post-run checks found no D-OPSD training or final-sampler process; `nvidia-smi` showed about `1804 MiB / 16380 MiB` used after exit
- verification checks: summary assertion matched completed status/latest step 5/latest loss `1.2803046703338623`/trainable params `983040`/five loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason, `args.json` and final sampler request matched the scaled sample/tiled VAE/CPU offload settings, Python unittest discovery ran 51 tests OK, `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests, WSL `bash -n` passed for all three smoke scripts, `npm run typecheck --prefix trainer-ui` passed, `scripts/check_ui_smoke.ps1` against 8675 returned `Ok True` with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and DatasetReady 3, JSON validation passed for `features.json` and `meta.json`, `git diff --check` passed, trailing whitespace checks passed, and exact stale-boundary text search found no matches

This verifies FLUX2 Identity five-step combined sample and checkpoint artifact writing at `0.625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. It does not prove native full-size inline sample generation, artifact writing beyond five steps, higher-than-0.625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples at this resolution.

The checked FLUX2 Identity 0.625-resolution five-step no-artifact path initially exposed a longer-run prompt-encoding OOM:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_5step_noartifact_20260622215608 RESOLUTION_SCALE=0.625 MAX_TRAIN_STEPS=5 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- result: failed after step 2 with latest loss `1.4672938585281372`
- failure reason: `RuntimeError: CUDA driver error: out of memory` inside Qwen3 `lm_head` during `_encode_prompt`
- root cause: FLUX2 prompt encoding only needs selected hidden states, but Qwen3 CausalLM was still computing logits for the full 512-token sequence
- fix: both FLUX2 `_encode_prompt` implementations now call Qwen3 with `logits_to_keep=1` and delete the CausalLM output/input tensors after hidden-state extraction
- test: `trainer_runtime.tests.test_runtime.RuntimeTests.test_flux2_prompt_encoding_limits_qwen_logits_memory` failed before the guard existed and passed after implementation

The verified FLUX2 Identity 0.625-resolution five-step no-artifact path after the prompt logits memory fix is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_identity_res0625_5step_noartifact_logits1_20260622220405 RESOLUTION_SCALE=0.625 MAX_TRAIN_STEPS=5 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_smoke.sh
```

- exp name: `flux2_identity_res0625_5step_noartifact_logits1_20260622220405`
- output: `trainer-data/smoke-runs/flux2_identity_res0625_5step_noartifact_logits1_20260622220405`
- result: completed, step 5, latest loss `1.2792584896087646`, trainable params `983040`
- loss points: five points, ending at epoch `1`, grad norm `2.580416679382324`
- args: `resolution_scale=0.625`, `max_train_steps=5`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `640x640` through `832x352`, main training logged tiled VAE before training, low VRAM mode was enabled, sample generation was disabled, and `Training completed.`
- post-run checks found no training process and GPU memory near idle at about `621 MiB`

This verifies FLUX2 Identity `0.625` no-artifact training beyond a single step on RTX 4060 Ti 16GB after the Qwen logits memory fix. It does not prove native full-size inline sample generation, sample/checkpoint artifact writing beyond the separately verified five-step run, higher-than-0.625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 low-VRAM checkpoint path is:

```bash
# Manual direct-Python run based on scripts/run_flux2_smoke.sh with --checkpoint-steps 1,
# sample generation disabled, and checkpoint saving enabled.
```

- exp name: `flux2_identity_ckpt_smoke_1step_lowvram_20260622023426`
- output: `trainer-data/smoke-runs/flux2_identity_ckpt_smoke_1step_lowvram_20260622023426`
- result: completed, step 1, loss `1.9178619384765625`, trainable params `983040`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`

This proves real FLUX2 low-VRAM checkpoint writing. It does not prove sample generation or production-resolution training.

The verified FLUX2 low-VRAM sample path is:

```bash
# Manual direct-Python run based on scripts/run_flux2_smoke.sh with --sample-steps 1,
# checkpoint saving disabled, and sample saving enabled.
```

- exp name: `flux2_identity_sample_smoke_1step_lowvram_devicefix_20260622031210`
- output: `trainer-data/smoke-runs/flux2_identity_sample_smoke_1step_lowvram_devicefix_20260622031210`
- result: completed, step 1, loss `1.9178619384765625`, trainable params `983040`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: `samples_trajectory/t0/step_1_student_teacher_x0.png`, `samples_trajectory/t0/one_img/step_1_student_single.png`, and `samples_trajectory/t0/one_img/step_1_teacher_single.png`
- parser result: `summarize_run` reported 3 sample files, 3 sample trajectory files, and completed status

This proves real FLUX2 low-VRAM sample generation at the smoke profile. It does not prove production-resolution training or sample quality at production settings.

The verified FLUX2 Editing low-VRAM smoke path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_smoke_1step_lowvram_python_20260622043148 timeout 1800 bash scripts/run_flux2_editing_smoke.sh"
```

This path uses:

- direct `python train_dopsd.py` launch,
- local FLUX2 4B snapshot,
- `--low-vram`,
- `--resolution-scale 0.25`,
- `--use-8bit-adam`,
- bf16,
- batch size 1,
- LoRA rank 4,
- `--no-save-samples`,
- `--no-save-checkpoints`.

- exp name: `flux2_editing_smoke_1step_lowvram_python_20260622043148`
- output: `trainer-data/smoke-runs/flux2_editing_smoke_1step_lowvram_python_20260622043148`
- result: completed, step 1, loss `0.08440777659416199`, trainable params `983040`
- parser result: `summarize_run` reported 0 sample files, 0 sample trajectory files, 0 checkpoint files, and completed status

The same FLUX2 Editing smoke path has also been verified through the UI/API runner:

- job id: `f4b2f5aa-216e-4974-8505-30f00e6b5c4c`
- exp name: `flux2_editing_20260621204242821`
- output: `trainer-data/smoke-runs/flux2_editing_20260621204242821`
- result: completed, step 1, loss `0.08440777659416199`, trainable params `983040`, runner exit code `0`

This proves FLUX2 Editing no-sample/no-checkpoint low-VRAM training at the smoke profile. The separate checkpoint and sample paths below cover artifact writing; this no-sample path still does not prove production-resolution training or sample quality at production settings.

The verified FLUX2 Editing low-VRAM checkpoint path is:

```bash
# Manual direct-Python run based on scripts/run_flux2_editing_smoke.sh with --checkpoint-steps 1,
# sample generation disabled, and checkpoint saving enabled.
```

- exp name: `flux2_editing_ckpt_smoke_1step_lowvram_20260622050819`
- output: `trainer-data/smoke-runs/flux2_editing_ckpt_smoke_1step_lowvram_20260622050819`
- result: completed, step 1, loss `0.08440777659416199`, trainable params `983040`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported 0 sample files, 0 sample trajectory files, 5 checkpoint files, and completed status

This proves real FLUX2 Editing low-VRAM checkpoint writing. It does not prove production-resolution training.

The verified FLUX2 Editing low-VRAM sample path is:

```bash
# Manual direct-Python run based on scripts/run_flux2_editing_smoke.sh with --sample-steps 1,
# checkpoint saving disabled, sample saving enabled, resized condition images, and pre-sampling tensor cleanup.
```

- exp name: `flux2_editing_sample_smoke_1step_lowvram_release_20260622055912`
- output: `trainer-data/smoke-runs/flux2_editing_sample_smoke_1step_lowvram_release_20260622055912`
- result: completed, step 1, loss `0.12728287279605865`, trainable params `983040`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: `samples_trajectory/t0/step_1_student_teacher_x0.png`, `samples_trajectory/t0/one_img/step_1_student_single.png`, and `samples_trajectory/t0/one_img/step_1_teacher_single.png`
- parser result: `summarize_run` reported 3 sample files, 3 sample trajectory files, 0 checkpoint files, and completed status
- visual check: `samples/samples_step_1_student.png` opened successfully

This proves real FLUX2 Editing low-VRAM sample generation at the smoke profile. It does not prove production-resolution training or production-quality samples.

The verified FLUX2 Editing 0.3125-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res03125_1step_noartifact_202606221930 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3000 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res03125_1step_noartifact_202606221930`
- output: `trainer-data/smoke-runs/flux2_editing_res03125_1step_noartifact_202606221930`
- result: completed, step 1, latest loss `0.21959611773490906`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training logged `Tiled VAE enabled: tile_sample_min_size=64, tile_latent_min_size=8, tile_overlap_factor=0.25`
- post-run checks found no training process and GPU memory back near idle

This verifies a small FLUX2 Editing no-artifact step above the 0.25 smoke baseline. It does not prove checkpoint writing, sample generation, longer training, or production-resolution training.

The verified FLUX2 Editing 0.3125-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res03125_ckpt_1step_tiledvae_202606221950 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 3000 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res03125_ckpt_1step_tiledvae_202606221950`
- output: `trainer-data/smoke-runs/flux2_editing_res03125_ckpt_1step_tiledvae_202606221950`
- result: completed, step 1, latest loss `0.21959611773490906`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Editing checkpoint writing at `0.3125` resolution on RTX 4060 Ti 16GB. It does not prove sample generation, longer training, or production-resolution training.

The verified FLUX2 Editing 0.3125-resolution sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res03125_sample_1step_tiledvae_202606221940 RESOLUTION_SCALE=0.3125 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 timeout 4200 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res03125_sample_1step_tiledvae_202606221940`
- output: `trainer-data/smoke-runs/flux2_editing_res03125_sample_1step_tiledvae_202606221940`
- result: completed, step 1, latest loss `0.1783905029296875`, trainable params `983040`
- args: `resolution_scale=0.3125`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: main training logged `Tiled VAE enabled: tile_sample_min_size=64, tile_latent_min_size=8, tile_overlap_factor=0.25` before sample images were written
- visual check: `samples/samples_step_1_student.png` opened successfully

This verifies FLUX2 Editing sample writing at `0.3125` resolution on RTX 4060 Ti 16GB after adding tiled VAE configuration to the main inline sample path. It does not prove longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.375-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_1step_noartifact_20260622233102 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3600 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_1step_noartifact_20260622233102`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_1step_noartifact_20260622233102`
- result: completed, step 1, latest loss `0.12125635147094727`, trainable params `983040`
- args: `resolution_scale=0.375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `384x384` through `512x224`, tiled VAE was enabled before training, low VRAM mode was enabled, and sample generation was disabled
- observed during polling: GPU memory reached about `15951 MiB / 16380 MiB`, then returned near idle after exit
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Editing no-artifact training at `0.375` resolution on RTX 4060 Ti 16GB. It does not prove checkpoint writing, sample generation, longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.375-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_ckpt_1step_tiledvae_20260622234218 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 3600 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_ckpt_1step_tiledvae_20260622234218`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_ckpt_1step_tiledvae_20260622234218`
- result: completed, step 1, latest loss `0.12125635147094727`, trainable params `983040`
- args: `resolution_scale=0.375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- observed during polling: GPU memory reached about `15997 MiB / 16380 MiB`, then returned near idle after exit

This verifies FLUX2 Editing checkpoint writing at `0.375` resolution on RTX 4060 Ti 16GB. It does not prove native sample generation, combined sample/checkpoint writing, longer training, full `1.0` production-resolution training, or production-quality samples.

The native FLUX2 Editing 0.375-resolution sample-writing path is not stable on this 16GB profile:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_sample_1step_tiledvae_20260622234826 RESOLUTION_SCALE=0.375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 timeout 4200 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_sample_1step_tiledvae_20260622234826`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_sample_1step_tiledvae_20260622234826`
- result: failed after step 1, latest loss `0.12772822380065918`, trainable params `983040`
- partial artifacts: `samples/samples_original.png` plus 3 sample trajectory PNGs
- failure reason after copying stderr to the parser-visible runner log: `RuntimeError: CUDA driver error: out of memory`
- failure location: final teacher/student sample transformer inference at Diffusers FLUX2 rotary embedding, after the initial tiled-VAE sample decode had succeeded

Do not use native full-size 0.375 Editing sample generation as a stable 16GB default. Use the scaled sample path below.

The verified FLUX2 Editing 0.375-resolution scaled sample-writing path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_sample_scale05_1step_20260622235743 RESOLUTION_SCALE=0.375 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 timeout 4200 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_sample_scale05_1step_20260622235743`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_sample_scale05_1step_20260622235743`
- result: completed, step 1, latest loss `0.12772822380065918`, trainable params `983040`
- args: `resolution_scale=0.375`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: `Inline sample resolution scaled to 192x192`, `Saved original sample images`, `Saved sample images`, and `Training completed`
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run checks found no training process and GPU memory back near idle

This verifies FLUX2 Editing sample writing at `0.375` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. It does not prove native full-size sample generation, longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.375-resolution combined sample/checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_artifacts_scale05_1step_20260623001340 RESOLUTION_SCALE=0.375 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 4200 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_artifacts_scale05_1step_20260623001340`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_artifacts_scale05_1step_20260623001340`
- result: completed, step 1, latest loss `0.12772822380065918`, trainable params `983040`
- args: `resolution_scale=0.375`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.12772822380065918`, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: `Inline sample resolution scaled to 192x192`, `Saved original sample images`, `Saved sample images`, and `Training completed`
- visual check: `samples/samples_step_1_student.png` opened successfully
- observed during polling: GPU memory reached about `16082 MiB / 16380 MiB`, then returned near idle after exit

This verifies FLUX2 Editing combined sample/checkpoint writing at `0.375` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. It does not prove native full-size sample generation, longer training, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.375-resolution two-step combined sample/checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res0375_artifacts_scale05_2step_20260623002814 RESOLUTION_SCALE=0.375 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 timeout 5400 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res0375_artifacts_scale05_2step_20260623002814`
- output: `trainer-data/smoke-runs/flux2_editing_res0375_artifacts_scale05_2step_20260623002814`
- result: completed, step 2, latest loss `0.1275705248117447`, trainable params `983040`
- args: `resolution_scale=0.375`, `sample_resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- loss points: step 1 loss `0.12772822380065918`, step 2 loss `0.1275705248117447`
- sample files: `samples/samples_original.png`, `samples/samples_step_2_student.png`, and `samples/samples_step_2_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 2, latest loss `0.1275705248117447`, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: `Inline sample resolution scaled to 192x192`, `Saved original sample images`, `Saved sample images`, and `Training completed`
- visual check: `samples/samples_step_2_student.png` opened successfully
- observed during polling: GPU memory reached about `16064 MiB / 16380 MiB`, then returned near idle after exit

This verifies FLUX2 Editing two-step combined sample/checkpoint writing at `0.375` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. A `0.4375` artifact boundary, `0.5` artifact boundary, and `0.5625` no-artifact boundary are separately verified below. It does not prove native full-size sample generation, `0.5625` sample/checkpoint artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.4375-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res04375_1step_noartifact_20260623010249 RESOLUTION_SCALE=0.4375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 5400 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res04375_1step_noartifact_20260623010249`
- output: `trainer-data/smoke-runs/flux2_editing_res04375_1step_noartifact_20260623010249`
- result: completed, step 1, latest loss `0.11627798527479172`, trainable params `983040`
- args: `resolution_scale=0.4375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- log evidence: training bucket resolutions scaled up to `448x448` through `592x256`, tiled VAE was enabled before training, low VRAM mode was enabled, sample generation was disabled, and `Training completed`

This verifies FLUX2 Editing no-artifact training at `0.4375` resolution on RTX 4060 Ti 16GB. Artifact writing at `0.4375`, `0.5` artifact writing, and no-artifact training at `0.5625` are separately verified below. It does not prove native sample generation, longer training, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.4375-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res04375_ckpt_1step_20260623010857 RESOLUTION_SCALE=0.4375 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 5400 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res04375_ckpt_1step_20260623010857`
- output: `trainer-data/smoke-runs/flux2_editing_res04375_ckpt_1step_20260623010857`
- result: completed, step 1, latest loss `0.11627798527479172`, trainable params `983040`
- args: `resolution_scale=0.4375`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- observed during polling: GPU memory reached about `16047 MiB / 16380 MiB`, then returned near idle after exit

This verifies FLUX2 Editing checkpoint writing at `0.4375` resolution on RTX 4060 Ti 16GB. Combined sample/checkpoint writing, `0.5` artifact writing, and a `0.5625` no-artifact training boundary are separately verified below. It does not prove native sample generation, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.4375-resolution scaled combined sample/checkpoint path is:

```bash
wsl -d Ubuntu-22.04 --cd /mnt/e/D-opsd-T8-Tranier -- env EXP_NAME=flux2_editing_res04375_artifacts_scale05_1step_20260623011509 RESOLUTION_SCALE=0.4375 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 6000 bash scripts/run_flux2_editing_smoke.sh
```

- exp name: `flux2_editing_res04375_artifacts_scale05_1step_20260623011509`
- output: `trainer-data/smoke-runs/flux2_editing_res04375_artifacts_scale05_1step_20260623011509`
- result: completed, step 1, latest loss `0.12953020632266998`, trainable params `983040`
- args: `resolution_scale=0.4375`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png` (`22265` bytes), `samples/samples_step_1_student.png` (`22189` bytes), and `samples/samples_step_1_teacher.png` (`22122` bytes)
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` (`3942832` bytes) and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` (`1976792` bytes)
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.12953020632266998`, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: `Inline sample resolution scaled to 224x224`, `Saved original sample images`, `Saved sample images`, and `Training completed`
- visual check: `samples/samples_step_1_student.png` opened successfully
- observed during polling: GPU memory reached about `16078 MiB / 16380 MiB`, then returned near idle after exit
- verification checks: summary assertions matched the 0.4375 no-artifact, checkpoint, and scaled combined artifact runs exactly, visual inspection opened `samples/samples_step_1_student.png`, Python unittest discovery ran 51 tests OK, `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests, WSL `bash -n` passed for all three smoke scripts, JSON validation passed for `features.json` and `meta.json`, `npm run typecheck --prefix trainer-ui` passed, `scripts/check_ui_smoke.ps1` returned `Ok True`, `git diff --check` passed, trailing whitespace checks passed, and final WSL/GPU checks showed no D-OPSD training process with GPU memory at about `2101 MiB / 16380 MiB`

This verifies FLUX2 Editing one-step combined sample/checkpoint writing at `0.4375` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Two-step artifact writing, `0.5` artifact writing, and a `0.5625` no-artifact training boundary are separately verified below. It does not prove native full-size sample generation, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.4375-resolution two-step scaled combined sample/checkpoint path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res04375_artifacts_scale05_2step_20260623012929 RESOLUTION_SCALE=0.4375 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res04375_artifacts_scale05_2step_20260623012929`
- output: `trainer-data/smoke-runs/flux2_editing_res04375_artifacts_scale05_2step_20260623012929`
- result: completed, step 2, latest loss `0.11018239706754684`, trainable params `983040`
- loss points: step 1 loss `0.12953020632266998` with grad norm `2.6287155151367188`; step 2 loss `0.11018239706754684` with grad norm `0.4161382019519806`
- args: `resolution_scale=0.4375`, `sample_resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- sample files: `samples/samples_original.png` (`22265` bytes), `samples/samples_step_2_student.png` (`22182` bytes), and `samples/samples_step_2_teacher.png` (`22155` bytes)
- trajectory files: `samples_trajectory/t0/step_2_student_teacher_x0.png` (`494297` bytes), `samples_trajectory/t0/one_img/step_2_student_single.png` (`246116` bytes), and `samples_trajectory/t0/one_img/step_2_teacher_single.png` (`245715` bytes)
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` (`3942832` bytes), `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors` (`1976792` bytes), student/teacher adapter config JSON files, and `checkpoints/lora_gen_step_2/README.md`
- parser result: `summarize_run` reported completed status, latest step 2, latest loss `0.11018239706754684`, two loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- log evidence: tiled VAE was enabled, low VRAM mode was enabled, bucket resolutions scaled up to `448x448` through `592x256`, inline samples were scaled to `224x224`, original samples were saved, step 2 sample images were saved, and training completed
- visual check: `samples/samples_step_2_student.png` opened successfully
- observed during polling: GPU memory reached about `16087 MiB / 16380 MiB`; a post-run check found no D-OPSD training process and GPU memory around `1984 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 2/latest loss `0.11018239706754684`/trainable params `983040`/two loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason, Python unittest discovery ran 51 tests OK, `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests, WSL `bash -n` passed for all three smoke scripts, JSON validation passed for `features.json` and `meta.json`, `npm run typecheck --prefix trainer-ui` passed, `scripts/check_ui_smoke.ps1` returned `Ok True`, `git diff --check` passed, trailing whitespace checks passed, and final WSL/GPU checks showed no D-OPSD training process with GPU memory at about `2068 MiB / 16380 MiB`

This verifies FLUX2 Editing two-step combined sample/checkpoint writing at `0.4375` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The `0.5` artifact boundaries and `0.5625` no-artifact boundary are separately verified below. It does not prove native full-size sample generation, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05_1step_noartifact_20260623014725 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05_1step_noartifact_20260623014725`
- output: `trainer-data/smoke-runs/flux2_editing_res05_1step_noartifact_20260623014725`
- result: completed, step 1, latest loss `0.12597878277301788`, trainable params `983040`
- args: `resolution_scale=0.5`, `sample_resolution_scale=1.0`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `sample_steps=999`, `save_checkpoints=false`, `checkpoint_steps=999`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.12597878277301788`, one loss point, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- loss point: step 1 loss `0.12597878277301788`, grad norm `0.6071614623069763`
- log evidence: tiled VAE was enabled, the dataset contained 16 samples, bucket resolutions scaled up to `512x512` through `672x288`, low VRAM mode was enabled, sample generation was disabled, trainable params were `983040`, and training completed
- post-run check: no D-OPSD training process remained, and GPU memory was about `1390 MiB / 16380 MiB`
- verification checks: summary assertions matched completed status, latest step `1`, latest loss `0.12597878277301788`, trainable params `983040`, one loss point, zero sample files, zero sample trajectory files, zero checkpoint files, and no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; JSON validation passed for `features.json` and `meta.json`; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true for the 8675 launcher UI; `git diff --check` and trailing whitespace checks passed; final WSL/GPU checks showed no D-OPSD training process with GPU memory about `2102 MiB / 16380 MiB`

This verifies FLUX2 Editing one-step no-artifact training at `0.5` resolution on RTX 4060 Ti 16GB. One-step/two-step/three-step/five-step scaled sample/checkpoint artifact writing at the same training resolution and a `0.5625` no-artifact boundary are separately verified below. It does not prove `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.5-resolution one-step scaled combined artifact path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05_artifacts_scale05_1step_20260623020147 RESOLUTION_SCALE=0.5 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05_artifacts_scale05_1step_20260623020147`
- output: `trainer-data/smoke-runs/flux2_editing_res05_artifacts_scale05_1step_20260623020147`
- result: completed, step 1, latest loss `0.11434215307235718`, trainable params `983040`
- args: `resolution_scale=0.5`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.11434215307235718`, one loss point, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` with size `3942832` bytes and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors` with size `1976792` bytes
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `512x512` through `672x288`, low VRAM mode was enabled, inline sample resolution was scaled to `256x256`, original/sample images were saved, and training completed
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1494 MiB / 16380 MiB`
- verification checks: summary assertions matched completed status, latest step `1`, latest loss `0.11434215307235718`, trainable params `983040`, one loss point, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts after individual reruns; JSON validation passed for `features.json` and `meta.json`; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true for the 8675 launcher UI; `git diff --check` and trailing whitespace checks passed; final WSL/GPU checks showed no D-OPSD training process with GPU memory about `1528 MiB / 16380 MiB`

This verifies FLUX2 Editing one-step sample/checkpoint artifact writing at `0.5` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Two-step, three-step, five-step scaled sample/checkpoint artifact writing at the same training resolution, and `0.5625` no-artifact training are separately verified below. It does not prove native full-size sample generation, artifact writing beyond five steps, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5-resolution two-step scaled combined artifact path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05_artifacts_scale05_2step_20260623021508 RESOLUTION_SCALE=0.5 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05_artifacts_scale05_2step_20260623021508`
- output: `trainer-data/smoke-runs/flux2_editing_res05_artifacts_scale05_2step_20260623021508`
- result: completed, step 2, latest loss `0.12089794874191284`, trainable params `983040`
- args: `resolution_scale=0.5`, `sample_resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- parser result: `summarize_run` reported completed status, latest step 2, latest loss `0.12089794874191284`, two loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11434215307235718` with grad norm `0.479538232088089`; step 2 loss `0.12089794874191284` with grad norm `1.12236487865448`
- sample files: `samples/samples_original.png` (`27985` bytes), `samples/samples_step_2_student.png` (`27861` bytes), and `samples/samples_step_2_teacher.png` (`27259` bytes)
- trajectory files: `samples_trajectory/t0/step_2_student_teacher_x0.png` (`687485` bytes), `samples_trajectory/t0/one_img/step_2_student_single.png` (`340601` bytes), and `samples_trajectory/t0/one_img/step_2_teacher_single.png` (`345108` bytes)
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` with size `3942832` bytes, `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors` with size `1976792` bytes, both `adapter_config.json` files, and `checkpoints/lora_gen_step_2/README.md`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `512x512` through `672x288`, low VRAM mode was enabled, inline sample resolution was scaled to `256x256`, original/sample images were saved at step 2, and training completed
- visual check: `samples/samples_step_2_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1158 MiB / 16380 MiB`
- verification checks: summary assertions matched completed status, latest step `2`, latest loss `0.12089794874191284`, trainable params `983040`, two loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; JSON validation passed for `features.json` and `meta.json`; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true for the 8675 launcher UI; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes, with GPU memory about `1808 MiB / 16380 MiB`

This verifies FLUX2 Editing two-step sample/checkpoint artifact writing at `0.5` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Three-step, five-step scaled sample/checkpoint artifact writing at the same training resolution, and `0.5625` no-artifact training are separately verified below. It does not prove native full-size sample generation, artifact writing beyond five steps, `0.5625` artifact writing, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5-resolution three-step scaled combined artifact path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05_artifacts_scale05_3step_20260623023254 RESOLUTION_SCALE=0.5 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=3 SAVE_SAMPLES=1 SAMPLE_STEPS=3 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=3 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05_artifacts_scale05_3step_20260623023254`
- output: `trainer-data/smoke-runs/flux2_editing_res05_artifacts_scale05_3step_20260623023254`
- result: completed, step 3, latest loss `0.10782467573881149`, trainable params `983040`
- args: `resolution_scale=0.5`, `sample_resolution_scale=0.5`, `max_train_steps=3`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=3`, `save_checkpoints=true`, `checkpoint_steps=3`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- parser result: `summarize_run` reported completed status, latest step 3, latest loss `0.10782467573881149`, three loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11434215307235718` with grad norm `0.4776155948638916`; step 2 loss `0.1210726648569107` with grad norm `1.1354470252990723`; step 3 loss `0.10782467573881149` with grad norm `0.9549729824066162`
- sample files: `samples/samples_original.png` (`27985` bytes), `samples/samples_step_3_student.png` (`27867` bytes), and `samples/samples_step_3_teacher.png` (`27263` bytes)
- trajectory files: `samples_trajectory/t0/step_3_student_teacher_x0.png` (`849437` bytes), `samples_trajectory/t0/one_img/step_3_student_single.png` (`421278` bytes), and `samples_trajectory/t0/one_img/step_3_teacher_single.png` (`426082` bytes)
- checkpoint files: `checkpoints/lora_gen_step_3/student/adapter_model.safetensors` with size `3942832` bytes, `checkpoints/lora_gen_step_3/teacher/adapter_model.safetensors` with size `1976792` bytes, both `adapter_config.json` files, and `checkpoints/lora_gen_step_3/README.md`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `512x512` through `672x288`, low VRAM mode was enabled, inline sample resolution was scaled to `256x256`, original/sample images were saved at step 3, and training completed
- visual check: `samples/samples_step_3_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1162 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 3/latest loss `0.10782467573881149`/trainable params `983040`/three loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; JSON validation passed for `features.json` and `meta.json`; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training process and GPU memory was about `2148 MiB / 16380 MiB`

This verifies FLUX2 Editing three-step sample/checkpoint artifact writing at `0.5` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. Five-step scaled sample/checkpoint artifact writing at the same training resolution is separately verified below. It does not prove native full-size sample generation, artifact writing beyond five steps, `0.5625` artifact writing, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5-resolution five-step scaled combined artifact path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05_artifacts_scale05_5step_20260623025143 RESOLUTION_SCALE=0.5 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=5 SAVE_SAMPLES=1 SAMPLE_STEPS=5 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=5 timeout 9000 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05_artifacts_scale05_5step_20260623025143`
- output: `trainer-data/smoke-runs/flux2_editing_res05_artifacts_scale05_5step_20260623025143`
- result: completed, step 5, latest loss `0.1243741437792778`, trainable params `983040`
- args: `resolution_scale=0.5`, `sample_resolution_scale=0.5`, `max_train_steps=5`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=5`, `save_checkpoints=true`, `checkpoint_steps=5`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- parser result: `summarize_run` reported completed status, latest step 5, latest loss `0.1243741437792778`, five loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11434215307235718`; step 2 loss `0.12127528339624405`; step 3 loss `0.10792019963264465`; step 4 loss `0.26611995697021484`; step 5 loss `0.1243741437792778`
- sample files: `samples/samples_original.png` (`27985` bytes), `samples/samples_step_5_student.png` (`27856` bytes), and `samples/samples_step_5_teacher.png` (`27220` bytes)
- trajectory files: `samples_trajectory/t0/step_5_student_teacher_x0.png` (`654313` bytes), `samples_trajectory/t0/one_img/step_5_student_single.png` (`324068` bytes), and `samples_trajectory/t0/one_img/step_5_teacher_single.png` (`327712` bytes)
- checkpoint files: `checkpoints/lora_gen_step_5/student/adapter_model.safetensors` with size `3942832` bytes, `checkpoints/lora_gen_step_5/teacher/adapter_model.safetensors` with size `1976792` bytes, both `adapter_config.json` files, and `checkpoints/lora_gen_step_5/README.md`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `512x512` through `672x288`, low VRAM mode was enabled, inline sample resolution was scaled to `256x256`, original/sample images were saved at step 5, and training completed
- visual check: `samples/samples_step_5_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1183 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 5/latest loss `0.1243741437792778`/trainable params `983040`/five loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; JSON validation passed for `features.json` and `meta.json`; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `1744 MiB / 16380 MiB`

This verifies FLUX2 Editing five-step sample/checkpoint artifact writing at `0.5` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The `0.5625` no-artifact, checkpoint, scaled sample, and scaled artifact paths are separately verified below. It does not prove native full-size sample generation, artifact writing beyond five steps, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_1step_noartifact_20260623031240 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_1step_noartifact_20260623031240`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_1step_noartifact_20260623031240`
- result: completed, step 1, latest loss `0.1217951700091362`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `sample_resolution_scale=1.0`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.1217951700091362`, one loss point, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- loss point: step 1 loss `0.1217951700091362` with grad norm `0.6499869227409363`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, low VRAM mode was enabled, sample generation was disabled, trainable params were `983040`, and training completed
- post-run check: no D-OPSD training process remained, and GPU memory was about `1209 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 1/latest loss `0.1217951700091362`/trainable params `983040`/one loss point/sample count 0/trajectory count 0/checkpoint count 0/no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets

This verifies FLUX2 Editing one-step no-artifact training at `0.5625` resolution on RTX 4060 Ti 16GB. The checkpoint-only path is separately verified below. It does not prove sample writing, combined artifact writing, longer training, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution checkpoint path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_ckpt_1step_20260623032741 RESOLUTION_SCALE=0.5625 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_ckpt_1step_20260623032741`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_ckpt_1step_20260623032741`
- result: completed, step 1, latest loss `0.1217951700091362`, trainable params `983040`
- args: `resolution_scale=0.5625`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=true`, `checkpoint_steps=1`, `sample_resolution_scale=1.0`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.1217951700091362`, one loss point, 0 sample files, 0 sample trajectory files, 5 checkpoint files, and no failure reason
- loss point: step 1 loss `0.1217951700091362` with grad norm `0.6503891944885254`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, low VRAM mode was enabled, sample generation was disabled, trainable params were `983040`, and training completed
- post-run check: no D-OPSD training process remained, and GPU memory was about `1829 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 1/latest loss `0.1217951700091362`/trainable params `983040`/one loss point/sample count 0/trajectory count 0/checkpoint count 5/no failure reason; JSON validation passed for `features.json` and `meta.json`; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `2084 MiB / 16380 MiB`

This verifies FLUX2 Editing checkpoint writing at `0.5625` resolution on RTX 4060 Ti 16GB. The scaled sample-writing path is separately verified below. It does not prove combined sample/checkpoint artifact writing, longer training, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution scaled sample-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_sample_scale05_1step_20260623034019 RESOLUTION_SCALE=0.5625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=0 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_sample_scale05_1step_20260623034019`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_sample_scale05_1step_20260623034019`
- result: completed, step 1, latest loss `0.11489006131887436`, trainable params `983040`
- args: `resolution_scale=0.5625`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.11489006131887436`, one loss point, 3 sample files, 3 sample trajectory files, 0 checkpoint files, and no failure reason
- loss point: step 1 loss `0.11489006131887436` with grad norm `0.5574998259544373`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, inline sample resolution was scaled to `288x288`, original/sample images were saved, and training completed
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1624 MiB / 16380 MiB`
- verification checks: summary assertion matched completed status/latest step 1/latest loss `0.11489006131887436`/trainable params `983040`/one loss point/sample count 3/trajectory count 3/checkpoint count 0/no failure reason; JSON validation passed for `features.json` and `meta.json`; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `1803 MiB / 16380 MiB`

This verifies FLUX2 Editing scaled sample writing at `0.5625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The one-step combined artifact path is separately verified below. It does not prove longer training, higher-than-0.5625 profiles, full `1.0` production-resolution training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution one-step scaled combined artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_artifacts_scale05_1step_20260623035153 RESOLUTION_SCALE=0.5625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAMPLE_STEPS=1 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=1 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_artifacts_scale05_1step_20260623035153`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_artifacts_scale05_1step_20260623035153`
- result: completed, step 1, latest loss `0.11489006131887436`, trainable params `983040`
- args: `resolution_scale=0.5625`, `sample_resolution_scale=0.5`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=1`, `save_checkpoints=true`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 1, latest loss `0.11489006131887436`, one loss point, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss point: step 1 loss `0.11489006131887436` with grad norm `0.5563964247703552`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, inline sample resolution was scaled to `288x288`, original/sample images were saved, and training completed
- visual check: `samples/samples_step_1_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `581 MiB / 16380 MiB`
- verification checks: combined summary assertion matched completed status/latest step 1/latest loss `0.11489006131887436`/trainable params `983040`/one loss point/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; JSON validation passed for `features.json` and `meta.json`; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `2320 MiB / 16380 MiB`

This verifies FLUX2 Editing one-step combined sample and checkpoint artifact writing at `0.5625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The two-step, three-step, and five-step combined artifact paths are separately verified below. It does not prove artifact writing beyond five steps, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution two-step scaled combined artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_artifacts_scale05_2step_ RESOLUTION_SCALE=0.5625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAMPLE_STEPS=2 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=2 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_artifacts_scale05_2step_`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_artifacts_scale05_2step_`
- result: completed, step 2, latest loss `0.145225390791893`, trainable params `983040`
- args: `resolution_scale=0.5625`, `sample_resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=2`, `save_checkpoints=true`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- sample files: `samples/samples_original.png`, `samples/samples_step_2_student.png`, and `samples/samples_step_2_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 2, latest loss `0.145225390791893`, two loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11489006131887436` with grad norm `0.5568050742149353`; step 2 loss `0.145225390791893` with grad norm `2.506260395050049`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, inline sample resolution was scaled to `288x288`, original/sample images were saved, and training completed
- visual check: `samples/samples_step_2_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1832 MiB / 16380 MiB` immediately after the run parse
- verification checks: two-step summary assertion matched completed status/latest step 2/latest loss `0.145225390791893`/trainable params `983040`/two loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; JSON validation passed for `features.json` and `meta.json`; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `2476 MiB / 16380 MiB`

This verifies FLUX2 Editing two-step combined sample and checkpoint artifact writing at `0.5625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The three-step and five-step combined artifact paths are separately verified below. It does not prove artifact writing beyond five steps, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution three-step scaled combined artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_artifacts_scale05_3step_20260623042328 RESOLUTION_SCALE=0.5625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=3 SAVE_SAMPLES=1 SAMPLE_STEPS=3 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=3 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_artifacts_scale05_3step_20260623042328`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_artifacts_scale05_3step_20260623042328`
- result: completed, step 3, latest loss `0.10300060361623764`, trainable params `983040`
- args: `resolution_scale=0.5625`, `sample_resolution_scale=0.5`, `max_train_steps=3`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=3`, `save_checkpoints=true`, `checkpoint_steps=3`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- sample files: `samples/samples_original.png`, `samples/samples_step_3_student.png`, and `samples/samples_step_3_teacher.png`
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_3/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_3/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, latest step 3, latest loss `0.10300060361623764`, three loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11489006131887436` with grad norm `0.5573927760124207`; step 2 loss `0.14530973136425018` with grad norm `2.501918077468872`; step 3 loss `0.10300060361623764` with grad norm `5.149724006652832`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, inline sample resolution was scaled to `288x288`, original/sample images were saved, and training completed
- visual check: `samples/samples_step_3_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `1970 MiB / 16380 MiB` immediately after the run parse
- verification checks: three-step summary assertion matched completed status/latest step 3/latest loss `0.10300060361623764`/trainable params `983040`/three loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; JSON validation passed for `features.json` and `meta.json`; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; `git diff --check` and trailing whitespace checks passed; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `2516 MiB / 16380 MiB`

This verifies FLUX2 Editing three-step combined sample and checkpoint artifact writing at `0.5625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. The five-step combined artifact path is separately verified below. It does not prove artifact writing beyond five steps, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified FLUX2 Editing 0.5625-resolution five-step scaled combined artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=flux2_editing_res05625_artifacts_scale05_5step_20260623044037 RESOLUTION_SCALE=0.5625 SAMPLE_RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=5 SAVE_SAMPLES=1 SAMPLE_STEPS=5 SAVE_CHECKPOINTS=1 CHECKPOINT_STEPS=5 timeout 7200 bash scripts/run_flux2_editing_smoke.sh"
```

- exp name: `flux2_editing_res05625_artifacts_scale05_5step_20260623044037`
- output: `trainer-data/smoke-runs/flux2_editing_res05625_artifacts_scale05_5step_20260623044037`
- result: completed, step 5, latest loss `0.10808877646923065`, trainable params `983040`
- args: `resolution_scale=0.5625`, `sample_resolution_scale=0.5`, `max_train_steps=5`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `sample_steps=5`, `save_checkpoints=true`, `checkpoint_steps=5`, `batch_size=1`, `mixed_precision=bf16`, `tiled_vae=true`, `vae_tile_size=64`, `vae_tile_overlap=0.25`, `block_offload=false`
- sample files: `samples/samples_original.png` (`34456` bytes), `samples/samples_step_5_student.png` (`34303` bytes), and `samples/samples_step_5_teacher.png` (`33990` bytes)
- trajectory files: 3 PNG files under `samples_trajectory/t0`
- checkpoint files: `checkpoints/lora_gen_step_5/student/adapter_model.safetensors` with size `3942832` bytes, `checkpoints/lora_gen_step_5/teacher/adapter_model.safetensors` with size `1976792` bytes, both `adapter_config.json` files, and `checkpoints/lora_gen_step_5/README.md`
- parser result: `summarize_run` reported completed status, latest step 5, latest loss `0.10808877646923065`, five loss points, 3 sample files, 3 sample trajectory files, 5 checkpoint files, and no failure reason
- loss points: step 1 loss `0.11489006131887436` with grad norm `0.5571338534355164`; step 2 loss `0.14463230967521667` with grad norm `2.4856324195861816`; step 3 loss `0.10168380290269852` with grad norm `4.546292781829834`; step 4 loss `0.2615056335926056` with grad norm `1.4736329317092896`; step 5 loss `0.10808877646923065` with grad norm `0.47298675775527954`
- log evidence: tiled VAE was enabled, bucket resolutions scaled up to `576x576` through `752x320`, low VRAM mode was enabled, inline sample resolution was scaled to `288x288`, original/sample images were saved at step 5, and training completed
- visual check: `samples/samples_step_5_student.png` opened successfully
- post-run check: no D-OPSD training process remained, and GPU memory was about `906 MiB / 16380 MiB` immediately after the run parse
- verification checks: five-step summary assertion matched completed status/latest step 5/latest loss `0.10808877646923065`/trainable params `983040`/five loss points/sample count 3/trajectory count 3/checkpoint count 5/no failure reason; Python unittest discovery ran 51 tests OK; `py_compile` passed for FLUX2 Identity, FLUX2 Editing, Z-Image training files and runtime tests; WSL `bash -n` passed for all three smoke scripts; `npm run typecheck --prefix trainer-ui` passed; `scripts/check_ui_smoke.ps1` returned Ok true with 6 pages, 6 CSS assets, 3 cached default models, 6 jobs, 3 datasets, and 3 ready datasets; JSON validation passed for `features.json` and `meta.json`; `git diff --check` passed; trailing whitespace checks passed; exact old Editing 0.5625 beyond-three text search found no matches; final WSL process checks found no D-OPSD training, Accelerate, or DeepSpeed processes and GPU memory was about `2031 MiB / 16380 MiB`

This verifies FLUX2 Editing five-step combined sample and checkpoint artifact writing at `0.5625` training resolution on RTX 4060 Ti 16GB when inline preview sampling is scaled with `SAMPLE_RESOLUTION_SCALE=0.5`. It does not prove artifact writing beyond five steps, higher-than-0.5625 profiles, full `1.0` production-resolution training, much longer training, or production-quality samples.

The verified local Z-Image smoke path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_smoke_1step_lowvram_cpu_text timeout 1800 bash scripts/run_zimage_smoke.sh"
```

This path uses:

- direct `python train_dopsd.py` launch,
- local `Tongyi-MAI/Z-Image-Turbo` and `Qwen/Qwen3-VL-4B-Instruct` snapshots,
- `DOPSD_QWEN_VL_MODEL` pointed at the local Qwen snapshot,
- `--low-vram`,
- `--resolution-scale 0.25`,
- `--use-8bit-adam`,
- bf16,
- batch size 1,
- LoRA rank 4,
- `--no-save-samples`,
- `--no-save-checkpoints`.

The smoke proves a one-step low-VRAM Z-Image training pass on RTX 4060 Ti 16GB. It does not prove production-resolution training, sample generation, or checkpoint writing.

The same Z-Image smoke path has also been verified through the UI/API runner:

- job id: `08286c98-836f-432e-ad7f-104c542e9a1c`
- exp name: `z_image_turbo_20260621182020787`
- output: `trainer-data/smoke-runs/z_image_turbo_20260621182020787`
- result: completed, step 1, loss `0.4008714556694031`, trainable params `9922560`, runner exit code `0`

The verified Z-Image low-VRAM checkpoint path is:

```bash
# Manual direct-Python run based on scripts/run_zimage_smoke.sh with checkpoint
# saving enabled and sample generation disabled.
```

- exp name: `zimage_style_ckpt_smoke_1step_lowvram_20260622032632`
- output: `trainer-data/smoke-runs/zimage_style_ckpt_smoke_1step_lowvram_20260622032632`
- result: completed, step 1, loss `0.4008714556694031`, trainable params `9922560`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported 5 checkpoint artifact files and completed status

This proves real Z-Image low-VRAM checkpoint writing. It does not prove production-resolution training.

The verified Z-Image low-VRAM sample path is:

```bash
# Manual direct-Python run based on scripts/run_zimage_smoke.sh with --sample-steps 1,
# checkpoint saving disabled, sample saving enabled, CPU prompt encoding, and explicit-device sampling.
```

- exp name: `zimage_style_sample_smoke_1step_lowvram_explicitdevice_state_20260622040721`
- output: `trainer-data/smoke-runs/zimage_style_sample_smoke_1step_lowvram_explicitdevice_state_20260622040721`
- result: completed, step 1, loss `0.4008714556694031`, trainable params `9922560`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 12 PNG files under `samples_trajectory/t0` through `samples_trajectory/t3`
- parser result: `summarize_run` reported 3 sample files, 12 sample trajectory files, 0 checkpoint files, and completed status
- visual check: `samples/samples_step_1_student.png` opened successfully

This proves real Z-Image low-VRAM sample generation at the smoke profile. It does not prove production-resolution training or sample quality at production settings.

The three smoke scripts accept `MAX_TRAIN_STEPS` with a default of `1`, so longer verification runs can be launched without changing the smoke baseline.

The verified Z-Image quasi-production training path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_quasiprod_2step_res05_202606221515 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 2400 bash scripts/run_zimage_smoke.sh"
```

- exp name: `zimage_style_quasiprod_2step_res05_202606221515`
- output: `trainer-data/smoke-runs/zimage_style_quasiprod_2step_res05_202606221515`
- result: completed, step 2, latest loss `0.359367311000824`, trainable params `9922560`
- args: `resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason

This verifies a less-downscaled two-step Z-Image no-artifact profile on RTX 4060 Ti 16GB. It does not prove full `1.0` production-resolution training, artifact writing, or production-quality samples.

The verified Z-Image 0.5-resolution artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_res05_artifacts_2step_202606221528 RESOLUTION_SCALE=0.5 MAX_TRAIN_STEPS=2 SAVE_SAMPLES=1 SAVE_CHECKPOINTS=1 SAMPLE_STEPS=2 CHECKPOINT_STEPS=2 timeout 3600 bash scripts/run_zimage_smoke.sh"
```

- exp name: `zimage_style_res05_artifacts_2step_202606221528`
- output: `trainer-data/smoke-runs/zimage_style_res05_artifacts_2step_202606221528`
- result: completed, step 2, latest loss `0.3601413071155548`, trainable params `9922560`
- args: `resolution_scale=0.5`, `max_train_steps=2`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `save_checkpoints=true`, `sample_steps=2`, `checkpoint_steps=2`, `batch_size=1`, `mixed_precision=bf16`
- sample files: `samples/samples_original.png`, `samples/samples_step_2_student.png`, and `samples/samples_step_2_teacher.png`
- trajectory files: 12 PNG files under `samples_trajectory`
- checkpoint files: `checkpoints/lora_gen_step_2/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_2/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 3 sample files, 12 sample trajectory files, 5 checkpoint artifact files, and no failure reason
- visual check: `samples/samples_step_2_student.png` opened successfully

This verifies Z-Image sample and checkpoint writing at `0.5` resolution on RTX 4060 Ti 16GB. It does not prove full `1.0` production-resolution training, FLUX2 production profiles, or production-quality samples.

The verified Z-Image 0.75-resolution no-artifact boundary path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_res075_1step_noartifact_202606221654 RESOLUTION_SCALE=0.75 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3000 bash scripts/run_zimage_smoke.sh"
```

- exp name: `zimage_style_res075_1step_noartifact_202606221654`
- output: `trainer-data/smoke-runs/zimage_style_res075_1step_noartifact_202606221654`
- result: completed, step 1, latest loss `0.47717854380607605`, trainable params `9922560`
- args: `resolution_scale=0.75`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`
- parser result: `summarize_run` reported completed status, 0 sample files, 0 sample trajectory files, 0 checkpoint files, and no failure reason
- observed during polling: GPU memory reached about `12594 MiB / 16380 MiB` while training, then returned below `500 MiB` after exit

This verifies a 0.75-resolution one-step Z-Image training boundary on RTX 4060 Ti 16GB when artifact writing is disabled. It does not prove production-quality samples.

The verified Z-Image 0.75-resolution artifact-writing path is:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_res075_artifacts_1step_202606221705 RESOLUTION_SCALE=0.75 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=1 SAVE_CHECKPOINTS=1 SAMPLE_STEPS=1 CHECKPOINT_STEPS=1 timeout 4200 bash scripts/run_zimage_smoke.sh"
```

- exp name: `zimage_style_res075_artifacts_1step_202606221705`
- output: `trainer-data/smoke-runs/zimage_style_res075_artifacts_1step_202606221705`
- result: completed, step 1, latest loss `0.47717854380607605`, trainable params `9922560`
- args: `resolution_scale=0.75`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=true`, `save_checkpoints=true`, `sample_steps=1`, `checkpoint_steps=1`, `batch_size=1`, `mixed_precision=bf16`
- sample files: `samples/samples_original.png`, `samples/samples_step_1_student.png`, and `samples/samples_step_1_teacher.png`
- trajectory files: 12 PNG files under `samples_trajectory`
- checkpoint files: `checkpoints/lora_gen_step_1/student/adapter_model.safetensors` and `checkpoints/lora_gen_step_1/teacher/adapter_model.safetensors`
- parser result: `summarize_run` reported completed status, 3 sample files, 12 sample trajectory files, 5 checkpoint artifact files, and no failure reason
- visual check: `samples/samples_step_1_student.png` opened successfully
- observed during polling: GPU memory reached about `15619 MiB / 16380 MiB` during final sampling, then returned below `500 MiB` after exit

This verifies Z-Image sample and checkpoint writing at `0.75` resolution on RTX 4060 Ti 16GB. It is close to the current 16GB memory ceiling and does not prove production-quality samples.

The checked full-resolution Z-Image no-artifact path is not stable on this 16GB profile:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/D-opsd-T8-Tranier; EXP_NAME=zimage_style_fullres_1step_noartifact_202606221620 RESOLUTION_SCALE=1.0 MAX_TRAIN_STEPS=1 SAVE_SAMPLES=0 SAVE_CHECKPOINTS=0 timeout 3600 bash scripts/run_zimage_smoke.sh"
```

- exp name: `zimage_style_fullres_1step_noartifact_202606221620`
- output: `trainer-data/smoke-runs/zimage_style_fullres_1step_noartifact_202606221620`
- result: failed before step 1, latest step `0`, no loss points, trainable params `9922560`
- args: `resolution_scale=1.0`, `max_train_steps=1`, `low_vram=true`, `use_8bit_adam=true`, `save_samples=false`, `save_checkpoints=false`, `batch_size=1`, `mixed_precision=bf16`
- failure reason: `RuntimeError: CUDA driver error: out of memory`
- stack tail: OOM occurred inside the Z-Image transformer LoRA feed-forward path at `peft/tuners/lora/layer.py`
- observed during polling: GPU memory reached about `15939 MiB / 16380 MiB` immediately before the OOM

Do not set Z-Image `resolution_scale=1.0` as a stable 16GB default until a new memory strategy is implemented and verified. For current 16GB guidance, `0.5` is the safer artifact-writing profile and `0.75` is the verified upper artifact-writing boundary.

Optional block offload status:

- `flux2-klein_self-distill-edit`, `flux2-klein-edit-self-distill-gt-ref`, and `z-image-turbo_self-distill-vlm` accept `--block-offload` and `--block-offload-num-blocks`.
- The implementation calls Diffusers `apply_group_offloading` with `offload_type="block_level"` only for final sampling when `global_step == max_train_steps`.
- `scripts/run_flux2_smoke.sh`, `scripts/run_flux2_editing_smoke.sh`, and `scripts/run_zimage_smoke.sh` keep block offload disabled by default; set `BLOCK_OFFLOAD=1` and optionally `BLOCK_OFFLOAD_NUM_BLOCKS=2` for OOM fallback experiments.
- The runtime command builder and New Job Memory & launch UI expose the optional block offload controls.
- A first FLUX2 Identity `BLOCK_OFFLOAD=1` run that attached block offload before training failed during backward with a cuda/cpu device mismatch, so the training path must stay unhooked.
- A no-sample FLUX2 Identity smoke with `BLOCK_OFFLOAD=1` completed after scoping block offload to final sampling only: `flux2_identity_blockoffload_nosample_smoke_1step_20260622073515`, step 1, loss `1.9178619384765625`, trainable params `983040`.
- A FLUX2 Identity sample smoke with `SAVE_SAMPLES=1`, `SAMPLE_STEPS=1`, and `BLOCK_OFFLOAD=1` failed after the training step: `flux2_identity_blockoffload_sample_smoke_1step_teacherpt_20260622082548`, step 1, loss `1.9178619384765625`, partial artifacts `samples/samples_original.png` plus 3 trajectory PNGs, then `RuntimeError: CUDA driver error: out of memory` during teacher image-condition VAE encode before block offload was enabled.
- After releasing trajectory/training tensors before final teacher sampling, `flux2_identity_blockoffload_sample_smoke_1step_release_20260622083742` still failed at step 1 with the same teacher image-condition VAE encode OOM; `summarize_run` reported loss `1.9178619384765625`, 1 sample file, 3 trajectory files, and 0 checkpoints.
- After enabling the Diffusers block-offload hook before teacher image-conditioned final sampling and using latent decode after transformer hooks, `flux2_identity_blockoffload_sample_smoke_1step_hookteacher_20260622084606` still failed at step 1 with the same OOM. The log reached `Transformer block offload enabled: 2 block(s) per group`, so the next fix should change the sampling memory architecture rather than continue in-process patch attempts.
- FLUX2 Identity deferred final sampler now enables tiled VAE by default: `--tiled-vae --vae-tile-size 64 --vae-tile-overlap 0.25`. This uses Diffusers `AutoencoderKLFlux2` tiled encode/decode, matching the ComfyUI-style overlap/blend approach without copying ComfyUI code.
- FLUX2 Identity main inline sampling now configures the same tiled VAE settings before the initial `samples_original.png` decode, preventing the verified 0.3125 and 0.375 sample-writing paths from OOMing before the first training step.
- FLUX2 Editing main inline sampling now configures the same tiled VAE settings before the initial `samples_original.png` decode, and it accepts `--sample-resolution-scale` for low-VRAM inline sample previews. The `0.3125` native sample-writing path completed with original/student/teacher samples plus trajectories. At `0.375`, native full-size sample writing fails during final transformer sampling, while `SAMPLE_RESOLUTION_SCALE=0.5` completes sample writing and combined sample/checkpoint artifact writing with original/student/teacher samples, trajectories, and student/teacher LoRA checkpoints.
- The deferred final sampler also enables Diffusers model CPU offload by default through `--final-sampler-cpu-offload`, so text encoder, transformer, and VAE are not all resident on CUDA during final sampling.
- A targeted rerun of the previously failed request `flux2_identity_blockoffload_sample_smoke_1step_tiledvae_202606221230` succeeded after enabling final sampler CPU offload, writing `samples_step_1_student.png` and `samples_step_1_teacher.png`.
- The verified FLUX2 Identity sample-writing fallback is `flux2_identity_blockoffload_sample_smoke_1step_tiledvae_cpuoffload_202606221250`: `SAVE_SAMPLES=1`, `SAMPLE_STEPS=1`, `BLOCK_OFFLOAD=1`, final sample scale `0.5`, tiled VAE enabled, model CPU offload enabled, completed step 1, loss `1.9178619384765625`, trainable params `983040`, 3 sample files, and 3 trajectory PNGs.
- Diffusers group block offload for the final sampler remains unverified/disabled in the verified request (`requested_block_offload=true`, `block_offload=false`); do not describe that hook itself as verified until a separate smoke proves it.

## Job And Observability Notes

The local job ledger currently lives at:

```text
trainer-data/jobs/jobs.json
```

Current job capabilities:

- seed the verified FLUX2 low-VRAM smoke run into the ledger,
- seed known real FLUX2 Editing sample OOM runs into the ledger as failed jobs when their output folders and `.runner.err.log` files exist,
- create UI draft jobs only after dataset preflight validates the selected JSONL,
- revalidate the job dataset immediately before runner launch and block the start before any `runner.sh` is written if the JSONL has gone invalid,
- clone jobs,
- delete jobs,
- start supported jobs through a detached WSL runner,
- queue additional supported jobs when another job is already running or queued,
- auto-promote the oldest queued job when no runner is active, returning it to draft with a dataset-preflight note instead of launching if validation fails,
- stop running jobs by terminating the detached WSL child process group as well as the runner shell,
- write a per-job `runner.sh` script under `trainer-data/jobs/runner/<job-id>/`,
- launch WSL jobs through a short `wsl.exe ... bash -lc` bootstrap that `nohup` detaches the Linux runner, captures startup stdout/stderr, writes `windows.pid`, waits for `linux.pid` / `runner-state.json`, and avoids tying training lifetime to the UI process,
- run the actual training command under `setsid`, write `child.pid`, and stop child process groups through direct WSL `/bin/kill` calls so Python/timeout descendants do not keep using GPU after a UI stop,
- track Windows PID, Linux runner PID, and Linux child process group PID files,
- recover completed/failed/stopped status from runner-state JSON,
- keep a running job alive during refresh when the Windows wrapper PID is gone but the WSL `linux.pid` still responds to `kill -0`,
- keep a running job alive when the Windows wrapper PID is gone, `runner-state.json` has not finished, and the WSL PID check is inconclusive but the `linux.pid` file still exists,
- read the jobs ledger with retry, tolerate an existing UTF-8 BOM, and write `jobs.json` through a temp file plus rename so transient reads do not reseed an empty ledger,
- recover jobs from `jobs.json`,
- summarize job artifacts from output directories,
- show latest step, latest loss, trainable params, log tail, runner log tail, runner PID/exit, failure reason, error tail, artifact chips, and command preview in the UI,
- open an on-demand Full logs panel through `/api/jobs/[id]/logs` with training, runner, and stderr lines capped to the latest 300 entries,
- stream open running/queued job logs through `/api/jobs/[id]/logs/stream` as SSE `snapshot`, `append`, and `heartbeat` events,
- open existing in-project job output folders from the Jobs page through `/api/jobs/[id]/open`,
- return a clear missing-folder error when draft jobs have no output directory yet,
- expose GPU telemetry through `/api/telemetry`,
- show live GPU memory, utilization, temperature, power, and reported GPU process count on Dashboard and Jobs.

Current dataset capabilities:

- validate bundled and custom JSONL paths through `/api/datasets`,
- import ai-toolkit-style image folders from the Datasets page by uploading images plus same-stem `.txt`, `.caption`, or `.json` tag files; managed datasets are stored under `trainer-data/datasets/managed` with a `managed-datasets.json` ledger and generated D-OPSD JSONL,
- edit managed datasets after import by adding image items, editing caption/tag text, deleting image items, or deleting the whole managed dataset,
- preflight selected New Job datasets before draft creation; the New Job page shows `Pair preflight`, rows/issues/buckets, and role labels for edit-pair readability when an edit dataset preview is active,
- select multiple datasets for one recipe in New Job; draft creation combines them into `trainer-data/datasets/selections/.../data.jsonl`,
- run the same dataset preflight again on Start, so stale, edited, or combined JSONL files are caught before the WSL runner is created,
- pass selected or combined datasets into the smoke-script launch wrappers through `DATA_PATH_TRAIN_JSONL` and `DATA_PATH_TEST_JSONL`,
- serve safe in-project image thumbnails through `/api/datasets/image`,
- show rows, valid rows, issues, prompt previews, reference/target image previews, and bucket dimensions on the Datasets page,
- report bucket dimensions from real image files first, falling back to JSONL `h*w` metadata.

Current settings capabilities:

- summarize runtime settings through `trainer_runtime/dopsd_trainer/settings.py`,
- expose `/api/settings` with runtime paths, backend detection, Hugging Face token presence, model readiness, runner policy, low-VRAM safety defaults, and verified `production_profiles`,
- show real Settings page data instead of static placeholder cards, including verified 16GB production starter profiles and Chinese-first labels for runtime paths, safety defaults, runner policy values, and profile labels,
- never display Hugging Face token values; only display presence/status,
- Settings page browser QA is verified on desktop and mobile after confirming CSS is loaded and no horizontal overflow, clipped buttons, or console errors are present.

Current observability parser:

- Python runtime: `trainer_runtime/dopsd_trainer/outputs.py`
- UI server helper: `trainer-ui/src/lib/jobs.ts`
- System telemetry helper: `trainer-ui/src/lib/system.ts`
- Dataset UI/API helper: `trainer-ui/src/lib/datasets.ts`
- Loss file path: `loss_log/loss_gen_log.jsonl`
- Log file path: `log.txt`
- Args file path: `args.json`
- Sample artifact directories: `samples` and `samples_trajectory`
- Checkpoint artifact directory: `checkpoints`
- External runner stderr path for manual smokes: sibling `<exp-name>.runner.err.log`
- Failure fields: `failure_reason` / `failureReason` and `error_tail` / `errorTail`
- Full logs API path: `/api/jobs/[id]/logs`
- Live log stream API path: `/api/jobs/[id]/logs/stream`
- Full logs sources: training `log.txt`, per-job runner log, sibling `.runner.out.log`, and sibling `.runner.err.log`

Current failed-run capture:

- `summarize_run` reads real sibling `.runner.err.log` files and reports failed status, a short failure reason, and a compact error tail for OOM/traceback failures.
- `/api/jobs` reports `failureReason` and `errorTail` for failed jobs.
- The Jobs page shows a concise Failure diagnostics block with the extracted OOM reason and stderr tail.
- Seeded failed-run evidence jobs cannot be started directly; clone them before rerun.
- Browser QA for `/jobs` passed at 1440x1000 and 390x844 with the seeded OOM failures visible, two Failure panels, seven Output buttons, seeded failure cards limited to Output/Clone/Delete, loaded CSS, no horizontal overflow, no clipped buttons, and no console errors.
- Known failed FLUX2 Editing sample smoke runs are seeded once into the job ledger as failure evidence:
  - `failed-flux2-editing-sample-vae-encode-oom`
  - `failed-flux2-editing-sample-vae-decode-oom`

Current log visibility:

- `/api/jobs/[id]/logs` returns combined training, runner, and stderr logs on demand, capped to the latest 300 lines.
- The Jobs page has a Logs button and expandable Full logs panel for each job.
- `/api/jobs/[id]/logs/stream` returns server-sent log events for open running/queued jobs, starting with a snapshot and then appending only newly observed lines.
- The Jobs page keeps the existing fetch snapshot as a fallback, then uses `EventSource` to append streamed lines and shows a Live stream status chip while connected.
- Browser QA passed for the Full logs panel on desktop and mobile with stderr/OOM text visible, CSS loaded, no horizontal overflow, no clipped buttons, and no console errors.
- `scripts/check_training_observability.ps1 -BaseUrl http://127.0.0.1:8675 -ProbeDurationSeconds 12 -TimeoutSeconds 90` verifies the live observability path without Browser automation: it creates a runner probe, starts it, opens `/api/jobs/[id]/logs/stream`, requires `snapshot`, `heartbeat`, and `append` SSE events, waits for completion, checks combined runner/training logs through `/api/jobs/[id]/logs`, serves `samples/probe-sample.png`, `samples_trajectory/probe-trajectory.png`, and `checkpoints/probe-adapter.safetensors`, checks `/api/telemetry`, deletes the probe job, and asserts `jobs.json` returns to its original IDs. The verified 8675 run returned Ok true for job `ffdf7743-bae4-4a65-8a13-7da32dca03bc` with runner exit code `0`, one sample, one trajectory, one checkpoint, and NVIDIA GeForce RTX 4060 Ti telemetry.
- A fresh short observability probe after runner startup hardening passed on 8675 with job `681d8b8c-2370-4293-bf7a-f86b124d1000`: status running -> completed, runner exit code `0`, SSE events `snapshot`, `append`, and `heartbeat`, one sample, one trajectory, one checkpoint, and RTX 4060 Ti telemetry.
- Fresh in-app Browser QA for the SSE UI was blocked by the local Browser runtime failing to start with `CreateProcessAsUserW failed: 5`; bundled Playwright was also unavailable because `playwright-core` is not present. Runtime verification for this change is now `scripts/check_training_observability.ps1` plus TypeScript/build/tests. As a Browser-runtime-free guard, `scripts/check_ui_smoke.ps1` also verifies `/api/project`, `/api/models`, `/api/jobs`, `/api/settings`, `/api/datasets`, Dashboard, New Job, Jobs, Datasets, Models, Settings, Next CSS assets, and Liquid Glass CSS against the running UI.

Current artifact browsing capabilities:

- `summarize_run` reports sample, sample trajectory, and checkpoint file entries with relative paths, sizes, modified timestamps, and image flags.
- `/api/jobs` includes artifact counts and file entries for samples, trajectories, and checkpoints.
- `/api/jobs/[id]/artifact?path=...` serves only safe in-job files under `samples`, `samples_trajectory`, or `checkpoints`.
- The Jobs page shows sample thumbnails and checkpoint file rows when artifacts exist.
- Artifact browsing mechanics are verified with a runner probe that writes tiny PNG samples and a dummy checkpoint. This does not prove real D-OPSD sample generation.
- Real FLUX2 checkpoint writing is verified by `flux2_identity_ckpt_smoke_1step_lowvram_20260622023426`; `summarize_run` reports 5 checkpoint artifact files for that run.
- Real FLUX2 sample writing is verified by `flux2_identity_sample_smoke_1step_lowvram_devicefix_20260622031210`; `summarize_run` reports 3 sample files and 3 sample trajectory files for that run.
- Real Z-Image checkpoint writing is verified by `zimage_style_ckpt_smoke_1step_lowvram_20260622032632`; `summarize_run` reports 5 checkpoint artifact files for that run.
- Real Z-Image sample writing is verified by `zimage_style_sample_smoke_1step_lowvram_explicitdevice_state_20260622040721`; `summarize_run` reports 3 sample files and 12 sample trajectory files for that run.
- FLUX2 Editing no-sample/no-checkpoint smoke training is verified by `flux2_editing_smoke_1step_lowvram_python_20260622043148`; `summarize_run` reports completed status, step 1, loss `0.08440777659416199`, trainable params `983040`, and no sample/checkpoint artifacts.
- Real FLUX2 Editing checkpoint writing is verified by `flux2_editing_ckpt_smoke_1step_lowvram_20260622050819`; `summarize_run` reports 5 checkpoint artifact files for that run.
- Real FLUX2 Editing sample writing is verified by `flux2_editing_sample_smoke_1step_lowvram_release_20260622055912`; `summarize_run` reports 3 sample files and 3 sample trajectory files for that run.

Current runner verification status:

- Runner mechanics are verified with short non-model probes for start -> running -> completed and start -> running -> stopped, including after the switch to per-job `runner.sh` launch.
- Detached WSL startup now writes a per-job `launch.sh` and spawns `wsl.exe ... bash <launch.sh>` instead of passing a multiline bootstrap through `bash -lc`; this avoids Windows/WSL parsing failures around redirection and background syntax. The per-job `runner.sh` executes the actual training command through `bash -lc ${bashQuote(command)}` inside WSL, so complex env assignments remain in one Linux shell context.
- Single-GPU FIFO queue ordering is verified with runner probes: a second start request enters `queued` without a runner PID while the first probe is running, then auto-promotes and completes after the first probe finishes.
- Running jobs are protected from direct delete; stop before delete.
- Queue smoke checker: `scripts/check_job_queue_smoke.ps1 -BaseUrl http://127.0.0.1:8675 -FirstProbeDurationSeconds 35 -SecondProbeDurationSeconds 6 -TimeoutSeconds 120`. It verifies create/start/running/queued/delete protection/stop/auto-promote/completed/clone/delete cleanup and confirms `jobs.json` returns to its original ids. The verified pass used first probe `08d4e170-2ef9-4466-b2f7-f31733622fac`, second probe `dab4d87c-517f-477a-bf9e-a8304e5257af`, clone `4e210abf-3d94-44f9-8d2f-61d8ba891c12`, and runner exit code `0`.
- FLUX2 Identity, FLUX2 Editing, and Z-Image draft creation now use the verified 16GB production starter profile registry for UI defaults and runner commands. The same smoke scripts are still the launch wrappers, but the runner sets profile-specific `RESOLUTION_SCALE`, `SAMPLE_RESOLUTION_SCALE` when needed, `MAX_TRAIN_STEPS`, `SAMPLE_STEPS`, `CHECKPOINT_STEPS`, `SAVE_SAMPLES=1`, `SAVE_CHECKPOINTS=1`, and `OUTPUT_DIR=../trainer-data/runs`.
- Earlier FLUX2 Identity, FLUX2 Editing, and Z-Image UI/API-launched one-step low-VRAM smokes remain valid runner evidence, but the default new draft profile is now the verified production starter rather than the one-step smoke.
- `scripts/check_runner_recovery.ps1` passed against `http://127.0.0.1:8675`: fresh probe `695434ba-c5b6-4d99-8ee4-deca978e9cb2` created a 20-second runner probe, injected a stale Windows PID into `jobs.json`, observed the job stay running with `Runner process monitor detached while Linux PID check is inconclusive.`, then observed completed status with runner exit code `0`; the ledger returned to the six expected jobs afterward.
- `scripts/check_ui_restart_recovery.ps1` passed on temporary port `18782`: it started a 60-second runner probe, stopped the initial UI process, restarted the UI, observed the same job return as `running`, then observed `completed` with runner exit code `0`.
- Completed long zhenzhen run: UI/API job `6c3127cb-7b93-42a8-b492-98b9a64d2029`, exp `flux2_identity_20260623191559452`, dataset `trainer-data/datasets/managed/zhenzhen-1782238992357-a7369f8f/data.jsonl`, recipe `flux2-klein-identity`, `MAX_TRAIN_STEPS=1000`, `CHECKPOINT_STEPS=100`, `SAVE_SAMPLES=0`, `SAVE_CHECKPOINTS=1`, `SKIP_INITIAL_SAMPLE=1`, `LOW_VRAM=1`, `USE_8BIT_ADAM=1`, `TARGET_RESOLUTION=1024`, `RESOLUTION_SCALE=0.625`, sample/final sample scale `0.375`. It completed at step 1000 with latest loss `0.6347438097000122`, runner exit code `0`, no failure reason, and wrote `checkpoints/lora_gen_step_1000` with student/teacher LoRA safetensors. This verifies the requested 1000-step checkpoint-only FLUX2 Identity run on RTX 4060 Ti 16GB; it does not verify production-quality samples or training-time sample generation at this length.
- Do not claim production-resolution training, production-quality sample output, final sampling group block offload, transformer block offload training, or browser-plugin visual QA for the live stream UI is verified yet. The live stream runtime/API path is verified by `scripts/check_training_observability.ps1`.

## Documentation Policy

Keep `roadmap.md`, `features.json`, and this `SKILL.md` in sync with reality. If implementation changes the plan, update the docs in the same work session.

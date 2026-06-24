# T8 D-OPSD Tranier

T8 D-OPSD Tranier 是一个面向 Windows + NVIDIA GPU 的本地 D-OPSD 模型训练器。它以 [vvvvvjdy/D-OPSD](https://github.com/vvvvvjdy/D-OPSD) 为训练核心，参考 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 的工作台体验，提供中文优先的液态玻璃风格 UI、模型管理、数据集导入、任务队列、日志、损失曲线、样图和 LoRA checkpoint 管理。

当前默认推荐单卡 16GB 显存起步，硬件基线为 NVIDIA GeForce RTX 4060 Ti 16GB。

## 主要功能

- 中文默认 UI，可切换英文。
- Windows 启动器和 Electron 桌面版。
- WSL2 Ubuntu 后端训练环境自动引导。
- Hugging Face 模型缓存检测、下载状态、自定义模型路径和打开模型文件夹。
- 支持 D-OPSD JSONL 数据集，也支持上传图片和同名打标文件生成托管数据集。
- 托管数据集支持新增图片、编辑打标文本、删除图片和删除数据集。
- 新建训练任务时可选择多个数据集合并训练。
- 单 GPU FIFO 任务队列、任务克隆、停止、删除、日志查看、输出目录打开。
- 16GB 显存安全默认值：low VRAM、8-bit Adam、分辨率缩放、tiled VAE、可选 block offload。
- 新建训练以「总步数」为主，底层 `epochs` 会自动按步数派生，避免用户配置两个容易冲突的停止条件。
- 新任务会保存版本化训练配置，后续采样、数据集权重、优化器和高级 D-OPSD 参数可以在同一个配置结构上扩展。
- Electron 主进程日志写入文件，避免 Windows GUI 下 stdout/stderr broken pipe 弹窗。

## 系统要求

- Windows 10/11。
- NVIDIA GPU，建议至少 16GB VRAM。
- 已安装可用的 NVIDIA 驱动，`nvidia-smi` 能正常运行。
- WSL2 和 Ubuntu-22.04。
- PowerShell 7 推荐用于完整发布检查。
- Node.js 20 或更高版本，源码运行时需要。
- 足够磁盘空间。默认三套模型缓存约 130GB。
- Hugging Face token。部分模型可能需要先在 Hugging Face 页面接受许可。

## 应该选择哪种安装方式

- 普通使用：下载 Release 里的安装版或便携版，按「直接安装使用」走。
- 开发和改代码：按「从源码运行」走，保留完整 Git 工作区。
- 已经有模型文件：安装后可以在「模型」页为每个模型设置自定义本地路径，不必重复下载。
- 第一次训练前：一定先运行 WSL 初始化脚本，再检查模型页和数据集页是否就绪。

## 直接安装使用

1. 打开 GitHub Release：<https://github.com/T8mars/D-opsd-trainer-T8/releases/tag/v0.1.0>
2. 下载以下任意一种：
   - `T8.D-OPSD.Tranier.Setup.0.1.0.exe`：安装版。
   - `T8.D-OPSD.Tranier.0.1.0.exe`：便携版。
3. 启动 `T8 D-OPSD Tranier`。
4. 首次运行后，应用会把可写工作区复制到：

```text
%APPDATA%\d-opsd-trainer-ui\workspace
```

5. 在 PowerShell 中进入该工作区，初始化 WSL 训练环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_trainer.ps1
```

6. 如果需要访问 gated 模型，请先设置 Hugging Face token，并在 Hugging Face 网页接受对应模型许可。
7. 回到应用的「模型」页，检查默认模型是否已缓存。没有缓存时可下载模型，或给每个模型设置自定义本地路径。
8. 在「数据集」页导入图片和打标文件。
9. 在「新建训练」页选择配方、数据集和 16GB 推荐配置，创建草稿后启动训练。
10. 在「任务」页查看 GPU 状态、实时日志、损失、样图、checkpoint 和输出目录。

## 从源码运行

```powershell
git clone https://github.com/T8mars/D-opsd-trainer-T8.git
cd D-opsd-trainer-T8
npm install --prefix trainer-ui
.\D-OPSD-Trainer.cmd
```

默认地址是：

```text
http://127.0.0.1:8675
```

无浏览器启动：

```powershell
.\D-OPSD-Trainer.cmd -NoBrowser
```

指定端口：

```powershell
.\D-OPSD-Trainer.cmd -Port 8680
```

仅做启动器健康检查：

```powershell
.\D-OPSD-Trainer.cmd -NoBrowser -SmokeTest -Port 18771 -TimeoutSeconds 120
```

## 初始化训练后端

源码工作区或 Electron 工作区都可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_trainer.ps1
```

脚本会在 WSL 中创建或更新：

```text
trainer-data/venvs/dopsd
trainer-data/hf-home
trainer-data/pip-cache
trainer-data/triton-cache
```

训练脚本会通过 `scripts/dopsd_wsl_env.sh` 设置 Hugging Face、pip、Triton 和 torch extension 缓存路径。

## 模型管理

默认模型：

- `Tongyi-MAI/Z-Image-Turbo`
- `Qwen/Qwen3-VL-4B-Instruct`
- `black-forest-labs/FLUX.2-klein-4B`

实验模型：

- `black-forest-labs/FLUX.2-klein-9B`

模型页支持：

- 查看缓存状态、快照数量、大小和主快照路径。
- 设置每个模型的自定义本地路径。
- 直接打开模型文件夹。
- 查看 gated/auth 状态。

默认 Hugging Face 缓存路径：

```text
trainer-data/hf-home/hub
```

自定义路径配置保存到：

```text
trainer-data/models/custom-model-paths.json
```

## 数据集使用

数据集页支持两类数据：

- D-OPSD JSONL：直接验证已有 `data.jsonl`。
- 托管数据集：上传图片和同名打标文件后，由 UI 生成训练用 JSONL。

支持的打标文件：

- `image_name.txt`
- `image_name.caption`
- `image_name.json`

托管数据集导入后可以：

- 单独新增图片。
- 编辑每张图片的提示词/打标文本。
- 删除单张图片。
- 删除整个数据集。
- 在新建训练任务中选择多个数据集合并训练。

托管数据默认保存到：

```text
trainer-data/datasets/managed
```

## 训练流程

1. 打开「新建训练」。
2. 选择训练配方：
   - FLUX2 Klein Identity
   - FLUX2 Klein Editing
   - Z-Image Turbo VLM
3. 选择一个或多个数据集。
4. 使用默认的 16GB 推荐配置，或按需调整步数、样图、checkpoint、显存选项。
5. 创建草稿。
6. 在「任务」页启动任务。
7. 训练过程中查看日志、loss、GPU 占用和产物。
8. 训练结束后打开输出目录，检查 LoRA checkpoint。

当前推荐的 16GB 起步配置：

- FLUX2 Identity：`RESOLUTION_SCALE=0.625`，样图缩放 `0.5`，5 步样图和 checkpoint。
- FLUX2 Editing：`RESOLUTION_SCALE=0.5625`，样图缩放 `0.5`，5 步样图和 checkpoint。
- Z-Image Turbo：`RESOLUTION_SCALE=0.5`，2 步样图和 checkpoint。

更高分辨率、更长步数、FLUX2 9B 或 block offload 组合需要重新评估显存占用。

### 训练参数说明

- 「总步数」是主要停止条件，建议优先调整它。
- 底层脚本仍会收到 `EPOCHS`，但它由训练器按 `MAX_TRAIN_STEPS + 1` 自动派生。
- `low VRAM`、`8-bit Adam`、样图缩放和 tiled VAE 默认适合 16GB 起步配置。
- `block offload` 是高级实验选项，开启前建议先用默认配置跑通一小段训练。

## 低显存策略

训练器包含以下低显存能力：

- low VRAM 模式。
- 8-bit Adam。
- 分辨率缩放和样图分辨率缩放。
- tiled VAE encode/decode。
- 冻结组件阶段性转移到 CPU。
- 可选 Diffusers group block offload。该能力保留为实验选项，不作为默认稳定配置。

推荐先使用 UI 默认的 16GB 配置，确认数据集和 checkpoint 能正常跑通，再逐步提高分辨率或训练步数。

## 常见问题

### 停止训练后 GPU 仍然满载

任务页的停止按钮会终止 WSL runner 和训练子进程组。若 GPU 仍然满载，请在任务页刷新状态，并检查是否还有其他 Python、Accelerate、DeepSpeed 或手动启动的训练进程。

可用命令：

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "ps -ef | grep -E 'train_dopsd|accelerate|deepspeed|runner.sh' | grep -v grep || true"
nvidia-smi
```

### Electron 弹出 broken pipe / EPIPE

当前版本已把 Electron 主进程和 Next 子进程输出改为安全文件日志，日志位置：

```text
%APPDATA%\d-opsd-trainer-ui\logs\electron-main.log
```

### 模型显示 gated 或 missing

先确认 Hugging Face token 可用，并在对应模型页面接受许可。之后在模型页刷新缓存状态，或设置已经下载好的本地模型路径。

### 端口 8675 被占用

源码启动时可以换端口：

```powershell
.\D-OPSD-Trainer.cmd -Port 8680
```

### WSL 后端不可用

确认 WSL2 和 Ubuntu-22.04 已安装：

```powershell
wsl --status
wsl -l -v
```

## 开发和验证

常用检查：

```powershell
python -m unittest discover -s trainer_runtime\tests -v
npm run typecheck --prefix trainer-ui
powershell -ExecutionPolicy Bypass -File scripts\check_ui_smoke.ps1 -BaseUrl http://127.0.0.1:8675 -TimeoutSeconds 30
```

提交前建议至少确认：

- `python -m unittest discover -s trainer_runtime\tests -v` 通过。
- `npm run typecheck --prefix trainer-ui` 通过。
- `git diff --check` 没有空白错误。
- 不要提交本地私有计划或提示文件。

构建 Electron 目录包：

```powershell
npm run pack:win --prefix trainer-ui
```

构建安装包和便携版：

```powershell
npm run dist:win --prefix trainer-ui
```

完整发布检查建议使用 PowerShell 7：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\check_release_readiness.ps1 -BaseUrl http://127.0.0.1:8675 -UiTimeoutSeconds 30
```

## 上游项目

训练核心来自 D-OPSD：

- GitHub：<https://github.com/vvvvvjdy/D-OPSD>
- Paper：<https://arxiv.org/abs/2605.05204>

如果这个训练器或 D-OPSD 对你有帮助，请引用原论文：

```bibtex
@article{jiang2026dopsd,
      title={D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models},
      author={Jiang, Dengyang and Jin, Xin and Liu, Dongyang and Wang, Zanyi and Zheng, Mingzhe and Du, Ruoyi and Yang, Xiangpeng and Wu, Qilong and Li, Zhen and Gao, Peng and Yang, Harry and Hoi, Steven},
      journal={arXiv preprint arXiv:2605.05204},
      year={2026}
}
```

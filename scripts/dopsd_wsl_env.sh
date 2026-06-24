#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

export PROJECT_ROOT
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/trainer-data/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/trainer-data/pip-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/trainer-data/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/trainer-data/torch-extensions}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$PROJECT_ROOT/trainer-data/venvs/dopsd}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# DeepSpeed probes CUDA_HOME during import even when ops are not prebuilt.
# PyTorch supplies the CUDA runtime wheels; this shim only answers nvcc -V.
export CUDA_HOME="${CUDA_HOME:-$PROJECT_ROOT/trainer-data/cuda-compat}"
mkdir -p "$CUDA_HOME/bin" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  cat > "$CUDA_HOME/bin/nvcc" <<'NVCC'
#!/usr/bin/env sh
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Copyright (c) 2005-2026 NVIDIA Corporation"
echo "Built on Wed_Jun_10_00:00:00_PDT_2026"
echo "Cuda compilation tools, release 12.6, V12.6.77"
NVCC
  chmod +x "$CUDA_HOME/bin/nvcc"
fi

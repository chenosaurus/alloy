#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${ALLOY_PYTHON:-python}"
command -v nvidia-cuda-mps-control >/dev/null

mps_dir=$(mktemp -d)
export CUDA_MPS_PIPE_DIRECTORY="$mps_dir/pipe"
export CUDA_MPS_LOG_DIRECTORY="$mps_dir/log"
export CUDA_MPS_CLIENT_PRIORITY=0
export ALLOY_CAMERA_WF_SPEC="${ALLOY_CAMERA_WF_SPEC:-0}"
export ALLOY_CAMERA_LIQUID_TRIS="${ALLOY_CAMERA_LIQUID_TRIS:-196608}"
mkdir "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
    if [[ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
        timeout 8 nvidia-cuda-mps-control <<<quit || true
    fi
    rm -rf -- "$mps_dir"
}
trap cleanup EXIT
nvidia-cuda-mps-control -d

"$python" -u -m examples.livekit.yam_portal robot \
    --device cuda:0 --scene glass-water --cup-collision surface \
    --camera-backend optix --denoise optix --joined-cameras --split-render \
    --realtime-fluid --liquid-accuracy fast --spp 1 --fps 30 --width 640 --height 480 \
    --snapshot-delay-ms 30 --prewarm-splash "$@"

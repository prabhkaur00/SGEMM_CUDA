#!/usr/bin/env bash
#
# Build the sgemm binary for an A100 (sm_80) and profile kernels 1-12 with
# Nsight Compute, producing one .ncu-rep file per kernel in
# benchmark_results/ncu/. Meant to be run from a Colab cell, e.g.:
#
#   !git clone <repo> SGEMM_CUDA
#   %cd SGEMM_CUDA
#   !bash scripts/colab_profile_all_kernels.sh
#   # then zip + download:
#   !zip -r ncu_reports.zip benchmark_results/ncu
#   from google.colab import files
#   files.download('ncu_reports.zip')
#
# Notes:
# - Colab's ncu usually needs sudo (GPU perf counters require elevated
#   privileges); this script uses `sudo ncu` automatically if available.
# - Kernel 0 is cuBLAS (nothing to profile); this script does 1-12.
# - Override which kernels to run: KERNELS="1 5 10" bash scripts/colab_profile_all_kernels.sh
# - Override the ncu set (default "basic", minimal: throughput/occupancy/
#   duration/warp-stalls, ~2-5 passes): NCU_SET=full bash ... for everything.
# - sgemm.cu runs each kernel across 6 matrix sizes: 128, 256, 512, 1024,
#   2048, 4096 (index 0-5), and also launches cuBLAS kernels
#   (ampere_sgemm_*, splitKreduce_kernel) as part of its correctness check
#   on every size. --launch-count/--launch-skip count ALL of those
#   launches, not just your kernel's, so this script instead filters by
#   kernel *name* (-k) to only match this repo's sgemm* kernels, then
#   skips to size 4096 (index 5, the 6th matching launch) by default -
#   large enough that occupancy/memory-bound differences between kernels
#   1-8 actually show up (size 128 is too small to distinguish them).
#   Override with NCU_LAUNCH_SKIP=0 to profile size 128 instead, or
#   NCU_LAUNCH_COUNT=6 to profile every size for the matched kernel.

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build"
OUT_DIR="benchmark_results/ncu"
NCU_SET="${NCU_SET:-basic}"
KERNELS="${KERNELS:-1 2 3 4 5 6 7 8}"
# Sizes are [128, 256, 512, 1024, 2048, 4096] -> index 5 = 4096.
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-5}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
# All of this repo's custom kernels are named sgemm*; cuBLAS's internal
# kernels (ampere_sgemm_*, splitKreduce_kernel) do NOT match this anchored
# regex, so -k filters them out and only profiles our kernel's launches.
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-regex:^sgemm}"

mkdir -p "$OUT_DIR"

echo "== Configuring build for A100 (sm_80) =="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_COMPUTE_CAPABILITY=80 ..
make -j"$(nproc)"
cd ..

NCU_PATH="$(command -v ncu)"
if [ -z "$NCU_PATH" ]; then
  echo "ERROR: ncu not found on PATH" >&2
  exit 1
fi

NCU_CMD=("$NCU_PATH")
if command -v sudo >/dev/null 2>&1; then
  # sudo drops the caller's PATH/LD_LIBRARY_PATH, so resolve ncu's absolute
  # path first and pass the environment through explicitly. Without
  # LD_LIBRARY_PATH, root's dynamic linker may fail to find libcuda.so.1
  # even though the calling user's shell can see it fine.
  if [ -e /usr/lib64-nvidia/libcuda.so.1 ]; then
    LIBCUDA_DIR="/usr/lib64-nvidia"
  else
    LIBCUDA_DIR="$(find / -xdev -name 'libcuda.so.1' -not -path '*/compat/*' -not -path '*/.julia/*' -exec dirname {} \; 2>/dev/null | head -n1)"
  fi
  NCU_CMD=(sudo --preserve-env env "PATH=$PATH" "LD_LIBRARY_PATH=${LIBCUDA_DIR:-}:${LD_LIBRARY_PATH:-}" "$NCU_PATH")
fi

echo "Using ncu binary: $NCU_PATH"
echo "== Profiling kernels: $KERNELS (ncu --set $NCU_SET, launch-skip=$NCU_LAUNCH_SKIP launch-count=$NCU_LAUNCH_COUNT) =="
for k in $KERNELS; do
  echo ""
  echo "-- kernel $k --"
  "${NCU_CMD[@]}" --set "$NCU_SET" \
    -k "$NCU_KERNEL_REGEX" \
    --launch-skip "$NCU_LAUNCH_SKIP" --launch-count "$NCU_LAUNCH_COUNT" \
    --export "$OUT_DIR/kernel_${k}" --force-overwrite \
    "$BUILD_DIR/sgemm" "$k"
done

echo ""
echo "== Done. Reports in $OUT_DIR =="
ls -la "$OUT_DIR"

echo ""
echo "To download all reports as a zip, run in a separate cell:"
echo "  !zip -r ncu_reports.zip $OUT_DIR"
echo "  from google.colab import files; files.download('ncu_reports.zip')"

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
# - Override the ncu set (default "detailed", ~15-25 passes: adds Roofline,
#   Compute/Memory Workload Analysis, Scheduler Stats, Warp State Stats,
#   Source Counters on top of "basic"): NCU_SET=basic for the fast/minimal
#   set (~2-5 passes, no roofline), or NCU_SET=full for everything
#   (~50 passes; noticeably slower, especially on kernel 1 at size 4096).
# - sgemm.cu runs each kernel across 6 matrix sizes: 128, 256, 512, 1024,
#   2048, 4096 (index 0-5). For EACH size it launches: 1 correctness-check
#   launch of your kernel (+ cuBLAS launches) THEN a timing loop of
#   repeat_times=50 more launches of your kernel (sgemm.cu:92,133-136) -
#   so that's 51 of your kernel's launches per size, not 1. cuBLAS's
#   internal kernels (ampere_sgemm_*, splitKreduce_kernel) don't match our
#   -k filter, but the 51-per-size repeat count still means a naive
#   "--launch-skip 5" lands inside size 128, not size 4096 (this bit us:
#   it produced a 1-block grid and ncu's "0.0 full waves" warning for the
#   2D-blocktiling kernel, whose 128x128 tile means a 128x128 problem is
#   just a single block).
#   To land on the FIRST launch of a given size index, skip
#   (size_index * LAUNCHES_PER_SIZE) prior launches. Default targets size
#   4096 (index 5): skip = 5 * 51 = 255, landing on that size's
#   correctness-check launch (same kernel config as the timed ones).
#   Override NCU_TARGET_SIZE_INDEX (0-5) to target a different size, or
#   set NCU_LAUNCH_SKIP directly to bypass this calculation entirely.

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build"
OUT_DIR="benchmark_results/ncu"
NCU_SET="${NCU_SET:-detailed}"
KERNELS="${KERNELS:-1 2 3 4 5 6 7 8}"
# 1 correctness-check launch + repeat_times(50) timing launches, per size
# (sgemm.cu:92,100-136). Update this if repeat_times changes upstream.
LAUNCHES_PER_SIZE=51
# Sizes are [128, 256, 512, 1024, 2048, 4096] -> index 5 = 4096.
NCU_TARGET_SIZE_INDEX="${NCU_TARGET_SIZE_INDEX:-5}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-$((NCU_TARGET_SIZE_INDEX * LAUNCHES_PER_SIZE))}"
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

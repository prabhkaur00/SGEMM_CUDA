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
# - Override the ncu set (default "full", which is slow): NCU_SET=basic bash ...

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build"
OUT_DIR="benchmark_results/ncu"
NCU_SET="${NCU_SET:-full}"
KERNELS="${KERNELS:-1 2 3 4 5 6 7 8 9 10 11 12}"

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
  # sudo drops the caller's PATH, so resolve ncu's absolute path first
  # and preserve the environment (needed for e.g. CUDA_VISIBLE_DEVICES).
  NCU_CMD=(sudo --preserve-env env "PATH=$PATH" "$NCU_PATH")
fi

echo "Using ncu binary: $NCU_PATH"
echo "== Profiling kernels: $KERNELS (ncu --set $NCU_SET) =="
for k in $KERNELS; do
  echo ""
  echo "-- kernel $k --"
  "${NCU_CMD[@]}" --set "$NCU_SET" --export "$OUT_DIR/kernel_${k}" --force-overwrite \
    "$BUILD_DIR/sgemm" "$k"
done

echo ""
echo "== Done. Reports in $OUT_DIR =="
ls -la "$OUT_DIR"

echo ""
echo "To download all reports as a zip, run in a separate cell:"
echo "  !zip -r ncu_reports.zip $OUT_DIR"
echo "  from google.colab import files; files.download('ncu_reports.zip')"

#!/bin/bash
# Build the rms_cpp pybind11 module.
# Run from the project root: ./cpp/build.sh
# The compiled .so is placed in the project root so
# "import rms_cpp" works from any module there.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    2>&1

make -j"$(nproc)" 2>&1

echo ""
echo "Build complete. Module written to:"
ls "$SCRIPT_DIR/../"rms_cpp*.so 2>/dev/null || echo "  (no .so found — check build output above)"

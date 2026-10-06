#!/bin/bash
# Build NeTrainSim on Linux (DGX Spark / Ubuntu)
# Prerequisites: sudo apt install cmake qt6-base-dev libqt6core5compat0-dev

set -e

cmake -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GUI=OFF \
  -DBUILD_SERVER=OFF

cmake --build build-linux --target NeTrainSimConsole -j$(nproc)

echo "Build complete: build-linux/src/NeTrainSimConsole/NeTrainSim"

#!/usr/bin/env bash
# Clone the two upstream repositories at the commits used in this project.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/qg-beta-sr/.git" ]; then
  git clone https://github.com/ananthu545/quasi-geostrophic-beta-plane-super-resolution \
    "$ROOT/third_party/qg-beta-sr"
fi
git -C "$ROOT/third_party/qg-beta-sr" checkout 5a236ca66f093a8e4192f7eec0cf8c54922f8274
if [ ! -d "$ROOT/third_party/qg-2d/.git" ]; then
  git clone https://github.com/ananthu545/qg-2d "$ROOT/third_party/qg-2d"
fi
git -C "$ROOT/third_party/qg-2d" checkout 41149bca3f74c4ab61e65393b94c058f6604771f
echo "third_party ready (MIT; see third_party/*/LICENSE)"

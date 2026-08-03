#!/usr/bin/env bash
# 构建静态阅读站,输出到 docs/(GitHub Pages 从 main/docs 服)。用法: bash docs/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
python3 "$HERE/build.py"

#!/usr/bin/env bash
# 构建静态阅读站。用法: ./site/build.sh  或  bash site/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
python3 "$HERE/build.py"

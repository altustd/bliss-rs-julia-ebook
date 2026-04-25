#!/bin/bash
# Render the BLISS-RS Julia ebook via pixi.
# Always use this script instead of calling quarto directly.
set -euo pipefail
cd "$(dirname "$0")"
pixi run render "$@"

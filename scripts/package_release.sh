#!/usr/bin/env bash
# Backward-compatible release entry point.  The Python builder keeps all writes
# inside dist/ and creates both archive formats plus checksums.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/build_release.py "$@"

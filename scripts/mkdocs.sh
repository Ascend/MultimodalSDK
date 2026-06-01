#!/usr/bin/env bash
# MkDocs wrapper (Material theme). Suppresses the MkDocs 2.0 advisory banner.
# See: https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/
set -euo pipefail
export NO_MKDOCS_2_WARNING=1
exec mkdocs "$@"

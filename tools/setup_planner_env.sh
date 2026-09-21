#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${workspace_root}"

UV_PROJECT_ENVIRONMENT="${workspace_root}/.venv-planner" \
  uv sync --python 3.11 --extra planner

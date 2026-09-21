#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${workspace_root}/.venv/bin/g1-tabletop" "$@"

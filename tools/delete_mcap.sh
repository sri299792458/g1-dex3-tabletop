#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="${workspace_root}/runs"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 runs/tabletop_<UTC>/raw_episode/bag/<file>.mcap" >&2
    exit 2
fi

if [[ ! -f "$1" ]]; then
    echo "MCAP file does not exist: $1" >&2
    exit 1
fi

target="$(realpath -- "$1")"
case "${target}" in
    "${runs_root}"/tabletop_*/raw_episode/bag/*.mcap) ;;
    *)
        echo "refusing to delete outside runs/tabletop_*/raw_episode/bag/*.mcap" >&2
        exit 1
        ;;
esac

size_bytes="$(stat --format='%s' -- "${target}")"
size_human="$(numfmt --to=iec-i --suffix=B "${size_bytes}")"
echo "MCAP: ${target}"
echo "Size: ${size_human} (${size_bytes} bytes)"
read -r -p "Type DELETE to permanently remove this file: " confirmation
if [[ "${confirmation}" != "DELETE" ]]; then
    echo "not deleted"
    exit 1
fi

rm -- "${target}"
echo "deleted ${target}; freed ${size_human}"

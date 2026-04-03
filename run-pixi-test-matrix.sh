#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENVS=(
    py310-np1
    py310-np2
    py311-np1
    py311-np2
    py312-np1
    py312-np2
    py313-np2
    py314-np2
)

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${ENVS[@]}"
    exit 0
fi

FAILURES=()

for env in "${ENVS[@]}"; do
    echo "==> Running pytest in ${env}"
    if ! pixi run -e "$env" pytest "$@"; then
        FAILURES+=("$env")
    fi
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo
    echo "Pytest failed in these environments:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo
echo "Pytest passed in all environments."

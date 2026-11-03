#!/usr/bin/env bash
# Install model environments on Linux/CUDA. Does not load models or control a robot.
set -euo pipefail
REVERSE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${REVERSE_ROOT}/runtime/uv_cache"
export UV_PYTHON_INSTALL_DIR="${REVERSE_ROOT}/runtime/python"
if command -v uv >/dev/null 2>&1; then
  REVERSE_UV="$(command -v uv)"
else
  python3 -m pip install --disable-pip-version-check --target "${REVERSE_ROOT}/runtime/bootstrap_tools" 'uv>=0.8,<1'
  REVERSE_UV="${REVERSE_ROOT}/runtime/bootstrap_tools/bin/uv"
fi
for kind in siglip yolo; do
  if [[ ! -x "${REVERSE_ROOT}/.venv/${kind}/bin/python" ]]; then
    "$REVERSE_UV" venv --python 3.12 "${REVERSE_ROOT}/.venv/${kind}"
  fi
  "$REVERSE_UV" pip install --python "${REVERSE_ROOT}/.venv/${kind}/bin/python" -r "${REVERSE_ROOT}/requirements-${kind}.txt"
done
python3 -B "${REVERSE_ROOT}/check_runtime.py"

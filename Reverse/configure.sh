#!/usr/bin/env bash
# Source --configure in the host application; imports alone do not start models.
set -euo pipefail
REVERSE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:---help}" in
  --configure)
    python3 -B "${REVERSE_ROOT}/check_activation.py" --configuration-only
    export ROBOT_STATION_REVERSE_VARIANT=7of10-hold20-history8-siglip362
    export ROBOT_STATION_REVERSE_AUTOWARMUP=1
    ;;
  --check-only)
    python3 -B "${REVERSE_ROOT}/check_activation.py"
    ;;
  --check-runtime)
    python3 -B "${REVERSE_ROOT}/check_runtime.py"
    ;;
  --warmup-only)
    PYTHONPATH="${REVERSE_ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}" python3 -B -m Reverse.warmup_check
    ;;
  --help|-h)
    echo 'source Reverse/configure.sh --configure: configure host startup'
    echo 'bash Reverse/configure.sh --check-only: verify model files without loading models'
    echo 'bash Reverse/configure.sh --check-runtime: GPU-hidden dependency imports'
    echo 'bash Reverse/configure.sh --warmup-only: explicit model-only GPU diagnostic'
    ;;
  *) echo 'Unknown option. Use --help.' >&2; exit 2 ;;
esac

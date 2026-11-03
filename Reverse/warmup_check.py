"""Explicit model-only diagnostic; never enabled implicitly by imports."""
import json
import time
from .runtime import ReverseRuntime


def main():
    service = ReverseRuntime()
    try:
        service.warmup_background()
        deadline = time.monotonic() + 610
        while service.warming and time.monotonic() < deadline:
            time.sleep(.1)
        result = service.status()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())

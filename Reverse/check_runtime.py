"""Check imports with GPUs hidden. Does not instantiate models or robot IO."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    report = {}
    for kind in ("siglip","yolo"):
        python = os.environ.get("REVERSE_"+kind.upper()+"_PYTHON", str(ROOT/".venv"/kind/"bin/python"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", TF_CPP_MIN_LOG_LEVEL="3",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", YOLO_AUTOINSTALL="false",
                   YOLO_CONFIG_DIR=str(ROOT/"runtime/yolo_config"))
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT/"vendor/big_vision"),str(ROOT/"vendor/retail_yolo_runtime"),
                                            str(ROOT/"runtime/siglip/scripts"),str(ROOT/"scripts"),str(ROOT.parent)])
        code = ('import infer_positive_48prompts; import big_vision.models.proj.image_text.two_towers; '
                'import big_vision.models.proj.image_text.text_transformer; '
                'import big_vision.pp.ops_general, big_vision.pp.ops_image, big_vision.pp.ops_text; '
                'print("IMPORTS_OK")') if kind == "siglip" else (
                'import torch, torchvision, ultralytics; from ultralytics.cfg import DEFAULT_CFG_DICT; print("IMPORTS_OK")')
        try:
            result = subprocess.run([python,"-c",code],env=env,capture_output=True,text=True,timeout=90)
            report[kind] = dict(ok=result.returncode==0 and "IMPORTS_OK" in result.stdout,
                                python=python, error=result.stderr[-2000:] if result.returncode else "")
        except (OSError,subprocess.TimeoutExpired) as exc:
            report[kind] = dict(ok=False,python=python,error=str(exc))
    print(json.dumps(dict(ok=all(x["ok"] for x in report.values()), model_loaded=False,
                         gpu_hidden=True, workers=report),indent=2,ensure_ascii=False))
    return int(not all(x["ok"] for x in report.values()))


if __name__ == "__main__":
    raise SystemExit(main())

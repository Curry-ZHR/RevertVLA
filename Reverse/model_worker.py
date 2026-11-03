#!/usr/bin/env python3
"""Private stdio model worker. No network listener, ROS or robot command imports."""
import argparse
import base64
from contextlib import redirect_stdout
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROTOCOL = sys.stdout


def send(value):
    PROTOCOL.write(json.dumps(value, separators=(",", ":")) + "\n")
    PROTOCOL.flush()


def siglip():
    from Reverse.siglip_scorer import SiglipScorer
    from Reverse.siglip_delivery import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, CHECKPOINT_STEP
    scorer = SiglipScorer(ROOT / "runtime/siglip", DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER)
    send(dict(ready=True, kind="siglip", checkpoint_step=CHECKPOINT_STEP, device=str(scorer.device),
              checkpoint=str(DEFAULT_CHECKPOINT), objects=len(scorer.names), object_names=scorer.names,
              warmup_ms=scorer.warmup_ms, prompts=len(scorer.specs)))
    return scorer


def yolo():
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from Reverse.yolo_delivery import DEFAULT_DIRECTORY, load_delivery
    directory = DEFAULT_DIRECTORY
    config, delivered, weights = load_delivery(directory)
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    names = {int(k): v for k,v in model.names.items()}
    if set(names) != set(range(71)) or any(names[k] != v for k,v in delivered.items()):
        raise ValueError("YOLO embedded classes differ from delivery")
    device = os.environ.get("REVERSE_YOLO_DEVICE", "0")
    options = dict(imgsz=config["imgsz"], conf=config["confidence"], iou=config["nms_iou"],
                   end2end=False, classes=None, max_det=100, agnostic_nms=False,
                   rect=False, device=device, verbose=False, save=False)
    if "quantize" in DEFAULT_CFG_DICT:
        options["quantize"] = None if device == "cpu" else 16
    else:
        options["half"] = device != "cpu"
    started = time.monotonic()
    model.predict(source=np.zeros((640,640,3), dtype=np.uint8), **options)
    send(dict(ready=True, kind="yolo", classes=71, device=device,
              model_directory=str(directory), weights=str(weights),
              weights_sha256=config["weights_sha256"], confidence=config["confidence"], postprocess=True,
              warmup_ms=(time.monotonic()-started)*1000))
    return model, names, options, config["postprocess"]


def score(scorer, request):
    values = []
    for frame in request["frames"]:
        raw = base64.b64decode(frame["image"], validate=True)
        result = scorer.judge(raw, frame["target"], frame["side"], .05, .01)
        values.append({**result, "side": frame["side"], "sequence": frame["sequence"],
                       "received_ns": frame["received_ns"]})
    return dict(decisions=values)


def mask(model_data, request):
    import numpy as np
    from PIL import Image
    from Reverse.core import canonical
    from Reverse.yolo_delivery import transform_box
    model, names, options, postprocess = model_data
    with Image.open(BytesIO(base64.b64decode(request["image"], validate=True))) as image:
        rgb = np.asarray(image.convert("RGB"))
    result = model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]), **options)[0]
    protected_targets = {canonical(name) for name in request["targets"]}
    blocked = np.zeros(rgb.shape[:2], dtype=bool)
    protected = np.zeros_like(blocked)
    boxes = []
    for box in result.boxes.cpu() if result.boxes is not None else []:
        class_id = int(box.cls.item())
        if class_id not in names:
            raise ValueError("Unknown YOLO class")
        values = box.xyxy[0].tolist()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite YOLO box")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("Invalid YOLO box")
        height,width = blocked.shape
        x1,y1,x2,y2 = transform_box(values, class_id, "head", postprocess, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        a,b,c,d = max(0,int(np.floor(x1))), max(0,int(np.floor(y1))), min(width,int(np.ceil(x2))), min(height,int(np.ceil(y2)))
        is_target = canonical(names[class_id]) in protected_targets
        (protected if is_target else blocked)[b:d,a:c] = True
        if not is_target:
            boxes.append(dict(class_id=class_id, name=names[class_id], bbox=[a,b,c,d]))
    blocked &= ~protected
    redacted = np.array(rgb, copy=True)
    redacted[blocked] = 0
    output = BytesIO()
    Image.fromarray(redacted).save(output, format="JPEG", quality=95)
    return dict(image=base64.b64encode(output.getvalue()).decode("ascii"),
                mask=dict(model="sku71_best_v42", camera="head", postprocess_applied=True, masked_boxes=boxes, masked_pixels=int(blocked.sum()),
                          target_overlap_protected=True, wrist_images_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("siglip","yolo"))
    args = parser.parse_args()
    # Third-party startup/progress output must never corrupt stdio messages.
    with redirect_stdout(sys.stderr):
        model = siglip() if args.kind == "siglip" else yolo()
        for line in sys.stdin:
            try:
                if len(line) > 16*1024*1024:
                    raise ValueError("Input too large")
                request = json.loads(line)
                started = time.monotonic()
                value = score(model,request) if args.kind == "siglip" and request.get("op") == "score" else (
                    mask(model,request) if args.kind == "yolo" and request.get("op") == "mask" else None)
                if value is None:
                    raise ValueError("Unsupported operation")
                send({**value,"infer_ms":(time.monotonic()-started)*1000})
            except Exception as exc:
                send(dict(error=f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send(dict(error=f"{type(exc).__name__}: {exc}"))
        raise

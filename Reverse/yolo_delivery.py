"""SKU71 v42 delivery validation and post-NMS geometry; no model imports."""
import hashlib
import json
import math
from pathlib import Path

DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints/yolo/delivery_best71_v42"


def load_delivery(directory=DEFAULT_DIRECTORY):
    directory = Path(directory)
    weights = directory / "best71_v42.pt"
    postprocess = json.loads((directory / "postprocess.json").read_text())
    expected = "720cbd8a2b56674b65cb75b3cbc64175be26b55f7690d2620042e9e51ad3a866"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if postprocess.get("base_weights_sha256") != expected or digest.hexdigest() != expected:
        raise ValueError("YOLO v42 delivery checksum mismatch")
    if (postprocess.get("schema_version") != 2 or
            postprocess.get("order") != ["class_box_scale", "class_camera_geometry"] or
            postprocess.get("applied_after_nms") is not True or
            postprocess.get("clip_to_image") is not True):
        raise ValueError("Unsupported YOLO v42 postprocess configuration")
    names = {int(k): v for k, v in json.loads((directory / "classes.json").read_text()).items()}
    if set(names) != set(range(71)) or any(not isinstance(v, str) or not v for v in names.values()):
        raise ValueError("YOLO v42 requires the complete 71-class table")
    # Defaults from the v42 delivery's prediction interface and README.
    config = dict(imgsz=640, confidence=.20, nms_iou=.45, internal_model_classes=71,
                  weights=weights.name, weights_sha256=expected, postprocess=postprocess)
    return config, names, weights


def transform_box(box, class_id, camera, config, width, height):
    """Match v42: class scale, camera geometry, then clipping to image bounds."""
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = float(config.get("class_box_scale", {}).get(str(class_id), 1.))
    box_width *= scale
    box_height *= scale
    geometry = config.get("class_camera_geometry", {}).get(f"{class_id}:{camera}")
    if geometry:
        center_x += float(geometry["shift_x_widths"]) * box_width
        center_y += float(geometry["shift_y_heights"]) * box_height
        box_width *= float(geometry["scale_x"])
        box_height *= float(geometry["scale_y"])
    values = [center_x-box_width/2, center_y-box_height/2,
              center_x+box_width/2, center_y+box_height/2]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("YOLO v42 produced nonfinite calibrated coordinates")
    return [max(0., min(float(bound), v)) for v, bound in zip(values, (width,height,width,height))]

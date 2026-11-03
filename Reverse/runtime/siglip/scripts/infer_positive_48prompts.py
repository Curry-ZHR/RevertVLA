#!/usr/bin/env python3
"""Sample positive Zerith images and visualize all 24-SKU SigLIP prompts."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import importlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf

# TensorFlow is only used for the exact training preprocessing pipeline. JAX
# owns the accelerator for model inference.
try:
  tf.config.set_visible_devices([], "GPU")
except RuntimeError:
  pass

import jax  # pylint: disable=g-import-not-at-top

from big_vision import utils as bv_utils  # pylint: disable=g-import-not-at-top
from big_vision.pp import builder as pp_builder  # pylint: disable=g-import-not-at-top


DEFAULT_TOKENIZER = str(Path(__file__).resolve().parents[3] / "checkpoints/siglip/cc_en.32000.sentencepiece.model")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Sample positive images per SKU and score every image against all "
          "24 SKUs x 2 hands (48 prompts)."
      )
  )
  parser.add_argument(
      "--image",
      "--image-dir",
      dest="image_dir",
      required=True,
      help="Directory searched recursively for JPEG/PNG/WebP images.",
  )
  parser.add_argument(
      "--sample-count", type=int, default=30, help="Images sampled without replacement."
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for reproducible sampling."
  )
  parser.add_argument(
      "--sample-objects",
      nargs="+",
      help=(
          "Optional object names for balanced manifest-aware sampling. Requires "
          "--samples-per-object and samples exactly that many images per object."
      ),
  )
  parser.add_argument(
      "--samples-per-object",
      type=int,
      help="Images sampled without replacement for each --sample-objects value.",
  )
  parser.add_argument(
      "--checkpoint",
      required=True,
      help=(
          "Checkpoint directory to load, for example "
          "/path/to/workdir/checkpoint.bv-000000060."
      ),
  )
  parser.add_argument(
      "--objects-file",
      required=True,
      help="Text file containing the 24 canonical SKU names, one per line.",
  )
  parser.add_argument(
      "--tokenizer-model", default=DEFAULT_TOKENIZER, help="SentencePiece model."
  )
  parser.add_argument(
      "--manifest",
      help="Optional manifest.jsonl path; otherwise search image-dir parents.",
  )
  parser.add_argument(
      "--output-dir",
      help=(
          "Output directory. Defaults to "
          "<repo>/inference_outputs/<workdir>_<timestamp>."
      ),
  )
  args = parser.parse_args()

  if args.sample_count <= 0:
    parser.error("--sample-count must be positive")
  if (args.sample_objects is None) != (args.samples_per_object is None):
    parser.error("--sample-objects and --samples-per-object must be used together")
  if args.samples_per_object is not None and args.samples_per_object <= 0:
    parser.error("--samples-per-object must be positive")
  return args


def resolve_checkpoint(checkpoint):
  checkpoint_path = Path(checkpoint).expanduser().resolve()
  if not re.fullmatch(r"checkpoint\.bv-\d{9}", checkpoint_path.name):
    raise ValueError(
        "--checkpoint must be an explicit checkpoint.bv-######### directory"
    )
  if not checkpoint_path.is_dir():
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
  workdir = checkpoint_path.parent
  if not (workdir / "config.json").is_file():
    raise FileNotFoundError(
        f"Training config not found next to checkpoint: {workdir / 'config.json'}"
    )
  return workdir, checkpoint_path


def load_config(workdir):
  config_path = workdir / "config.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"Training config not found: {config_path}")
  with config_path.open(encoding="utf-8") as f:
    return json.load(f)


def find_and_sample_images(image_dir, sample_count, seed):
  image_dir = Path(image_dir).expanduser().resolve()
  if not image_dir.is_dir():
    raise NotADirectoryError(f"Image directory not found: {image_dir}")

  candidates = sorted(
      path.resolve()
      for path in image_dir.rglob("*")
      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  )
  if len(candidates) < sample_count:
    raise ValueError(
        f"Requested {sample_count} images, but only found {len(candidates)} "
        f"under {image_dir}"
    )
  selected = random.Random(seed).sample(candidates, sample_count)
  return image_dir, candidates, selected


def find_manifest(image_dir, explicit_manifest):
  if explicit_manifest:
    manifest = Path(explicit_manifest).expanduser().resolve()
    if not manifest.is_file():
      raise FileNotFoundError(f"Manifest not found: {manifest}")
    return manifest

  for parent in (image_dir, *image_dir.parents):
    manifest = parent / "manifest.jsonl"
    if manifest.is_file():
      return manifest
  raise FileNotFoundError(
      f"No manifest.jsonl found in {image_dir} or any parent directory"
  )


def sample_images_by_object(
    candidates, manifest, sample_objects, samples_per_object, seed
):
  candidate_by_path = {str(path): path for path in candidates}
  groups = {
      object_name: {"left": defaultdict(list), "right": defaultdict(list)}
      for object_name in sample_objects
  }
  seen = {object_name: set() for object_name in sample_objects}
  with manifest.open(encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
      target_object = row.get("target_object") or row.get("image_object")
      if target_object not in groups:
        continue
      is_positive = row.get("pair_is_positive")
      if is_positive is False or row.get("pair_label") == 0:
        continue
      if is_positive is None and row.get("label", "positive") != "positive":
        continue
      hand = row.get("active_hand") or row.get("hand")
      if hand not in ("left", "right"):
        raise ValueError(f"Missing left/right hand for {target_object}: {row}")
      for key in _manifest_image_keys(row, manifest.parent):
        image_path = candidate_by_path.get(key)
        if image_path is not None and key not in seen[target_object]:
          source_group = row.get("source_video") or (
              f"{row.get('scene_id')}:{row.get('episode_id')}:"
              f"{hand}"
          )
          groups[target_object][hand][source_group].append(image_path)
          seen[target_object].add(key)
          break

  selected = []
  for object_index, object_name in enumerate(sample_objects):
    hand_groups = groups[object_name]
    available_count = sum(
        len(paths)
        for object_groups in hand_groups.values()
        for paths in object_groups.values()
    )
    if available_count < samples_per_object:
      raise ValueError(
          f"Requested {samples_per_object} images for {object_name}, but only "
          f"found {available_count} candidates in the manifest"
      )

    # For three samples, alternate 2L+1R and 1L+2R by SKU. This guarantees
    # both hand prompts are exercised while keeping the full set balanced.
    extra_hand = "left" if object_index % 2 == 0 else "right"
    quotas = {
        "left": samples_per_object // 2,
        "right": samples_per_object // 2,
    }
    quotas[extra_hand] += samples_per_object % 2
    object_selection = []
    for hand, quota in quotas.items():
      object_groups = hand_groups[hand]
      if sum(len(paths) for paths in object_groups.values()) < quota:
        raise ValueError(
            f"Not enough {hand} positive images for {object_name}: need {quota}"
        )
      group_keys = sorted(object_groups)
      rng = random.Random(f"{seed}:{object_name}:{hand}")
      rng.shuffle(group_keys)
      for group_key in group_keys[:quota]:
        object_selection.append(rng.choice(sorted(object_groups[group_key])))
    random.Random(f"{seed}:{object_name}:output").shuffle(object_selection)
    selected.extend(object_selection)
  return selected


def _farthest_visual_sample(paths, sample_count, seed):
  paths = sorted(paths)
  if sample_count <= 0:
    return []
  if sample_count >= len(paths):
    return paths

  features = []
  for path in paths:
    with Image.open(path) as image:
      thumbnail = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
      features.append(np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0)
  features = np.stack(features)

  rng = random.Random(seed)
  selected_indices = [rng.randrange(len(paths))]
  tie_breakers = np.asarray([rng.random() for _ in paths])
  min_distances = np.full(len(paths), np.inf, dtype=np.float32)
  while len(selected_indices) < sample_count:
    latest = features[selected_indices[-1]]
    distances = np.mean(np.square(features - latest), axis=1)
    min_distances = np.minimum(min_distances, distances)
    min_distances[selected_indices] = -1.0
    best_distance = float(np.max(min_distances))
    tied = np.flatnonzero(np.isclose(min_distances, best_distance, rtol=1e-6))
    selected_indices.append(int(tied[np.argmax(tie_breakers[tied])]))
  return [paths[index] for index in selected_indices]


def _manifest_image_keys(row, manifest_dir):
  keys = []
  if row.get("image_path"):
    keys.append(str(Path(row["image_path"]).expanduser().resolve()))
  if row.get("relative_image_path"):
    keys.append(str((manifest_dir / row["relative_image_path"]).resolve()))
  if row.get("output_path"):
    keys.append(str((manifest_dir / row["output_path"]).resolve()))
  return keys


def load_positive_metadata(manifest, selected_images):
  wanted = {str(path): path for path in selected_images}
  metadata = {}
  with manifest.open(encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc

      is_positive = row.get("pair_is_positive")
      if is_positive is False or row.get("pair_label") == 0:
        continue
      if is_positive is None and row.get("label", "positive") != "positive":
        continue

      for key in _manifest_image_keys(row, manifest.parent):
        if key in wanted and key not in metadata:
          metadata[key] = row

  missing = [str(path) for path in selected_images if str(path) not in metadata]
  if missing:
    preview = "\n".join(missing[:5])
    raise ValueError(
        f"No positive manifest metadata for {len(missing)} sampled images. "
        f"First missing paths:\n{preview}"
    )
  return metadata


def make_preprocessors(tokenizer_model):
  for module in ("ops_general", "ops_image", "ops_text"):
    importlib.import_module(f"big_vision.pp.{module}")

  image_pp = pp_builder.get_preprocess_fn(
      'decode|resize(224)|value_range(-1,1)|keep("image")', log_data=False
  )
  text_pp = pp_builder.get_preprocess_fn(
      (
          f'tokenize(max_len=64, model="{tokenizer_model}", '
          'eos="sticky", pad_value=1, inkey="text")|keep("labels")'
      ),
      log_data=False,
  )
  return image_pp, text_pp


def preprocess_images(image_paths, image_pp):
  images = []
  for image_path in image_paths:
    encoded = tf.io.read_file(str(image_path))
    images.append(image_pp({"image": encoded})["image"].numpy())
  return np.stack(images, axis=0)


def preprocess_prompts(prompts, text_pp):
  tokens = [
      text_pp({"text": tf.constant(prompt)})["labels"].numpy()
      for prompt in prompts
  ]
  return np.stack(tokens, axis=0)


def load_objects(objects_file):
  objects_path = Path(objects_file).expanduser().resolve()
  if not objects_path.is_file():
    raise FileNotFoundError(f"Object label file not found: {objects_path}")
  objects = [
      line.strip()
      for line in objects_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if len(objects) != 24:
    raise ValueError(f"Expected 24 SKU labels, found {len(objects)}")
  return objects


def build_prompt_specs(objects):
  specs = []
  for object_name in objects:
    for hand in ("left", "right"):
      specs.append({
          "task_index": len(specs),
          "object": object_name,
          "hand": hand,
          "prompt": f"Grasp {object_name} with the {hand} hand.",
      })
  if len(specs) != 48:
    raise AssertionError(f"Expected 48 prompts, built {len(specs)}")
  return specs


def load_model(config, checkpoint_path):
  model_name = config.get("model_name")
  if model_name != "proj.image_text.two_towers":
    raise ValueError(f"Unexpected model_name: {model_name}")
  model_module = importlib.import_module(f"big_vision.models.{model_name}")
  model = model_module.Model(**config["model"])

  loaded = bv_utils.load_checkpoint_ts(
      str(checkpoint_path), regex=r"params/.*"
  )
  if set(loaded) != {"params"}:
    raise ValueError(f"Expected params-only checkpoint tree, got {loaded.keys()}")
  return model, loaded["params"]


def run_model(model, params, images, tokens):
  device = jax.local_devices()[0]
  device_description = f"{device.platform}:{device.device_kind}:{device.id}"
  params = jax.device_put(params, device)
  images = jax.device_put(images, device)
  tokens = jax.device_put(tokens, device)

  @jax.jit
  def predict(model_params, model_images, model_tokens):
    zimg, ztxt, extras = model.apply(
        {"params": model_params}, model_images, model_tokens, train=False
    )
    cosine = zimg @ ztxt.T
    logits = cosine * extras["t"] + extras["b"]
    return cosine, logits, jax.nn.sigmoid(logits), extras["t"], extras["b"]

  outputs = predict(params, images, tokens)
  return [np.asarray(value) for value in jax.device_get(outputs)], device_description


def _expected_prompt(metadata):
  prompt = metadata.get("prompt")
  if prompt:
    return prompt
  object_name = metadata.get("target_object") or metadata.get("image_object")
  hand = metadata.get("active_hand") or metadata.get("hand")
  if object_name and hand:
    return f"Grasp {object_name} with the {hand} hand."
  raise ValueError("Positive manifest row has no prompt or object/hand metadata")


def build_results(selected_images, metadata_by_path, prompt_specs, outputs):
  cosine, logits, probabilities, temperature, bias = outputs
  image_results = []

  for image_index, image_path in enumerate(selected_images):
    metadata = metadata_by_path[str(image_path)]
    expected_prompt = _expected_prompt(metadata)
    scores = []
    for prompt_index, spec in enumerate(prompt_specs):
      scores.append({
          **spec,
          "cosine_similarity": float(cosine[image_index, prompt_index]),
          "logit": float(logits[image_index, prompt_index]),
          "match_probability": float(probabilities[image_index, prompt_index]),
          "match_percent": float(probabilities[image_index, prompt_index] * 100.0),
          "is_match": bool(logits[image_index, prompt_index] >= 0.0),
          "is_expected_prompt": spec["prompt"] == expected_prompt,
      })
    scores.sort(key=lambda row: row["logit"], reverse=True)
    for rank, row in enumerate(scores, start=1):
      row["rank"] = rank

    expected_score = next(
        (row for row in scores if row["is_expected_prompt"]), None
    )
    if expected_score is None:
      raise ValueError(
          f"Manifest prompt is not one of the 48 candidates: {expected_prompt}"
      )

    source_video = metadata.get("source_video")
    scene_id = metadata.get("scene_id") or metadata.get("source_scene") or "unknown"
    episode_id = metadata.get("episode_id") or (
        Path(source_video).stem if source_video else image_path.stem
    )
    image_results.append({
        "sample_index": image_index + 1,
        "image_path": str(image_path),
        "scene_id": scene_id,
        "episode_id": episode_id,
        "camera_type": metadata.get("camera_type") or (
            Path(source_video).parent.name if source_video else None
        ),
        "target_object": metadata.get("target_object") or metadata.get("image_object"),
        "active_hand": metadata.get("active_hand") or metadata.get("hand"),
        "expected_prompt": expected_prompt,
        "expected_prompt_rank": expected_score["rank"],
        "expected_prompt_match_probability": expected_score["match_probability"],
        "expected_prompt_match_percent": expected_score["match_percent"],
        "top1_prompt": scores[0]["prompt"],
        "top1_match_probability": scores[0]["match_probability"],
        "top1_match_percent": scores[0]["match_percent"],
        "top1_is_expected_prompt": scores[0]["is_expected_prompt"],
        "scores": scores,
    })

  return (
      image_results,
      float(temperature.reshape(-1)[0]),
      float(bias.reshape(-1)[0]),
  )


def resolve_output_dir(workdir, requested_output_dir):
  if requested_output_dir:
    output_dir = Path(requested_output_dir).expanduser().resolve()
  else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        workdir.parent.parent
        / "inference_outputs"
        / f"{workdir.name}_{timestamp}"
    )
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir


def _load_font(size, bold=False):
  names = (
      ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
      if bold
      else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
  )
  for name in names:
    try:
      return ImageFont.truetype(name, size=size)
    except OSError:
      continue
  return ImageFont.load_default()


def _safe_name(value):
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _fit_text(draw, text, font, max_width):
  if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
    return text
  suffix = "..."
  while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > max_width:
    text = text[:-1]
  return text + suffix


def write_annotated_image(image_result, output_dir):
  with Image.open(image_result["image_path"]) as source:
    image = source.convert("RGB")

  canvas_width = max(image.width, 2400)
  row_font = _load_font(18)
  header_font = _load_font(24, bold=True)
  summary_font = _load_font(20, bold=True)
  note_font = _load_font(16)
  temporary = Image.new("RGB", (canvas_width, 100), (24, 24, 24))
  draw = ImageDraw.Draw(temporary)
  episode_name = str(image_result["episode_id"]).rstrip("/").split("/")[-1]
  padding = 20
  column_gap = 18
  column_count = 3
  rows_per_column = 16
  column_width = (
      canvas_width - 2 * padding - (column_count - 1) * column_gap
  ) // column_count
  header_height = 30
  note_height = 22
  section_gap = 12
  table_header_height = 30
  row_height = 30
  winner_height = 34
  banner_height = (
      padding
      + header_height
      + section_gap
      + header_height
      + note_height
      + section_gap
      + table_header_height
      + rows_per_column * row_height
      + section_gap
      + winner_height
      + padding
  )
  canvas = Image.new(
      "RGB", (canvas_width, banner_height + image.height), (24, 24, 24)
  )
  canvas.paste(image, ((canvas_width - image.width) // 2, banner_height))
  draw = ImageDraw.Draw(canvas)

  y = padding
  scene_line = (
      f"Scene: {image_result['scene_id']} | Episode: {episode_name} | "
      f"File: {Path(image_result['image_path']).name}"
  )
  draw.text((padding, y), scene_line, fill=(250, 250, 250), font=header_font)
  y += header_height + section_gap

  gt_line = (
      f"Original prompt: {image_result['expected_prompt']} | Match: "
      f"{image_result['expected_prompt_match_percent']:.2f}% | Rank: "
      f"{image_result['expected_prompt_rank']}/48"
  )
  draw.text((padding, y), gt_line, fill=(125, 200, 255), font=summary_font)
  y += header_height
  draw.text(
      (padding, y),
      "Match = sigmoid(logit), independently computed for each image-prompt pair; percentages do not sum to 100%.",
      fill=(185, 185, 185),
      font=note_font,
  )
  y += note_height + section_gap

  column_x = tuple(
      padding + column_index * (column_width + column_gap)
      for column_index in range(column_count)
  )
  draw.rectangle(
      (padding, y, canvas_width - padding, y + table_header_height),
      fill=(45, 45, 45),
  )
  for column_index, x in enumerate(column_x):
    start = column_index * rows_per_column
    end = start + rows_per_column - 1
    draw.text(
        (x + 8, y + 4),
        f"Tasks {start:02d}-{end:02d}",
        fill=(245, 245, 245),
        font=row_font,
    )
  y += table_header_height

  scores_by_task = {row["task_index"]: row for row in image_result["scores"]}
  for row_index in range(rows_per_column):
    for column_index in range(column_count):
      task_index = row_index + column_index * rows_per_column
      score = scores_by_task[task_index]
      x = column_x[column_index]
      row_top = y + row_index * row_height
      if score["rank"] == 1 and score["is_expected_prompt"]:
        background = (65, 82, 44)
        marker = " [GT, TOP1]"
      elif score["rank"] == 1:
        background = (38, 82, 54)
        marker = " [TOP1]"
      elif score["is_expected_prompt"]:
        background = (37, 65, 91)
        marker = " [GT]"
      else:
        background = (31, 31, 31) if row_index % 2 == 0 else (38, 38, 38)
        marker = ""
      draw.rectangle(
          (x, row_top, x + column_width, row_top + row_height), fill=background
      )
      percent_text = f"{score['match_percent']:.2f}%"
      percent_width = draw.textbbox((0, 0), percent_text, font=row_font)[2]
      label = f"[{task_index:02d}] {score['prompt']}{marker}"
      label = _fit_text(
          draw, label, row_font, column_width - percent_width - 28
      )
      draw.text((x + 8, row_top + 4), label, fill=(240, 240, 240), font=row_font)
      draw.text(
          (x + column_width - percent_width - 8, row_top + 4),
          percent_text,
          fill=(255, 222, 120),
          font=row_font,
      )

  y += rows_per_column * row_height + section_gap
  highest = image_result["scores"][0]
  highest_line = (
      f"Highest Match: task_index={highest['task_index']:02d} | "
      f"{highest['prompt']} | {highest['match_percent']:.2f}%"
  )
  draw.rectangle(
      (padding, y, canvas_width - padding, y + winner_height), fill=(38, 82, 54)
  )
  draw.text(
      (padding + 8, y + 5), highest_line, fill=(255, 255, 255), font=summary_font
  )

  episode_safe = _safe_name(episode_name)
  scene_safe = _safe_name(image_result["scene_id"])
  object_safe = _safe_name(image_result["target_object"])
  hand_safe = _safe_name(image_result["active_hand"])
  output_path = output_dir / (
      f"{image_result['sample_index']:03d}_{object_safe}_{hand_safe}_"
      f"{scene_safe}_{episode_safe}_"
      f"{Path(image_result['image_path']).stem}.jpg"
  )
  canvas.save(output_path, format="JPEG", quality=95)
  image_result["annotated_image_path"] = str(output_path)


def write_csv(image_results, output_path):
  fields = [
      "sample_index",
      "scene_id",
      "episode_id",
      "camera_type",
      "image_path",
      "target_object",
      "active_hand",
      "expected_prompt",
      "task_index",
      "prompt",
      "object",
      "hand",
      "rank",
      "cosine_similarity",
      "logit",
      "match_probability",
      "match_percent",
      "is_match",
      "is_expected_prompt",
  ]
  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for image_result in image_results:
      common = {
          key: image_result.get(key)
          for key in (
              "sample_index",
              "scene_id",
              "episode_id",
              "camera_type",
              "image_path",
              "target_object",
              "active_hand",
              "expected_prompt",
          )
      }
      for score in image_result["scores"]:
        writer.writerow({**common, **score})


def print_summary(report):
  summary = report["summary"]
  print(f"workdir:         {report['workdir']}")
  print(f"checkpoint:      {report['checkpoint']}")
  print(f"device:          {report['device']}")
  print(f"image_directory: {report['image_directory']}")
  print(f"manifest:        {report['manifest']}")
  print(f"sampled:         {summary['sampled_images']}/{summary['candidate_images']}")
  print(f"prompt_count:    {summary['prompt_count']}")
  print(f"top1_accuracy:   {summary['top1_prompt_accuracy_percent']:.2f}%")
  print(f"mean_gt_match:   {summary['mean_expected_prompt_match_percent']:.2f}%")
  print(f"output_dir:      {report['output_dir']}")
  for image_result in report["images"]:
    episode_name = str(image_result["episode_id"]).rstrip("/").split("/")[-1]
    print(
        f"[{image_result['sample_index']:02d}] "
        f"{image_result['scene_id']}/{episode_name} "
        f"GT={image_result['expected_prompt_match_percent']:.2f}% "
        f"rank={image_result['expected_prompt_rank']}/48 "
        f"top1={image_result['top1_prompt']} "
        f"({image_result['top1_match_percent']:.2f}%)"
    )


def main():
  args = parse_args()
  workdir, checkpoint_path = resolve_checkpoint(args.checkpoint)
  config = load_config(workdir)
  image_dir, candidates, selected_images = find_and_sample_images(
      args.image_dir,
      args.sample_count if args.sample_objects is None else 1,
      args.seed,
  )
  manifest = find_manifest(image_dir, args.manifest)
  if args.sample_objects is not None:
    if len(set(args.sample_objects)) != len(args.sample_objects):
      raise ValueError("--sample-objects contains duplicate object names")
    selected_images = sample_images_by_object(
        candidates,
        manifest,
        args.sample_objects,
        args.samples_per_object,
        args.seed,
    )
  metadata_by_path = load_positive_metadata(manifest, selected_images)

  tokenizer_path = Path(args.tokenizer_model).expanduser().resolve()
  if not tokenizer_path.is_file():
    raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
  objects = load_objects(args.objects_file)
  prompt_specs = build_prompt_specs(objects)
  image_pp, text_pp = make_preprocessors(str(tokenizer_path))
  images = preprocess_images(selected_images, image_pp)
  tokens = preprocess_prompts([spec["prompt"] for spec in prompt_specs], text_pp)

  model, params = load_model(config, checkpoint_path)
  outputs, device = run_model(model, params, images, tokens)
  image_results, temperature, bias = build_results(
      selected_images, metadata_by_path, prompt_specs, outputs
  )

  output_dir = resolve_output_dir(workdir, args.output_dir)
  for image_result in image_results:
    write_annotated_image(image_result, output_dir)

  top1_correct = sum(row["top1_is_expected_prompt"] for row in image_results)
  mean_expected = float(np.mean([
      row["expected_prompt_match_percent"] for row in image_results
  ]))
  report = {
      "workdir": str(workdir),
      "checkpoint": str(checkpoint_path),
      "device": device,
      "image_directory": str(image_dir),
      "manifest": str(manifest),
      "output_dir": str(output_dir),
      "random_seed": args.seed,
      "sample_objects": args.sample_objects,
      "samples_per_object": args.samples_per_object,
      "temperature": temperature,
      "bias": bias,
      "score_definition": "sigmoid(cosine_similarity * temperature + bias)",
      "score_note": (
          "Independent image-prompt match probability; not a 48-class softmax."
      ),
      "summary": {
          "candidate_images": len(candidates),
          "sampled_images": len(selected_images),
          "prompt_count": len(prompt_specs),
          "score_rows": len(selected_images) * len(prompt_specs),
          "top1_prompt_accuracy": top1_correct / len(image_results),
          "top1_prompt_accuracy_percent": top1_correct / len(image_results) * 100.0,
          "mean_expected_prompt_match_percent": mean_expected,
      },
      "images": image_results,
  }

  report_path = output_dir / "results.json"
  report_path.write_text(
      json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
  )
  write_csv(image_results, output_dir / "scores.csv")
  print_summary(report)
  print(f"results_json:    {report_path}")
  print(f"scores_csv:      {output_dir / 'scores.csv'}")


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:  # Keep CLI failures concise while preserving a code.
    print(f"error: {exc}", file=sys.stderr)
    raise

# RevertVLA

**Pre-close verification and executed-action rollback for vision-language-action policies.**

[中文说明](README.zh-CN.md) · [Model downloads](https://github.com/Curry-ZHR/RevertVLA/releases/tag/v0.1.1) · [Integration guide](docs/integration.md)

RevertVLA checks the intended grasp target using wrist-camera SigLIP scores,
holds a rejected close for a bounded observation window, replays the failing
hand's recently issued actions, and masks non-target objects in the head image
before asking the VLA policy to try again. The other hand retains its current
targets during replay.

This first release contains the recovery core, isolated perception workers,
checkpoint/configuration checks, offline tests, and executor integration contracts.
The VLA policy server, robot SDK, ROS installation, and the general Robot Station
web application are host dependencies. VLA checkpoints and experimental gripper
ROI/segmentation code are outside this release.

## Recovery loop

```text
fresh observation -> VLA / RTC -> prospective gripper close
                                       |
                           SigLIP wrist verification
                              /                  \
                           PASS                 WAIT
                             |                    |
                         close once      hold pose, keep hand open
                                         for 20 control periods
                                            /            \
                                         PASS           timeout
                                           |               |
                                    close + replan    replay wrong hand
                                                           |
                                        fresh head image + YOLO mask
                                                           |
                                                        VLA retry
```

| Setting | Released value |
|---|---|
| SigLIP delivery | B/16-224, checkpoint **362**, **71** SKUs, **142** hand-specific prompts |
| Pass window | 10 distinct, ordered wrist frames; at least **7 PASS** and **latest two PASS** |
| Frame freshness | Oldest vote ≤ 1 s; newest vote ≤ 0.5 s |
| Match threshold / margin | 0.05 / 0.01, independent sigmoid scores |
| HOLD | **20** successful open publications, then evaluate the latest window |
| History | Latest **8** successfully issued chunk groups, including a partial chunk |
| Retry masking | Head image only; persists until an approved close succeeds |
| YOLO delivery | `best71_v42.pt`, 71 classes, confidence 0.20, NMS IoU 0.45, image size 640 |

HOLD takes about 0.667 s at 30 Hz, or 1.333 s at 15 Hz. In-flight RTC results are
retired, and no new VLA request is sent during HOLD. A successful close triggers a
fresh observation rather than resuming an old action suffix. Reopening rearms
wrist verification and rejects decisions captured before that reopening.

## Quick start: core only

Python 3.10+ is sufficient for the core and offline checks. Run from a source checkout:

```bash
git clone https://github.com/Curry-ZHR/RevertVLA.git
cd RevertVLA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
python -m examples.offline_recovery
python -m pytest -q
```

These commands do not load model weights, connect to a robot, or send actions.

## Model weights

Weights are GitHub Release assets, kept out of Git history. Download and restore
the fixed assets with the standard-library installer:

```bash
python3 scripts/download_weights.py
python3 -B Reverse/check_activation.py
```

The downloader verifies asset SHA-256 values and individual restored files.
`weights/assets.json` records URLs, sizes, and hashes. Downloads can be resumed
between completed assets; differing existing model files require explicit
`--force`. For assets downloaded in a browser, use `--asset-dir /path/to/assets`.
Allow approximately 5 GiB for weights plus cached archives.

```text
Reverse/checkpoints/
├── siglip/
│   ├── cc_en.32000.sentencepiece.model
│   ├── training_42sku_200steps/objects.txt
│   └── training_71sku_400steps/
│       ├── config.json
│       ├── objects.txt
│       └── checkpoint.bv-000000362/     # restored from Release
└── yolo/delivery_best71_v42/
    ├── best71_v42.pt                   # restored from Release
    ├── classes.json
    ├── postprocess.json
    └── metrics.json
```

The full SigLIP checkpoint contains 2,484 files / 2,066,163,854 bytes (~1.924 GiB),
including the original training state. YOLO weights are 61,847,354 bytes (~59 MiB).
The old 42-class text catalog is retained only for the compatibility check;
the previous step-180 weights are unnecessary. `postprocess.json` is required
with YOLO v42 and applies after NMS in its declared order.

All checkpoint bytes, labels, tokenizer, and YOLO weights were compared against
the original training-host copies before release. Portable path strings in the
training configuration and its manifest entry are adjusted for this checkout;
model tensors are unchanged.

## Perception runtime: Linux / NVIDIA CUDA

The deployed model stack uses two isolated Python 3.12 environments. Install on
the same Linux host/container as the integrating executor, with a compatible
NVIDIA driver, CUDA 12 support, and `libzstd` available:

```bash
bash Reverse/setup_runtime.sh
bash Reverse/configure.sh --check-runtime
bash Reverse/configure.sh --warmup-only
```

The installation and import check do not initialize a robot. The last command
explicitly loads SigLIP and YOLO on the GPU, reports readiness, then exits.
It acquires no camera stream and sends no robot commands. GPU inference must be
checked in the target environment; release packaging was validated with offline
tests and checkpoint checks rather than a new hardware trial.

Dependencies are pinned in `Reverse/requirements-siglip.txt` and
`Reverse/requirements-yolo.txt`. Override interpreters with
`REVERSE_SIGLIP_PYTHON` / `REVERSE_YOLO_PYTHON`, and the YOLO device with
`REVERSE_YOLO_DEVICE`. Importing the recovery modules does not start workers.

## Integrating a robot / RTC executor

See [the integration guide](docs/integration.md) for camera, action, publisher,
stop, and RTC retirement contracts. The core takes normalized seven-joint-plus-
gripper actions per hand. Only successful post-interpolation, post-limit SDK
targets enter history; an SDK acknowledgement does not establish physical arrival.

The runtime supports an injected camera reader. Robot Station's existing reader
is used only when no reader is supplied. The host must retain its original
authority, stop, speed, joint-limit, and operation-lock checks and use one action
publisher. Standard C2 RTC and synchronous `legacy_clip` execution are the
released baseline; coordinated-path reverse retiming requires host validation.

## Source layout

| Path | Purpose |
|---|---|
| `Reverse/core.py`, `gate_policy.py` | Target parsing, frame votes, sent-action history, replay planning |
| `Reverse/rolling_gate.py`, `integration.py` | Bounded HOLD and existing-executor replay adapters |
| `Reverse/runtime.py`, `model_worker.py` | Worker lifecycle, fresh camera frames, SigLIP scoring, YOLO masking |
| `Reverse/runtime/siglip/scripts/` | Training-compatible checkpoint loader and preprocessing |
| `Reverse/vendor/big_vision/` | Deployed Big Vision inference subset and upstream license |
| `Reverse/checkpoints/` | Small versioned configs, class catalogs, and YOLO postprocessing |
| `Reverse/tests/`, `tests/` | CPU-only state-machine and publication-boundary regressions |
| `examples/` | Offline walkthrough and reference UI integration |
| `docs/provenance.json` | Source and tensor provenance, plus packaging changes |

## Training configuration and evidence

The supplied SigLIP config records 400 optimizer updates, effective batch 512,
micro batch 4, 128 accumulation steps, learning rate `1e-5`, weight decay 0, and
40 warmup steps. Step 362 was selected by minimum observed **training loss**.
Private training images and the VLA training pipeline are not distributed here.

The release passes 37 offline tests and two subtests. These establish bookkeeping and protocol behavior. They do not establish
perception accuracy, physical tracking, or a task success rate. In particular,
whole-wrist matching can respond to background merchandise; this release retains
that deployed behavior. Dynamic gripper ROI / object segmentation is ongoing
experimental work and is not presented as validated recovery behavior.

## License

RevertVLA code is licensed under [Apache-2.0](LICENSE). Third-party components and
derived model assets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

# Third-party notices

- `Reverse/vendor/big_vision` contains the deployed Google Research Big Vision
  inference source subset. It retains its Apache-2.0 `LICENSE` and file-level notices.
  Upstream: https://github.com/google-research/big_vision.
- The SigLIP checkpoint fine-tunes the English WebLI SigLIP B/16-224 model.
  The SentencePiece tokenizer is distributed with the checkpoint assets.
  Upstream model documentation:
  https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip.md.
- YOLO inference uses Ultralytics, installed as a dependency rather than vendored.
  Ultralytics code and derived YOLO assets retain the applicable Ultralytics
  AGPL-3.0 / enterprise licensing terms. See https://www.ultralytics.com/license
  and https://github.com/ultralytics/ultralytics/blob/main/LICENSE.

The repository's Apache-2.0 license applies to the RevertVLA code; it does not
replace third-party licenses or grant rights to external training datasets.

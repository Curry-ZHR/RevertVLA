# Perception weights

Download with `python3 scripts/download_weights.py`, or restore these fixed v0.1.0 assets. Both SigLIP parts are independent tar archives and both are required.

| Asset | Size (bytes) | Files |
|---|---:|---:|
| [siglip362-part01.tar.gz](https://github.com/Curry-ZHR/RevertVLA/releases/download/v0.1.0/siglip362-part01.tar.gz) | 1,033,905,680 | 1243 |
| [siglip362-part02.tar.gz](https://github.com/Curry-ZHR/RevertVLA/releases/download/v0.1.0/siglip362-part02.tar.gz) | 1,033,478,955 | 1242 |
| [yolo71-v42.tar.gz](https://github.com/Curry-ZHR/RevertVLA/releases/download/v0.1.0/yolo71-v42.tar.gz) | 56,825,696 | 1 |

Checksums are in [SHA256SUMS](SHA256SUMS) and [assets.json](assets.json). The installer validates every restored checkpoint file against `Reverse/siglip362_manifest.json`, and validates the tokenizer and YOLO weights.

Keep the versioned class catalogs and YOLO postprocess configuration beside these weights. The SigLIP checkpoint is the full original training state; no optimizer arrays were removed.

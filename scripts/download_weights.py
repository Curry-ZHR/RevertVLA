#!/usr/bin/env python3
"""Download fixed Release assets, verify SHA256, then restore only declared files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expected_files(root):
    folder = "Reverse/checkpoints/siglip/training_71sku_400steps/"
    manifest = json.loads((root / "Reverse/siglip362_manifest.json").read_text())
    result = {folder + item["path"]: item["sha256"] for item in manifest["files"]
              if item["path"].startswith("checkpoint.bv-")}
    result["Reverse/checkpoints/siglip/cc_en.32000.sentencepiece.model"] = manifest["tokenizer_sha256"]
    result["Reverse/checkpoints/yolo/delivery_best71_v42/best71_v42.pt"] = (
        "720cbd8a2b56674b65cb75b3cbc64175be26b55f7690d2620042e9e51ad3a866")
    return result


def restore(archive, root, expected, *, force=False):
    root = Path(root).resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or str(name) not in expected or not member.isfile():
                raise ValueError(f"Unexpected model archive entry: {member.name}")
            target = root / str(name)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Model destination escapes the checkout")
            if target.exists():
                if digest(target) == expected[str(name)]:
                    continue
                if not force:
                    raise FileExistsError(f"Different file exists: {target}. Review it before using --force.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".download")
            with bundle.extractfile(member) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
            if digest(temporary) != expected[str(name)]:
                temporary.unlink()
                raise ValueError(f"Restored checksum mismatch: {name}")
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/weights")
    parser.add_argument("--asset-dir", type=Path, help="Use already downloaded Release assets")
    parser.add_argument("--force", action="store_true", help="Replace differing model files after review")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "weights/assets.json").read_text())
    expected = expected_files(ROOT)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for asset in metadata["assets"]:
        target = (args.asset_dir or args.cache_dir) / asset["name"]
        if not args.asset_dir and (not target.is_file() or digest(target) != asset["sha256"]):
            temporary = target.with_name(target.name + ".download")
            print(f"Downloading {asset['name']} ({asset['bytes']:,} bytes)", flush=True)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "RevertVLA/0.1.1"})
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
            temporary.replace(target)
        if not target.is_file() or target.stat().st_size != asset["bytes"] or digest(target) != asset["sha256"]:
            raise ValueError(f"Release asset checksum/size mismatch: {target}")
        restore(target, ROOT, expected, force=args.force)
        print(f"Verified and restored {asset['name']}", flush=True)
    print("Run: python3 -B Reverse/check_activation.py")


if __name__ == "__main__":
    main()

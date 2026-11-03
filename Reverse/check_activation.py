#!/usr/bin/env python3
"""Offline delivery checks; never imports model frameworks or robot interfaces."""
import argparse
import ctypes
import ctypes.util
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from Reverse.core import canonical
from Reverse.siglip_delivery import CHECKPOINT_STEP, CATALOG_RELATIVE, load_catalog, build_prompt_specs
from Reverse.yolo_delivery import load_delivery


def stored_step(checkpoint):
    data = (checkpoint / 'opt~1~0~0/0').read_bytes()
    library = ctypes.util.find_library('zstd')
    if not library:
        raise ValueError('libzstd required to verify the stored optimizer step')
    zstd = ctypes.CDLL(library)
    zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    zstd.ZSTD_decompress.restype = ctypes.c_size_t
    output = ctypes.create_string_buffer(4)
    if zstd.ZSTD_decompress(output, 4, data, len(data)) != 4:
        raise ValueError('Invalid stored optimizer counter')
    return struct.unpack('<i', output.raw)[0]


def verify(root=ROOT, *, hashes=True):
    root = Path(root)
    folder = root / Path(CATALOG_RELATIVE).parent
    checkpoint = folder / 'checkpoint.bv-000000362'
    names = load_catalog(folder / 'objects.txt')
    previous = root / 'checkpoints/siglip/training_42sku_200steps/objects.txt'
    old = [canonical(x.strip()) for x in previous.read_text().splitlines() if x.strip()]
    if len(old) != 42 or not set(old).issubset({canonical(x) for x in names}):
        raise ValueError('The 71-SKU catalog does not cover the original 42 classes')
    config = json.loads((folder / 'config.json').read_text())
    model = config['model']
    if (config['model_name'] != 'proj.image_text.two_towers' or
            model['image']['variant'] != 'B/16' or model['image']['pool_type'] != 'map' or
            model['text']['variant'] != 'B' or model['text']['vocab_size'] != 32000 or
            model['out_dim'] != [None, 768] or config['total_steps'] < CHECKPOINT_STEP):
        raise ValueError('Training architecture differs from the B16/224 scorer')
    shape_checks = {'params~txt~Embed_0~embedding': [32000,768],
                    'params~img~embedding~kernel': [16,16,3,768]}
    for node, shape in shape_checks.items():
        if json.loads((checkpoint / node / '.zarray').read_text())['shape'] != shape:
            raise ValueError('Checkpoint tensor shape mismatch: ' + node)
    arrays = list(checkpoint.glob('*/.zarray'))
    if len(arrays) != 1242:
        raise ValueError('Incomplete checkpoint tensor metadata')
    for path in arrays:
        info = json.loads(path.read_text())
        shape, chunks = info['shape'], info['chunks']
        if info['zarr_format'] != 2 or len(shape) != len(chunks) or any(n <= 0 for n in chunks):
            raise ValueError('Invalid Zarr tensor metadata')
        indices = itertools.product(*(range((n+c-1)//c) for n,c in zip(shape,chunks)))
        for ix in indices:
            chunk = path.parent / ('.'.join(map(str, ix)) if ix else '0')
            if not chunk.is_file() or chunk.stat().st_size == 0:
                raise ValueError('Missing checkpoint chunk: ' + str(chunk.relative_to(folder)))
    if stored_step(checkpoint) != CHECKPOINT_STEP:
        raise ValueError('Stored training step is not 362')
    manifest = json.loads((root / 'siglip362_manifest.json').read_text())
    layout = {f.parent.name: {k: json.loads(f.read_text())[k] for k in ('shape','dtype')}
              for f in checkpoint.glob('params*/.zarray')}
    if hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest() != manifest['parameter_layout_sha256']:
        raise ValueError('Checkpoint parameter layout differs from the validated B16 model')
    expected = {item['path']:item for item in manifest['files']}
    actual = {str(p.relative_to(folder)) for p in folder.iterdir() if p.is_file()}
    actual.update(str(Path(checkpoint.name) / p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file())
    if actual != set(expected):
        raise ValueError('Checkpoint file inventory differs from the verified delivery')
    for rel,item in expected.items():
        path = folder / rel
        if path.stat().st_size != item['bytes']:
            raise ValueError('Delivery size mismatch: ' + rel)
        if hashes:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError('Delivery hash mismatch: ' + rel)
    tokenizer = root / 'checkpoints/siglip/cc_en.32000.sentencepiece.model'
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer hash mismatch')
    yolo, yolo_names, _ = load_delivery(root / 'checkpoints/yolo/delivery_best71_v42')
    if not {canonical(x) for x in names}.issubset({canonical(x) for x in yolo_names.values()}):
        raise ValueError('SigLIP targets missing from YOLO delivery')
    return dict(ok=True, checkpoint_step=CHECKPOINT_STEP, objects=len(names), prompts=len(build_prompt_specs(names)),
                files=len(expected), bytes=sum(x['bytes'] for x in expected.values()), hashes_verified=hashes,
                stored_step_verified=True, yolo_sha256=yolo['weights_sha256'], original_source_compared=manifest.get("original_training_copy_compared", False),
                models_loaded=False, robot_commands_sent=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration-only', action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(hashes=not args.configuration_only), ensure_ascii=False))

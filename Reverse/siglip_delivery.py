"""Active SigLIP delivery metadata; safe to import without ML or robot IO."""
from pathlib import Path
from .core import canonical

ROOT = Path(__file__).resolve().parent
CHECKPOINT_STEP = 362
EXPECTED_OBJECTS = 71
EXPECTED_PROMPTS = EXPECTED_OBJECTS * 2
CATALOG_RELATIVE = "checkpoints/siglip/training_71sku_400steps/objects.txt"
DEFAULT_CATALOG = ROOT / CATALOG_RELATIVE
DEFAULT_CHECKPOINT = DEFAULT_CATALOG.parent / "checkpoint.bv-000000362"
DEFAULT_TOKENIZER = ROOT / "checkpoints/siglip/cc_en.32000.sentencepiece.model"


def load_catalog(path=DEFAULT_CATALOG):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    keys = [canonical(name) for name in names]
    if len(names) != EXPECTED_OBJECTS or len(set(keys)) != EXPECTED_OBJECTS or not all(keys):
        raise ValueError("SigLIP step362 requires exactly 71 distinct SKU labels")
    return names


def build_prompt_specs(names):
    if len(names) != EXPECTED_OBJECTS:
        raise ValueError("SigLIP step362 requires the 71-SKU catalog")
    return [dict(task_index=i*2+j, object=name, hand=hand,
                 prompt=f"Grasp {name} with the {hand} hand.")
            for i, name in enumerate(names) for j, hand in enumerate(("left", "right"))]

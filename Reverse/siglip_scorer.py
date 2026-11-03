"""Resident 71-SKU step362 scorer using the training-compatible Big Vision loader."""
from pathlib import Path
import sys
import time
import numpy as np
from .core import canonical as canonical_name
from .siglip_delivery import load_catalog, build_prompt_specs
from zerith_runtime_io_1016 import startup_status


class SiglipScorer:
    def __init__(self, runtime, checkpoint, tokenizer):
        # Reuse the training-compatible preprocessing and TensorStore loader.
        sys.path.insert(0, str(Path(runtime) / "scripts"))
        import infer_positive_48prompts as source
        import jax
        import tensorflow as tf
        self.jax, self.tf = jax, tf
        self.names = load_catalog(Path(checkpoint).parent / "objects.txt")
        self.specs = build_prompt_specs(self.names)
        workdir, checkpoint_path = source.resolve_checkpoint(checkpoint)
        config = source.load_config(workdir)
        model, params = source.load_model(config, checkpoint_path)
        self.device = jax.local_devices(backend="cuda")[0]
        # TensorStore restores committed CPU arrays. device_put without a target
        # preserves that placement and silently makes the scorer CPU-bound.
        self.params = jax.device_put(params, self.device)
        if any(leaf.devices() != {self.device} for leaf in jax.tree_util.tree_leaves(self.params)):
            raise RuntimeError("SigLIP parameters were not placed on the assigned CUDA device")
        self.image_pp, text_pp = source.make_preprocessors(str(tokenizer))
        tokens = source.preprocess_prompts([spec["prompt"] for spec in self.specs], text_pp)

        @jax.jit
        def text_embeddings(p, text):
            return model.apply({"params": p}, None, text, train=False)[1]

        catalog = text_embeddings(self.params, jax.device_put(tokens, self.device))
        catalog.block_until_ready()
        startup_status(f"[TEXT_CACHE] SigLIP {len(self.specs)}/{len(self.specs)} DONE batches=1")

        @jax.jit
        def score(p, image):
            embedding, _, extras = model.apply({"params": p}, image, None, train=False)
            return jax.nn.sigmoid((embedding @ catalog.T) * extras["t"] + extras["b"])

        self._score = score
        dummy = tf.io.encode_jpeg(tf.zeros((224, 224, 3), dtype=tf.uint8)).numpy()
        started = time.monotonic()
        startup_status("[WARMUP] SigLIP 1/1 START image=224x224")
        self.probabilities(dummy)
        self.warmup_ms = (time.monotonic() - started) * 1000
        startup_status(f"[WARMUP] SigLIP 1/1 DONE elapsed={self.warmup_ms / 1000:.3f}s")

    def probabilities(self, encoded):
        image = self.image_pp({"image": self.tf.constant(encoded)})["image"].numpy()[None]
        return np.asarray(self.jax.device_get(self._score(self.params, self.jax.device_put(image, self.device))))[0]

    def judge(self, encoded, target, hand, threshold=0.05, margin=0.01):
        scores = self.probabilities(encoded)
        indices = [i for i, spec in enumerate(self.specs) if spec["hand"] == hand]
        target_index = next(i for i in indices if canonical_name(self.specs[i]["object"]) == canonical_name(target))
        top = max(indices, key=lambda i: float(scores[i]))
        other = max(float(scores[i]) for i in indices if i != target_index)
        target_score, top_score = float(scores[target_index]), float(scores[top])
        if top == target_index:
            state = "PASS" if target_score >= threshold and target_score - other >= margin else "UNCERTAIN"
        else:
            state = "WRONG" if top_score >= threshold and top_score - target_score >= margin else "UNCERTAIN"
        return {"state": state, "target": target, "hand": hand,
                "top1_object": self.specs[top]["object"], "top1_probability": top_score,
                "target_probability": target_score, "strongest_other_probability": other,
                "threshold": threshold, "margin": margin}


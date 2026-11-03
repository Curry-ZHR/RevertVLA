# Big Vision inference subset

This directory preserves the deployed Big Vision modules needed by SigLIP B/16-224 inference: model definitions, TensorStore checkpoint loading, preprocessing and tokenization. Files retain their upstream notices and Apache-2.0 LICENSE. Private training converters, dataset reports and unrelated training/configuration code are excluded.

Upstream: https://github.com/google-research/big_vision

The included ImageNet class-name constants are a transitive import of the generic text preprocessing module; they do not add a dataset download or training requirement.

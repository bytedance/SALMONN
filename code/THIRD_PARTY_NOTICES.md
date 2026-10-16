# Code attribution

Repository-authored code uses the included Apache-2.0 LICENSE. Existing third-party headers remain in place.

- `utils/rotation_conversions.py`: Facebook/PyTorch3D rotation utilities, with their original copyright header.
- `models/vq/`: existing RVQVAE code, including attribution to vector-quantize-pytorch.
- `streaming_transformer_encoder_v2/`: SPEAR-style audio encoder using Hugging Face Llama components.
- `evaluation/face_metrics.py`: UniLS-derived formulas and vertex index sets.
- `utils/fast_render.py`, `utils/media.py`: GestureLSM renderer helpers.
- `evaluation/flame/FLAME.py`: adapter using the installed smplx skinning implementation.

SMPL-H, SMPL-X and FLAME geometry assets are obtained under their upstream agreements. Their local geometry copies are excluded from the distributable manifests; the KaoLRM mono configuration, weights, and weight license are included. See the [checkpoint asset guide](../checkpoints/README.md#external-assets-sources-and-download-guide) for sources, attribution, licenses, and geometry download/conversion steps. The separately published ELLSA2 data and checkpoint packages retain their respective upstream terms.

# SlideFuse

Official code for the paper **SlideFuse: Cross-Modal Fusion of Whole-Slide Images and Transcriptomics for Cancer Subtyping** (MICCAI 2025).

SlideFuse fuses tile-level whole-slide image embeddings with bulk
transcriptomic profiles through a gated cross-attention module, then
classifies molecular cancer subtypes from the fused representation.

## Layout

- `src/encoder.py` — WSI tile encoder wrapper
- `src/fusion.py` — gated cross-attention fusion module
- `src/data.py` — slide / expression pairing utilities
- `train.py` — training entry point
- `eval.py` — cohort-level evaluation
- `configs/base.yaml` — reference hyper-parameters

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
python eval.py --config configs/base.yaml --checkpoint checkpoints/slidefuse.pt
```

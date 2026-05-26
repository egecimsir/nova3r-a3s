# NOVA3R (minimal module fork)

A pruned, importable fork of [NOVA3R](https://github.com/wrchen530/nova3r) — keeps only what is needed to **import, run inference, and train** the model from another project. Demo scripts, evaluation pipeline, benchmark datasets, and Gradio UI have been removed.

> **NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction** — Chen, Zheng, Zhang, Vedaldi, Cremers. ICLR 2026.
> [[Paper]](https://arxiv.org/abs/2603.04179) [[Project page]](https://wrchen530.github.io/nova3r/) [[Upstream repo]](https://github.com/wrchen530/nova3r)

## Install

```bash
pip install -e .
# optional: PLY export
pip install -e ".[io]"
```

Requirements: Python 3.10+, PyTorch 2.2+ with CUDA 12.1+, an NVIDIA GPU (>=24 GB VRAM recommended).

`torch-cluster` and (optionally) `pytorch3d` must match your installed PyTorch / CUDA version — see their docs.

## Usage

```python
import nova3r

# End-to-end: image(s) -> (N, 3) numpy point cloud.
# device defaults to None (auto-picks cuda > mps > cpu).
pts = nova3r.predict(
    ckpt_path="checkpoints/scene_n1/checkpoint-last.pth",
    image_paths=["path/to/image.png"],
    resolution=(518, 392),
    output_path="output.ply",   # optional, requires nova3r[io]
)
```

Device selection is fully flexible — pass any of `None` (auto), a string, a `torch.device`, or omit entirely:

```python
import torch, nova3r

nova3r.predict(..., device=None)                       # auto: cuda > mps > cpu
nova3r.predict(..., device="cpu")
nova3r.predict(..., device="mps")                      # Apple Silicon
nova3r.predict(..., device=torch.device("cuda:1"))
```

Lower-level API:

```python
from nova3r import Nova3rImgCond, load_model, load_images, make_pairs, inference_nova3r
from nova3r.utils.device import get_default_device

device = get_default_device()
model, cfg = load_model("checkpoints/scene_n1/checkpoint-last.pth", device=device)
images = load_images(["a.png", "b.png"], size=518)
pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=False)
out = inference_nova3r(cfg, pairs, model, device=device, batch_size=1, num_queries=20000)
```

For training, instantiate `Nova3rImgCond` / `Nova3rPtsCond` directly and supply your own dataset, loss, and optimizer loop. The models' `forward` returns predictions (not a loss).

## Checkpoints

Download via [`scripts/download_checkpoints.sh`](scripts/download_checkpoints.sh) (HuggingFace).

| Model | Input | Path |
|---|---|---|
| `Nova3rPtsCond` (AE) | point cloud | `checkpoints/scene_ae/` |
| `Nova3rImgCond` (N=1) | 1 image | `checkpoints/scene_n1/` |
| `Nova3rImgCond` (N=2) | 2 images | `checkpoints/scene_n2/` |

Each checkpoint directory must contain `.hydra/config.yaml` (the downloader handles this).

## Layout

```
nova3r/
  inference.py          # inference_nova3r + flow-matching glue
  io.py                 # load_images, make_pairs, save_pointcloud_ply, load_model, predict
  models/               # Nova3rImgCond, Nova3rPtsCond, BatchModelWrapper, aggregator
  heads/                # DPT head, pts3d encoder/decoder, TripoSG AE wrapper
  layers/               # transformer blocks, attention, rope, etc.
  flow_matching/        # paths, schedulers, ODE solver
  utils/                # device, geometry, misc, image, image_pairs, sampling
  _vendor/
    croco/models/blocks.py
    triposg/            # full vendored package
```

## Citation

```bibtex
@inproceedings{chennova3r,
  title={NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction},
  author={Chen, Weirong and Zheng, Chuanxia and Zhang, Ganlin and Vedaldi, Andrea and Cremers, Daniel},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## License

NOVA3R is Apache 2.0 (see `LICENSE`). Vendored third-party code retains its original licenses (see `NOTICES`):
- CroCo (`nova3r/_vendor/croco/`) — CC BY-NC-SA 4.0
- TripoSG (`nova3r/_vendor/triposg/`) — MIT
- DUSt3R-derived utilities (`nova3r/utils/`) — CC BY-NC-SA 4.0

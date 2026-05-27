# NOVA3R (minimal module fork)

A pruned, importable fork of [NOVA3R](https://github.com/wrchen530/nova3r) — keeps only what is needed to **import, run inference, and train** the model from another project. Demo scripts, evaluation pipeline, benchmark datasets, and Gradio UI have been removed.

> **NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction** — Chen, Zheng, Zhang, Vedaldi, Cremers. ICLR 2026.
> [[Paper]](https://arxiv.org/abs/2603.04179) [[Project page]](https://wrchen530.github.io/nova3r/) [[Upstream repo]](https://github.com/wrchen530/nova3r)

## Install

### Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.2+ |
| GPU (recommended) | NVIDIA with CUDA 12.1+, ≥24 GB VRAM (48 GB for the largest checkpoints) |
| Apple Silicon | works via MPS for inference (slower, ≥32 GB unified memory recommended) |
| CPU-only | works for inference but is very slow |

### Quick install (development checkout)

Create and activate a virtual environment (on Windows, use `.venv\Scripts\activate` instead of the `source` line):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch for your platform — pick the matching line from the [PyTorch install picker](https://pytorch.org/get-started/locally/).

CPU / macOS-MPS:

```bash
pip install torch torchvision
```

CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install `nova3r`:

```bash
pip install -e .
```

Optional extras — `[io]` pulls in `open3d` for PLY export, `[sampling]` pulls in `pytorch3d` for the FPS / k-NN sampling code paths:

```bash
pip install -e ".[io]"
pip install -e ".[sampling]"
```

### Platform-specific notes

- **`torch-cluster`** is a required runtime dependency and ships only as a source/wheel build that must match your PyTorch + CUDA. If `pip install -e .` fails on it, install the matching wheel from the [PyG wheel index](https://data.pyg.org/whl/) first:

  ```bash
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  ```

- **`pytorch3d`** has no universal PyPI wheel. Install only if you need the FPS / k-NN sampling paths in `nova3r.utils.sampling`. See the [pytorch3d install guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
- **Apple Silicon (MPS)** works out of the box for inference (`device="mps"`); skip the CUDA-only wheels. `torch-cluster` does not run on MPS — keep tensors that pass through sampling code on CPU.
- **`open3d`** is large (hundreds of MB). It is only needed for `save_pointcloud_ply` / `predict(output_path=...)`. Skip the `[io]` extra if you do your own PLY writing (e.g. with `plyfile`).

### Verify install

```bash
python -c "import nova3r; print(nova3r.get_default_device())"
```

## Use in a downstream project

You do not need to clone this repo. Below is a full walkthrough of consuming `nova3r` from a fresh downstream project.

Create the project and a virtual environment:

```bash
mkdir my-3d-project
cd my-3d-project
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch first (pick the line that matches your platform from the [PyTorch install picker](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision
```

Install `nova3r` straight from git (pin a tag or commit with `@v0.1.0` for reproducibility):

```bash
pip install "nova3r @ git+https://github.com/<you>/nova3r-a3s.git"
```

This also puts a `nova3r-download` CLI on your `PATH`. Download the checkpoints you need into your project (anywhere you like — `--dest` is honored verbatim):

```bash
nova3r-download --dest ./checkpoints
```

Write a minimal `main.py`:

```python
import nova3r

pts = nova3r.predict(
    ckpt_path="./checkpoints/scene_n1/checkpoint-last.pth",
    image_paths=["./input.png"],
    resolution=(518, 392),
    num_queries=20000,
    output_path="./output.ply",
)
print("predicted", pts.shape, "points")
```

Run it:

```bash
python main.py
```

`nova3r.predict` auto-picks the best device (CUDA > MPS > CPU). Pass `device="cpu"` etc. to override — see [Device selection](#device-selection).

### Download checkpoints

Checkpoints are fetched separately and land wherever **you** specify — they never live inside the installed `nova3r` package.

If the HuggingFace repo is gated or your `HF_TOKEN` is not set, log in once:

```bash
huggingface-cli login
```

Download all models into the default `./checkpoints` folder of the current directory:

```bash
nova3r-download
```

Download a single model into a custom path:

```bash
nova3r-download --model scene_n1 --dest ./assets/ckpts
```

Force redownload (overwrite existing files):

```bash
nova3r-download --force
```

Print the full set of options:

```bash
nova3r-download --help
```

Programmatic equivalent:

```python
import nova3r
nova3r.download_checkpoints(model="scene_n1", dest="./assets/ckpts")
```

Each model lands at `<dest>/<model>/checkpoint-last.pth` together with its `.hydra/config.yaml` sidecar (required by `load_model`).

## Usage

### Quick start

End-to-end: image(s) in, `(N, 3)` numpy point cloud out. `device=None` means auto (CUDA > MPS > CPU).

```python
import nova3r

pts = nova3r.predict(
    ckpt_path="./checkpoints/scene_n1/checkpoint-last.pth",
    image_paths=["./input.png"],
    resolution=(518, 392),
    num_queries=20000,
    output_path="./output.ply",
)
print(pts.shape)
```

Pass 1 image for single-view, 2 for multi-view. Use `(518, 392)` as `(width, height)` for the released checkpoints. `output_path` is optional and requires the `[io]` extra.

### Device selection

Every entry point accepts `device=None` (auto), a string, a `torch.device`, or a tensor:

```python
import torch
import nova3r

nova3r.predict(..., device=None)
nova3r.predict(..., device="cpu")
nova3r.predict(..., device="mps")
nova3r.predict(..., device=torch.device("cuda:1"))
```

You can also resolve the default explicitly:

```python
from nova3r.utils.device import get_default_device, resolve_device

device = get_default_device()
device = resolve_device("cuda:0")
device = resolve_device(some_tensor)
```

### Lower-level inference API

Use this when you want full control over preprocessing, batching, or want to share a single loaded model across many calls.

```python
from nova3r import load_model, load_images, make_pairs, inference_nova3r, save_pointcloud_ply
from nova3r.utils.device import get_default_device

device = get_default_device()

model, cfg = load_model("./checkpoints/scene_n1/checkpoint-last.pth", device=device)
model.eval()

images = load_images(["./a.png", "./b.png"], size=518)
pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=False)

out = inference_nova3r(
    cfg, pairs, model, device=device,
    batch_size=1, num_queries=20000,
    method=cfg.get("fm_sampling", "euler"),
)
pts3d = out["pred"]["pts3d_xyz"][0].cpu().numpy()

save_pointcloud_ply(pts3d, "./output.ply")
```

### Direct model construction

If you don't want to go through `load_model` (e.g. for tests or to override config), instantiate the model class directly with the params from `cfg.model.params`:

```python
import torch
from omegaconf import OmegaConf
from nova3r import Nova3rImgCond, get_default_device

device = get_default_device()
cfg = OmegaConf.load("./checkpoints/scene_n1/.hydra/config.yaml").experiment

model = Nova3rImgCond(**cfg.model.params).to(device)

state = torch.load("./checkpoints/scene_n1/checkpoint-last.pth", map_location=device)
model.load_state_dict(state["model"], strict=True)
model.eval()
```

### Training

This package ships the **model definitions only** — no trainer, no optimizer loop, no dataloader. Bring your own dataset and training loop. The snippet below assumes a flat-dict batch (`{str: Tensor}`); replace the `.to(device)` line with your own helper if your batches are nested.

```python
import torch
from torch.utils.data import DataLoader
from nova3r import Nova3rImgCond
from nova3r.utils.device import get_default_device, autocast

device = get_default_device()
model = Nova3rImgCond(**your_model_params).to(device)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
loader = DataLoader(YourDataset(...), batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with autocast(device, dtype=torch.bfloat16):
        pred = model(batch)
        loss = your_loss_fn(pred, batch)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
```

`model.forward(...)` returns predictions; design your own loss (e.g. Chamfer + flow-matching velocity loss as in the paper).

### API reference (public surface)

```python
nova3r.Nova3rImgCond
nova3r.Nova3rPtsCond
nova3r.BatchModelWrapper
nova3r.inference_nova3r(cfg, pairs, model, device, batch_size=1, num_queries=20000, method="euler")
nova3r.load_model(ckpt_path, device=None)
nova3r.predict(ckpt_path, image_paths, device=None, resolution=(518, 392), num_queries=20000, output_path=None)
nova3r.load_images(paths, size=...)
nova3r.make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=False)
nova3r.save_pointcloud_ply(pts, path)
nova3r.download_checkpoints(model="all", dest="./checkpoints", repo="wrchen530/nova3r", force=False)
nova3r.get_default_device()
nova3r.resolve_device(device)
nova3r.autocast(device, dtype=None, enabled=True)
```

### Troubleshooting

- **`ModuleNotFoundError: torch_cluster`** — install the wheel matching your PyTorch + CUDA from <https://data.pyg.org/whl/>.
- **`ImportError: save_pointcloud_ply requires open3d`** — install the io extra (`pip install -e ".[io]"`) or pass `output_path=None` and save the returned array yourself.
- **`FileNotFoundError: No .hydra/config.yaml found`** — your checkpoint directory is missing the Hydra sidecar. Re-fetch with `nova3r-download --force`, or construct the model directly (see "Direct model construction").
- **`KeyError: Unknown model class 'X'`** — the checkpoint's `cfg.model.name` is not in `nova3r.io._MODEL_REGISTRY`. Only `Nova3rImgCond` and `Nova3rPtsCond` are exposed by default; extend the registry if you trained a custom subclass.
- **HuggingFace 401 / gated repo** — run `huggingface-cli login`, or export `HF_TOKEN=...` before invoking `nova3r-download`.
- **bf16 autocast errors on MPS** — `nova3r.utils.device.autocast` already disables this automatically; if you call `torch.amp.autocast` yourself, use `dtype=torch.float16` or omit autocast on MPS.

## Checkpoints

Download via the bundled `nova3r-download` CLI (see [Download checkpoints](#download-checkpoints)).

| Model | Input | Path (after `nova3r-download`) |
|---|---|---|
| `Nova3rPtsCond` (AE) | point cloud | `<dest>/scene_ae/` |
| `Nova3rImgCond` (N=1) | 1 image | `<dest>/scene_n1/` |
| `Nova3rImgCond` (N=2) | 2 images | `<dest>/scene_n2/` |

`<dest>` defaults to `./checkpoints` and is overridable with `--dest`. Each directory contains `checkpoint-last.pth` + `.hydra/config.yaml`.

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
  scripts/              # download_checkpoints (exposes the nova3r-download CLI)
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

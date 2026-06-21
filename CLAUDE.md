# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch Lightning implementation of **MVSNeRF** (ICCV 2021): a generalizable radiance field that reconstructs a novel scene from only 3 input views by combining Multi-View Stereo with NeRF. A pretrained generalizable model can render novel views of an unseen scene directly, or be fine-tuned per-scene for higher quality.

The whole pipeline is **CUDA-only**: `InPlaceABN` requires a GPU, and `Embedder` calls `.cuda()` at construction time. There is no CPU fallback.

## Environment

Tested on Ubuntu 20.04 + PyTorch 1.10.1 + PyTorch Lightning 1.3.5 (pinned — newer Lightning breaks the `Trainer`/`LightningModule` API used here).

```
conda create -n mvsnerf python=3.8 && conda activate mvsnerf
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch-lightning==1.3.5 imageio pillow scikit-image opencv-python configargparse lpips kornia warmup_scheduler matplotlib test-tube imageio-ffmpeg
```
Also required (not in README): `inplace-abn` (cost-volume norm), `kornia` (`create_meshgrid` for homography warping). The experimental `*_plF.py` variant additionally imports `jax`.

## Commands

There is no test suite, linter, or build step — this is a research training/rendering codebase driven by CLI scripts and notebooks.

**Train the generalizable model** (DTU):
```
CUDA_VISIBLE_DEVICES=0 python train_mvs_nerf_pl.py --expname myexp \
    --num_epochs 6 --use_viewdirs --dataset_name dtu --datadir $DTU_DIR \
    --with_depth --imgScale_test 1.0 --N_samples 128 --batch_size 1024 --N_vis 6 --pad 0
```

**Fine-tune on one scene** (blender / llff / dtu_ft), starting from the pretrained checkpoint:
```
CUDA_VISIBLE_DEVICES=0 python train_mvs_nerf_finetuning_pl.py \
    --dataset_name blender --datadir /path/to/nerf_synthetic/lego \
    --expname lego-ft --with_rgb_loss --batch_size 1024 --num_epochs 1 \
    --imgScale_test 1.0 --white_bkgd --pad 0 --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1
```
`--pad` differs by dataset: **0** for DTU/blender, **24** for LLFF. LLFF also needs `--use_disp`; blender needs `--white_bkgd`.

**Batch experiments:** `run_batch.py` is a scratch script of commented-out `os.system(...)` commands looping over scenes — edit and run it, don't treat it as a stable entrypoint.

**Render / evaluate:** Jupyter notebooks, not scripts. `renderer.ipynb` = quantitative eval + image rendering; `renderer_video.ipynb` = free-viewpoint video.

A pretrained generalizable checkpoint ships at `ckpts/mvsnerf-v0.tar` (trained with `--net_type v0`).

## Architecture

The model is a **two-stage cascade**, both stages defined in `models.py`:

1. **`MVSNet`** (the "encoding" net) turns 3 posed source images into a **neural encoding volume**: `FeatureNet` (FPN, 1/4 res 32-ch features) → differentiable **homography warping** of source features into the reference frustum across 128 depth planes (`utils.homo_warp`) → variance-based cost volume, concatenated with warped RGB → `CostRegNet` (3D U-Net) → a `(1, C, D, H, W)` feature volume in the reference view's NDC space.
2. **`MVSNeRF`** (the radiance MLP) renders rays. For each 3D sample it tri-linearly samples the encoding volume at the point's NDC coordinate (`utils.index_point_feature` / `RefVolume`), concatenates positionally-encoded position, view direction, and per-view colors looked up from source images (`utils.build_color_volume`), and regresses RGBσ.

`MVSNeRF` selects one MLP variant via `--net_type`: **`v0` = `Renderer_ours`** (multiplicative feature bias; this is what the shipped checkpoint uses), `v1` = `Renderer_attention`, `v2` = `Renderer_linear` (additive bias; the class default). `create_nerf_mvs()` wires the model, the positional embedders, the `network_query_fn` closure, and checkpoint loading; it returns `render_kwargs_train/test`.

`renderer.py` is the volumetric-rendering glue: `rendering()` assembles per-point features (`gen_pts_feats`) and runs the integral; `run_network_mvs` (the `network_query_fn`) chunks points through the MLP; `raw2outputs`/`raw2alpha` do alpha compositing.

`utils.py` holds everything coordinate/geometry: ray generation (`build_rays`, `build_rays_test`), world→reference-NDC projection (`get_ndc_coordinate`), `homo_warp`, color-volume construction, and camera-path generators for video rendering.

### Two training paths with *different dataset contracts* (important)

The two training scripts consume datasets through **incompatible interfaces** — this is the main source of confusion when editing data loaders:

- **Generalization** (`train_mvs_nerf_pl.py`): constructs `dataset(root_dir=..., split=..., max_len=..., downSample=...)`. Each item is a full multi-view bundle (`images`, `proj_mats`, `near_fars`, `depths_h`, `w2cs`, `c2ws`, `intrinsics`). The DataLoader batch is **1 scene**; `--batch_size` is the number of **rays** sampled per step via `build_rays`. The encoding volume is recomputed by `MVSNet` every step.
- **Fine-tuning** (`train_mvs_nerf_finetuning_pl.py`): constructs `dataset(args, split=...)`. Each item is a single ray (`rays`, `rgbs`), and the DataLoader batch *is* `--batch_size` rays. The encoding volume is built **once** in `init_volume()` from `dataset.read_source_views()`, then wrapped in `RefVolume` as a **learnable `nn.Parameter`** and optimized along with the MLP. Rays are marched with `data/ray_utils.ray_marcher`.

`data/__init__.py` maps names to classes: `dtu` → `MVSDatasetDTU` (generalization contract), and `llff`/`blender`/`dtu_ft` → `LLFFDataset`/`BlenderDataset`/`DTU_ft` (fine-tuning contract). Adding a dataset means matching the contract of the script you intend to run it with.

## Conventions & gotchas

- `args.feat_dim` is **hardcoded in code** (`8 + 3*4 = 20`) inside both `MVSSystem.__init__`, not via CLI — 8 encoding-volume channels + 3 source views × (RGB + in-view mask).
- Outputs: generalization logs/checkpoints go to `runs_new/<expname>/`, fine-tuning to `runs_fine_tuning/<expname>/`. Checkpoints are `.tar` dicts with `network_fn_state_dict`, `network_mvs_state_dict`, and (fine-tuning only) `volume` — see `save_ckpt`.
- `--use_color_volume` / `--use_density_volume` change how the per-point features are assembled (project colors into the volume vs. index images per-query; density-guided importance sampling). Only the fine-tuning path supports them.
- `train_mvs_nerf_finetuning_plF.py` is an **untracked experimental variant** adding a JAX-based occlusion-regularization loss (`lossfun_occ_reg`) on top of `train_mvs_nerf_finetuning_pl.py`. Treat `_pl.py` as canonical.
- `train_mvs_nerf_fusion_finetuning_pl.py` is a separate fine-tuning variant using the color-fusion renderer.
- Custom-data rendering expects a right-handed, OpenCV-format camera convention (intrinsics + near/far + extrinsics as c2w or w2c).
- `configs/pairs.th` and `configs/dtu_pairs.txt` define source-view selection; `configs/lists/` holds the DTU train/val/test scan splits.

# Copyright (c) 2026 Weirong Chen
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""Legacy inference glue for ``Nova3r{Img,Pts}Cond``.

Verbatim move of ``loss_of_one_batch_lari``, ``loss_of_one_batch_demo`` and the
original ``inference_nova3r`` runner. These all depend on the Hydra ``args``
config, the upstream ``BatchModelWrapper`` and ``model._encode`` glue, and are
preserved here for backward compatibility only. The Nova3r-compatible
replacement lives in :func:`nova3r.inference.inference_nova3r`.

Shared data utilities (``get_all_pts3d``, ``normalize_input``,
``check_if_same_size``, ``amp_dtype_mapping``) are imported from
:mod:`nova3r.inference` so the implementations live in exactly one place.
"""

import tqdm
import torch

from nova3r.flow_matching.solver import ODESolver
from nova3r.inference import (
    amp_dtype_mapping,
    check_if_same_size,
    get_all_pts3d,
    normalize_input,
)
from nova3r.modules.model_wrapper import BatchModelWrapper
from nova3r.utils.device import autocast, collate_with_cat, to_cpu

__all__ = [
    "loss_of_one_batch_lari",
    "loss_of_one_batch_demo",
    "inference_nova3r",
]


def loss_of_one_batch_lari(args, batch, model, criterion, device, use_amp=False, ret=None, num_queries=20000, model_wrapper=None, **kwargs):
    """Compute evaluation metrics (Chamfer Distance, F-Score) for trained models."""
    ignore_keys = set(['dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'view_label'])

    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    token_mask = None

    img_src_list = []
    for view in batch:
        if 'input' in view['view_label'][0]:
            view['img'] = view['img'] * 0.5 + 0.5
            img = view['img']
            img_src_list.append(img)

    images = torch.stack(img_src_list, dim=1)

    # prepare the diffusion data
    if 'query_source' in args.model.params.cfg.pts3d_head.params:
        query_src = args.model.params.cfg.pts3d_head.params.query_source
    else:
        query_src = 'src_complete'

    if 'down_resolution' in args.model.params.cfg.pts3d_head.params:
        down_resolution = args.model.params.cfg.pts3d_head.params.down_resolution
    else:
        down_resolution = 224

    if 'norm_mode' in args.model.params.cfg.pts3d_head.params:
        norm_mode = args.model.params.cfg.pts3d_head.params.norm_mode
    else:
        norm_mode = 'none'

    if 'fm_sampling' in args:
        fm_sampling = args.fm_sampling
    else:
        fm_sampling = 'euler'

    pts3d_src, valid_src = get_all_pts3d(batch, mode=query_src, down_resolution=down_resolution)

    pts3d_src_norm, _ = normalize_input(pts3d_src, valid_src, pts3d_src, valid_src, mode=norm_mode)


    def process_point(args, images, pts3d_src, token_mask, num_queries, device, model_wrapper=None, method='euler'):
        step_size = args.fm_step_size
        num_steps = int(1 // args.fm_step_size)
        point_size = num_queries

        cfg_scale = args.cfg_scale if 'cfg_scale' in args else 1.0
        B = images.shape[0]

        T = torch.linspace(0, 1, num_steps).to(device)
        if model_wrapper is not None:
            wrapped_vf = model_wrapper
        else:
            wrapped_vf = BatchModelWrapper(model=model)
        wrapped_vf.eval()

        model.eval()
        x_init = torch.rand((B, point_size, 3), dtype=torch.float32, device=device) * 2 - 1
        solver = ODESolver(velocity_model=wrapped_vf)

        if hasattr(model, 'module'):
            encoder_data = model.module._encode(images=images, pointmaps=pts3d_src, test=True, cfg_scale=cfg_scale)
        else:
            encoder_data = model._encode(images=images, pointmaps=pts3d_src, test=True, cfg_scale=cfg_scale)

        sol = solver.sample(
            time_grid=T,
            x_init=x_init.detach(),
            method=method,
            step_size=step_size,
            return_intermediates=True,
            images=images.detach(),
            token_mask=token_mask,
            encoder_data=encoder_data,
            pointmaps=pts3d_src.detach()
        )
        return sol

    with torch.no_grad():
        with autocast(device, dtype=amp_dtype_mapping[args.amp_dtype]):
            pts3d_xyz_list = process_point(args, images, pts3d_src_norm, token_mask, num_queries, device, model_wrapper=model_wrapper, method=fm_sampling)

    pts3d_xyz = pts3d_xyz_list[-1]
    pred_dict = {}
    pred_dict['pts3d_xyz'] = pts3d_xyz
    pred_dict['pts3d_xyz_list'] = pts3d_xyz_list
    pred_dict["images"] = images

    pred_dict['input_pts3d'] = pts3d_src
    pred_dict['input_valid'] = valid_src

    if criterion is not None:
        with autocast(device, dtype=amp_dtype_mapping[args.amp_dtype], enabled=bool(use_amp)):
            pts3d_data, loss = criterion(batch, pred_dict)
    else:
        pts3d_data, loss = None, None

    result = dict(view=batch, pred=pred_dict, data=pts3d_data, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_demo(args, batch, model, criterion, device, use_amp=False, ret=None, num_queries=20000, model_wrapper=None, n_views=2, method='euler', pointmaps=None):
    """Run inference for demo/visualization -- generates 3D point clouds from images."""
    ignore_keys = set(['dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'view_label'])

    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    token_mask = None

    img_src_list = []
    for view in batch:
        if 'input' in view['view_label'][0]:
            view['img'] = view['img'] * 0.5 + 0.5
            img = view['img']
            img_src_list.append(img)


    images = torch.stack(img_src_list, dim=1)
    images = images[:, :n_views, ...]  # Use only n_views

    def process_point(args, images, pts3d_src, token_mask, num_queries, device, model_wrapper=None, method='euler'):
        step_size = args.fm_step_size
        num_steps = int(1 // args.fm_step_size)
        point_size = num_queries

        B = images.shape[0]

        T = torch.linspace(0, 1, num_steps).to(device)
        if model_wrapper is not None:
            wrapped_vf = model_wrapper
        else:
            wrapped_vf = BatchModelWrapper(model=model)
        wrapped_vf.eval()

        model.eval()
        x_init = torch.rand((B, point_size, 3), dtype=torch.float32, device=device) * 2 - 1
        solver = ODESolver(velocity_model=wrapped_vf)

        if hasattr(model, 'module'):
            encoder_data = model.module._encode(images=images, pointmaps=pts3d_src)
        else:
            encoder_data = model._encode(images=images, pointmaps=pts3d_src)

        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=method,
            step_size=step_size,
            return_intermediates=False,
            images=images,
            token_mask=token_mask,
            encoder_data=encoder_data,
            pointmaps=pts3d_src
        )
        return sol[-1] if isinstance(sol, list) else sol

    if pointmaps is not None:
        norm_mode = args.model.params.cfg.pts3d_head.params.get('norm_mode', 'none')
        valid = torch.ones(pointmaps.shape[0], pointmaps.shape[1], dtype=torch.bool, device=device)
        pointmaps, _ = normalize_input(pointmaps, valid, pointmaps, valid, mode=norm_mode)

    with torch.no_grad():
        with autocast(device):
            pts3d_xyz = process_point(args, images, pointmaps, token_mask, num_queries, device, model_wrapper=model_wrapper, method=method)

    pred_dict = {}
    pred_dict['pts3d_xyz'] = pts3d_xyz
    pred_dict["images"] = images

    result = dict(view=batch, pred=pred_dict)
    return result[ret] if ret else result



@torch.no_grad()
def inference_nova3r(args, pairs, model, device, batch_size=8, verbose=True, num_queries=20000, n_views=2, method='euler', pointmaps=None):
    """Run batched NOVA3R inference over a list of image pairs (legacy).

    Iterates over ``pairs`` in chunks of ``batch_size``, runs the model under
    :func:`torch.no_grad`, and concatenates the per-batch outputs. ``args`` is
    the OmegaConf experiment config returned by
    :func:`nova3r._legacy.io.load_model`.
    """
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch_demo(args, collate_with_cat(pairs[i:i + batch_size]), model, None, device, num_queries=num_queries, n_views=n_views, method=method, pointmaps=pointmaps)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result

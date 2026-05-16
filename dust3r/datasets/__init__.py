# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .scrream import SCRREAM  # noqa
from .scrream_lari_multi_view import SCRREAM_MULTI  # noqa


class DynamicBatchDatasetWrapper:
    """
    Wrapper dataset that handles DynamicBatchedMultiFeatureRandomSampler output.

    The dynamic sampler returns batches (lists of tuples) instead of individual samples.
    This wrapper ensures that the underlying dataset's __getitem__ method gets called
    with individual tuples as expected.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped dataset
        return getattr(self.dataset, name)


def cut3r_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank):
    """
    Create data loader for CUT3R format datasets (Multi datasets).
    Uses batch sampler instead of regular sampler.
    """
    import torch
    
    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                      rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    
    return data_loader


def wai_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank, max_num_of_images_per_gpu):
    """
    Create data loader for WAI format datasets.
    Uses batch_sampler for dynamic batch sizes based on num_views.
    """
    import torch
    
    shuffle = True # always shuffle for WAI datasets
    sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                  rank=rank, drop_last=drop_last, max_num_of_images_per_gpu=max_num_of_images_per_gpu)
    
    # Use batch_sampler to ensure all samples in a batch have the same num_views
    wrapped_dataset = DynamicBatchDatasetWrapper(dataset)

    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    
    return data_loader


def dust3r_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank):
    """
    Create data loader for standard DUST3R format datasets.
    Uses standard sampler and batch_size parameter.
    """
    import torch
    
    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                      rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )
    
    return data_loader


def get_test_data_loader(
    args, dataset, batch_size, num_workers=8, shuffle=False, drop_last=False, pin_mem=True
):
    "Get simple PyTorch dataloader corresponding to the testing dataset"
    from croco.utils.misc import get_world_size, get_rank

    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader


def get_data_loader(args, dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    """
    Main data loader factory function.
    Dispatches to appropriate loader based on dataset type.
    """
    from croco.utils.misc import get_world_size, get_rank

    # Determine dataset format
    use_cut3r_format = 'Multi' in dataset
    use_wai_format = 'WAI' in dataset

    # Evaluate string datasets
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # Dispatch to appropriate loader
    if use_cut3r_format:
        return cut3r_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank)
    elif use_wai_format:
        max_num_of_images_per_gpu = args.get('max_num_of_images_per_gpu', 4)
        return wai_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank, max_num_of_images_per_gpu)
    else:
        return dust3r_loader(dataset, batch_size, num_workers, shuffle, drop_last, pin_mem, world_size, rank)


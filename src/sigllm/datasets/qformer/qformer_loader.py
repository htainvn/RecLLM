"""Shared DataLoader factories for Q-Former alignment pipelines.

Used by both stage 1 representation training and stage 2 generative
pretraining so the dataset/collate/loader plumbing stays in one place.
"""

import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


FilterFn = Callable[[QFormerAlignmentDataset], Dataset]


def qformer_collate(batch):
    """Stack tensors, keep other fields as lists.

    ``his`` (variable-length history id lists) is padded here, on CPU, into a
    single ``[B, L]`` LongTensor (0 = MF padding index). Padding per batch in
    the loss with ``torch.tensor(row, device=cuda)`` issued one small H2D copy
    per row (~350/step at batch 1024); a single stacked tensor moves once with
    the rest of the batch. Rows without history are all-zero and mask out
    downstream.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "his":
            max_len = max(max((len(b[k]) for b in batch), default=1), 1)
            his_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
            for row, b in enumerate(batch):
                if b[k]:
                    his_pad[row, : len(b[k])] = torch.tensor(b[k], dtype=torch.long)
            out[k] = his_pad
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def build_qformer_loader(
    cfg,
    filename: str,
    shuffle: bool,
    filter_fn: Optional[FilterFn] = None,
    permute_seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a single Q-Former DataLoader from a samples pickle.

    Parameters
    ----------
    cfg
        Config namespace with ``batch_size`` and ``num_workers`` attributes.
    filename
        Path to a ``.pkl`` file produced by ``QFormerAlignmentBuilder``.
    shuffle
        Whether to shuffle the loader.
    filter_fn
        Optional callable that receives the loaded ``QFormerAlignmentDataset``
        and returns a (possibly subsetted) ``Dataset``. Use this in stage 2
        to keep only ``item_text`` samples.
    permute_seed
        Apply a single fixed permutation to the dataset before batching, then
        iterate in that order. Use on eval loaders so their batches carry the
        same *mix* of sample types as shuffled train batches.

        The builder writes samples in contiguous per-type blocks, so an
        unpermuted eval loader yields near-single-type batches: the item_text
        block fills whole batches (~250 rows) while train's shuffled batches
        hold only ~15. In-batch contrastives draw negatives from same-type
        rows in the batch, so chance accuracy is ``1/n`` — a 250-vs-15 split
        makes train and eval retrieval numbers differ by ~17x for reasons
        that have nothing to do with model quality.

        Preferred over ``shuffle=True`` for eval: a one-shot permutation keeps
        the ordering identical across epochs, so epoch-to-epoch validation
        deltas reflect the model rather than resampling noise (which matters
        because early stopping reads those deltas).
    drop_last
        Drop the final short batch. Set this on loaders whose metrics are read
        as in-batch retrieval numbers: chance accuracy is ``1/n``, so a ragged
        tail batch contributes rows measured against a different (easier)
        chance level and silently shifts the average. With it on, EVERY batch
        holds exactly ``cfg.batch_size`` rows, which is what makes the numbers
        comparable across splits of different sizes.
    """
    dataset: Dataset = QFormerAlignmentDataset(filename=filename)
    if filter_fn is not None:
        dataset = filter_fn(dataset)
    if permute_seed is not None:
        generator = torch.Generator().manual_seed(int(permute_seed))
        order = torch.randperm(len(dataset), generator=generator).tolist()
        dataset = Subset(dataset, order)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        drop_last=bool(drop_last),
        collate_fn=qformer_collate,
        num_workers=int(cfg.num_workers),
    )


def build_qformer_loaders(
    cfg,
    data_dir: str,
    train_filename: str = "train_qformer_ood2.pkl",
    val_filename: str = "valid_qformer_ood2.pkl",
    test_filename: Optional[str] = "test_qformer_ood2.pkl",
    filter_fn: Optional[FilterFn] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test loaders for the Q-Former alignment task.

    Val/test get a fixed permutation (``run.*.eval_permute_seed``, default 0)
    so their batches hold the same sample-type mix as shuffled train batches
    and the in-batch contrastive metrics are directly comparable across
    splits. Set ``eval_permute_seed: null`` to restore the old block-ordered
    behaviour.

    Pass ``test_filename=None`` if the calling stage has no test split
    (stage 2 generative pretraining); the returned ``test_loader`` is
    ``None`` in that case.
    """
    eval_permute_seed = cfg.get("eval_permute_seed", 0) if hasattr(cfg, "get") else 0

    train_loader = build_qformer_loader(
        cfg, filename=os.path.join(data_dir, train_filename), shuffle=True, filter_fn=filter_fn
    )
    val_loader = build_qformer_loader(
        cfg,
        filename=os.path.join(data_dir, val_filename),
        shuffle=False,
        filter_fn=filter_fn,
        permute_seed=eval_permute_seed,
    )
    test_loader = None
    if test_filename is not None:
        test_loader = build_qformer_loader(
            cfg,
            filename=os.path.join(data_dir, test_filename),
            shuffle=False,
            filter_fn=filter_fn,
            permute_seed=eval_permute_seed,
        )
    return train_loader, val_loader, test_loader
